# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.models.hy_v4.nvidia import attention as native_attention
from vllm.models.hy_v4.nvidia import mtp as native_mtp

from vllm_ascend.patch.worker import patch_hy_v4
from vllm_ascend.patch.worker.patch_hy_v4 import AscendHYV4MLAAttention, HeadwiseGate
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.configs.modelopt_mxfp8_config import AscendModelOptMxFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantization", [None, "modelopt_mxfp8", "fp8", "ascend"])
def test_native_constructor_preserves_projections_and_quantization(dtype, quantization):
    """Keep the same projection owners for floating and quantized models."""
    if quantization == "modelopt_mxfp8":
        quant_config = AscendModelOptMxFp8Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": []},
            }
        )
    elif quantization == "fp8":
        quant_config = AscendFp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    elif quantization == "ascend":
        quant_config = AscendModelSlimConfig({"model.layers.1.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"})
    else:
        quant_config = None

    class Projection(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            self.quant_config = quant_config

    class Wrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.modules = args[8]
            self.quant_config = args[10]
            self.topk_indices_buffer = self.modules.topk_indices_buffer
            self.mla_attn = SimpleNamespace(impl=SimpleNamespace())

    config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["sparse_attention", "sparse_attention"],
        indexer_types=["full", "shared"],
        index_topk=4,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
    )
    parallel = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    topk = torch.zeros(2, 4, dtype=torch.int32)
    with ExitStack() as stack:
        for name in (
            "ColumnParallelLinear",
            "MergedColumnParallelLinear",
            "ReplicatedLinear",
            "RowParallelLinear",
            "RMSNorm",
        ):
            stack.enter_context(patch.object(native_attention, name, Projection))
        stack.enter_context(patch.object(native_attention, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(native_attention, "get_attn_backend"))
        stack.enter_context(patch.object(native_attention, "get_rope", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(native_attention, "hpc_gated_mla_supported", return_value=False))
        stack.enter_context(
            patch(
                "vllm_ascend.patch.worker.patch_hy_v4.get_current_vllm_config",
                return_value=SimpleNamespace(parallel_config=parallel),
            )
        )
        stack.enter_context(patch("vllm_ascend.patch.worker.patch_hy_v4.MultiHeadLatentAttentionWrapper", Wrapper))
        stack.callback(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(dtype)
        attention = AscendHYV4MLAAttention(
            vllm_config=SimpleNamespace(),
            config=config,
            hidden_size=8,
            num_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=4,
            kv_lora_rank=4,
            max_position_embeddings=32,
            cache_config=SimpleNamespace(cache_dtype="auto"),
            quant_config=quant_config,
            prefix="model.layers.1.self_attn",
            topk_indices_buffer=topk,
            layer_idx=1,
        )
    wrapper = attention.mla_attn
    assert wrapper.quant_config is quant_config
    assert wrapper.modules.fused_qkv_a_proj is attention.fused_qkv_a_proj
    assert wrapper.modules.o_proj is attention.o_proj
    assert wrapper.modules.g_proj is attention.linear_gate
    assert attention.o_proj.quant_config is quant_config
    assert attention.skip_topk and attention.indexer is None
    assert attention.topk_indices_buffer is topk
    assert wrapper.mla_attn.impl.learnable_sink_param is attention.learnable_sink_param


def test_headwise_gate_broadcasts_each_head_without_mixing_heads():
    projection = Mock(return_value=(torch.tensor([[1.0, -2.0]]), None))
    gate, bias = HeadwiseGate(projection, 3)(torch.zeros(1, 4))
    torch.testing.assert_close(gate, torch.tensor([[1.0, 1.0, 1.0, -2.0, -2.0, -2.0]]))
    assert bias is None


@pytest.mark.parametrize(
    "projection",
    ["self_attn.q_b_proj", "self_attn.indexer.wk", "mlp.shared_experts.gate_proj", "mlp.experts.0.gate_proj"],
)
@pytest.mark.parametrize("scale_e8m0", [False, True])
def test_mtp_scale_decodes_linear_values_and_preserves_packed_experts(projection, scale_e8m0):
    config = AscendFp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128], store_dtype="mxfp4")
    config.is_scale_e8m0 = scale_e8m0
    prefix = f"model.layers.78.mtp_block.{projection}"
    raw = torch.tensor([[126, 127, 128]], dtype=torch.uint8)
    name, scale = native_mtp._prepare_mtp_fp8_expert_scale(config, prefix + ".scale", raw)
    assert name == prefix + ".weight_scale_inv"
    if scale_e8m0:
        assert scale.dtype == torch.float8_e8m0fnu
        torch.testing.assert_close(scale.float(), torch.tensor([[0.5, 1.0, 2.0]]))
    elif ".mlp.experts." in prefix:
        assert scale is raw
    else:
        assert scale.dtype == torch.float32
        torch.testing.assert_close(scale, torch.tensor([[0.5, 1.0, 2.0]]))


def test_explicit_mtp_fp8_uses_ascend_methods_and_keeps_loader_state():
    config = native_mtp._create_mtp_quant_config(
        SimpleNamespace(
            mtp_quant_algo="FP8",
            quantization_config={
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
                "modules_to_not_convert": ["model.mtp_layers.0.self_attn.linear_gate"],
                "expert_dtype": "mxfp4",
            },
        )
    )
    assert isinstance(config, AscendFp8Config)
    assert config.weight_block_size == [128, 128]
    assert config.is_scale_e8m0
    assert config.ignored_layers == ["model.mtp_layers.0.self_attn.linear_gate"]
    assert config.store_dtype is None


@pytest.mark.parametrize("algorithm", [None, "NONE"])
def test_mtp_inherits_platform_quantization_config(algorithm):
    backbone = AscendModelOptMxFp8Config.from_config(
        {
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": []},
        }
    )
    assert native_mtp._create_mtp_quant_config(SimpleNamespace(mtp_quant_algo=algorithm), backbone) is backbone


@pytest.mark.parametrize("num_hidden_layers", [2, 78])
@pytest.mark.parametrize("mtp_layer", [0, 1])
def test_mtp_modelslim_loads_quantization_from_backbone_checkpoint(num_hidden_layers, mtp_layer):
    checkpoint_prefix = f"model.mtp_layers.{mtp_layer}"
    description = {
        f"{checkpoint_prefix}.{projection}.weight": "W8A8_MXFP8"
        for projection in ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "mlp.experts.0.gate_proj")
    }
    description["optional"] = {"metadata": "keep"}
    original_description = description.copy()
    backbone = AscendModelSlimConfig(description)
    draft = native_mtp._create_mtp_quant_config(
        SimpleNamespace(num_hidden_layers=num_hidden_layers, mtp_quant_algo=None), backbone
    )

    assert draft is not backbone
    assert backbone.quant_description == original_description
    assert draft.quant_description is not backbone.quant_description
    assert draft.quant_description["optional"] == description["optional"]
    for infix in ("", "mtp_block."):
        prefix = f"model.layers.{num_hidden_layers + mtp_layer}.{infix}"
        assert (
            get_quant_type_for_layer(
                draft.quant_description,
                prefix + "self_attn.fused_qkv_a_proj",
                native_mtp.HYV4MTP.packed_modules_mapping,
            )
            == "W8A8_MXFP8"
        )
        assert get_quant_type_for_layer(draft.quant_description, prefix + "mlp.experts.0.gate_proj") == "W8A8_MXFP8"


def test_mtp_modelslim_preserves_explicit_runtime_descriptions():
    description = {
        "model.mtp_layers.0.self_attn.q_a_proj.weight": "W8A8_MXFP8",
        "model.mtp_layers.0.self_attn.kv_a_proj_with_mqa.weight": "W8A8_MXFP8",
        "model.layers.78.self_attn.q_a_proj.weight": "FLOAT",
        "model.layers.78.self_attn.kv_a_proj_with_mqa.weight": "FLOAT",
    }
    draft = native_mtp._create_mtp_quant_config(
        SimpleNamespace(num_hidden_layers=78, mtp_quant_algo=None), AscendModelSlimConfig(description)
    )
    assert (
        get_quant_type_for_layer(
            draft.quant_description,
            "model.layers.78.self_attn.fused_qkv_a_proj",
            native_mtp.HYV4MTP.packed_modules_mapping,
        )
        is None
    )


@pytest.mark.parametrize("kv_type, error", [(None, KeyError), ("FLOAT", ValueError)])
def test_mtp_modelslim_rejects_incomplete_or_mixed_fused_quantization(kv_type, error):
    description = {"model.mtp_layers.0.self_attn.q_a_proj.weight": "W8A8_MXFP8"}
    if kv_type is not None:
        description["model.mtp_layers.0.self_attn.kv_a_proj_with_mqa.weight"] = kv_type
    draft = native_mtp._create_mtp_quant_config(
        SimpleNamespace(num_hidden_layers=78, mtp_quant_algo=None), AscendModelSlimConfig(description)
    )
    with pytest.raises(error):
        get_quant_type_for_layer(
            draft.quant_description,
            "model.layers.78.self_attn.fused_qkv_a_proj",
            native_mtp.HYV4MTP.packed_modules_mapping,
        )


def test_mtp_modelslim_keeps_existing_runtime_layout():
    backbone = AscendModelSlimConfig({"model.layers.78.self_attn.q_a_proj.weight": "W8A8_MXFP8"})
    assert native_mtp._create_mtp_quant_config(SimpleNamespace(mtp_quant_algo=None), backbone) is backbone


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("provided_embeddings", [False, True])
def test_mtp_masks_bos_without_broadcasting_mrope_axes(multimodal, provided_embeddings):
    input_ids = torch.arange(3)
    positions = torch.tensor([[0, 2, 5], [0, 0, 6], [0, 4, 0]]) if multimodal else torch.tensor([0, 2, 5])
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
    previous_hidden_states = torch.ones(3, 4)
    layer = Mock(side_effect=lambda ids, pos, hidden, embeds: embeds)
    model = SimpleNamespace(
        embed_tokens=Mock(return_value=embeddings),
        spec_step_idx=1,
        num_mtp_layers=2,
        mtp_start_layer_idx=78,
        layers={"79": layer},
    )

    result = native_mtp.HYV4MultiTokenPredictor.forward(
        model, input_ids, positions, previous_hidden_states, embeddings if provided_embeddings else None
    )

    expected = embeddings.clone()
    expected[0] = 0
    torch.testing.assert_close(result, expected)
    assert result.shape == previous_hidden_states.shape
    assert layer.call_args.args[1] is positions
    assert layer.call_args.args[2] is previous_hidden_states
    assert model.embed_tokens.call_count == int(not provided_embeddings)


@pytest.fixture
def mtp_loader(monkeypatch):
    """Exercise the upstream streaming loader with small CPU checkpoint tensors."""
    monkeypatch.setattr(native_mtp, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(native_mtp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(native_mtp, "get_pp_missing_layer_names", lambda model: [])
    monkeypatch.setattr(patch_hy_v4, "get_pp_missing_layer_names", lambda model: [])
    monkeypatch.setattr(native_mtp, "is_pp_missing_parameter", lambda name, model: False)
    monkeypatch.setattr(native_mtp, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: [])

    def build(params):
        model = native_mtp.HYV4MTP.__new__(native_mtp.HYV4MTP)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_hidden_layers=78, num_nextn_predict_layers=1, n_routed_experts=0, num_attention_heads=1
        )
        model.quant_config = AscendFp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        model.named_parameters = lambda: iter(params.items())
        return model

    return build


def test_mtp_loader_loads_linear_ue8m0_scale_numerically(mtp_loader):
    name = "model.layers.78.mtp_block.self_attn.q_b_proj.weight_scale_inv"
    scale = torch.nn.Parameter(torch.zeros(1, 3), requires_grad=False)
    model = mtp_loader({name: scale})
    loaded = model.load_weights(
        iter([("model.mtp_layers.0.self_attn.q_b_proj.scale", torch.tensor([[126, 127, 128]], dtype=torch.uint8))])
    )
    assert loaded == {name}
    torch.testing.assert_close(scale, torch.tensor([[0.5, 1.0, 2.0]]))


def test_mtp_loader_rejects_missing_weight_scales(mtp_loader):
    name = "model.layers.78.mtp_block.self_attn.q_b_proj.weight_scale_inv"
    model = mtp_loader({name: torch.nn.Parameter(torch.zeros(1), requires_grad=False)})
    with pytest.raises(ValueError, match="Missing MTP checkpoint weight scales.*q_b_proj"):
        model.load_weights(iter([]))


@pytest.mark.parametrize("field", ["weight", "weight_scale_inv"])
def test_mtp_loader_rejects_unpaired_fp8_indexer_weights(mtp_loader, field):
    name = "model.layers.78.mtp_block.self_attn.indexer.wk_weights_proj.weight"
    model = mtp_loader({name: torch.nn.Parameter(torch.zeros(8, 32), requires_grad=False)})
    tensor = torch.zeros(8, 32).to(torch.float8_e4m3fn) if field == "weight" else torch.ones(1, 1)
    with pytest.raises(ValueError, match="Incomplete MTP FP8 indexer weight/scale pairs"):
        model.load_weights(iter([(f"model.mtp_layers.0.self_attn.indexer.wk.{field}", tensor)]))


@pytest.mark.parametrize("scale_first", [False, True])
def test_mtp_loader_accepts_complete_fp8_indexer_pairs_in_either_order(mtp_loader, scale_first):
    name = "model.layers.78.mtp_block.self_attn.indexer.wk_weights_proj.weight"
    param = torch.nn.Parameter(torch.zeros(8, 32, dtype=torch.bfloat16), requires_grad=False)
    param.weight_loader = lambda target, tensor, shard: target.data.copy_(tensor)
    model = mtp_loader({name: param})
    prefix = "model.mtp_layers.0.self_attn.indexer.wk"
    weights = [
        (prefix + ".weight", torch.ones(8, 32).to(torch.float8_e4m3fn)),
        (prefix + ".weight_scale_inv", torch.full((1, 1), 2.0)),
    ]
    loaded = model.load_weights(iter(weights[::-1] if scale_first else weights))
    assert loaded == {name}
    torch.testing.assert_close(param, torch.full_like(param, 2.0))


def test_mtp_loader_allows_runtime_default_kv_cache_scales(mtp_loader):
    name = "model.layers.78.mtp_block.self_attn.attn.k_scale"
    param = patch_hy_v4.KVCacheScaleParameter()
    model = mtp_loader({name: param})
    assert model.load_weights(iter([])) == set()


def test_mtp_pair_check_ignores_other_pipeline_stages(mtp_loader, monkeypatch):
    model = mtp_loader({})
    monkeypatch.setattr(patch_hy_v4, "get_pp_missing_layer_names", lambda model: ["model.layers.78."])
    weights = [("model.mtp_layers.0.self_attn.indexer.wk.weight", torch.zeros(8, 32).to(torch.float8_e4m3fn))]
    assert list(patch_hy_v4._checked_mtp_weights(model, iter(weights))) == weights
