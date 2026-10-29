# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
    HCCLWeightTransferUpdateInfo,
)


def _make_engine():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    return engine


def test_start_weight_update_initializes_layerwise_reload():
    engine = _make_engine()

    with patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as initialize:
        engine.start_weight_update()

    initialize.assert_called_once_with(engine.model)


def test_finish_weight_update_finalizes_layerwise_reload():
    """Deferred materialization must have room after transfer scratch is freed."""
    engine = _make_engine()
    scratch = [True]

    def materialize(model, config):
        if scratch[0]:
            raise torch.OutOfMemoryError("transfer scratch occupies the budget")

    with (
        patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload", side_effect=materialize) as finalize,
        patch("torch.npu.empty_cache", side_effect=lambda: scratch.__setitem__(0, False)),
    ):
        engine.finish_weight_update()

    finalize.assert_called_once_with(engine.model, engine.model_config)


def test_shutdown_closes_session_once():
    engine = _make_engine()
    group = MagicMock()
    engine.model_update_group = group

    engine.shutdown()
    engine.shutdown()

    group.close.assert_called_once_with()
    assert engine.model_update_group is None


@patch("torch.accelerator.current_device_index", return_value=0)
def test_reinit_closes_previous_session_before_creating_next(_mock_device):
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=0, world_size=1, rank=0)
    old_group = MagicMock()
    engine.model_update_group = old_group
    init_info = HCCLWeightTransferInitInfo(master_address="127.0.0.1", master_port=12345, rank_offset=1, world_size=2)

    def create_group(*args, **kwargs):
        old_group.close.assert_called_once_with()
        assert engine.model_update_group is None
        return MagicMock()

    with patch.object(HCCLWeightTransferEngine, "_stateless_init_process_group", side_effect=create_group) as init:
        engine.init_transfer_engine(init_info)

    init.assert_called_once_with("127.0.0.1", 12345, 1, 2, device=0)
    assert engine.model_update_group is not old_group


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cpu_staging", [False, True])
def test_transfer_validates_the_complete_checkpoint(packed, cpu_staging):
    """A model's completeness check must see the entire streamed update."""
    engine = _make_engine()
    engine.model_update_group = MagicMock()
    info = HCCLWeightTransferUpdateInfo(
        names=["visual.a", "visual.b"],
        dtype_names=["float32", "float32"],
        shapes=[[2], [3]],
        packed=packed,
        cpu_staging=cpu_staging,
        packed_buffer_size_bytes=1,
        packed_num_buffers=1,
    )

    def load(weights):
        received = list(weights)
        assert [name for name, _ in received] == info.names
        assert [tuple(tensor.shape) for _, tensor in received] == [(2,), (3,)]

    engine.model.load_weights.side_effect = load
    with (
        patch(
            "vllm_ascend.distributed.weight_transfer.hccl_engine.torch.empty",
            side_effect=lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
        ),
        patch("torch.npu.current_stream"),
        patch("torch.npu.Stream"),
        patch("torch.npu.stream", side_effect=lambda stream: nullcontext()),
    ):
        engine.receive_weights(info)
    engine.model.load_weights.assert_called_once()
    assert engine.model_update_group.broadcast.call_count == 2
