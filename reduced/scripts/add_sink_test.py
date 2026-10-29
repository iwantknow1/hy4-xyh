from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/tests/ut/models/test_hy_v4.py')
s = p.read_text()
assert 'test_sink_reload_updates_runtime_parameter' not in s
s = s.replace('from vllm.models.hy_v4.nvidia import attention as native_attention\n', '''from vllm.models.hy_v4.nvidia import attention as native_attention
from vllm.models.hy_v4.nvidia import model as native_model
from vllm.model_executor.model_loader.reload.layerwise import (
    record_metadata_for_reloading,
    initialize_layerwise_reload,
    finalize_layerwise_reload,
)
''')
s += '''

@pytest.mark.parametrize("tp_rank", [0, 1])
def test_sink_reload_updates_runtime_parameter(monkeypatch, tp_rank):
    """Checkpoint sink updates must reach the original runtime TP shard."""
    monkeypatch.setattr(native_model, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(native_model, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(native_model, "get_pp_missing_layer_names", lambda model: [])
    monkeypatch.setattr(native_model, "is_pp_missing_parameter", lambda name, model: False)
    model = native_model.HYV4Model.__new__(native_model.HYV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=1, num_attention_heads=4, num_experts=0,
        tie_word_embeddings=False,
    )
    model._cached_expert_params_mapping = []
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    runtime_sink = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    layer.self_attn.register_parameter("learnable_sink_param", runtime_sink)
    model.layers = torch.nn.ModuleList([layer])
    name = "layers.0.self_attn.learnable_sink_param"
    record_metadata_for_reloading(model)
    for offset in (10, 20):
        initialize_layerwise_reload(model)
        values = torch.arange(4, dtype=torch.float32) + offset
        assert model.load_weights(iter([(name, values)])) == {name}
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
        assert layer.self_attn.learnable_sink_param is runtime_sink
        torch.testing.assert_close(runtime_sink, values[tp_rank * 2:tp_rank * 2 + 2])
'''
p.with_suffix('.py.codex_before').write_text(p.read_text())
p.write_text(s)
