from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py')
s = p.read_text()
s = s.replace('from types import SimpleNamespace', 'import torch\n\nfrom types import SimpleNamespace')
s = s.replace('HCCLWeightTransferEngine, HCCLWeightTransferInitInfo', 'HCCLWeightTransferEngine, HCCLWeightTransferInitInfo, HCCLWeightTransferUpdateInfo')
s += '''

def test_unpacked_transfer_validates_the_complete_checkpoint():
    """A model's completeness check must see the entire streamed update."""
    engine = _make_engine()
    engine.model_update_group = MagicMock()
    info = HCCLWeightTransferUpdateInfo(
        names=["visual.a", "visual.b"],
        dtype_names=["float32", "float32"],
        shapes=[[2], [3]],
        packed=False,
    )
    def load(weights):
        received = list(weights)
        assert [name for name, _ in received] == info.names
        assert [tuple(tensor.shape) for _, tensor in received] == [(2,), (3,)]
    engine.model.load_weights.side_effect = load
    with patch(
        "vllm_ascend.distributed.weight_transfer.hccl_engine.torch.empty",
        side_effect=lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
    ), patch("torch.npu.current_stream"):
        engine.receive_weights(info)
    engine.model.load_weights.assert_called_once()
    assert engine.model_update_group.broadcast.call_count == 2
'''
p.with_suffix('.py.codex_before').write_text(p.read_text())
p.write_text(s)
