from pathlib import Path
p=Path('/codex/src/vllm-ascend/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py')
p.with_suffix('.py.codex_before').write_text(p.read_text())
p.write_text(p.read_text()+'''

def test_packed_update_validates_all_chunks_and_retains_values():
    """A reused producer buffer must survive a whole-checkpoint validation."""
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    engine.device = torch.device("cpu")
    engine.packed = True
    storage = torch.tensor([1.0, 2.0])
    seen = []

    def load(weights):
        values = dict(weights)
        assert set(values) == {"visual.a", "visual.b"}
        seen.append({name: value.clone() for name, value in values.items()})

    engine.model.load_weights.side_effect = load
    with (
        _patch_reload_module(),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch(f"{_MODULE}.packed_npu_ipc_consumer") as consumer,
    ):
        engine.start_weight_update()
        for name, value in [("visual.a", 1.0), ("visual.b", 2.0)]:
            storage.fill_(value)
            consumer.return_value = [(name, storage)]
            engine.receive_weights(MagicMock(tensor_sizes=[8], ipc_handles={}))
        storage.zero_()
        engine.finish_weight_update()
    assert len(seen) == 1
    torch.testing.assert_close(seen[0]["visual.a"], torch.ones(2))
    torch.testing.assert_close(seen[0]["visual.b"], torch.full((2,), 2.0))
''')
