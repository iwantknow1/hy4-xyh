from pathlib import Path
p=Path('/codex/src/vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py')
s=p.read_text(); p.with_suffix('.py.pre6l').write_text(s)
old='''def test_finish_weight_update_finalizes_layerwise_reload():
    engine = _make_engine()

    with (
        patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as finalize,
        patch("torch.npu.empty_cache"),
    ):
        engine.finish_weight_update()

    finalize.assert_called_once_with(engine.model, engine.model_config)
'''
new='''def test_finish_weight_update_finalizes_layerwise_reload():
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
'''
assert old in s
p.write_text(s.replace(old,new))
