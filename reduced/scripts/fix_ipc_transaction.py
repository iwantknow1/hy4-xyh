from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py'
s=p.read_text(); p.with_suffix('.py.codex_before').write_text(s)
s=s.replace('        self.packed = False\n','        self.packed = False\n        self._pending_weights: list[tuple[str, torch.Tensor]] = []\n',1)
s=s.replace('        initialize_layerwise_reload(self.model)\n','        self._pending_weights = []\n        initialize_layerwise_reload(self.model)\n',1)
s=s.replace('        finalize_layerwise_reload(self.model, self.model_config)\n','''        if getattr(self, "_pending_weights", None):
            self.model.load_weights(iter(self._pending_weights))
        finalize_layerwise_reload(self.model, self.model_config)
        self._pending_weights = []
        torch.npu.empty_cache()
''',1)
s=s.replace('        self.model.load_weights(weights)\n','''        if self.packed:
            # Packed RPCs are chunks of one checkpoint. Validate only at finish;
            # CPU copies also outlive the producer's reusable IPC buffer.
            self._pending_weights.extend((name, weight.to("cpu", copy=True)) for name, weight in weights)
        else:
            self.model.load_weights(weights)
''',1)
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py'
s=p.read_text().replace('with _patch_reload_module(finalize=mock_finalize):','with _patch_reload_module(finalize=mock_finalize), patch("torch.npu.empty_cache"):')
s=s.replace('        engine.receive_weights(update_info)\n\n    mock_consumer', '        with _patch_reload_module(), patch("torch.npu.empty_cache"):\n            engine.start_weight_update()\n            engine.receive_weights(update_info)\n            engine.finish_weight_update()\n\n    mock_consumer',1)
s=s.replace('    engine.model.load_weights.assert_called_once_with(packed_weights)','    engine.model.load_weights.assert_called_once()\n    received = list(engine.model.load_weights.call_args.args[0])\n    assert received[0][0] == packed_weights[0][0]\n    torch.testing.assert_close(received[0][1], packed_weights[0][1])')
s=s.replace('        _patch_reload_module(),\n        patch(f"{_MODULE}.npu_generate_uuid"', '        _patch_reload_module(),\n        patch("torch.npu.empty_cache"),\n        patch(f"{_MODULE}.npu_generate_uuid"',1)
p.write_text(s)
