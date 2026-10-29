from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/hccl_engine.py'
s=p.read_text(); p.with_suffix('.py.pre6l').write_text(s)
old='        finalize_layerwise_reload(self.model, self.model_config)'
new='''        print("HCCL_PREFINALIZE_BEFORE", torch.npu.memory_allocated(), torch.npu.memory_reserved(), torch.npu.mem_get_info(), flush=True)
        torch.npu.empty_cache()
        print("HCCL_PREFINALIZE_CLEAN", torch.npu.memory_allocated(), torch.npu.memory_reserved(), torch.npu.mem_get_info(), flush=True)
'''+old
assert s.count(old)==1
p.write_text(s.replace(old,new))
p=root/'tests/ut/distributed/weight_transfer/test_hccl_engine.py'
s=p.read_text().replace('        patch("torch.npu.empty_cache", side_effect=lambda: scratch.__setitem__(0, False)),','''        patch("torch.npu.empty_cache", side_effect=lambda: scratch.__setitem__(0, False)),
        patch("torch.npu.memory_allocated", return_value=0, create=True),
        patch("torch.npu.memory_reserved", return_value=0, create=True),
        patch("torch.npu.mem_get_info", return_value=(0, 0), create=True),''')
p.write_text(s)
