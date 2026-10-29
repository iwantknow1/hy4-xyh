from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/hccl_engine.py'
s=p.read_text()
s='\n'.join(line for line in s.split('\n') if 'print("HCCL_PREFINALIZE_' not in line)
old='        torch.npu.empty_cache()\n        finalize_layerwise_reload(self.model, self.model_config)'
assert old in s
s=s.replace(old,'        # Deferred layers need room after transfer scratch is released.\n'+old,1)
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_hccl_engine.py'
s='\n'.join(line for line in p.read_text().split('\n') if not any(token in line for token in ['patch("torch.npu.memory_allocated"','patch("torch.npu.memory_reserved"','patch("torch.npu.mem_get_info"']))
p.write_text(s)
