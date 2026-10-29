from pathlib import Path
p=Path('/codex/src/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py')
s=p.read_text()
s=s.replace('import torch\n','import torch\n\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n',1)
old='        finalize_layerwise_reload(self.model, self.model_config)'
new=old+'''
        torch.npu.synchronize()
        logger.info("HCCL finalized: allocated=%d reserved=%d free=%s", torch.npu.memory_allocated(), torch.npu.memory_reserved(), torch.npu.mem_get_info())
        torch.npu.empty_cache()
        logger.info("HCCL cache released: allocated=%d reserved=%d free=%s", torch.npu.memory_allocated(), torch.npu.memory_reserved(), torch.npu.mem_get_info())'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
