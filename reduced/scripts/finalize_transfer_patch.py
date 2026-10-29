from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/hccl_engine.py'
s=p.read_text().replace('\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n','\n')
s='\n'.join(line for line in s.split('\n') if 'print("HCCL ' not in line and line.strip() != 'torch.npu.synchronize()')
s=s.replace('        torch.npu.empty_cache()', '        # Release reload scratch buffers before the allocator restores KV cache.\n        torch.npu.empty_cache()',1)
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_hccl_engine.py'
s=p.read_text().replace('    with patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as finalize:', '    with (\n        patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as finalize,\n        patch("torch.npu.empty_cache"),\n    ):')
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py'
s=p.read_text().replace('    engine.device = MagicMock(index=0)\n    engine.packed = True', '    engine.device = MagicMock(index=0)\n    engine.model_config = MagicMock()\n    engine.packed = True')
p.write_text(s)
