from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py'
s=p.read_text().replace('    packed: bool = False\n\n\n@lru_cache', '    packed: bool = False\n    cpu_staging: bool = False\n\n\n@lru_cache',1)
s=s.replace('    backend: ClassVar[str] = "npu_ipc"','    cpu_staging: bool = False\n\n    backend: ClassVar[str] = "npu_ipc"',1)
s=s.replace('        self.packed = False\n','        self.packed = False\n        self.cpu_staging = False\n',1)
s=s.replace('        self.packed = init_info.packed\n','        self.packed = init_info.packed\n        self.cpu_staging = init_info.cpu_staging\n',1)
s=s.replace('        if self.packed:\n            # Packed RPCs', '        if self.packed and self.cpu_staging:\n            # Packed RPCs',1)
s=s.replace('        packed: bool = False,\n        packed_buffer_size_bytes:', '        packed: bool = False,\n        cpu_staging: bool = False,\n        packed_buffer_size_bytes:',1)
s=s.replace('        self.packed = packed\n','        self.packed = packed\n        self.cpu_staging = cpu_staging\n',1)
s=s.replace('            packed=init_info.packed,\n','            packed=init_info.packed,\n            cpu_staging=init_info.cpu_staging,\n',1)
s=s.replace('engine.client.init_weight_transfer_engine({"packed": init_info.packed})','engine.client.init_weight_transfer_engine({"packed": init_info.packed, "cpu_staging": init_info.cpu_staging})',1)
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py'
s=p.read_text().replace('    engine.packed = True\n','    engine.packed = True\n    engine.cpu_staging = True\n')
p.write_text(s)
p=Path('/codex/scripts/skill-scripts/weight-update/test_npu_ipc_weight_update.py')
s=p.read_text().replace('rank=rank, packed=True, packed_buffer_size_bytes=packed_buffer_bytes,','rank=rank, packed=True, cpu_staging=True, packed_buffer_size_bytes=packed_buffer_bytes,')
p.with_name('test_npu_ipc_cpu_stage.py').write_text(s)
