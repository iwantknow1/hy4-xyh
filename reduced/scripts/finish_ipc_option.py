from pathlib import Path
root=Path('/codex/src/vllm-ascend')
p=root/'vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py'
s=p.read_text().replace('    No initialization needed for NPU IPC.','    Records packing and optional CPU staging; no rendezvous is needed.')
s=s.replace('''    """NPU IPC trainer init info — overrides the backend key only.

    ``IPCTrainerInitInfo`` already provides ``packed`` and
    ``packed_buffer_size_bytes``; this subclass only rebinds the
    factory ``backend`` from ``"ipc"`` to ``"npu_ipc"``.
    """''','''    """NPU IPC settings, including optional deferred checkpoint CPU staging."""''')
s=s.replace('    cpu_staging: bool = False\n\n\n@lru_cache','    cpu_staging: bool = False\n    """Buffer packed chunks on CPU and load the complete checkpoint at finish."""\n\n\n@lru_cache',1)
p.write_text(s)
p=root/'tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py'
s=p.read_text().replace('import torch\n','import pytest\nimport torch\n',1)
s=s.replace('def test_receive_packed_weights_loads_model():','@pytest.mark.parametrize("cpu_staging", [False, True])\ndef test_receive_packed_weights_loads_model(cpu_staging):',1)
start=s.index('def test_receive_packed_weights_loads_model(');end=s.index('def test_packed_update_validates',start)
s=s[:start]+s[start:end].replace('engine.cpu_staging = True','engine.cpu_staging = cpu_staging')+s[end:]
p.write_text(s)
