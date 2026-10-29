import json
import os

os.environ['PATH'] = '/codex/.venv/bin:/usr/local/Ascend/cann-9.1.0/bin:/usr/local/bin:/usr/bin:/bin'
import torch
import torch_npu
from vllm_ascend.ops.triton.compute_slot_mapping import _compute_slot_mapping_kernel as kernel

assert hasattr(kernel, '__getitem__'), type(kernel).__name__
torch.npu.set_device(0)
query = torch.tensor([0, 2], dtype=torch.int32, device='npu')
positions = torch.tensor([0, 1], dtype=torch.int64, device='npu')
table = torch.tensor([[2]], dtype=torch.int32, device='npu')
output = torch.empty(8, dtype=torch.int32, device='npu')
kernel[(2,)](2, 8, query, positions, table, 1, 128, output,
    KV_CACHE_BLOCK_SIZE=128, BLOCKS_PER_KV_BLOCK=1,
    TOTAL_CP_WORLD_SIZE=1, TOTAL_CP_RANK=0, CP_KV_CACHE_INTERLEAVE_SIZE=1,
    PAD_ID=-1, TILE_BLOCK_SIZE=1024, BLOCK_TABLE_WINDOW_SIZE=16, IS_CIRCULAR=False)
actual = output.cpu().tolist()
assert actual == [256, 257, -1, -1, -1, -1, -1, -1], actual
print(json.dumps({'status': 'passed', 'kernel': type(kernel).__name__, 'slots': actual}))
