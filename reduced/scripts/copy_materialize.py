import gc
import sys
import torch
import torch_npu

torch.npu.set_device(0)
mode = sys.argv[1]
shape = (128, 4096, 6144)
stride = (4096 * 6144, 6144, 1)
print('allocation', mode, shape, flush=True)
if mode == 'strided':
    dst = torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='npu')
else:
    dst = torch.empty(shape, dtype=torch.bfloat16, device='npu')
src = torch.ones((2048, 6144), dtype=torch.bfloat16)
for expert in [0, 1, 64, 85, 86, 100, 120, 127]:
    for offset in [0, 2048]:
        view = dst[expert, offset:offset + 2048]
        print('copy', expert, offset, view.storage_offset(), flush=True)
        view.copy_(src)
        torch.npu.synchronize()
        assert torch.equal(view.cpu(), src)
print('PASS', mode, flush=True)
