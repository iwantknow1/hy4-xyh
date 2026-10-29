import json
import torch
import torch_npu

rows=[]
for device in range(8):
    torch.npu.set_device(device)
    free,total=torch.npu.mem_get_info(device)
    rows.append({'device':device,'free_gib':free/2**30,'total_gib':total/2**30})
print(json.dumps({'devices':rows,'min_free_gib':min(r['free_gib'] for r in rows)},indent=2),flush=True)
