import json
from pathlib import Path
import torch
import torch_npu
report={'devices':[]}
try:
    for device in [0,1]:
        torch.npu.set_device(device)
        source=torch.ones((2048,6144),dtype=torch.bfloat16)
        copied=source.clone().to(f'npu:{device}')
        assert torch.equal(copied.cpu(),source)
        torch.npu.synchronize()
        report['devices'].append({'physical_device':6+device,'copy_bytes':source.nbytes,'status':'passed'})
        del source,copied
        torch.npu.empty_cache()
    report['status']='passed'
except Exception as error:
    report['status']='failed'; report['error']=repr(error)
    raise
finally:
    Path('/codex/logs/resume-exclusive/host-copy-probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
