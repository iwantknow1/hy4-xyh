import hashlib
import json
from pathlib import Path
import torch
import torch_npu

root=Path('/codex')
reference=json.loads((root/'reference-newhost/model_openbox/hy4/manifest.json').read_text())
for name, identity in reference['source_files'].items():
    actual=hashlib.sha256((root/'src'/name).read_bytes()).hexdigest()
    assert actual == identity['sha256'], name
for name, expected in reference['model_identity'].items():
    assert hashlib.sha256((Path('/models/Hy4-2L')/name).read_bytes()).hexdigest() == expected, name
report={'host':'177.8.21.1','source_hashes_match':True,'model_identity_match':True,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'devices':[]}
try:
    for device_index in [0,1]:
        torch.npu.set_device(device_index)
        tensor=torch.ones((256,256),device=f'npu:{device_index}',dtype=torch.float16)
        result=tensor @ tensor
        torch.npu.synchronize()
        assert torch.equal(result.cpu(),torch.full((256,256),256,dtype=torch.float16))
        report['devices'].append({'logical_device':device_index,'physical_device':6+device_index,'matmul':'passed'})
        del tensor,result
        torch.npu.empty_cache()
    report['status']='passed'
except Exception as error:
    report['status']='failed'
    report['error']=repr(error)
    raise
finally:
    (root/'logs/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
