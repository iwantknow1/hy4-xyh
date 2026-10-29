import hashlib
import json
from pathlib import Path
root=Path('/codex')
reference=json.loads((root/'reference-newhost/model_openbox/hy4/manifest.json').read_text())
version={name:hashlib.sha256((root/'src'/name).read_bytes()).hexdigest() for name in reference['source_files']}
(root/'logs/resume-exclusive/prefinalize-debug-source.json').write_text(json.dumps(version,indent=2))
p=root/'scripts/report_oldhost_exclusive.py'
s=p.read_text().replace(" 'HCCL 6L TP2 graph packed':'hccl-6l-graph.json',", " 'HCCL 6L TP2 eager with pre-finalize cleanup':'hccl-6l-eager-cachefix.json',\n 'HCCL 6L TP2 graph with pre-finalize cleanup':'hccl-6l-graph-cachefix.json',")
s=s.replace("assert all(v['sha256']==v['reference_sha256'] for v in manifest['source_files'].values())", "changed={name for name,value in manifest['source_files'].items() if value['sha256']!=value['reference_sha256']}\nassert changed <= {'vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py','vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py'}\nmanifest['changed_since_reference']=sorted(changed)")
s=s.replace('No new production-code changes were needed to resume.', 'A subsequent 6L failure prompted pre-finalize allocator-cache cleanup; versions and before/after evidence are preserved.')
p.write_text(s)
