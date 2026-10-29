import hashlib,json,pathlib,importlib.metadata
manifest=json.loads(pathlib.Path('/codex/source-manifest.json').read_text());checks={}
for name,value in manifest['source_files'].items():
    path=pathlib.Path('/vllm-workspace')/name
    actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==value['sha256'],name
    checks[name]=actual
versions={name:importlib.metadata.version(name) for name in ['torch','torch-npu','triton-ascend']}
import torch
versions['torch_runtime']=str(torch.__version__)
assert versions['torch']=='2.10.0' and versions['torch_runtime']=='2.10.0+cpu',versions
assert versions['torch-npu']=='2.10.0.post4',versions
assert versions['triton-ascend']=='3.6.0',versions
report={'status':'passed','source_files':checks,'versions':versions,'source_manifest_sha256':hashlib.sha256(pathlib.Path('/codex/source-manifest.json').read_bytes()).hexdigest()}
pathlib.Path('/codex/image-runtime-preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
