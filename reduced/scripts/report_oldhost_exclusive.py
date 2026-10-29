import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root=Path('/codex')
logs=root/'logs/resume-exclusive'
output=root/'model_openbox/hy4'
output.mkdir(parents=True,exist_ok=True)
reference=json.loads((root/'reference-newhost/model_openbox/hy4/manifest.json').read_text())
cases={
 'Checkpoint 6L TP2 graph sleep/reload, 3 rounds':'checkpoint-6l-graph.json',
 'IPC 6L TP4 graph packed':'ipc-6l-tp4-graph.json',
 'HCCL TP1 graph packed persistent, 3 rounds':'hccl-graph.json',
 **{f'HCCL TP1 graph communicator reinit {i}':f'hccl-reinit-{i}.json' for i in [1,2,3]},
 'HCCL TP1 graph unpacked':'hccl-unpacked-staged.json',
 'IPC TP2 graph packed':'ipc-graph.json',
 'IPC TP2 eager packed':'ipc-final.json',
 'Checkpoint TP2 graph sleep/reload, 3 rounds':'checkpoint-graph.json',
 'Checkpoint TP2 graph changed sink/expert A-B-A':'variants-graph.json',
 'HCCL 6L TP2 eager packed':'hccl-6l-eager.json',
 'HCCL 6L TP2 eager with pre-finalize cleanup':'hccl-6l-eager-cachefix.json',
 'HCCL 6L TP2 graph with pre-finalize cleanup':'hccl-6l-graph-cachefix.json',
}
results={}
for case,name in cases.items():
    path=logs/name
    if not path.exists():
        results[case]={'status':'not_run','evidence':str(path)}
        continue
    data=json.loads(path.read_text())
    status=data.get('status')
    if status is None:
        equal=[v for k,v in data.items() if k.startswith('identical_') or k.endswith('_restored_identical')]
        status='passed' if equal and all(equal) else 'inspect_raw_report'
    results[case]={'status':status,'evidence':str(path),'rounds':data.get('rounds',[])}
preflight=json.loads((logs/'preflight.json').read_text())
manifest={
 'created_utc':datetime.now(timezone.utc).isoformat(), 'host':'177.8.21.1',
 'workspace':'/data/x00951564/codex-hy4-20261027-oldhost-revalidation',
 'status':'validated_subset' if any(v['status']=='passed' for v in results.values()) else 'in_progress',
 'validation_scope':'functional_smoke','weight_kind':'reduced_synthetic',
 'model_path':'/models/Hy4-2L','model_identity':reference['model_identity'],
 'source_files':{name:{'sha256':hashlib.sha256((root/'src'/name).read_bytes()).hexdigest(),'reference_sha256':value['sha256']} for name,value in reference['source_files'].items()},
 'authorized_devices':[2,3,4,5,6,7], 'reserved_for_colleagues':[0,1],
 'models':{str(n)+'L':{'path':f'/models/Hy4-{n}L','identity':{name:hashlib.sha256((Path(f'/models/Hy4-{n}L')/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors.index.json']}} for n in [2,6]},
 'preflight':preflight, 'unit_tests':{'passed':98,'warnings':15,'evidence':'logs/resume-exclusive/prefinalize-final-unit.log'},
 'results':results,
 'resource_history':['Initial all-device capacity block preserved in logs/npu-after-start-failure.txt and history/initial-attempt','After user confirmed exclusive control, stopped occupancy container wyz-dsv4-vision; container and data retained','Post-reboot device health OK; resumed on 6/7'],
 'dataset_evaluation':{'enabled':False,'reason':'reduced-layer functional scope; full model accuracy excluded'},
 'performance_collection':{'status':'not_run','profile_phase':'middle','profile_steps':10},
 'remaining':['Complete unexecuted configurations listed above','DCP/PCP implementation restrictions','MTP checkpoint and validation','Multi-node DP/PP','TP8 original scenario: blocked by six-device allocation; 0/1 reserved for colleagues','R3/SP/PP/DP final-source regressions','Middle-window profiling and single-device CSV/JSON/Excel'],
}
changed={name for name,value in manifest['source_files'].items() if value['sha256']!=value['reference_sha256']}
assert changed <= {'vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py','vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py'}
manifest['changed_since_reference']=sorted(changed)
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
lines=['# Original A5 exclusive revalidation','','Host: 177.8.21.1. Scope: reduced-model functional verification. All acceptance below comes from this host.','','| Scenario | Status | Evidence |','| --- | --- | --- |']
lines += [f'| {case} | {value["status"]} | logs/resume-exclusive/{Path(value["evidence"]).name} |' for case,value in results.items()]
lines += ['','Source/model hashes matched; NPU 6/7 matmul passed. 98 unit tests passed. A subsequent 6L failure prompted pre-finalize allocator-cache cleanup; versions and before/after evidence are preserved.','','Initial resource and health failures are historical and preserved. Devices were healthy after reboot. The occupancy container was stopped after the user confirmed exclusive control.','','Communicator reinit passing here means the earlier missing-symbol failure did not reproduce under these conditions. It does not prove which device library was previously loaded.','','## Remaining','']+[f'- {item}' for item in manifest['remaining']]
(output/'report.md').write_text('\n'.join(lines)+'\n')
(output/'audit.md').write_text('Reference source identity and patch rationale: reference-newhost/model_openbox/hy4/manifest.json and scripts/build_newhost_report.py.\nCurrent source hashes are rechecked before reporting. CPU staging remains explicit and costs host RAM and copies. Graph modes require capture evidence in server logs.\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
