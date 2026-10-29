from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root=Path('/codex')
out=root/'model_openbox/hy4'
out.mkdir(parents=True,exist_ok=True)
files=[
'vllm/vllm/models/hy_v4/nvidia/model.py',
'vllm-ascend/vllm_ascend/patch/worker/patch_hy4_reload.py',
*['vllm-ascend/vllm_ascend/distributed/weight_transfer/'+name+'.py' for name in ['hccl_engine','npu_ipc_engine','packed_tensor']],
'vllm-ascend/tests/ut/models/test_hy_v4.py',
*['vllm-ascend/tests/ut/distributed/weight_transfer/'+name+'.py' for name in ['test_hccl_engine','test_npu_ipc_engine','test_packed_tensor']],
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results={}
for name in ['fixed-sleep.json','graph-sleep.json','variants-eager.json','variants-graph.json','hccl-unpacked-staged-round1.json','hccl-unpacked-staged-round2.json','hccl-packed-staged-round1.json','hccl-packed-reinit-failed.json','hccl-packed-persistent.json','ipc-tp2-round1.json','ipc-tp2-round2.json','ipc-counters-fixed-round1.json','ipc-counters-fixed-round2.json','ipc-tp2.json','ipc-final.json','pause-consistency.json','ipc-graph.json','hccl-graph.json']:
    p=root/'logs'/name
    if p.exists():
        data=json.loads(p.read_text())
        results[name]={'status':data.get('status','see_raw_report'),'checks':data.get('checks',{}),'rounds':data.get('rounds',[]),'path':str(p)}
manifest={
 'created_utc':datetime.now(timezone.utc).isoformat(), 'host':'177.11.28.1',
 'workspace':'/data/x00951564/codex-hy4-20261027-newhost', 'container':'x00951564-hy4-codex-newhost',
 'environment':{'chip':'Ascend950DT','devices':[6,7],'CANN':'9.1.0','driver':'26.2.0.b020','torch':'2.10.0+cpu','torch_npu':'2.10.0.post4','image':'vllm-ascend-hy4:a5-cann9.1'},
 'validated_configurations':{'checkpoint_reload':{'tp':2,'ep':True,'modes':['eager','FULL_DECODE_ONLY'],'kv_cache_bytes':8589934592},'hccl':{'tp':1,'producer_device':7,'worker_device':6,'cpu_staging':True,'modes':['eager','FULL_DECODE_ONLY'],'packed_persistent_rounds':3},'npu_ipc':{'tp':2,'producer_devices':[6,7],'worker_devices':[6,7],'cpu_staging':True,'packed':True,'modes':['eager','FULL_DECODE_ONLY']}},
 'weight_kind':'reduced_synthetic','weight_origin':'retained original weights in reduced-layer checkpoint', 'validation_scope':'functional_smoke',
 'model_path':'/models/Hy4-2L','model_config':json.loads(Path('/models/Hy4-2L/config.json').read_text()),
 'model_identity':{name:sha(Path('/models/Hy4-2L')/name) for name in ['config.json','model.safetensors.index.json']},
 'source_snapshot':'private donor container copied to isolated writable sources; no upstream revision metadata supplied',
 'source_files':{name:{'sha256':sha(root/'src'/name),'baseline_sha256':sha(root/'src'/(name+'.codex_before')) if (root/'src'/(name+'.codex_before')).exists() else None} for name in files},
 'dataset_evaluation':{'enabled':False,'dataset':None,'max_out_len':None,'reason':'reduced-layer functional validation; full-model accuracy excluded'},
 'performance_collection':{'status':'not_run','profile_steps':10,'profile_phase':'middle','validation_scope':'functional_smoke','reason':'weight-update failures addressed first; profiling and Excel still pending'},
 'results':results,
 'unit_tests':{'command':'TORCH_DEVICE_BACKEND_AUTOLOAD=0 /codex/.venv/bin/python -m pytest tests/ut/models/test_hy_v4.py tests/ut/distributed/weight_transfer/{test_hccl_engine,test_packed_tensor,test_npu_ipc_engine}.py -q','result':'98 passed, 15 warnings','log':'/codex/logs/transfer-final-unit.log'},
 'pending':['HCCL initialization: missing RunAicpuIndOpCommInit; mixed CANN versions suspected, isolated reproduction pending','DCP/PCP implementation restrictions','MTP checkpoint preparation and validation','multi-node DP/PP','TP8 high-utilization sleep','R3/SP/PP/DP regression on final patches','middle 10-step profiling and hardware-efficiency CSV/JSON/Excel'],
}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(out/'audit.md').write_text('''# HY4 reduced-model repair audit

All work is isolated from DeepSeek and the local /home/a/src/vllm checkout.
The 2-layer checkpoint retains dense and expert paths. Garbled semantics are expected at this reduction; acceptance requires valid finite outputs and exact lifecycle consistency, not model quality.

1. Native sink loading uses the parameter weight_loader after TP head slicing, allowing layerwise reload to count and replay updates into existing graph-visible storage.
2. Expert copy wrappers execute meta copies for CopyCounter rather than returning early. The previous early return silently skipped actual expert updates.
3. HCCL model.load_weights sees the whole stream rather than individual tensors or packed batches; strict visual completeness validation remains enabled.
4. Optional HCCL CPU staging reduces device retention during deferred loading. finish_weight_update empties unused NPU allocator cache before CaMem restores KV physical allocations. Diagnostic evidence: active 37135471616 bytes unchanged, reserved 96626278400 to 56486789120 bytes.
5. Packed IPC chunks are copied to CPU and loaded together at finish, preserving checkpoint validation and preventing producer buffer reuse from corrupting deferred loads.
6. Each IPC chunk exports a fresh reference counter with reduce_tensor while reusing one allocation. Previously one exported counter was consumed multiple times.

Inherited host expert-map caching and copy-source materialization are retained; these are not claimed as newly authored repairs. CPU staging is explicit: HCCL update info cpu_staging=True; NPUIPCTrainerInitInfo cpu_staging=True. CPU staging adds host RAM and transfer cost; no throughput benefit is claimed. All acceptance is limited to recorded shapes, topology and mode.

HCCL packed passed three updates with a persistent communicator but another attempt failed during communicator reinitialization before transfer. Device logs show a missing RunAicpuIndOpCommInit symbol. A concurrent CANN 9.2 worker shares the device; its library lacks this symbol while our CANN 9.1 library exports it. Mixed-version selection is suspected but not proven; see logs/hccl-reinit-driver/diagnosis.md. Preserve this failure separately from payload correctness. IPC producer-lifetime warnings are retained in initial logs and assessed again after per-chunk counter repair.
''')
lines=['# HY4 A5 verification status','','Scope: reduced-model functional verification. Full-model accuracy and performance conclusions are excluded.','','| Evidence | Status |','| --- | --- |']
lines += [f'| logs/{name} | {data["status"]} |' for name,data in results.items()]
lines += ['','98 unit tests passed. Raw logs include failing-before regressions and real-device capacity failures.','','## Remaining work','']+[f'- {item}' for item in manifest['pending']]
(out/'report.md').write_text('\n'.join(lines)+'\n')
(out/'README.md').write_text((out/'report.md').read_text())
(root/'review-files.txt').write_text('\n'.join('src/'+name+suffix for name in files for suffix in ['', '.codex_before'] if (root/'src'/(name+suffix)).exists())+'\n')
print(out)
