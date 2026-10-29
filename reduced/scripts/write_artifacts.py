import hashlib
import json
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

root = Path('/codex')
out = root / 'model_openbox/hy4-2l'
out.mkdir(parents=True, exist_ok=True)
def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
sources = [
    root / 'src/vllm/vllm/models/hy_v4/nvidia/model.py',
    root / 'src/vllm-ascend/tests/ut/models/test_hy_v4.py',
]
models = {}
for name in ('Hy4-2L', 'Hy4-6L'):
    model = Path('/models') / name
    index = model / 'model.safetensors.index.json'
    data = json.loads(index.read_text())
    shards = sorted(set(data['weight_map'].values()))
    models[name] = {
        'path': str(model), 'index_sha256': digest(index),
        'config_sha256': digest(model / 'config.json'),
        'tensors': len(data['weight_map']), 'shards': len(shards),
        'checkpoint_bytes': sum((model / shard).stat().st_size for shard in shards),
        'weight_kind': 'reduced_real', 'validation_scope': 'functional_smoke',
    }
manifest = {
    'created': datetime.now().astimezone().isoformat(),
    'directory': '/data/x00951564/codex-hy4-20261027',
    'container': 'x00951564-hy4-codex-20261027',
    'image': (root / 'logs/image-id.txt').read_text().strip(),
    'models': models, 'physical_devices': [6, 7], 'port': 18080,
    'parallelism': {'TP': 2, 'PP': 1, 'DP': 1, 'EP': 2},
    'versions': {p: version(p) for p in ('torch','torch-npu','transformers','triton-ascend')},
    'source_sha256': {str(p.relative_to(root)): digest(p) for p in sources},
    'dataset_evaluation': {'enabled': False, 'reason': 'Reduced-layer functional validation'},
    'performance_collection': {'status': 'not_run', 'reason': 'Focused repair, not a performance task'},
    'baseline_special_config': {
        'gpu_memory_utilization': 0.85, 'kv_cache_memory_bytes': 8589934592,
        'max_model_len': 2048, 'max_num_seqs': 8, 'enforce_eager': True,
        'enable_sleep_mode': True, 'rl_config': {'enabled': True},
        'revision': 'codex-reload-v1',
    },
    'environment_changes': {
        'assessment': 'required',
        'items': [
            {'component':'HY4 sink loader', 'pr':'local patch, not submitted',
             'reason':'Direct copy bypasses layerwise accounting',
             'status':'applied and verified', 'evidence':'patches/sink-loader.patch'},
            {'component':'KV cache budget', 'pr':'configuration',
             'reason':'Uncapped 0.85 reproduces 507899; 8 GiB cap passes',
             'status':'verified', 'evidence':'logs/util085-serve.log and logs/kvcap8-client.log'},
        ],
    },
}
dump(out / 'manifest.json', manifest)
prompts = ['What is 2 + 2?', 'Write one word about the sky.']
for label, title, server in (
    ('baseline', '2L checkpoint, util=0.5, original sink', 'baseline-serve.log'),
    ('kvcap8', '2L checkpoint, util=0.85, KV=8 GiB, original sink', 'kvcap8-serve.log'),
    ('sleep-fixed', '2L sleep + checkpoint, fixed sink', 'sleep-fixed-serve.log'),
    ('sleep-control', '2L sleep + checkpoint, original sink control', 'sleep-control-serve.log'),
    ('sixlayer-sleep', '6L sleep + checkpoint, fixed sink', 'sixlayer-serve.log'),
):
    data = json.loads((root / 'logs' / (label + '.json')).read_text())
    rounds = [i for i in range(1,4) if 'identical_' + str(i) in data]
    passed = len(rounds) == 3 and all(data['identical_' + str(i)] for i in rounds)
    latest = max(rounds)
    before = {'/v1/' + k:v for k,v in data['baseline'].items()}
    after = {'/v1/' + k:v for k,v in data['outputs_' + str(latest)].items()}
    comparison = {ep:[a == b for a,b in zip(before[ep], after[ep])] for ep in before}
    report = {
        'feature':title, 'mode':'checkpoint_reload', 'status':'passed' if passed else 'failed',
        'prompts':prompts, 'baseline_by_endpoint':before, 'post_update_by_endpoint':after,
        'outputs_identical_by_endpoint':comparison,
        'endpoint_checks':{ep:all(values) for ep,values in comparison.items()},
        'checks':{'three_rounds':len(rounds) == 3, 'all_rounds_identical':all(data['identical_' + str(i)] for i in rounds)},
        'server_log':str(root / 'logs' / server), 'raw_evidence':str(root / 'logs' / (label + '.json')),
        'verification_command':'.venv/bin/python scripts/check_sleep_reload.py ' + label if 'sleep' in label else '.venv/bin/python scripts/check_reload.py ' + label,
        'launch_command':(root / 'scripts/start.sh').read_text() if label == 'sixlayer-sleep' else 'See raw server startup arguments in ' + server,
    }
    dump(out / (label + '.json'), report)
dump(out / 'hccl-capacity.json', {
    'feature':'HCCL 6L TP2 unpacked candidate', 'status':'blocked',
    'reason':'Real NPU memory allocation failure during transfer. Stopped without retry.',
    'checks':{'end_to_end_passed':False, 'capacity_failure_recorded':True},
    'server_log':'/codex/logs/hccl-sixlayer-serve.log',
    'producer_log':'/codex/logs/hccl-sixlayer-producer.log',
})
for feature, reason in (
    ('packed HCCL','Not reached after unpacked capacity failure'),
    ('IPC','Not run in this focused repair'),
    ('TP8 sleep','Not verified: repair tested on TP2'),
    ('FULL_DECODE_ONLY graph reload','Not verified: repair tested in eager'),
    ('DCP','Existing HY4 attention sinks implementation requires DCP=1'),
    ('PCP','Existing HY4 attention sinks implementation requires PCP=1'),
    ('MTP','Reduced checkpoints have no nextn weights'),
    ('PP','Not rerun; previous handoff results are not counted as this repair evidence'),
    ('SP','Not rerun; previous handoff results are not counted as this repair evidence'),
    ('multi-node','Only one node used'),
    ('dataset evaluation','Reduced-layer functional scope'),
    ('profiling','Focused repair scope'),
):
    dump(out / (feature.replace(' ','_').replace('/','_') + '.json'),
         {'feature':feature, 'status':'not_run', 'reason':reason})
(root / 'manifest.json').write_text((out / 'manifest.json').read_text())
(root / 'logs/launch_commands.txt').write_text(
    (root / 'scripts/start.sh').read_text() + '\n' +
    '.venv/bin/python scripts/check_sleep_reload.py sixlayer-sleep\n' +
    (root / 'scripts/run_hccl.sh').read_text())
(out / 'audit.md').write_text(
    '# Audit\n\nModel configs and index hashes are in manifest.json. '
    '2L and 6L retain real reduced weights and the vision tower, without MTP. '
    'No model assets were modified. Only functional consistency is tested. '
    'Source was copied from the existing HY4 container on 2026-10-27; '
    'that container had no Git metadata. Existing host expert-map and source-clone '
    'patches were inherited, not authored in this repair.\n')
print('wrote', out, flush=True)
