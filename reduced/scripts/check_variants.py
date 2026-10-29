import json
import math
import pathlib
import sys
import time
import urllib.request

root = pathlib.Path('/codex/logs')
label = sys.argv[1]
base = 'http://127.0.0.1:18080'
def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}
for _ in range(150):
    try:
        request('/v1/models')
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError('server not ready')
def generate():
    outputs = {}
    for endpoint in ('completions','chat/completions'):
        items = []
        for prompt in ('What is 2 + 2?', 'Write one word about the sky.'):
            payload = dict(model='hy4-codex', temperature=0, seed=42, max_tokens=16)
            if endpoint == 'completions':
                payload.update(prompt=prompt, logprobs=1)
            else:
                payload.update(messages=[{'role':'user','content':prompt}], logprobs=True, top_logprobs=1)
            response = request('/v1/' + endpoint, payload)
            c = response['choices'][0]
            if endpoint == 'completions':
                entry = dict(text=c['text'], tokens=c['logprobs']['tokens'], logprobs=c['logprobs']['token_logprobs'])
            else:
                logs = c['logprobs']['content']
                entry = dict(text=c['message']['content'], tokens=[x['token'] for x in logs], logprobs=[x['logprob'] for x in logs])
            assert entry['tokens'] and len(entry['tokens']) == len(entry['logprobs'])
            assert all(math.isfinite(x) for x in entry['logprobs'])
            items.append(entry)
        outputs[endpoint] = items
    return outputs

report = {'label':label, 'baseline':generate()}
path = root / (label + '.json')
for variant in ('sink-B', 'expert-C'):
    print('LOAD', variant, flush=True)
    request('/collective_rpc', {'method':'reload_weights', 'kwargs':{
        'is_checkpoint_format':True, 'weights_path':'/codex/variants/' + variant}})
    request('/reset_prefix_cache', {})
    updated = generate()
    report[variant] = updated
    report[variant + '_changed'] = {
        endpoint:any(a != b for a,b in zip(report['baseline'][endpoint], updated[endpoint]))
        for endpoint in updated
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('changed', report[variant + '_changed'], flush=True)
    assert all(report[variant + '_changed'].values())
    request('/collective_rpc', {'method':'reload_weights', 'kwargs':{
        'is_checkpoint_format':True, 'weights_path':'/models/Hy4-2L'}})
    request('/reset_prefix_cache', {})
    restored = generate()
    report[variant + '_restored'] = restored
    report[variant + '_restored_identical'] = restored == report['baseline']
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('restored', report[variant + '_restored_identical'], flush=True)
    assert report[variant + '_restored_identical']
print('PASS sink and expert A-B-A both endpoints', flush=True)
