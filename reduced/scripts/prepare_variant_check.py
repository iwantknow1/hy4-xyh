from pathlib import Path
source = Path('/codex/scripts/check_reload.py').read_text()
prefix = source.split("report = {'label':label, 'baseline':generate()}")[0]
prefix = prefix.replace('return json.load(r)', "raw = r.read()\n        return json.loads(raw) if raw else {}")
body = """
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
"""
Path('/codex/scripts/check_variants.py').write_text(prefix + body)
