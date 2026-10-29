from pathlib import Path
p = Path('/codex/scripts/check_reload.py')
s = p.read_text().replace('return json.load(r)', "raw = r.read()\n        return json.loads(raw) if raw else {}")
s = s.replace("print('RELOAD', i + 1, flush=True)", """print('SLEEP_RELOAD', i + 1, flush=True)
    request('/sleep?level=2', {})
    request('/wake_up?tags=weights', {})""")
s = s.replace("    outputs = generate()", "    request('/wake_up?tags=kv_cache', {})\n    outputs = generate()")
Path('/codex/scripts/check_sleep_reload.py').write_text(s)
