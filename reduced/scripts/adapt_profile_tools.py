import pathlib,json,hashlib
root=pathlib.Path('/codex/tooling');changes=[]
for p in root.rglob('*.py'):
    s=p.read_text();before=hashlib.sha256(p.read_bytes()).hexdigest();new=s.replace('python3 -','/codex/.venv/bin/python -').replace('python -','/codex/.venv/bin/python -').replace('"python3"','"/codex/.venv/bin/python"').replace("'python3'","'/codex/.venv/bin/python'")
    # A second replacement may see the suffix of an already absolute command.
    new=new.replace('/codex/.venv/bin//codex/.venv/bin/python','/codex/.venv/bin/python')
    if new!=s:p.write_text(new);changes.append({'path':str(p),'before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pathlib.Path('/codex/logs/resume-exclusive/profile-tool-adaptations.json').write_text(json.dumps({'reason':'All Python subprocesses must use the existing uv-created .venv; no profiler settings or acceptance conditions altered.','changes':changes},indent=2))
