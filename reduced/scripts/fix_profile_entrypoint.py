from pathlib import Path
import hashlib,json
root=Path('/codex/tooling/skills/inference-model-skills')
changes=[]
for p,old,new in [
(root/'deployment/vllm-ascend-serving/scripts/serve_start.py','argv_tokens = ["vllm", "serve", shlex.quote(model)]','argv_tokens = ["/codex/.venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", shlex.quote(model)]'),
(root/'performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py','"vllm", "bench", "serve", "--backend"','sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend"')]:
    s=p.read_text();assert old in s;before=hashlib.sha256(p.read_bytes()).hexdigest();p.write_text(s.replace(old,new));changes.append({'path':str(p),'before':before,'after':hashlib.sha256(p.read_bytes()).hexdigest()})
Path('/codex/logs/resume-exclusive/profile-entrypoint-fix.json').write_text(json.dumps({'reason':'The uv verification environment exposes the vLLM Python module, without a vllm console script. Invoke the same module with .venv/bin/python. Serving connection mode remains docker; internal state field mode=direct describes SSH transport, not bare-host execution. No inference or profiler acceptance behavior changed.','files':changes},indent=2))
