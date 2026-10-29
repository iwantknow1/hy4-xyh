"""Summarize shared evidence without exposing model predictions or credentials."""
from pathlib import Path
import datetime
import json
import os
import re

root = Path('/task')
run = os.environ.get('HY4_EVAL_RUN', '04')
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'nodes': {}}
for node in range(25, 29):
    paths = [root / f'logs/node{node}/full-dp4-{run}.{stream}.log' for stream in ['stdout', 'stderr']]
    text = '\n'.join(p.read_text(errors='replace') for p in paths if p.exists())
    progress = re.findall(r'Loading safetensors checkpoint shards:\s*(\d+)% Completed\s*\|\s*(\d+)/246', text)
    errors = [line[-600:] for line in text.splitlines() if any(s in line for s in ['EnforceNotMet','preamble.length','OutOfMemoryError','died unexpectedly'])]
    loaded = [line[-300:] for line in text.splitlines() if any(s in line for s in ['Model loading took','Loading model weights took','Loading weights took','GPU KV cache size','NPU KV cache size'])]
    report['nodes'][node] = {'log_bytes':sum(p.stat().st_size for p in paths if p.exists()),'last_checkpoint_progress':progress[-1] if progress else None,'load_events':loaded[-2:],'failure_events':errors[-2:]}
status = root / 'logs/evaluation-pipeline.json'
if status.exists():
    report['evaluation'] = json.loads(status.read_text())
for name in ['gpqa-full-' + run,'hallusion-full-' + run, 'hallusion-judge-' + run]:
    path = root / 'logs' / (name+'.log')
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-12000))
            tail = stream.read().decode(errors='replace')
        tail = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', tail)
        rows = [s[:400] for s in tail.splitlines() if any(k in s for k in ['POST=', '/1129', 'accuracy', 'extract_rate', 'ERROR', 'qAcc'])]
        report[name] = rows[-2:]
print(json.dumps(report,ensure_ascii=False,indent=2))
(root/'logs/current-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
