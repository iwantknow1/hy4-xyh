import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

root = Path('/task')
client = root / 'evaluation/hallusion/run_hallusionbench_hy4.py'
expected = '9c58e21e44c3b2c3c2db0f0998c49b4b426aa5c47eb3f010c6e53fae6d26c7a1'
assert hashlib.sha256(client.read_bytes()).hexdigest() == expected
shutil.copy2(client, client.with_name('run_hallusionbench_hy4.original-timeout1200.py'))
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
def find_exact(script):
    matches = []
    for line in processes:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            args = parts[1].split()
            if len(args) >= 2 and args[0] == '/task/evaluation/.venv-client/bin/python' and args[1] == script:
                matches.append(int(parts[0]))
    assert len(matches) == 1, (script, matches)
    return matches[0]
controller = find_exact('/task/scripts/evaluate-when-ready.py')
hall = find_exact(str(client))
os.kill(controller, signal.SIGTERM)
os.kill(hall, signal.SIGTERM)
time.sleep(3)
folder = root / 'evaluation/runs/hallusion-full-07'
completed = len(json.loads((folder / 'partial_results.json').read_text())) if (folder / 'partial_results.json').exists() else 0
folder.rename(folder.with_name('hallusion-full-07-before-timeout7200'))
for name in ('hallusion-full-07', 'evaluation-pipeline-07'):
    path = root / f'logs/{name}.log'
    path.rename(path.with_name(name + '.before-timeout7200.log'))
shutil.copy2(root / 'scripts/run_hallusionbench_hy4.py', client)
env = dict(os.environ, HY4_EVAL_RUN='07', HY4_RESUME_AFTER_GPQA='1',
    TORCH_DEVICE_BACKEND_AUTOLOAD='0', PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH='/task/evaluation/benchmark:/mnt/share/z00949185/hy4-work/22.1/aisbench-venv/lib/python3.12/site-packages')
with (root / 'logs/evaluation-pipeline-07.log').open('w') as log:
    process = subprocess.Popen(['/task/evaluation/.venv-client/bin/python',
        '/task/scripts/evaluate-when-ready.py'], env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
report = {'status': 'restarted_hall_only', 'gpqa_results': 'preserved_and_revalidated',
    'old_timeout_seconds': 1200, 'new_timeout_seconds': 7200,
    'previous_partial_samples_archived': completed, 'new_controller_pid': process.pid,
    'concurrency': 'unchanged_sequential', 'generation_parameters': 'unchanged',
    'dataset_and_order': 'unchanged', 'server': 'not_restarted'}
(root / 'logs/hallusion-transport-timeout-change.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
