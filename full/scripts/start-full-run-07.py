import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/task')
container = 'x00951564-hy4-codex-20261028'
ssh = ['ssh', '-i', '/tmp/hy4-ssh/id_ed25519', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10']

def remote(host, args, timeout=180):
    for attempt in range(3):
        result = subprocess.run(ssh + ['root@' + host, shlex.join(args)],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result
        if result.returncode != 255 or attempt == 2:
            raise RuntimeError(f'{host}: {result.stderr[-1500:]}')
        time.sleep(3)

config = root / 'evaluation/gpqa-full-config-07.py'
assert not (root / 'evaluation/runs/gpqa-full-07').exists()
assert not (root / 'evaluation/runs/hallusion-full-07').exists()
config.write_text((root / 'evaluation/gpqa-full-config-06.py').read_text().replace('gpqa-full-06', 'gpqa-full-07'))
for node, subnet in zip(range(25, 29), range(49, 53)):
    host = f'197.13.{subnet}.1'
    runtime_file = root / f'logs/node{node}/dp4-runtime-06.txt'
    runtime = runtime_file.read_text().strip()
    for stream in ('stdout', 'stderr'):
        remote(host, ['docker', 'exec', container, 'cp', runtime + '/' + stream + '.log',
            f'/task/logs/node{node}/full-dp4-06.{stream}.final.log'])
    remote(host, ['docker', 'restart', '-t', '10', container], timeout=60)
    probe = remote(host, ['docker', 'exec', container, 'hy4-entrypoint',
        '/codex/.venv/bin/python', '/task/scripts/probe-slot-mapping.py'])
    (root / f'logs/node{node}/slot-mapping-path-preflight.log').write_text(probe.stdout + probe.stderr)
    assert '"status": "passed"' in probe.stdout
    print(json.dumps({'node': node, 'restart': 'own_container_only', 'slot_mapping': 'passed'}), flush=True)

env = dict(__import__('os').environ, HY4_EVAL_RUN='07', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
           PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH='/task/evaluation/benchmark:/mnt/share/z00949185/hy4-work/22.1/aisbench-venv/lib/python3.12/site-packages')
with (root / 'logs/evaluation-pipeline-07.log').open('w') as log:
    evaluator = subprocess.Popen(['/task/evaluation/.venv-client/bin/python',
        '/task/scripts/evaluate-when-ready.py'], stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True)
with (root / 'logs/dp4-serving-launch-07.log').open('w') as log:
    launcher = subprocess.Popen(['bash', '/task/scripts/launch-dp4-skill.sh'],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'evaluation_pid': evaluator.pid, 'launcher_pid': launcher.pid}), flush=True)

seen = set()
deadline = time.monotonic() + 1200
while time.monotonic() < deadline and len(seen) < 4:
    for node, subnet in zip(range(25, 29), range(49, 53)):
        if node in seen:
            continue
        host = f'197.13.{subnet}.1'
        result = remote(host, ['docker', 'exec', container, 'sh', '-c',
            'ls -dt /home/vllm/.workspace-runtime/serving/* 2>/dev/null | head -1'])
        runtime = result.stdout.strip()
        old = (root / f'logs/node{node}/dp4-runtime-06.txt').read_text().strip()
        if not runtime or runtime == old:
            continue
        remote(host, ['docker', 'exec', '-d', container, 'bash',
            '/task/scripts/sync-service-logs.sh', str(node), runtime, '07'])
        seen.add(node)
        print(json.dumps({'node': node, 'runtime': runtime, 'log_sync': 'started'}), flush=True)
    time.sleep(5)
assert len(seen) == 4, seen
print(json.dumps({'status': 'launched', 'run': '07'}), flush=True)
