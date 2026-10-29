"""Release only our NPUs after validated generation, then archive our image."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/task')
own = 'x00951564-hy4-codex-20261028'
tag = 'hy4-codex:a5-cann9.1-four-node-run07-20261029'
archive_dir = '/mnt/share/x00951564/codex-hy4-20261028-four-node/images'
archive = archive_dir + '/hy4-codex-a5-cann9.1-four-node-run07-20261029.tar.gz'
ssh = ['ssh', '-i', '/tmp/hy4-ssh/id_ed25519', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']

def remote(host, args, timeout=120):
    for attempt in range(3):
        result = subprocess.run(ssh + ['root@' + host, shlex.join(args)],
            capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode != 255 or attempt == 2:
            raise RuntimeError(f'{host}: {result.stderr[-1500:]}')
        time.sleep(3)

deadline = time.monotonic() + 18 * 3600
while time.monotonic() < deadline:
    state = json.loads((root / 'logs/evaluation-pipeline.json').read_text())
    if state.get('run') == '07' and state.get('status') in (
            'hallusion-judge-07_running', 'evaluation_finished', 'judge_failed'):
        break
    if state.get('run') == '07' and state.get('status') in ('blocked', 'evaluation_failed'):
        raise RuntimeError('Evaluation failed before generation validation; image not saved')
    time.sleep(15)
else:
    raise TimeoutError('Generation did not finish within watcher window')

integrity = json.loads((root / 'logs/gpqa-integrity-07.json').read_text())
assert integrity['status'] == 'passed' and integrity['successful_nonempty'] == 198
rows = json.loads((root / 'evaluation/runs/hallusion-full-07/HallusionBench_result.json').read_text())
assert len(rows) == 1129 and all(row.get('model_prediction') for row in rows)
head = '197.13.49.1'
recipe = '/usr/local/share/hy4-codex/run07'
remote(head, ['docker', 'exec', own, 'mkdir', '-p', recipe])
for name in ('launch-dp4-skill.sh', 'serve-with-cann.sh', 'RUN-07-NOTES.zh-CN.txt'):
    remote(head, ['docker', 'exec', own, 'cp', '/task/scripts/' + name, recipe + '/' + name])
for node, subnet in zip(range(25, 29), range(49, 53)):
    host = f'197.13.{subnet}.1'
    runtime = (root / f'logs/node{node}/dp4-runtime-07.txt').read_text().strip()
    for stream in ('stdout', 'stderr'):
        remote(host, ['docker', 'exec', own, 'cp', runtime + '/' + stream + '.log',
            f'/task/logs/node{node}/full-dp4-07.{stream}.final.log'])
    remote(host, ['docker', 'stop', '-t', '30', own])
    print(json.dumps({'node': node, 'own_container_stopped': True}), flush=True)

image_id = remote(head, ['docker', 'commit', own, tag], timeout=1800)
remote(head, ['mkdir', '-p', archive_dir])
command = ('set -euo pipefail; test ! -e ' + shlex.quote(archive) + '; '
    'docker save ' + shlex.quote(tag) + ' | gzip -1 > ' + shlex.quote(archive + '.partial') + '; '
    'mv ' + shlex.quote(archive + '.partial') + ' ' + shlex.quote(archive))
remote(head, ['bash', '-c', command], timeout=3600)
sha = remote(head, ['sha256sum', archive], timeout=1200).split()[0]
size = int(remote(head, ['stat', '-c', '%s', archive]))
report = {'status': 'saved', 'generation_validated': True,
    'accuracy_status': 'see evaluation-pipeline.json; judge may still be running',
    'tag': tag, 'image_id': image_id, 'archive': archive, 'bytes': size,
    'sha256': sha, 'source_container': own, 'cpu_evaluation_container_saved': False,
    'saved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root / 'logs/reusable-image-run07.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
