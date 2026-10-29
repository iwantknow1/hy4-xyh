import subprocess
import shlex

for node, subnet in zip(range(25, 29), range(49, 53)):
    runtime = f'/home/vllm/.workspace-runtime/serving/20261028_175536_197.13.{subnet}.1'
    command = ['docker', 'exec', '-d', 'x00951564-hy4-codex-20261028', 'bash',
               '/task/scripts/sync-service-logs.sh', str(node), runtime, '07']
    subprocess.run(['ssh', '-i', '/tmp/hy4-ssh/id_ed25519', '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=accept-new', f'root@197.13.{subnet}.1',
        shlex.join(command)], check=True, timeout=30)
    print(f'node {node} log sync started', flush=True)
