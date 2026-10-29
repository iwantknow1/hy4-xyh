import base64
import datetime
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/task')
OUT = ROOT / 'evaluation' / 'runs'
STATUS = ROOT / 'logs' / 'evaluation-pipeline.json'
API = 'http://177.11.25.1:18080'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RUN = os.environ.get('HY4_EVAL_RUN', '03')
state = {'status': 'waiting_service', 'judge': 'credential_preflight_passed', 'run': RUN, 'phases': {}}
RESUME_GPQA = os.environ.get('HY4_RESUME_AFTER_GPQA') == '1'
if RESUME_GPQA:
    previous = json.loads(STATUS.read_text())
    assert previous['run'] == RUN
    state['phases'] = previous['phases']
    state['resumed_after_gpqa'] = True


def save(**updates):
    state.update(updates)
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temp.replace(STATUS)
    print(json.dumps({'status': state['status'], 'updated_utc': state['updated_utc']}), flush=True)


def get_json(path):
    with OPENER.open(API + path, timeout=10) as response:
        return json.load(response)


def final_text(text):
    for marker in ('</think:opensource>', '</think>'):
        if marker in text:
            text = text.rsplit(marker, 1)[1].strip()
    if not text or '<think' in text:
        raise ValueError('Smoke response did not contain a completed final answer')
    return text


def chat(name, content):
    body = {'model': 'hy4', 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'max_tokens': 4096, 'stream': False}
    request = urllib.request.Request(API + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with OPENER.open(request, timeout=1200) as response:
        result = json.load(response)
    (ROOT / 'logs' / (name + '-' + RUN + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    choice = result['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('Smoke response was truncated')
    text = final_text(choice['message']['content'])
    if '\ufffd' in text:
        raise ValueError('Smoke response contains replacement characters')
    state['phases'][name] = {'status': 'passed', 'final_answer': text,
        'elapsed_seconds': time.monotonic()-started, 'usage': result.get('usage')}
    return text


def run_phase(name, command, *, env=None, monitor_service=True):
    save(status=name + '_running')
    with (ROOT / 'logs' / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        failures = 0
        while process.poll() is None:
            time.sleep(10)
            if not monitor_service or process.poll() is not None:
                continue
            try:
                with OPENER.open(API + '/health', timeout=5) as response:
                    healthy = response.status == 200
            except (OSError, urllib.error.URLError):
                healthy = False
            failures = 0 if healthy else failures + 1
            if failures >= 3:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f'{name}: serving health failed three consecutive checks; stopped evaluation')
    state['phases'][name] = {'exit_code': process.returncode,
        'status': 'completed' if process.returncode == 0 else 'failed',
        'log': str(ROOT / 'logs' / (name + '.log'))}
    save(status=name + '_finished')
    return process.returncode


def validate_gpqa():
    folder = OUT / ('gpqa-full-' + RUN)
    files = list(folder.glob('**/predictions/hy4-gpu-aligned/GPQA_diamond.jsonl'))
    if len(files) != 1:
        raise RuntimeError(f'Expected one GPQA prediction artifact, got {len(files)}')
    with files[0].open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    failed = 0
    for path in folder.glob('**/predictions/hy4-gpu-aligned/GPQA_diamond_failed.jsonl'):
        with path.open() as stream:
            failed += sum(bool(line.strip()) for line in stream)
    valid = [row for row in rows if row.get('success') is True
             and isinstance(row.get('prediction'), str) and row['prediction'].strip()]
    report = {'predictions': len(rows), 'successful_nonempty': len(valid),
              'unique_ids': len({row['id'] for row in rows}), 'failed': failed,
              'status': 'passed' if len(valid) == 198 and len(rows) == 198
              and len({row['id'] for row in rows}) == 198 and failed == 0 else 'failed'}
    (ROOT / 'logs' / ('gpqa-integrity-' + RUN + '.json')).write_text(json.dumps(report, indent=2))
    state['phases']['gpqa_integrity'] = report
    if report['status'] != 'passed':
        raise RuntimeError('GPQA incomplete or failed: ' + json.dumps(report))


if __name__ == '__main__':
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        save()
        deadline = time.monotonic() + 7200
        ready = False
        while time.monotonic() < deadline:
            try:
                with OPENER.open(API + '/health', timeout=5) as response:
                    ready = response.status == 200
                if ready and any(x['id'] == 'hy4' for x in get_json('/v1/models')['data']):
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            ready = False
            time.sleep(30)
        if not ready:
            raise TimeoutError('Full model did not become healthy within 2h; no evaluation requests sent')
        save(status='smoke_running')
        text = chat('full-text-smoke', 'What is 1 + 1? Give the final answer briefly.')
        if not re.search(r'\b2\b|二', text):
            raise ValueError('Text smoke answer does not contain the expected arithmetic result')
        from PIL import Image
        data = io.BytesIO()
        Image.new('RGB', (256, 256), color='red').save(data, format='PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
        text = chat('full-image-smoke', [
            {'type': 'image_url', 'image_url': {'url': uri}},
            {'type': 'text', 'text': 'What is the dominant color of this image? Reply briefly.'}])
        if not re.search(r'red|红', text, re.IGNORECASE):
            raise ValueError('Vision smoke answer does not identify the red image')
        save(status='smoke_passed')
        if not RESUME_GPQA:
            gpqa = run_phase('gpqa-full-' + RUN, [
                '/task/evaluation/bin/ais_bench', '/task/evaluation/gpqa-full-config-' + RUN + '.py',
                '--mode', 'all', '--num-prompts', '198', '--num-warmups', '0',
                '--max-num-workers', '1', '--dump-eval-details', '--dump-extract-rate', '--debug'])
            if gpqa != 0:
                raise RuntimeError('GPQA client failed; preserve the run before continuing')
        validate_gpqa()
        hall = run_phase('hallusion-full-' + RUN, [sys.executable,
            '/task/evaluation/hallusion/run_hallusionbench_hy4.py',
            '--data', '/mnt/share/z00949185/hy4-work/22.1/Hallusionbench_test/HallusionBench/HallusionBench.json',
            '--root', '/mnt/share/z00949185/hy4-work/22.1/Hallusionbench_test/HallusionBench/hallusion_bench',
            '--api', API + '/v1', '--model', 'hy4', '--max-tokens', '102400',
            '--temperature', '0.1', '--request-timeout', '7200',
            '--output', str(OUT / ('hallusion-full-' + RUN))])
        if hall == 0:
            result_path = OUT / ('hallusion-full-' + RUN) / 'HallusionBench_result.json'
            rows = json.loads(result_path.read_text())
            if len(rows) != 1129 or any(not row.get('model_prediction') for row in rows):
                raise RuntimeError('HallusionBench output does not contain 1129 nonempty predictions')
            credential = Path('/tmp/hy4-judge-auth/key').read_text().strip()
            match = re.search(r'sk-[A-Za-z0-9_-]+', credential)
            if match is None:
                raise RuntimeError('Judge credential file has no recognizable API key')
            env = os.environ.copy()
            env['HALLUSION_JUDGE_API_KEY'] = match.group(0)
            env['HY4_HALLUSION_RESULT'] = str(OUT / ('hallusion-full-' + RUN) / 'HallusionBench_result.json')
            env['HY4_JUDGE_OUTPUT'] = str(OUT / ('hallusion-judge-' + RUN))
            judge = run_phase('hallusion-judge-' + RUN,
                ['bash', '/task/scripts/run-hallusion-judge.sh'], env=env, monitor_service=False)
            state['judge'] = 'completed' if judge == 0 else 'failed'
            save(status='evaluation_finished' if judge == 0 else 'judge_failed')
        else:
            save(status='evaluation_failed')
    except Exception as error:
        save(status='blocked', error=f'{type(error).__name__}: {error}')
        raise
