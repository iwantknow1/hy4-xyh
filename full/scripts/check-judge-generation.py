import json
from pathlib import Path
import re
import time
import urllib.request

key = re.search(r'sk-[A-Za-z0-9_-]+', Path('/tmp/hy4-judge-auth/key').read_text()).group()
request = urllib.request.Request('https://api.deepseek.com/chat/completions',
    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
    data=json.dumps({'model': 'deepseek-flash', 'messages': [{'role': 'user',
        'content': 'The ground truth is yes. The prediction is yes. Reply only correct or incorrect.'}],
        'temperature': 0, 'max_tokens': 128}).encode())
start = time.monotonic()
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
answer = result['choices'][0]['message']['content'].strip().lower()
assert answer == 'correct', 'Synthetic judge response did not match expected label'
report = {'status': 'passed', 'model': result['model'], 'elapsed_seconds': time.monotonic()-start,
          'usage': result.get('usage'), 'private_key_copy': 'retained'}
Path('/task/logs/judge-generation-preflight-restored.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
