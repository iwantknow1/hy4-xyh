import hashlib
import json
from pathlib import Path
import statistics

root = Path('/task')
files = list((root / 'evaluation/runs/gpqa-full-07').glob('**/predictions/hy4-gpu-aligned/GPQA_diamond.jsonl'))
assert len(files) == 1
with files[0].open() as stream:
    rows = sorted((json.loads(line) for line in stream if line.strip()), key=lambda row: row['id'])
assert [row['id'] for row in rows] == list(range(198))
payload = ''.join(json.dumps({'prompt': row['origin_prompt'][0]['prompt'], 'gold': row['gold']},
    ensure_ascii=False) + '\n' for row in rows).encode()
sha = hashlib.sha256(payload).hexdigest()
assert sha == '1705137667c5b9b4e2ad5b3a85a0d94103bad3c303f41aa24609203771a91e2d'
tokens = [row['output_tokens'] for row in rows]
unclosed = [row['id'] for row in rows if '<think' in row['prediction']
            and '</think' not in row['prediction']]
report = {'status': 'passed', 'rows': len(rows), 'exact_input_sha256': sha,
    'raw_prediction_keys': sorted(rows[0]),
    'successful': sum(row['success'] is True for row in rows),
    'output_tokens': {'min': min(tokens), 'max': max(tokens),
        'mean': statistics.mean(tokens), 'median': statistics.median(tokens), 'sum': sum(tokens)},
    'output_tokens_at_limit_102400': sum(value >= 102400 for value in tokens),
    'unclosed_thinking_prediction_ids': unclosed,
    'native_result': str(files[0]).replace('/predictions/', '/results/').removesuffix('l')}
(root / 'logs/gpqa-full-audit-07.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
