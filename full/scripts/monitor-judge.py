"""Read progress without printing credentials or prediction content."""
import json
from pathlib import Path
import time

root = Path('/task')
state = json.loads((root / 'logs/evaluation-pipeline.json').read_text())
result = {'pipeline': state['status'], 'judge': {}}
for path in sorted((root / 'evaluation/runs/hallusion-judge-07').glob('hallusion_output*.json')):
    for attempt in range(5):
        try:
            rows = json.loads(path.read_text())
            break
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.2)
    result['judge'][path.stem] = {
        'rows': len(rows),
        'correctness_completed': sum('gpt4v_output_gpt_check' in row for row in rows),
        'consistency_completed': sum('same' in row for row in rows),
    }
final = root / 'logs/final-report-run07.json'
if final.is_file():
    result['final_report'] = json.loads(final.read_text())
print(json.dumps(result, ensure_ascii=False))
