import json
from pathlib import Path
import subprocess
import sys
sys.path[:0] = ['/codex/scripts/skill-scripts/weight-update/common', '/codex/scripts/skill-scripts']
from weight_update_validation import WeightUpdateValidation
report={'feature':'pause_resume', 'checks':{}}
validation=WeightUpdateValidation('http://127.0.0.1:18080','hy4-codex',['What is 2 + 2?','Write one word about the sky.'],16,600,report)
validation.record_baseline()
subprocess.run([sys.executable,'/codex/scripts/skill-scripts/lifecycle/test_pause_resume.py','--server-url','http://127.0.0.1:18080','--model','hy4-codex','--max-tokens','16','--report','/codex/logs/pause-domain.json'],check=True)
validation.verify_updated_outputs()
report['status']='passed'
Path('/codex/logs/pause-consistency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS pause/resume dual-endpoint exact consistency')
