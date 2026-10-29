import json
import sys
import torch
sys.path[:0]=['/codex/scripts/skill-scripts/weight-update/common','/codex/scripts/skill-scripts']
from checkpoint_weight_source import CheckpointWeightSource
source=CheckpointWeightSource('/models/Hy4-6L',torch.device('cpu'))
summary=source.summary()
with open('/codex/logs/resume-exclusive/6l-checkpoint-summary.json','w') as stream:
    json.dump(summary,stream,indent=2)
print(json.dumps(summary,indent=2))
