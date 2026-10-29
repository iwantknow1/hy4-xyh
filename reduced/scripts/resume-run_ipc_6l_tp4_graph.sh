set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
for attempt in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
  sleep 2
done
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/skill-scripts/weight-update/test_npu_ipc_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-6L --devices 0,1,2,3 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --packed-buffer-mb 12288 --timeout 600 --report /codex/logs/resume-exclusive/ipc-6l-tp4-graph.json
