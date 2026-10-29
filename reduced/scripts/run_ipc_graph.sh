set -e
source /codex/scripts/env.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
for attempt in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
  sleep 2
done
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/skill-scripts/weight-update/test_npu_ipc_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --devices 0,1 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --packed-buffer-mb 12288 --timeout 600 --report /codex/logs/ipc-graph.json
