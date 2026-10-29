set -e
source /codex/scripts/env.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --weight-transfer-config '{"backend":"npu_ipc"}' --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --enforce-eager --port 18080
