set -e
source /codex/scripts/env.sh
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --max-model-len 2048 --max-num-seqs 8 --enforce-eager --port 18080
