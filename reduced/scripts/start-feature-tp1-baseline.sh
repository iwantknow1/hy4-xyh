set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_NPU_SOCKET_PORT_RANGE=62500-62599
export HCCL_HOST_SOCKET_PORT_RANGE=62500-62599
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 1 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18082
