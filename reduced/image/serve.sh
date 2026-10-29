#!/bin/bash
set -e
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the assigned physical NPU IDs}"
graph='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
args=(serve "${MODEL_PATH:-/models/Hy4-6L}" --served-model-name "${SERVED_MODEL_NAME:-hy4-codex}" --trust-remote-code --tensor-parallel-size "${TP_SIZE:-2}" --enable-expert-parallel --enable-sleep-mode --load-format auto --gpu-memory-utilization "${MEMORY_UTILIZATION:-0.85}" --additional-config '{"rl_config":{"enabled":true}}' --kv-cache-memory-bytes "${KV_CACHE_BYTES:-8589934592}" --max-model-len "${MAX_MODEL_LEN:-2048}" --max-num-seqs "${MAX_NUM_SEQS:-8}" --compilation-config "${GRAPH_CONFIG:-$graph}" --port "${PORT:-18080}")
backend=${WEIGHT_TRANSFER_BACKEND:-hccl}
case "$backend" in
  hccl) args+=(--weight-transfer-config '{"backend":"hccl"}') ;;
  npu_ipc) export VLLM_ALLOW_INSECURE_SERIALIZATION=1; args+=(--weight-transfer-config '{"backend":"npu_ipc"}') ;;
  none) ;;
  *) echo "Unsupported WEIGHT_TRANSFER_BACKEND=$backend" >&2; exit 2 ;;
esac
exec /codex/.venv/bin/python -m vllm.entrypoints.cli.main "${args[@]}" "$@"
