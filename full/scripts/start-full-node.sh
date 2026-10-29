#!/bin/bash
set -euo pipefail
rank=${1:?node rank 0..3}
case "$rank" in 0|1|2|3) ;; *) exit 2;; esac
node=$((25+rank))
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PP_LAYER_PARTITION=21,20,20,17
export VLLM_HOST_IP=177.11.${node}.1
export HCCL_IF_IP=$VLLM_HOST_IP
export HCCL_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=enp35s0f2
export IP_SOCKET_IFNAME=enp35s0f2
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61999
export HCCL_HOST_SOCKET_PORT_RANGE=61000-61999
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
args=(serve /mnt/share/z30059446/hy4_safetensors
  --served-model-name hy4 --trust-remote-code
  --distributed-executor-backend mp --nnodes 4 --node-rank "$rank"
  --master-addr 177.11.25.1 --master-port 29720
  --tensor-parallel-size 8 --pipeline-parallel-size 4 --enable-expert-parallel
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 8589934592
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
  --host 0.0.0.0 --port 18080)
if [ "$rank" -gt 0 ]; then args+=(--headless); fi
exec hy4-entrypoint /codex/.venv/bin/python -m vllm.entrypoints.cli.main "${args[@]}"
