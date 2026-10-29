#!/bin/bash
set -euo pipefail
rank=${1:?usage: run-inference-node.sh RANK LOCAL_IP MASTER_IP [MODEL_PATH]}
local_ip=${2:?local communication IP required}
master_ip=${3:?master communication IP required}
model=${4:-/mnt/model}
case "$rank" in 0|1|2|3) ;; *) echo 'RANK must be 0,1,2,3' >&2; exit 2;; esac
test -r "$model/config.json"
export PATH=/codex/.venv/bin:/usr/local/Ascend/cann-9.1.0/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=$local_ip HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=${HY4_NIC:-enp39s0f0}
export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME IP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61999 HCCL_HOST_SOCKET_PORT_RANGE=61000-61999
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
export VLLM_ENGINE_READY_TIMEOUT_S=3600 HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV
export TOKENIZERS_PARALLELISM=false PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
args=(serve "$model" --served-model-name hy4 --tensor-parallel-size 8
  --data-parallel-size 4 --data-parallel-size-local 1
  --data-parallel-address "$master_ip" --data-parallel-rpc-port 29740
  --trust-remote-code --distributed-executor-backend mp --enable-expert-parallel
  --max-model-len 147456 --max-num-seqs 64 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.95 --kv-cache-memory-bytes 15032385536
  --enable-prefix-caching --async-scheduling --seed 0
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_load_config":{}}')
if [[ "$rank" == 0 ]]; then
  args+=(--host 0.0.0.0 --port 18080 --api-server-count 1)
else
  args+=(--headless --data-parallel-start-rank "$rank")
fi
exec hy4-entrypoint /codex/.venv/bin/python -m vllm.entrypoints.cli.main "${args[@]}"
