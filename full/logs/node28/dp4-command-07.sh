#!/bin/bash
echo "using mandatory comm NIC: data0.3001 (local_ip=197.13.52.1)"
[ -n "$HCCL_IF_IP" ] || export HCCL_IF_IP=197.13.52.1
[ -n "$GLOO_SOCKET_IFNAME" ] || export GLOO_SOCKET_IFNAME=data0.3001
[ -n "$IP_SOCKET_IFNAME" ] || export IP_SOCKET_IFNAME=data0.3001
[ -n "$HCCL_SOCKET_IFNAME" ] || export HCCL_SOCKET_IFNAME=data0.3001
exec vllm serve /mnt/share/z30059446/hy4_safetensors --host 0.0.0.0 --port 18080 --served-model-name hy4 --tensor-parallel-size 8 --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-address 197.13.49.1 --data-parallel-rpc-port 29740 --data-parallel-start-rank 3 --headless --trust-remote-code --distributed-executor-backend mp --enable-expert-parallel --max-model-len 147456 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes 15032385536 --enable-prefix-caching --async-scheduling --seed 0 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_load_config":{}}'
