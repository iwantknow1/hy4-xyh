#!/bin/bash
set -euo pipefail
node_ip=${2##*_}
case "$node_ip" in 197.13.49.1|197.13.5[0-2].1) ;; *) echo 'Unexpected node identity in runtime directory' >&2; exit 2;; esac
export VLLM_HOST_IP=$node_ip HCCL_IF_IP=$node_ip
export GLOO_SOCKET_IFNAME=enp39s0f0 HCCL_SOCKET_IFNAME=enp39s0f0 IP_SOCKET_IFNAME=enp39s0f0
exec hy4-entrypoint bash "$1"
