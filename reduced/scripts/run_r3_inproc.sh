set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python /codex/scripts/r3_inproc.py
