source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=6,7
export HCCL_NPU_SOCKET_PORT_RANGE=62100-62199
export HCCL_HOST_SOCKET_PORT_RANGE=62100-62199
export PATH=/codex/.venv/bin:/codex/tools:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd /codex
