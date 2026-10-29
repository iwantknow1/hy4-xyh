source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export HCCL_NPU_SOCKET_PORT_RANGE=62200-62299
export HCCL_HOST_SOCKET_PORT_RANGE=62200-62299
for attempt in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
    sleep 2
done
exec .venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py \
    --server-url http://127.0.0.1:18080 --served-model-name hy4-codex \
    --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 \
    --master-port 29611 --prompts-json /codex/scripts/prompts.json \
    --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-reinit-1.json
