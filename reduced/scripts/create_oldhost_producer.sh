set -e
TASK=/data/x00951564/codex-hy4-20261027-oldhost-revalidation
OLD=/data/x00951564/codex-hy4-20261027
docker run -d --name x00951564-hy4-codex-oldhost-producer --network host --ipc=shareable --shm-size=16g --privileged --security-opt label=disable \
 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
 -v "$TASK:/codex" -v "$TASK/src/vllm:/vllm-workspace/vllm" -v "$TASK/src/vllm-ascend:/vllm-workspace/vllm-ascend" \
 -v /data/x00951564/models:/models:ro -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v "$OLD/tools/triton:/usr/local/python3.12.13/lib/python3.12/site-packages/triton:ro" \
 -v "$OLD/tools/triton_ascend-3.6.0.dist-info:/usr/local/python3.12.13/lib/python3.12/site-packages/triton_ascend-3.6.0.dist-info:ro" \
 --entrypoint /bin/bash vllm-ascend-hy4:a5-cann9.1 -c 'sleep infinity'
