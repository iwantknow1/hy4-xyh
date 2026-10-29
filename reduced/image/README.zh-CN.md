# HY4 A5 减层版验证镜像

修复源码、uv `.venv`、验证脚本和报告均已烘焙入镜像；运行时无需挂载源码目录。模型权重与 Ascend 驱动仍由宿主机只读挂载。镜像默认进入 shell，不自动占卡。

目标基础环境：Ascend 950DT、CANN 9.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4。完整结论见 `/codex/model_openbox/hy4/VERIFICATION-REPORT.zh-CN.md`。

示例（先核对 2、3 卡的分配）：

```bash
docker run --rm --name hy4-reuse --network host --ipc=shareable --shm-size=64g --privileged \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/x00951564/models:/models:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=2,3 \
  -e MODEL_PATH=/models/Hy4-6L -e TP_SIZE=2 \
  -e HCCL_NPU_SOCKET_PORT_RANGE=62800-62899 \
  -e HCCL_HOST_SOCKET_PORT_RANGE=62800-62899 \
  hy4-codex:a5-cann9.1-20261027-verified hy4-serve
```

启动器支持环境变量：MODEL_PATH、TP_SIZE、SERVED_MODEL_NAME、PORT、KV_CACHE_BYTES、MAX_MODEL_LEN、MAX_NUM_SEQS、MEMORY_UTILIZATION、GRAPH_CONFIG、WEIGHT_TRANSFER_BACKEND（hccl/npu_ipc/none）。R3 另设 `VLLM_USE_V2_MODEL_RUNNER=1` 并追加 `--enable-return-routed-experts`。MTP 减层资产 `/codex/models/Hy4-2L-MTP` 通过符号链接引用 `/models/Hy4-2L` 和 `/models/Hy4-32L-MTP`，因此该项复用时两套外部资产都要存在。

HCCL/IPC CPU staging 是显式选项；对应已验证 producer 脚本位于 `/codex/scripts/skill-scripts/weight-update/`。复用前替换脚本中服务器 URL、设备、端口和报告目录。普通推理可将 WEIGHT_TRANSFER_BACKEND 设为 none。

镜像只证明报告中列明的减层功能，不代表全量精度或全量性能通过。DCP、PCP 在当前 HY4 attention sink 实现中被明确拒绝。
