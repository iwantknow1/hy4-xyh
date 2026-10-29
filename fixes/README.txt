HY4 A5 减层模型修复审阅目录

本目录运行 git diff 可查看修改前后 9 个文件差异；暂存区保存 donor 原文件，工作区为最终修复，未提交。
补丁：patches/weight-transfer-fix.diff
最终报告：oldhost-revalidation-20261027/model_openbox/hy4/VERIFICATION-REPORT.zh-CN.md
镜像验证和复用：oldhost-revalidation-20261027/model_openbox/hy4/IMAGE-VALIDATION.md
Excel：oldhost-revalidation-20261027/performance/hardware_efficiency.xlsx
原始证据：oldhost-revalidation-20261027/logs/resume-exclusive/

最终 9 份本地源码 SHA-256 与已验证镜像完全一致。98 个单元测试通过，ruff 通过。
修复包括 sink 原生 weight_loader、专家 meta 加载计数、HCCL 整流单次加载、
finalize 前后空闲缓存回收、IPC 多块 CPU 暂存和每块独立引用计数。
其中 finalize 前回收修复了 6L reload 的容量失败，保留 live tensor storage。

已验证：2L/6L checkpoint、HCCL、IPC 和图模式；另含 SP、合法分层 PP、DP/EP、
R3V2、MTP、pause/resume、实际多模态长输入 profiling。详细拓扑和版本边界见报告。
MTP 60 个草稿 token 接受数为 0，不宣称加速收益。
当前 HY4 Ascend sink 实现显式拒绝 PCP>1 或 DCP>1；未删除保护条件。
跨机器 DP/PP、原始 TP8 配置、全量模型精度仍未验证。
此前新机器 AICPU 重建失败在老机器 3 次重建未复现，原始根因仍未证明。
Excel 指定私有包返回 404，已用 Windows 下载的公开库生成并独立复算。

镜像：hy4-codex:a5-cann9.1-20261027-verified
ID：sha256:81d4c9171a24404d669f6b161c4f3abea70bb928e534126bea29d57198ecfa45
共享归档：/mnt/share/x00951564/codex-transfer-20261027/hy4-codex-a5-cann9.1-20261027-verified.tar.gz
归档 SHA-256：18d57f4dd20279e6594daabdc3a03c55b209c43b8a706dad5cfaa0d723282f40
归档大小：10150827612 字节，gzip 完整性检查通过。
镜像内置修复源码和 Python/Triton 环境；模型权重和宿主 driver 需按说明只读挂载。
最终镜像 6L TP2 图模式 HCCL 810 张量更新通过，两个接口 text/token/logprob 严格一致。
所有本任务 NPU 进程已停止、卡已释放。

远端独立目录：177.8.21.1:/data/x00951564/codex-hy4-20261027-oldhost-revalidation
不修改 /home/a/src/vllm，不覆盖 DeepSeek 的源码或容器。
