# 镜像自包含与实机复验

- 镜像：`hy4-codex:a5-cann9.1-20261027-verified`
- Image ID：`sha256:81d4c9171a24404d669f6b161c4f3abea70bb928e534126bea29d57198ecfa45`
- 未压缩大小：26706248200 字节
- 源码身份：`af15727eeeb457eec1a680903b722c882af94280f3759e1b32a31e864b0ca6e6`（9 个源文件校验通过）

推理容器没有源码或 .venv 挂载，仅只读挂载模型目录和宿主驱动。使用镜像内置源码、uv 环境及 Triton 3.6.0，6L TP2 FULL_DECODE_ONLY 捕获 [1,2,4,8]，完整更新 810 个张量；两个接口、各两条 prompt 的 text/token/logprob 与更新前完全一致。

相同冻结生产源码的镜像内 98 项 CPU 单元测试通过。CPU mock 测试单独设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0；实际 NPU 服务不设置该开关。包元数据 torch=2.10.0、运行时 torch.__version__=2.10.0+cpu，校验脚本已区分两者。最初的校验/fixture 失败日志保留。最终派生层仅补充 Excel、报告、公开 JS 导出依赖及校验脚本，没有修改模型/权重传输生产代码。

实机验证完成后自有推理和 producer 容器均已停止，NPU 进程表确认 5–7 卡释放。6、7 卡的历史 Warning 状态未擅自复位。

导出位置和最终 SHA256 见 IMAGE-METADATA.json、IMAGE-ARCHIVE.sha256；归档使用 docker save + gzip，包含完整镜像层，模型权重不打包。加载：`docker load -i /mnt/share/x00951564/codex-transfer-20261027/hy4-codex-a5-cann9.1-20261027-verified.tar.gz`。启动参数见 IMAGE-README.zh-CN.md。

镜像归档已完成，gzip 完整性检查通过。

路径：`/mnt/share/x00951564/codex-transfer-20261027/hy4-codex-a5-cann9.1-20261027-verified.tar.gz`

SHA-256：`18d57f4dd20279e6594daabdc3a03c55b209c43b8a706dad5cfaa0d723282f40`

大小：10150827612 字节。
