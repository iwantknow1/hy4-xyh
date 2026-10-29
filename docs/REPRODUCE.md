# 复现、镜像与评测恢复

## 镜像和配置

优先阅读 [镜像复用说明](../full/scripts/IMAGE-REUSE.zh-CN.txt) 与 [正式启动命令](../full/logs/launch_commands-final-run07.txt)。四机最终镜像：

- tag：`hy4-codex:a5-cann9.1-four-node-run07-20261029`
- image ID：`sha256:b4f23be23aa05d77b33c5b1fa7070398986f5b40e3720c18c29289fce866322a`
- 归档：`/mnt/share/x00951564/codex-hy4-20261028-four-node/images/hy4-codex-a5-cann9.1-four-node-run07-20261029.tar.gz`
- bytes：`10233176716`
- SHA256：`b73f6ad30f12a09cf8760b15f4e8cc698445cef2a4280031d85e38ebfc946969`

核对归档后 `docker load -i <archive>`。模型外部只读挂载；不将新裁判凭据写进镜像。节点 helper 在 `/usr/local/share/hy4-codex/run07/run-inference-node.sh`，源码副本也见本仓库 `full/scripts/run-inference-node.sh`。实际镜像挂载/权限和节点参数以复用说明为准。

正式全量：TP8/DP4/PP1/EP32、MTP3、seq64、batch token8192、model len147456、KV14GiB/卡、单 API、ready timeout3600、200Gb NIC。必须继承完整 CANN PATH。不能把管理网 1Gb 路径和 200Gb 数据网混用。

## 代码差异

组合补丁相对于 `fixes/before/` 根目录；已执行 `git apply --check ../patches/combined.diff`。`fixes/after/` 是实测最终文件，不是最新上游。实际移植需核对 donor commit 和依赖，不能直接向任意版本覆盖。原始代码说明中的“尚未推送”描述当时修复代码没有进入开发仓库；本证据仓库上传不代表上游修复合入。

## 精度和打分

先核对数据 SHA、请求顺序、选项/gold 与生成参数，再复用 `full/evaluation/run-gpqa.sh`、`full/evaluation/gpqa-full-config-07.py` 和 `full/evaluation/hallusion/`。CPU 客户端依赖与 NPU 推理环境隔离；使用 uv 管理的环境及其 `.venv/bin/python`，不升级推理环境中的 torch 来满足客户端。

Hallusion 评分需通过环境变量注入自己的 `HALLUSION_JUDGE_API_KEY`，模型 `deepseek-flash`，API base `https://api.deepseek.com`。本仓库不含 API key。评分目录必须属于当前模型输出，不能混用参考已有缓存。

本次原始 VD/VS JSON 都已经完整；**不要为了读取报告再调用 API 重打分**。若后续自己的评分中断，可在原运行目录恢复原 evaluator，用该轮缓存跳过完成项；先备份并校验 JSON，避免中断写入损坏。`run-hallusion-judge.sh` 的新目录保护是防止混用历史分数，恢复已有运行不能直接重用它新建目录。

下一次使用有限并发前，需要在每个独立阶段内调度请求、保留阶段依赖、按样本标识回填、单独序列化缓存写入、限流退避并验证完整覆盖。并发不改评分标准；在线裁判本身非固定快照，不能保证逐次分数完全一致。本次仍为参考原脚本串行评分。
