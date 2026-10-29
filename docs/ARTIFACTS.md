# 交付物与路径映射

## skill 输出

- `reduced/model_openbox/hy4/`：自动生成 README、audit、manifest、逐项 cases、总报告、镜像验证记录。
- `reduced/logs/`：减层原始 stdout/stderr、逐接口输出、权重 manifest 和回归日志。
- `reduced/performance/`：Excel、硬件测算 JSON、导出依赖与校验记录。
- `reduced/profiles/`：两个 rank 的原始采集、解析 CSV/JSON、框架与设备数据。
- `full/model_openbox/hy4/`：全量阶段 skill 自动 README、manifest、最终 report、四项报告 JSON。
- `full/model_openbox/hy4/audit.md`：归档补充的审计入口，原始模型/容量审计在 `full/logs/`。
- `full/logs/launch_commands-final-run07.txt`：正式服务和评测实际命令；仅保留成功正式轮次。

## 原始数据与日志

- `full/evaluation/runs/gpqa-full-07/20261028_180756/`：198 题逐条预测、配置、分数与汇总。
- `full/evaluation/gpqa-exact-prompts.jsonl`：与参考对齐的 GPQA 输入；对应 SHA 见审计。
- `full/evaluation/runs/hallusion-full-07/`：1129 题最终答案、逐条 API 响应、token 和停止原因。
- `full/evaluation/runs/hallusion-judge-07/`：实际评分输入、VD/VS 逐题打分缓存和原始 evaluator/helper。
- `full/logs/hallusion-judge-07.log`：完整裁判日志和最终统计表。
- `full/logs/node25/` 至 `node28/`：实际服务命令、源码哈希、完整服务 stdout/stderr和成功轮次完整运行记录。
- 失败、不完整、被替代轮次和诊断失败数据不上传；仅在 `docs/PROCESS.md` 说明遇到的问题与处理方式。

没有复制模型权重、图片素材原库、第三方 AISBench 全部依赖、容器镜像或进程 core。权重/图片素材通过原共享存储路径、校验值和参考仓库追溯；本仓库保留获得的模型输出、裁判结果及性能原始数据。不得把本仓库当作包含所有依赖和权重的一键离线包。

## 关键脚本和修改

- `reduced/scripts/`：各特性启动、权重传输、一致性、Profiling 与 Excel 脚本；包含当时使用的 skill 脚本副本。
- `full/scripts/`：四机预检/拉起、通信诊断、完整性审计、评测编排、镜像保存和报告生成。
- `full/evaluation/`：正式 GPQA 配置、生成/评分入口、原始 Hallusion evaluator 和仅 HTTP timeout 改动的生成客户端。
- `fixes/before/`、`fixes/after/`、`fixes/patches/`：9 个修复文件的基线/最终版本与从实际最终差异重新导出的补丁。
- `workflow/model-rl-case-guard/`：使用的 skill 文档、参考与生成器。来源 MindSpeed-Agent；SKILL front matter 声明 MIT。保留来源与原文件内容（凭据示例脱敏除外），不为第三方内容另行声明许可。

脚本是当时运行的证据，不会随 clone 自动执行；历史脚本含固定主机、卡号、容器和路径，复用前必须按实际环境调整。与其他工作负载清理、公告、密钥恢复相关的脚本未交付。

## 历史绝对路径映射

| 运行时路径 | 仓库位置 |
| --- | --- |
| `/data/x00951564/codex-hy4-20261027-oldhost-revalidation/` | `reduced/` |
| `/mnt/share/x00951564/codex-hy4-20261028-four-node/` 或 `/task/` | `full/` |
| `/home/a/src/hy4-codex-weight-transfer-review/` 的修改快照 | `fixes/` |

原报告内的绝对路径用于追溯成功运行位置；原生成 README 只保留成功证据章节，未覆盖边界在过程说明中单列；本仓库根 README 提供当前结论和可点击的相对路径。`delivery/SHA256SUMS` 是本次交付文件哈希，脱敏文件的哈希可能不同于现场原件；原现场 SHA 不改写。失败报告已移出仓库，没有将失败改写为通过。
