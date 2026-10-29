# HY4 A5 开箱验证归档（hy4-xyh）

本仓库保存 2026-10-27 至 2026-10-29 的 HY4 A5 开箱记录：减层功能验证、问题修复与回归、Profiling/Excel、四机官方全量 BF16 模型精度，以及镜像复用材料。包含 Codex 辅助完成的脚本、修改和文档；这是证据归档仓库，不是 vLLM/vLLM-Ascend 上游代码合入或 PR。

## 结果与范围

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 减层 HCCL 权重更新 | passed；packed/unpacked、重建通信器、eager/图模式 | [功能报告](reduced/model_openbox/hy4/VERIFICATION-REPORT.zh-CN.md) |
| 减层 IPC 权重更新 | passed；2L TP2、6L TP4 图模式 | [逐接口输出与 checks](reduced/model_openbox/hy4/README.md) |
| 减层 checkpoint reload、sink/expert A-B-A | passed；完整事务、双接口 text/token/logprob 对照 | [逐项 JSON](reduced/model_openbox/hy4/cases/) |
| 减层 baseline、SP、单机 PP、DP、R3、MTP、pause/resume | passed；各自的拓扑与比较标准见报告 | [原始证据](reduced/logs/resume-exclusive/) |
| DCP、PCP | blocked；当前 HY4 attention sink 实现明确拒绝 | [实现限制说明](docs/PROCESS.md) |
| 减层多模态中段 10 步 Profiling、单卡算子测算 | passed；仅减层流程，不是全量性能结论 | [Profiling](reduced/profiles/)、[Excel](reduced/performance/hardware_efficiency.xlsx) |
| 四机全量 BF16 TP8/DP4/PP1/EP32、MTP3 | 两套数据完整生成、评测完成 | [全量报告](full/model_openbox/hy4/report.md) |
| 镜像保存与校验 | 已完成；镜像不内嵌权重 | [复用说明](full/scripts/IMAGE-REUSE.zh-CN.txt)、[元数据](full/logs/reusable-image-run07.json) |

上表是分阶段实际结果。减层通过不能外推为全量所有功能通过。四机阶段按用户要求只做全量精度：没有重新验证全量 reload/HCCL/IPC/CP/SP/PP；跨节点 PP 的旧尝试失败，换网卡后没有重测。减层报告中的“多机 DP/全量精度尚未验证”是 10 月 27 日的历史状态，四机阶段已补充全量 DP4 和精度证据。DCP/PCP 的实现限制没有绕过。

## 全量精度

仅上传成功正式轮次 run07 的评测数据；失败、不完整和被替代轮次的数据及日志已移出仓库，问题经过见过程说明。

| 指标 | 本次 BF16 | 参考结果 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| GPQA Diamond | 93.4343%（185/198） | 93.4343%（185/198） | 0.0000 |
| Hallusion qAcc | 54.9451% | 56.4835% | -1.5384 |
| Hallusion fAcc | 62.4277% | 63.8728% | -1.4451 |
| Hallusion easy_aAcc | 82.6374% | 82.6374% | 0.0000 |
| Hallusion hard_aAcc | 65.5814% | 65.8140% | -0.2326 |
| Hallusion aAcc | 79.1851% | 79.4508% | -0.2657 |

GPQA 198/198 有效，失败 0；HallusionBench 1129/1129 有最终回答，2258 次裁判判断全部完成。两个 Hallusion 样本达到 102400 输出上限，保留并按原规则评分，没有删题。参考与本次权重不同；输入、顺序、输出长度上限及影响分数的配置有审计记录。Hallusion 使用原 evaluation.py/utils.py 和 deepseek-flash，脚本哈希与参考一致，在线模型版本未冻结。完成评测不等于客户已接受精度差距。

- [skill 生成的全量 README](full/model_openbox/hy4/README.md)、[manifest](full/model_openbox/hy4/manifest.json)
- [GPQA 逐题预测、原生结果和配置](full/evaluation/runs/gpqa-full-07/20261028_180756/)
- [Hallusion 全量回答、原始响应和长度统计](full/evaluation/runs/hallusion-full-07/)
- [Hallusion 逐题裁判缓存与原始评分脚本](full/evaluation/runs/hallusion-judge-07/)
- [原始裁判日志及最终表格](full/logs/hallusion-judge-07.log)
- [GPQA 完整性](full/logs/gpqa-full-audit-07.json)、[Hallusion 完整性](full/logs/hallusion-generation-integrity-07.json)、[裁判协议审计](full/logs/judge-protocol-audit-07.json)

## 阅读与复现入口

1. [开箱经过与问题修复](docs/PROCESS.md)：为什么改、如何复验、哪些问题仍有边界。
2. [目录与证据索引](docs/ARTIFACTS.md)：skill 必需输出、原始数据、日志、脚本和校验清单。
3. [复现与恢复说明](docs/REPRODUCE.md)：正式配置、评测与断点缓存、镜像加载。
4. [九文件代码修改说明](fixes/CODE-CHANGES.zh-CN.txt)、[最终组合差异](fixes/patches/combined.diff)、[修改前](fixes/before/)、[修改后](fixes/after/)。这些快照针对当时 donor，不保证直接适用于未来主线。
5. [减层 skill 自动 README](reduced/model_openbox/hy4/README.md)、[audit](reduced/model_openbox/hy4/audit.md)、[manifest](reduced/model_openbox/hy4/manifest.json)。

来源：个人 fork [iwantknow1/vllm-ascend-hy4 的 hy4-v0.30.0](https://github.com/iwantknow1/vllm-ascend-hy4/tree/hy4-v0.30.0)；全量阶段 donor SHA 为 `a74b08e3f3ec98f5f624cb4a4ab3ae07b2c217b4`。参考部署材料：[ZYY-ing/Hy4-preview-deployment-guide](https://github.com/ZYY-ing/Hy4-preview-deployment-guide)。


已将 Ascend 侧源码推送到个人 fork 的 `hy4-v0.30.0`：[提交 b74fec8d1](https://github.com/iwantknow1/vllm-ascend-hy4/commit/b74fec8d11cd0e292813ee16ce24ac7e929b4253)。包含实测 runtime 原有的补丁加载入口；源码发布时仅调整 logger 接线。配套 vLLM sink 修复仍在本仓库单独保存，未推入另一个 vLLM 仓库。发布检查及边界见 [source-publication.json](delivery/source-publication.json)。

减层 Profiling 的完整参数、核心采集器和服务依赖见 [配置与脚本索引](docs/REDUCED-PROFILING.md)。

## 归档约定

本仓库先设为私有。保留成功轮次所需的主机/路径与运行记录，敏感凭据已检查和脱敏；不包含密码、SSH 私钥、裁判 API key、模型权重、容器镜像或内存 core dump。镜像约 10GB，保存于原共享存储，仓库记录镜像 ID、归档路径和 SHA256。原始日志里的绝对路径是当时运行位置，仓库对应关系见证据索引。只交付成功轮次的原始数据和日志；失败经过与未覆盖边界保留文字说明。

文件哈希清单：[SHA256SUMS](delivery/SHA256SUMS)；归档检查：[validation.json](delivery/validation.json)；脱敏/排除记录：[sanitization.json](delivery/sanitization.json)。未对移植后的源码或修改后的脚本重新宣称 NPU 验证。
