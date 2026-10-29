# HY4 A5 减层版验证报告

日期：2026-10-27。实机：177.8.21.1，Ascend 950DT。结论范围：减层版功能与权重更新一致性；不代表全量精度、全量性能或所有拓扑通过。

## 结论

6 层 TP2 HCCL 的 eager、图模式，6 层 TP4 IPC 图模式，以及 6 层 TP2 checkpoint 连续三轮重载均通过。SP、单机 PP、DP、R3 的 HTTP/进程内路径、MTP、pause/resume 已补测。DCP 和 PCP 在当前 HY4 attention sink 实现中明确不支持；Excel 已通过公开库备用导出器完成并独立重算；跨节点验证和原始 TP8 组合仍有下述限制。

## 环境与隔离

- 工作目录：`/data/x00951564/codex-hy4-20261027-oldhost-revalidation`；独立源码和容器，未覆盖 DeepSeek 的代码。
- 仅使用物理卡 2–7。检查时旧机其他活动容器没有 NPU 计算进程，但不能据此断言无人登录；未停止这些容器。另一台机器八张卡均有其他任务。
- CANN 9.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、Triton Ascend 3.6.0。6/7 卡的历史健康告警仍记录在案；本轮未复位整机或同事的设备。
- 基线启用 EP、RL、sleep 和 FULL_DECODE_ONLY，固定每卡 8 GiB KV。每个特性保留公共配置，并记录 TP/DP/PP、端口、设备和性能负载的必要差异。

## 已执行测试与未覆盖项

| 测试 | 状态 | 依据/边界 |
| --- | --- | --- |
| 2L TP1 HCCL 图模式 packed，持续通信器 3 轮 | passed | 此前版本；完整更新逐轮严格一致 [原始证据](../../logs/resume-exclusive/hccl-graph.json) |
| 2L TP1 HCCL 图模式，重新初始化通信器 1 | passed | 每轮新 producer、销毁并重建通信组 [原始证据](../../logs/resume-exclusive/hccl-reinit-1.json) |
| 2L TP1 HCCL 图模式，重新初始化通信器 2 | passed | 每轮新 producer、销毁并重建通信组 [原始证据](../../logs/resume-exclusive/hccl-reinit-2.json) |
| 2L TP1 HCCL 图模式，重新初始化通信器 3 | passed | 每轮新 producer、销毁并重建通信组 [原始证据](../../logs/resume-exclusive/hccl-reinit-3.json) |
| 2L TP1 HCCL 图模式 unpacked | passed | 显式 CPU staging，实际通过 NPU/HCCL 传输 [原始证据](../../logs/resume-exclusive/hccl-unpacked-staged.json) |
| 2L TP2 IPC 图模式 packed | passed | 跨多个 chunk 的完整 checkpoint 事务 [原始证据](../../logs/resume-exclusive/ipc-graph.json) |
| 2L TP2 checkpoint 图模式，连续 3 轮 | passed | 完整 sleep2/reload/wake [原始证据](../../logs/resume-exclusive/checkpoint-graph.json) |
| 2L TP2 sink/expert 权重 A-B-A | passed | 修改权重后输出改变，恢复 A 后完全一致 [原始证据](../../logs/resume-exclusive/variants-graph.json) |
| 6L TP2 HCCL eager，缓存修复后 | passed | 含临时内存诊断打印；最终代码已移除打印 [原始证据](../../logs/resume-exclusive/hccl-6l-eager-cachefix.json) |
| 6L TP2 HCCL 图模式，最终修复代码 | passed | 810 张量；原 TP2、8 GiB KV 配置复验 [原始证据](../../logs/resume-exclusive/hccl-6l-graph-cachefix.json) |
| 6L TP2 checkpoint 图模式，连续 3 轮 | passed | 两接口每轮 text/token/logprob 完全一致 [原始证据](../../logs/resume-exclusive/checkpoint-6l-graph.json) |
| 6L TP4 IPC 图模式 packed | passed | 4 个同卡 producer；810 张量；两接口严格一致 [原始证据](../../logs/resume-exclusive/ipc-6l-tp4-graph.json) |
| 2L TP2 图模式基线 | passed | 物理卡 2、3 [原始证据](../../logs/resume-exclusive/feature-baseline.json) |
| 2L TP2 SP/FlashComm1 | passed | 与同 TP2 基线对照；记录精确 logprob 相等与否 [原始证据](../../logs/resume-exclusive/feature-sp.json) |
| 2L TP1×PP2 流水线并行 | passed | 与 TP1×PP1 对照，避免混入 TP 变化 [原始证据](../../logs/resume-exclusive/feature-pp.json) |
| 2L DP2×TP2，EP4 图模式 | passed | 实际启动两个 DP engine、四个 worker [原始证据](../../logs/resume-exclusive/feature-dp.json) |
| 2L DP2×TP2 双接口并发 | passed | 并发 4，24 次接口请求均成功 [原始证据](../../logs/resume-exclusive/feature-dp-concurrent.json) |
| 2L TP2 R3，HTTP 双接口 | passed | V2 runner；解码 base64 npy，检查层数/top-k/专家范围 [原始证据](../../logs/resume-exclusive/feature-r3.json) |
| 2L TP2 R3，进程内 LLM | passed | 显式 spawn；路由数组为 uint8，形状包含 2 层、top-k 8 [原始证据](../../logs/resume-exclusive/feature-r3-inproc.json) |
| 2L+MTP TP2 图模式 | passed | 草稿 60 token、接受 0；仅功能通过，无加速结论 [原始证据](../../logs/resume-exclusive/feature-mtp.json) |
| pause/resume 双接口一致性 | passed | 在 MTP 服务上确认暂停状态并恢复，严格对比输出 [原始证据](../../logs/resume-exclusive/pause-consistency.json) |
| DCP | blocked | HY4 attention sink 的明确实现限制 （实现限制见过程说明） |
| PCP | blocked | 与 DCP 分开登记；不是通用 DSA 支持即 HY4 支持 （实现限制见过程说明） |
| 2L 多模态 32768/4096，中段 10 步采集 | passed | 两个 rank，物理卡 2、3；仅减层采集流程 [原始证据](../../logs/resume-exclusive/profile-final.json) |
| 单卡算子测算 / Excel 导出 | passed | 公开库备用导出器生成 Excel；逐调用公式、阈值修改及文件重开重算均通过 [原始证据](../../logs/resume-exclusive/hardware-efficiency-export.json) |
| 多机 DP | blocked | 177.11.28.1 当前八张卡都有其他 VLLMWorker_DP 进程；未停止他人任务，未启动跨节点服务。 （该阶段未执行；后续范围见仓库根 README） |
| 多机 PP | blocked | 第二节点无空闲卡；单机 PP 通过不能替代跨节点 PP。 （该阶段未执行；后续范围见仓库根 README） |
| 原始 TP8 / util0.9 组合 | not_run | 当前按明确分配仅使用物理卡 2–7；未覆盖原始八卡配置，不能外推该历史错误已修复。 |
| 全量精度 / 数据集评测 | not_run | 本任务为减层功能范围；未运行 32 层 Math500/TextVQA，也不以减层输出评价准确率。 |

权重更新通过的标准是完整 sleep(level=2) → wake(weights) → init/start/update/finish → wake(KV)，两个接口、全部提示词的 text、token、有限 logprob 完全一致。HCCL/IPC 都实际传输 checkpoint；CPU staging 不是跳过数据面。一般并行特性要求生成 text/token 与相同 TP 的基线一致，logprob 数值有效；精确数值相等与否另存报告，不与权重更新的严格标准混淆。

## 遇到的问题、修复及复验

| 问题 | 原因/证据 | 处理与结果 |
| --- | --- | --- |
| reload 的 sink 加载计数及运行时引用 | TP 切片后直接 copy 绕过 weight_loader | 对原运行时参数调用 loader，保留图捕获的存储引用；TP 分片测试和 A-B-A 实机复验通过。 |
| meta 专家权重漏加载 | 旧补丁在 meta 分支提前返回，合法加载未进入计数 | 恢复该分支的 loader 调用；计数回归从 0 恢复到 6，保留严格完整性检查。 |
| HCCL packed 流与完整 checkpoint 校验冲突 | 分批调用模型 load_weights，流不完整；接收缓存生命周期、stream 也需明确 | 整个流只调用一次 load_weights；显式 default stream；packed batch 保持存活；可选 CPU staging 控制 NPU 常驻原始权重；packed/unpacked、持续通信组和重建通信组均有实机证据。 |
| IPC 多 chunk 的事务及 producer 引用计数 | 每块单独加载破坏完整性；重用 packed storage 时复用引用计数 | 显式 CPU staging 跨 RPC 累积，finish 时一次完整加载；每 chunk 重新导出 IPC 引用计数；完成后释放缓存；2L TP2 和 6L TP4 图模式通过。 |
| 6L TP2 finish 阶段显存不足/拷贝报错 | 原现场有分配失败；传输结束后 reserved 仍约 94.93 GB，专家延迟 materialization 还需空间 | finalize 前 empty_cache 释放 21,214,303,872 字节（约 19.76 GiB），活跃分配 63,994,016,768 字节保持不变；finalize 后仍保留清理以恢复 KV。相同 TP2、8 GiB KV 的 eager/图模式通过。成功回归已保留，失败经过仅作文字说明。 |
| PP 首轮输出不一致 | 首轮把 TP2/PP1 与 TP1/PP2 比较，混入 TP 改变 | 首轮失败数据不上传；补 TP1/PP1 基线后 PP2 对照通过，未修改模型实现来迎合比较。 |
| R3 进程内 fork 初始化失败 | 已初始化 NPU 后 fork 无法重新初始化 | 测试入口显式 spawn 并加 main guard；HTTP 与进程内路由形状、dtype、专家范围检查通过。进程退出告警保留，随后重启自有容器清理。 |
| 性能工具无法启动 vllm | 此 uv 环境没有 vllm console script | 独立工具副本改用 `.venv/bin/python -m vllm.entrypoints.cli.main`，不修改模型代码。 |
| 长输入首步 slot-mapping TypeError | 工具覆盖 PATH 丢失 bisheng；broken 环境 HAS_TRITON=false，kernel 为普通 function | 继承完整 CANN PATH 后 HAS_TRITON=true、kernel 为 JITFunction；32K 图片输入/4096 输出实测通过，两个 rank 均采集到十次 MODEL_EXECUTE。 |
| MTP 减层资产缺失 | 原 2L/6L checkpoint 不含 nextn | 独立目录链接 2L 骨干和已有 32L-MTP 中的 27 个 MTP 张量，维度先核对；原资产不改动。草稿 60、接受 0，最终输出一致；不能推导加速或全量质量。 |
| Excel 导出失败 | @oai/artifact-tool 未安装，导出报 ERR_MODULE_NOT_FOUND，依赖检索日志归档 | 原工具失败仅在过程说明中描述；通过 Windows 从官方 npm 下载公开库并核对 SHA512，使用 SheetJS/xlsx-calc 实际生成 Excel，全部逐调用与汇总公式独立重算通过。 |

HCCL 在此前另一台机器出现的 RunAicpuIndOpCommInit 动态库符号错误，本机三轮重建通信组未复现。这只能证明当前环境下通过，不能把推测的混合动态库原因写成已证实根因。

## CP 核实结论

`vllm_ascend/patch/worker/patch_hy_v4.py:205–206` 在 `AscendHYV4MLAAttention.__init__` 检查 DCP、PCP，任一大于 1 就抛出：`HY4 attention sinks currently require PCP=1 and DCP=1.` 原代码条件分别以 (DCP,PCP)=(1,1)、(2,1)、(1,2) 验证，只有默认组合不触发拒绝。通用 DSA 后端存在 CP 实现不等于 HY4 的 sink 路径已适配。未删除断言或伪造 CP 服务结果。

## 采集与单卡测算

真实多模态负载：一张 256×256 图片，图片/模板实测开销 95 token；实际总输入 32768、输出 4096，成功 1、失败 0。worker 延迟 2048 步后启动、十步后自动停止。两个 rank 各有 10 个 MODEL_EXECUTE 图执行标记、2380 条 kernel 记录。当前 CANN trace 没有 ProfilerStep 名称，step_trace_time.csv 仅有聚合行，因此没有把聚合行数误当推理步数。

单卡选择物理卡 2；按 Accelerator Core 列区分计算、通信与 AI_CPU，使用全部两层、十步的调用记录计算。默认经验上限 0.8，按 <10μs 短调用及其他调用分别加权，JSON 的 weighted_attainment=1.097228；该粗略指标可超过 1，不裁剪、不当作端到端加速比，更不能外推 32 层性能。Excel 已生成：2250 条计算调用、130 条排除记录，包含 Summary、Kernels、Operators、Excluded 四张表；公式重算与原 JSON 一致，更改经验上限/短调用阈值后重算、恢复默认和文件重开重算均通过。原工具失败仅作过程说明，成功结果以备用导出的 Excel 和校验为准。

## 版本与复验边界

98 项单元测试及 ruff 检查通过；日志分别为 prefinalize-final-unit.log、prefinalize-lint.log。此前新机版本之后，本轮仅 HCCL engine 的前置缓存清理及其既有单元测试这两个文件继续变化；其余七个修复文件校验值与参考版本一致。2L 早期 HCCL 用例属于前一版本，最终两文件修复通过 98 项单元回归及 6L 原配置图模式验证；未声称所有早期用例都在最终哈希上重跑。

| 修复文件 | SHA256 |
| --- | --- |
| `vllm/vllm/models/hy_v4/nvidia/model.py` | `bd0d31316c596dd58faf33966ac1fdd0de5974fa3bd3e3aef24e2a7cc001929f` |
| `vllm-ascend/vllm_ascend/patch/worker/patch_hy4_reload.py` | `73e9c256f6122a03b941a539999686ac451c43ba5136a4151379ccd25fe09005` |
| `vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py` | `08e60dd09cee42ecb8d03107c3cb88dadafd1336564f93adbdcc00d24cf81ecb` |
| `vllm-ascend/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py` | `fc51a52a9c928af956186134d45084bdeb0f96c1ed5fc66001c4e4b7eacc2473` |
| `vllm-ascend/vllm_ascend/distributed/weight_transfer/packed_tensor.py` | `fa655feb33e9fb4a08b009e12c83691b20f97ecaf75d53b72860d58ac89619a8` |
| `vllm-ascend/tests/ut/models/test_hy_v4.py` | `942d8193eea485c15465e0af5c541ffb7dc2e3d421708a28be6da3ab5b348b35` |
| `vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py` | `5f57cc49fd516280e16828f75b34afdfdcf9c9ce394af4ef245e18884ab047b5` |
| `vllm-ascend/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py` | `6479c83da1c84c4692c56854f92323202d84b30bee4721a0aa57aeff64e992c5` |
| `vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py` | `c603f7b1237d094124b0af762c3c41a62871b907252984c71af32924663f4eb0` |

## 复用与原始证据

镜像说明见随附 IMAGE-README.zh-CN.md；镜像将内置修复源码、uv 环境、Triton 覆盖版本和本报告，模型及驱动继续只读外部挂载。镜像构建后的独立实机检查与镜像 ID、导出校验值单独归档，避免让镜像自身的 ID 成为循环构建依赖。

完整逐接口输出、checks 和实际命令见 [逐项证据 README](README.md)；成功轮次的完整日志、容量报告和回归记录位于 logs/resume-exclusive；失败轮次数据不上传。没有通过的项目已逐项列明，不能将本报告解读为“所有功能、所有拓扑、所有规模全部通过”。


## Excel 备用导出交付

[下载 Excel](../../performance/hardware_efficiency.xlsx)。备用库版本为 SheetJS 0.18.5、xlsx-calc 0.9.2；官方 npm tarball 的 SHA512 已核对。原工具不可获取的经过不改写成通过，备用导出器有独立校验报告 hardware_efficiency_fallback.json。所有计算只使用物理卡 2 的完整十步调用，不复制乘层数；计算结果超过 1 时保留原值。


## 镜像最终实机复验

[镜像自包含验证与完整 ID](IMAGE-VALIDATION.md)：6L TP2 HCCL 图模式，在没有源码挂载的最终镜像中完整复验通过。5–7 卡已释放。镜像元数据和归档 SHA256 另行交付，避免镜像自身 ID 的循环依赖。
