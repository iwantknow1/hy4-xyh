# Profiling、单卡性能估算与数据集评测

在首次启动前确认权重范围、模态和评测选择，按主入口保存 manifest。以下命令以仓库根目录为工作目录；示例中的连接、模型、并行和特殊参数用已确认配置替换，原始命令与日志归档到任务目录。采集与评测各有服务生命周期，顺序执行，复用已有启动/停止入口。

## 1. 复用 Profiling 采集

先读取 [ascend-profiling-collection](../../../performance-optimization/ascend-profiling-collection/SKILL.md)，使用其 `scripts/collect_torch_profile_case.py`，不另写远程采集器。确认部署版本支持 `delay_iterations`、`max_iterations`、`ignore_frontend` 以及 PipeUtilization 指标；缺少功能时记录依赖修改及 PR，不能用整段采集冒充中段 10 步。

| 模型模态 | 目标输入 | 固定输出 | 采集窗口 | 负载 |
| --- | --- | --- | --- | --- |
| 纯文本 | 2048 token | 6144 token | 中段 10 个 worker 推理步 | random |
| 多模态 | 32768 token（包含模态与模板开销） | 4096 token | 中段 10 个 worker 推理步 | random-mm，图片与文本 |

默认单请求、并发 1，设置 `--ignore-eos --random-range-ratio 0` 保持长度。中段延迟分别为 3072 / 2048 步，采集步骤上限为 10。窗口处于 decode 区间，覆盖该卡所执行的全部层。多模态默认使用一张 256×256 合成图片，可按模型的有效视觉输入尺寸调整并记录，不允许只发送长文本冒充多模态。

纯文本示例：

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode direct --host <host> --model <checkpoint> --served-model-name <name> \
  --tp <tp> --mode full_decode_only --tag rl-text-middle \
  --workload-kind text --random-input-len 2048 --random-output-len 6144 \
  --profile-phase middle --profile-steps 10 --ignore-frontend --random-range-ratio 0 \
  --num-prompts 1 --max-concurrency 1 --max-num-seqs 1 --max-model-len 8192 --analyse-all
```

多模态沿用该命令，修改为 `--workload-kind multimodal --random-input-len 32768 --random-output-len 4096 --max-model-len 36864`，增加 `--mm-input-token-overhead <measured-overhead> --image-height 256 --image-width 256` 并更换 tag。该开销必须由当前模型 processor、聊天模板、相同图片尺寸和请求格式实测，包含图片展开 token 和模板 token；脚本从 32768 中扣除开销生成文本，再通过 `/v1/chat/completions` 发送真实图片。

采集前用相同请求格式核对 `usage.prompt_tokens` 并校准开销；采集后核对实际 prompt/completion token 与完整输出长度。若 tokenizer 重编码使长度变化，重新校准请求，不能仅依据 CLI 参数宣称实际长度达标。检查 trace 与 worker 日志证明窗口在中段、实际记录 10 步且层执行完整；不同版本可能有边界计数差异，发现步数不符时定位后调整采集配置/裁剪可验证的窗口再分析，保留原始数据与选择依据。早停、窗口不足、采集失败或缺指标均不通过。

把 baseline 的 `--extra-env`、worker 环境和 `--serve-args` 一并传给采集器；其中 `--serve-args` 是一个按 shlex 解析的字符串。上下文长度、内存利用率、并发序列数、API 服务数使用采集器的对应专用参数，不在透传字符串中重复；采集器拒绝覆盖这些参数或 profiler 配置。图配置只传一份合并后的 JSON，脚本保留其他图配置字段并校验图模式冲突。归档采集 manifest、服务日志、trace、`kernel_details.csv`、设备/rank、实际 token 和步数证据，专项报告使用 `feature: profiling`。

## 2. 单卡加权极致性能达成率

输入为功能 1 产生的一个设备的 `kernel_details.csv`。只算一张卡，无需单卡部署整个模型；TP 模型可选择其中一个 rank，记录拓扑。PP 时单卡仅有部分层，报告必须标注 stage / 层范围，不能宣称代表全模型。累加窗口内所有计算算子调用，保留所有层和所有步，不取第一层、不重复乘层数。耗时为各调用 duration 之和，存在重叠时与端到端墙钟时间不同。

### 计算口径

1. 根据 `Task Type` 选择 AI_CORE、AI_VECTOR、AI_VECTOR_CORE、MIX_AIC/MIX_AIV 等计算任务，排除通信、拷贝和非计算任务，保存排除行。其他任务类型需核对实际含义后适配，不能根据耗时猜测。存在 `Device_id` 时筛选指定设备；无设备列时由调用方确认整个文件仅来自该卡，文件来源与设备映射一并归档。
2. 将 `*_ratio` 统一为 0–1，排除派生的 `dur_ratio`。缺失指标保留为空，所有指标缺失或所选瓶颈指标缺失时失败。`max_ratio` 为该次调用可用 ratio 的最大值。默认 `x = max_ratio / 0.8`；若有算子同事提供的瓶颈指标经验上限，显式配置该算子、指标、经验上限和来源，改为 `x = selected_ratio / selected_ceiling`。经验值须匹配芯片、算子实现、形状和 dtype；未匹配的算子仍用默认 0.8。
3. 每次调用的 `w = duration / Σ(选中计算调用的 duration)`。先对每次调用判断时长，再聚合各层：`duration >= 10μs` 贡献为 `w*x`，`duration < 10μs` 贡献为 `w`，总体达成率为两部分贡献之和。不得先将短调用聚合成长调用再计算。结果不裁剪到 1，不取倒数作为吞吐加速比。
4. **重点优化项：`duration > 20μs AND max_ratio < 0.6`**。20μs 与 0.6 本身不触发；阈值可配置。使用原始 max_ratio 判断，不用除经验上限后的 x 判断。

用户示例：GMM1 的 `mte2_ratio=0.709`、`mac_ratio=0.224`，判断为访存受限（MTE2 瓶颈），以经验上限 0.85 计算，得到 `x1=0.709/0.85≈0.8341`；若 `w1=0.14`，贡献为 `0.1168`。其余算子按同一口径加权，若总和为 0.8，表示约达到经验极致性能的 80%。这是计算算子的粗略估算，未纳入通信、空闲与重叠，不能据此声称端到端吞吐提升 25%。

`ceilings.json` 示例（算子名 glob 区分大小写；规则不得重叠）：

```json
[
  {
    "operator": "GMM1*",
    "metric": "mte2_ratio",
    "ceiling": 0.85,
    "source": "算子同事提供的经验值；使用前补充适用芯片、shape、dtype 和实现版本"
  }
]
```

运行离线测算，不需要 NPU。依赖 Python 3.10+、Node.js 和 `@oai/artifact-tool`；可使用当前 Codex 提供的依赖运行时，其他环境须先准备同名 npm 包。缺少 Excel 依赖时报告失败并保留分析 JSON，不得把 JSON 当成已交付 Excel。

```bash
python3 skills/inference-model-skills/deployment/model-rl-case-guard/scripts/performance/estimate_hardware_efficiency.py \
  --kernel-details <single-device/kernel_details.csv> --device 0 \
  --validation-scope full_model --ratio-unit fraction --ceilings <ceilings.json> \
  --output <task-dir/performance/hardware_efficiency.xlsx>
```

没有专用经验值时省略 `--ceilings`。百分比数值如 `70.9` 使用 `--ratio-unit percent`；带 `%` 的单元格自动按百分比读取。`--default-ceiling` 默认 0.8；分界和重点阈值分别由 `--short-kernel-us`、`--priority-duration-us`、`--priority-ratio` 配置。不同 CSV 表头可显式传 `--name-column`、`--type-column`、`--duration-column`（值必须为 μs）、`--device-column`。不得把其他时间单位直接当作 μs。

可通过 `--node <node-executable> --node-modules <node_modules>` 指定已有依赖位置；`--preview-dir <dir>` 导出工作表 PNG 便于检查。输出包含 JSON 证据与 Excel：Summary（指标和可调阈值）、Kernels（各调用/原始 ratio/公式/经验来源）、Operators（各层累计）、Excluded（排除行）。重新计算 Excel 后汇总应与 JSON 独立计算一致；Excel 修改参数后 JSON 不自动更新，正式变更需重跑并生成新目录。

减层随机权重使用 `--validation-scope functional_smoke`，保留该标识，只验证采集与测算路径，不能外推全层模型成绩。`full_model` 取值前必须核对真实全层资产和所选设备实际执行的层范围；脚本无法仅凭 CSV 证明权重真实性。

## 3. 复用 AISBench 数据集评测

仅 `weight_kind: real_full` 且开始时用户选择评测时执行。先读取 [vllm-ascend-aisbench-accuracy](../../../common_tools/vllm-ascend-aisbench-accuracy/SKILL.md)，调用其 `scripts/accuracy_run.py`，保留原来的连接、环境准备、服务拉起/停止、评测和结果归档能力。不要自建评测客户端，也不复用随机 Profiling 请求计算准确率。

| 模态 | 默认数据集参数 | 数据集类型 |
| --- | --- | --- |
| 纯文本 | `--dataset math500_gen_0_shot_cot_chat_prompt` | `--dataset-kind math` |
| 多模态 | `--dataset textvqa_gen` | `--dataset-kind textvqa` |

```bash
python3 skills/inference-model-skills/common_tools/vllm-ascend-aisbench-accuracy/scripts/accuracy_run.py \
  --connection-mode direct --host <host> --model <real-full-checkpoint> --tp <tp> \
  --dataset math500_gen_0_shot_cot_chat_prompt --dataset-kind math \
  --max-out-len <user-provided-length> \
  --serve-args --max-model-len <validated-context-length> <confirmed-baseline-special-args>
```

评测器的 `--serve-args` 后面接独立参数，与采集器的字符串形式不同。环境变量通过重复的 `--extra-env` 传入；同样继承公共特殊配置、图模式和 worker 配置，按已授权数据集目录/下载方式准备资源。上下文长度按数据集实际输入（TextVQA 包含图片展开）加用户输出长度预检；不能仅凭工具默认输入余量判定足够。归档实际数据集 revision/样本数、推理长度、模板、采样配置、轮数和原始结果；分数如实报告，未设置精度验收阈值时不替用户认定精度达标。

专项报告使用 `feature: dataset_evaluation`，保存 `dataset`、`max_out_len`、`metrics`、`result_path`、`baseline_config_revision`、`inherited_special_config`、实际命令和日志。未选择、长度缺失、减层权重、资源不可用分别记录对应原因。硬件测算的 JSON 可直接纳入 README；采集和评测需将复用工具的原始产物路径纳入专项报告，避免把中间 manifest 重复扫描成额外特性。
