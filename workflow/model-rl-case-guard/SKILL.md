---
name: model-rl-case-guard
description: This skill should be used when the user asks to verify model RL features on Ascend NPU inference services, validate weight transfer and lifecycle, collect middle-window profiling, estimate single-card compute efficiency in Excel, run Math500 or TextVQA evaluation, or generate reduced-layer checkpoints for functional checks.
license: MIT
metadata:
  version: "0.27.1"
  domain: inference-model
  function: deployment
  tags: [rl, case-guard, verification, vllm-ascend, npu]
  author: mindspeed-agent
---

# 模型 RL 用例看护 (Model RL Case Guard)

面向昇腾 NPU 推理服务与 RL 用例的看护、验证和证据归档流程。支持裸机（宿主机 conda）与容器（Docker）两种运行环境，**同时支持单机与多机（跨节点 DP 集群）**。流程分为**阶段1（新机环境检测）**与**阶段2+阶段3（特性拉起与验证循环）**。

## 统一入口职责

本 skill 是 MindSpeed-Agent 的模型 RL 用例看护统一入口，**主体是 vLLM-Ascend 推理服务与多机特性验证**。模型资产与实现路径审计只作为进入服务验证前的前置步骤；完成前置审计后，按本文件定义的环境检测、基础配置和特性验证循环执行。

- **模型资产审计**：读取模型名称、revision、`config.json`、tokenizer、实际建模代码、权重索引和自定义 kernel；核对层数、注意力/KV 头、head dim、RoPE、MoE/MLA/DSA/MTP、特殊 token 及权重键。不得仅根据模型名称或论文推断结构。
- **推理开箱范围**：覆盖推理服务的环境、拓扑、容量、启动、功能和特性开关验证，以及固定负载的 Profiling、单卡计算算子性能估算和用户选择的数据集评测；不替代模型迁移、训练收敛或长期性能调优。

## 输入与开箱工作区

除非用户明确指定，向用户收集或从已授权 checkout 推导：模型权重路径、模型代码/配置来源及 revision、head 与 worker 节点信息、连接模式（`docker`/`conda`/`direct`）、容器名或 conda 环境、CANN 根目录、可用设备、目标特性清单和是否允许真实 NPU 运行。不得擅自下载权重、数据集或登录未授权主机。

每个任务在工作目录维护 `model_openbox/<model-slug>/`，不得覆盖现有任务；至少保存 `manifest.json`（模型/代码/环境身份）、`audit.md`（结构与路径审计）、`logs/`（原始命令与 stdout/stderr）和 `report.md`（逐项证据结论）。无真实 NPU 环境时可做静态检查与可选的 CPU 减层权重生成，将 NPU 运行项标记为 `not_run`。

### 开始时确认评测与性能采集

- 审计并记录 `weight_kind`（`real_full` 或 `reduced_synthetic`）、模型模态及全层数。只有真实全层权重才在开始时询问“是否进行数据集评测？默认纯文本 Math500、多模态 TextVQA；如评测，请提供生成长度上限（token）”。已有明确答复直接复用；同意但长度未提供时评测保持待补充，其他独立步骤可继续，不代填 Profiling 的输出长度。减层随机权重不运行准确率评测。
- 将答复保存到 `manifest.json` 的 `dataset_evaluation`：`enabled`、`dataset`、`max_out_len`、`reason`。另存 `performance_collection`：模态、输入/输出长度、`profile_steps: 10`、`profile_phase: middle`、采集配置、权重范围和经验上限规则。Profiling 与单卡测算作为本流程的新增步骤执行；缺少必要环境或原始数据时如实记录 `blocked` / `not_run`。
- 采集与评测复用已确认的 `baseline_special_config`（启动参数、环境变量、配置文件、图模式及配置版本），保持环境前置 PR 一致。不同负载需要修改上下文长度时记录差异并重新预检容量；配置冲突先解决，不静默覆盖。

### 可选：只有 config 时生成减层权重

用户选择快速功能验证、没有真实权重但有本地 `config.json` 时，按 [references/reduced-checkpoint.md](./references/reduced-checkpoint.md) 生成减层随机 checkpoint，再复用同一套特性用例。默认的真实权重验收流程不变；缺少权重时不得静默切换模式。

- 保留模型类、嵌入/输出层、隐藏维度、注意力头、专家数、词表、位置编码及多模态分支；只减少解码层数并同步相关逐层配置。保留各类注意力、dense/MoE 等结构层，无法确认分层规则时先适配模型，不直接截断。
- 使用 [scripts/model-preparation/generate_reduced_checkpoint.py](./scripts/model-preparation/generate_reduced_checkpoint.py) 先 `--dry-run` 审查配置差异，再离线生成可加载的 safetensors。模型实现未被当前环境支持或存在专用初始化时，按参考文档处理，不能手工猜测权重名称和张量形状。
- 生成物保存独立目录，带 `reduced_checkpoint.json` 来源标识；任务 manifest 和最终报告标记 `validation_scope: functional_smoke`。此模式下，下文的“真实 checkpoint”前置条件改为“同一份已生成并实际加载的减层 checkpoint”；双接口、全部 prompt 的 text/token/logprob 一致性要求保持不变。
- 仅验证实际执行的功能路径，不用于真实模型精度、完整规模或性能验收。随机输出不要求语义正常或无乱码，但必须检查请求成功、输出有效、数值有限及对应特性约束。缺 tokenizer 时先记录接口验证阻塞，不能伪造 tokenizer 或聊天模板。

## 流程总览

```text
阶段1 新机环境检测
   A. 登录（head 节点；多机另需 worker 节点信息）
   B. 确认可用卡（单机或各节点汇总）
   C. 软件环境（conda 或 容器）
   D. 多机拓扑（多机场景）：确定各节点卡数 → 推导 TP/DP/PP → 组装多机拉起方式
   启动前：汇总环境前置修改（含 PR 号），询问并固化 baseline 特殊配置
   ↓
阶段2 + 阶段3 循环（对每个特性开关执行）
   阶段2: 用该特性拉起 vllm-ascend 服务（单机本地拉起；多机复用 serve_start.py 拉起集群）
   阶段3: 对该服务做联调/验证
   → 回到阶段2 拉起下一个特性服务 …… 直到所有特性开关验证完毕
   ↓
性能与评测：复用 Profiling 采集 → 选一张卡估算并导出 Excel
   → 真实全层权重且用户选择评测：复用 AISBench（Math500 / TextVQA）
```

> **循环要点**：开箱涉及多种特性开关，需要**多次拉起 vllm-ascend 服务**，每个特性对应一轮「拉起→验证」。阶段2 与阶段3 交替循环执行，直至特性清单全部验证完。

## 阶段1：新机环境检测

### A. 登录

- **输入**：目标机器 IP、账号、密码。**多机**：需要 head 节点 + 各 worker 节点的 IP/账号/密码（或同一套凭据）。
- **操作**：技能自动处理登录细节（SSH 密码登录、askpass、sudo 切换等），不打扰用户。
- **成功标准**：能成功登录并继续执行后续命令即可。
- **多机连接模式**：与 `vllm-ascend-serving` 一致，支持三种连接拓扑——`docker`（SSH 宿主机 + `docker exec` 进容器）、`conda`（SSH 宿主机 + `conda activate`）、`direct`（SSH 直连容器/VM）。多机下所有节点统一使用同一模式，worker 自动继承 head 的连接凭据（可在需要时用 `--worker-*` 覆盖）。详见 `../vllm-ascend-serving/SKILL.md` 的“多机部署”章节。
- **已知障碍**：在受限（沙箱）环境下，Windows OpenSSH + askpass、Git Bash ssh、Python paramiko 等登录通道可能被环境拦阻（子进程 / 内存映射 / 依赖缺失）。优先复用 `vllm-ascend-serving` 的连接脚本；遇到失败不要反复重试同一通道，切换可行通道或请用户提供已配好密钥免密的通道。**多机**登录由复用的服务脚本统一处理，不要手写 SSH 命令。

### B. 确认可用卡

- **单机**：执行 `npu-smi info`；报告卡数与型号即可。
- **多机**：复用 `vllm-ascend-serving` 的 `serve_probe_npus.py`，一次探测 head + 所有 worker 节点：

  ```bash
  python3 .../vllm-ascend-serving/scripts/serve_probe_npus.py \
      --host <head-ip> [--container <name>] --worker-hosts <w1>,<w2> [--mode docker|conda|direct] ...
  ```

  返回 `nodes`（各节点 `total`/`free`/`busy`）与 `summary`（`total_devices`/`total_free`/`nodes_probed`）。直接复用 `../vllm-ascend-serving/scripts/serve_probe_npus.py`，不要另写一套 SSH 探测器。
- **输出**：向用户报告**节点数 K**、**每节点卡数**、**总可用卡数（各节点之和）**与型号；多机下**单机卡数**用于 TP 上限判定，**总可用卡数**用于总量判定（见基础配置多机推导）。
- **成功标准**：能输出各节点卡数量与型号即可，不做额外复杂判断。

### C. 软件环境（conda 或 容器）

1. **询问环境类型**：容器（Docker）还是裸机（宿主机 conda）？
2. **收集环境信息**：
   - **裸机**：让用户填写环境信息表（如 conda 环境名、CANN 包路径、VLLM_VERSION 等）。
   - **容器**：让用户提供**容器名**。
3. **验证环境是否存在**：确认对应 conda 环境 / docker 容器确实存在；若不存在则提示用户修正。**多机**下确认 head 与各 worker 的容器名 / conda 环境（可不同，用 `--worker-container` / `--worker-conda-env` 指定）。
4. **运行前置自检**：在进阶段2 之前，确认 **vllm 命令可用** 且 **CANN 环境已加载**：
   - `which vllm` —— 若不在 PATH，用统一 python 可执行补（如 `export PATH=/usr/local/python3.12.13/bin:$PATH`）。
   - 能 `python -c "import torch, torch_npu, vllm"` —— 若 torch_npu 报缺 `libhccl.so` 等库，需 `source <CANN>/ascend-toolkit/set_env.sh` 或在 `LD_LIBRARY_PATH` 补 CANN 的 lib64（含 driver 与 toolkit 两套，缺依赖会报 `libc_sec.so` / `libdrvdsmi_host.so` / `libhccl.so`）。
   - **注意**：“不进入 conda” ≠ “不 source CANN”。vllm-ascend 运行时必须能加载 CANN 的 libhccl 等库。启动前先按 `../vllm-ascend-serving/SKILL.md` 的环境检查说明验证。
   - **多机**：上述自检需在 head 与每个 worker 上各自成立（服务脚本会自动 source CANN 环境）。
5. **检查环境是否需要提前修改**：询问用户是否有已知的环境前置修改及对应 PR，已有记录直接复用；同时结合模型、目标特性和实际安装源码/版本检查所需修复。有修改时，在启动 baseline 前提供清单，逐项列出仓库、PR 号及链接、修改原因、适用特性、当前是否已包含、需执行的操作、涉及节点和验证证据。已包含的 PR 不重复应用；回退项写明原 PR 与回退 PR；无对应 PR 的依赖/配置修改如实标注。无修改也明确记录“无需前置修改”，尚未确认则记录待核实，不把未知当作无需修改。具体字段与核查方式见 [references/execution-notes.md](./references/execution-notes.md)。

### D. 多机拓扑确认（多机场景）

仅当存在多个节点（`--worker-hosts`）时执行：

- **输入**：head + worker 节点列表、各节点可用卡数（阶段1-B）。
- **输出**：多机并行拓扑 `TP × PP/DP`，以及多机拉起所需的连接信息。
- **原则**：**TP ≤ 单机卡数**（TP 组通信通常同节点），跨机用 **PP / DP** 扩展。详细推导规则见下文「基础配置（agent 自动生成）」与 `../vllm-ascend-serving/SKILL.md` 的多机章节。

## 阶段2 + 阶段3：特性拉起与验证循环

针对**特性清单**中的每一个特性，循环执行「阶段2 拉起服务 → 阶段3 验证」，直至全部特性验证完毕。

**默认执行范围**：CP 的 DCP、PCP 两个维度，以及 SP、PP 均默认执行、分别记录结果，无需用户额外要求开启。只有用户明确排除的项才跳过并注明原因；模型、版本、资源或拓扑不支持时，完成前置核查并记录 `blocked` 及证据，不以“可选”或“低优先级”为由省略。默认执行指各特性分别拉起验证，不要求将互斥特性同时开启。

> 开箱验证场景为**单独拉起 vllm-ascend 推理服务**，验证每个特性能否正常开启并工作，不依赖 trainer 配合。各特性对应的开启参数与验证方法详见 [references/feature-matrix.md](./references/feature-matrix.md)（由读代码仓得出）。
> **验证定位**：开箱检测以**记录结果为主**。一般特性的精度是否有问题由客户结合证据判断；但 HCCL、IPC 和 Checkpoint 权重更新有明确的幂等一致性硬门槛：服务先加载真实 checkpoint，sleep/wake/update 后两个接口、所有 prompt 的 text/token/logprob 必须与基线完全一致，否则不得标记通过。

### 验证脚本分类

特性验证必须按领域从对应目录选择脚本，不再从 `scripts/` 根目录查找平铺脚本：

| 路径 | 分类 | 内容 |
| --- | --- | --- |
| `scripts/model-preparation/` | 模型准备 | 可选的 config 减层与随机 checkpoint 生成 |
| `scripts/baseline/` | 基线 | 双接口基线结果生成 |
| `scripts/routing/` | 路由 | MoE routing replay |
| `scripts/lifecycle/` | 生命周期 | sleep/wake、pause/resume |
| `scripts/weight-update/` | 权重更新 | HCCL、NPU IPC、Checkpoint 重载测试入口；`common/` 存放权重来源与更新验证公共方法 |
| `scripts/parallel/` | 并行 | CP、SP、PP |
| `scripts/performance/` | 性能分析 | 单卡 kernel_details 经验达成率计算与 Excel 导出；采集复用已有 skill |
| `scripts/reporting/` | 报告 | Markdown README 报告生成 |
| `scripts/common/` | 公共实现 | 跨特性共用的参数定义、双接口探针与在线服务对照工具；不作为独立验证项 |

执行命令和报告中的 `verification_command` 必须记录分类后的完整相对路径，禁止继续使用旧的 `scripts/test_*.py` 路径。

### 基础配置（agent 自动生成）

- **输入**：用户提供**模型权重路径**（可选减层模式下为生成的 checkpoint 路径）与**节点信息**（单机：head 即可；多机：head + `worker-hosts`），不要求用户提供 TP/EP/DP、图模式等并行/运行配置。
- **拉起 baseline 前询问特殊配置**：询问“baseline 拉起是否需要额外的启动参数、环境变量或配置文件？如有，请提供具体内容，后续特性验证会共同继承这些配置。”已有本轮明确答复时直接沿用，不重复询问；明确不需要也记录。尚未答复则保持待补充，不默认为无配置并提前启动。按 [references/execution-notes.md](./references/execution-notes.md) 将答复与最终生效配置固化到 `manifest.json` 的 `baseline_special_config`。
- **自动生成规则**：agent 读取权重目录 `config.json` 识别模型类型（是否 MoE、架构、注意力类型、并行约束），结合阶段1-B 确认的**可用卡数**，自动推导基础配置：
  - **TP（张量并行）**：默认取全部可用卡；若模型并行约束不允许（如 config.json 标注不支持的并行度），按最接近的合法并行度回退。**多机下 TP ≤ 单机卡数**（TP 组通信通常同节点；跨机用 PP/DP，不把 TP 扩到跨机）。
  - **DP（数据并行）**：按卡数与目标拓扑推导，验证时覆盖 `DP≠1`，保证并行拓扑被实际验证到。**多机下 DP 跨节点扩展**：vLLM 用 `--data-parallel-size`（总 DP）/ `--data-parallel-size-local`（每节点 DP）/ `--data-parallel-start-rank`（每节点起始 rank）/ `--data-parallel-address`（head IP）/ `--data-parallel-rpc-port` 编排，head 运行 API Server + Engine，worker 只跑 Engine（`--headless`），无需 Ray。
  - **PP（流水线并行）**：普通 baseline 的 PP 按容量与拓扑规划；PP 专项默认验证 `PP>1`，在已规划资源内生成合法的 TP/DP/PP 配置并独立预检，不能因 baseline 已能放下模型而跳过。跨节点 PP 使用目标版本支持的原生启动方式。
  - **EP（专家并行）**：MoE 模型**默认开启** `--enable-expert-parallel`（非 MoE 模型不适用）。
  - **图模式**：按目标 vLLM-Ascend 版本和交付件约定开启（如 `--compilation-config` 相关图模式配置），保证**图模式在基础配置阶段即被拉起验证**。
  - **端口等**：用默认端口（如 8000），host 用本机；多机下 service port（API）与 dp_rpc_port（DP 通信）由复用的启动脚本自动分配。
  - **多机怎么“自动找配置”**：多机配置不再只按单机卡数推导，而是 **先逐节点探测可用卡**（复用 `serve_probe_npus.py`，返回各节点 `total/free/busy` 与 HBM）→ 以**最小单机可用卡数**判定 TP 上限 → 用 `dp_local = min(每节点逻辑设备数 // TP)`、`dp_total = dp_local × 节点数` 推导跨机 DP；按默认 PP 专项另行规划合法配置并预检容量，跨节点 PP 选择原生 launcher → 分别组装服务命令。推导细节、rank、回滚和网络接口规则见 [references/multi-node.md](./references/multi-node.md)。
- **验证要求**：agent 生成的**基础配置本身需先成功拉起并验证**（确认 TP≠1、DP≠1、EP、图模式均生效），再把每个特性叠加在其上验证。多机下基础配置拉起即验证了跨节点 DP 拓扑（head + workers 均就绪）。
- **baseline 达标判定（由 AI 判断，不交客户）**：对 baseline 服务发起标准 curl 功能检测（见 [references/benchmark-aisbench.md](./references/benchmark-aisbench.md)，`POST /v1/completions`，`model:"auto"`、`temperature:0`），**是否达标由 AI 判断 curl 输出是否无乱码**：正常返回 `choices[].text`、无 `Traceback`/报错、输出文本无乱码（无替换字符/乱码字符）即视为 baseline 功能正常达标。**不再交由客户判定。**
- **修正原则**：基础配置候选生成阶段若发现参数/并行度不合法，可依据报错与 `config.json` 约束修正候选。普通服务拉起或特性失败仍按 [references/failure-diagnosis.md](./references/failure-diagnosis.md) 定位并记录；**权重更新后双接口结果与基线不一致是允许修复的专项场景**，必须按 [references/weight-update-consistency.md](./references/weight-update-consistency.md) 依次检查漏更新权重、地址变化导致的脏数据/旧引用、图模式地址变化，根因证实后可实施最小代码修复并在原配置下完整复验。显存不足仍输出《容量报告》并终止，不得自动换并行或盲目重试。
- **容量预检（权重放不下 → 终止+报告）**：生成基础配置候选后必须**先做容量预检，再允许任何服务拉起**——统计实际加载的权重文件与 dtype/量化、逐节点每卡 HBM 和可用卡数，结合 TP/DP/EP/PP 分摊估算单卡需求，并预留 KV、激活、图模式/编译、HCCL 和运行时开销及利用率余量。**若权重无法在当前卡数下放下，立即终止整个开箱流程**，输出固定《容量报告》并标明「权重无法在当前卡数情况下放下」，不得调用 `serve_start.py`/`vllm serve`、继续特性循环或盲目重试。**多机判法**：总可用显存取各节点实际可用 HBM 之和，但 TP 上限锁死在单机可用卡数（通常取各节点卡数最小值及其合法约数）；若 `TP_max` 下单卡仍放不下，报告必须指出「需增加节点数/换更大 HBM」，跨机优先使用 PP/DP。详见 [references/capacity-check.md](./references/capacity-check.md)。
- **后续配置继承**：每个特性的服务、对照服务、重启及修复后复验，均以同一份已固化 baseline 配置为起点，保留特殊参数、环境变量和配置文件，再叠加特性开关；适用于真实权重和减层权重两种模式。JSON 配置逐键合并，不能用特性配置整体覆盖 baseline 的 `--additional-config`/`--compilation-config`。发生同键冲突或与特性要求不兼容时，说明差异并询问用户如何处理，不静默删改；仅端口、设备映射等有明确规则的运行差异按节点生成并记录。公共配置或环境补丁改变后，重新生成 baseline，不复用旧结果。

### 拉起方式（单机 vs 多机，复用多机脚本）

- **单机**：按原方式由 agent 直接在目标机上执行 `vllm serve` 拉起（后台 `nohup`），或由本 skill 的测试脚本本地 `subprocess` 拉起。
- **多机**：**复用 `vllm-ascend-serving` 的多机脚本**，不要在 head 上手工拼跨机命令：
  - 拉起集群：`serve_start.py --host <head-ip> [--container <name>] --worker-hosts <w1>,<w2> --model <path> --tp <每节点TP> [--port <service-port>] -- [<feature 附加参数>...]`；跨节点 DP 由脚本按逐节点探测结果自动推导，不手工传总 DP 覆盖。
  - 查状态：`serve_status.py ...`（输出 `cluster_type:"multi-node"`、head/workers 存活、`nnodes`、`dp_rpc_port`）。
  - 停止：`serve_stop.py ...`（遍历所有节点 SIGINT→SIGTERM→SIGKILL，清理孤儿进程）。
  - 多机服务统一使用 `../vllm-ascend-serving/scripts/serve_start.py`、`serve_status.py`、`serve_stop.py` 和 `serve_probe_npus.py`；若目标 `vllm-ascend` checkout 提供特性 e2e 脚本，再按版本接入。
- **多机注意事项（重要）**：
  - **Head 禁止 `--headless`**：Head 必须承担 API Server；只有 Worker 以 engine-only（`--headless`）运行。启动脚本会自动对 head 剥离 `--headless`。
  - **通信网卡**：若目标平台要求专用通信网卡，按服务脚本配置对应的接口和本机管理 IP；不要从未验证的接口输出中猜测 IP。细节见 [references/multi-node.md](./references/multi-node.md)。
  - **连接方式**：head 与各 worker 统一用同一种 `--mode`（docker/conda/direct），worker 凭据默认继承 head。
  - **客户端可达性**：多机验证时，测试脚本连的是 head 的 API 端口；若控制机不能直连 head IP，需先建 SSH 隧道或走可达代理。

### 特性开启方式的识别来源

- 特性（routing replay / sleep mode / 权重更新 / CP / SP / PP 等）的**开关位置与开启方式**，通过**阅读代码仓**来识别，而非仅依赖口述。
- 主要代码仓：`vllm-project/vllm` 与 `vllm-project/vllm-ascend`。
- 用户会提供相关代码仓，技能读取代码判断各特性的开启参数与验证方式；必要时再向用户确认。

### Checkpoint 权重流（HCCL/IPC 强制前置）

HCCL/IPC 只使用 checkpoint-native 数据面：扫描 safetensors/PyTorch
checkpoint 头部获取名称、shape 和 dtype，再逐张量搬到 producer NPU。
发送名称必须保持为原始 checkpoint key，由接收端已适配模型的
`load_weights()` 执行 mapper、fused/split/layout、量化处理和 TP/PP/EP 切分。
不得构造 Transformers 模型、使用 FSDP，或在 producer 侧自行把 key
转成 worker 内部参数名。详细规则见
[references/weight-name-alignment.md](./references/weight-name-alignment.md)。

### 特性清单（开箱验证项）

| # | 特性 | 开启方式（概要） | 验证内容 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | routing replay | `--enable-return-routed-experts` | 功能验证 | MoE 路由重放 |
| 2 | **sleep mode** | `--enable-sleep-mode` | 功能验证 | 单独验证 sleep(level=2)、分阶段 wake、状态与内存证据 |
| 3a | **HCCL 权重更新** | `--weight-transfer-config '{"backend":"hccl"}'` | 真实权重幂等一致性验证 | 独立 producer NPU 流式读同一 checkpoint；使用 `test_hccl_weight_update.py` |
| 3b | **IPC 权重更新** | `--weight-transfer-config '{"backend":"npu_ipc"}'` | 真实权重幂等一致性验证 | 每个 worker 同卡启动 checkpoint producer；使用 `test_npu_ipc_weight_update.py`；仅可信环境允许不安全序列化 |
| 3c | **Checkpoint 权重重载** | dev-mode `/collective_rpc` + `reload_weights(is_checkpoint_format=true)` | 真实权重幂等一致性验证 | 不需要 `weight-transfer-config`；使用 `test_reload_weights.py`；服务先以真实权重启动 |
| 4 | CP（上下文并行） | `--prefill-context-parallel-size` + `--decode-context-parallel-size` | 功能 + 精度验证；**分 DCP / PCP 分别测** | **DCP、PCP 均默认执行** |
| 5 | SP（序列并行） | MoE 专家并行推导（all2all backend + EP + TP + DP） | 功能 + 精度验证 | **默认执行** |
| 6 | PP（流水线并行） | `--pipeline-parallel-size` | 功能 + 精度验证；稀疏注意力模型先做 Indexer/Top-K 边界预检 | **默认执行** |

> **权重更新执行清单**：HCCL、IPC 和 Checkpoint 重载是三个独立执行项，不能用一个后端的通过结论代替另一个。服务必须先以目标真实 checkpoint 启动并生成双接口基线；随后分别完成：容量预检 → checkpoint manifest 扫描 → `sleep(level=2)` → 初始化 checkpoint producer/transfer engine（或 reload engine）→ `wake(weights)` → 流式 `start/update/finish` 或 checkpoint reload → `wake(kv_cache)` → 更新后生成。正式通过的唯一标准是两个接口、所有 prompt 的 text/token/logprob 与基线完全一致；任何差异均为失败。保存 producer/服务完整日志、manifest checksum 和更新前后输出。缺少共享 checkpoint、对应后端或目标版本 API 时，分别记录为 `not_run` 或 `blocked`。

三个权重更新脚本复用 [scripts/weight-update/common/weight_update_validation.py](./scripts/weight-update/common/weight_update_validation.py) 的 `WeightUpdateValidation`：`record_baseline` 记录双接口基线，`sleep_service` 确认休眠，`wake_weight_buffers` 和 `wake_kv_cache` 分阶段唤醒，`verify_updated_outputs` 校验全部提示词的输出数量及 text/token/logprob，`record_incomplete_update` 记录未完成更新的恢复提示。后端初始化、实际权重传输、显存释放和分布式同步由各脚本保持原有顺序；公共方法在协调进程执行，IPC 通过集体步骤同步异常。

维护这三个脚本及其公共模块时，Python 文件、函数和变量使用描述用途的 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。优先使用 `arguments`、`world_size`、`device_index`、`weight_source`、`transfer_group` 等完整名称；函数名说明动作及对象，如 `create_checkpoint_producer`、`wake_weight_buffers`。保留领域通用术语 `rank`、NPU、IPC、HCCL，以及外部 API 的既有参数名；不使用单字母、版本数字拼接或含义不清的内部缩写。

命令行公共参数集中在 [scripts/common/feature_arguments.py](./scripts/common/feature_arguments.py)，由权重更新、基线、sleep、pause/resume、CP、SP、PP 和 routing replay 脚本按需组合。`create_feature_parser` 创建解析器，`add_service_arguments` 添加服务地址与模型名，`add_generation_arguments` 添加提示词及生成长度，`add_report_arguments` 添加报告路径及验证命令，`add_timeout_argument` 添加超时，`add_cluster_connection_arguments` 添加集群连接参数。提示词支持 JSON 文件、重复 `--prompt` 和单条 `--prompt`；各脚本保留自己的参数名称、默认值和必填项，不加载无关特性的参数。正整数、非负整数、有限正数和端口范围使用公共校验类型。同一模块中的 `add_checkpoint_transfer_arguments` 定义 HCCL/IPC 共用的 checkpoint 与缓冲区参数，按需添加。
> **权重名称对齐前置条件**：HCCL/IPC 在 level-2 sleep 前必须完成
> [references/weight-name-alignment.md](./references/weight-name-alignment.md) 的
> checkpoint schema 预检。HCCL 与 IPC 共用同一份 checkpoint manifest；
> 名称、shape、dtype、重复 key 或分片索引未通过时不得开始 sleep。
> 报告必须归档 checkpoint 路径/版本、tensor/file 数、总字节数、
> 最大单张量字节数和 manifest checksum。
> **默认验证项**：HCCL、IPC 和 Checkpoint 重载均属于独立执行项；CP 的 DCP、PCP，以及 SP、PP 也默认分别执行。顺序不代表可省略，不能用其中一项的结论替代另一项。
> **多机特性支持**：跨节点开箱下，专项脚本通过 `--worker-hosts` 委托 `serve_start.py`/`serve_stop.py` 管理集群，并以 head 的在线 HTTP 端点验证；但该 serving wrapper 当前编排的是跨节点 DP，不能仅凭它的多机状态或 HTTP 成功声称跨节点 PP 已验证。真正的跨节点 PP 必须使用目标版本支持的 MP/Ray/专用 PP launcher；否则 PP 记为 `blocked`/`not_run`。sleep mode + 权重更新是否支持多节点，以目标版本的 e2e 用例和日志为准。CP/SP/PP 是否支持还要核对目标版本、模型和拓扑；不支持时记录 `blocked`，不把普通 DP 结果冒充特性结果。稀疏注意力模型的 PP 边界、IndexCache、Indexer、SP/CP/offload 限制见 [references/pp-sparse-attention.md](./references/pp-sparse-attention.md)。详见 [references/multi-node.md](./references/multi-node.md)。

### 每轮循环执行步骤

**阶段2：拉起服务**

1. 首次启动前完成环境前置修改核查、baseline 特殊配置询问与固化，再取得**模型权重路径**，按上文「基础配置（agent 自动生成）」规则生成含特殊配置的基础配置候选（TP/DP/EP/PP/图模式/端口等），并**先完成容量预检**。只有前置项已满足、配置已明确且容量预检通过后，才允许拉起并验证基础配置本身（含 TP≠1、DP≠1、EP、图模式生效）；容量失败时按《容量报告》终止，不进入本轮或后续特性。
2. 从固化的 baseline 配置派生当前服务，保留全部公共特殊配置并叠加当前特性的开启参数；启动前核对差异与 head/worker 的实际生效值，确认没有遗漏、覆盖或未处理冲突。公共环境或配置发生变化时回到 baseline 重验。
3. 拉起 vllm-ascend 服务：

   - **先展示、再执行命令（强制）**：在页面中输出当前特性的完整、可复制拉起命令（包含环境变量、容器进入方式、模型路径、并行参数、特性开关、端口和日志重定向），让用户能在执行前检查命令；不要只展示省略号、脚本名或后台 PID。密码、token 等凭据必须显示为 `<redacted>`。多机必须同时展示实际委托给 `serve_start.py` 的完整命令。
   - 同一段命令必须追加到 `model_openbox/<model-slug>/logs/launch_commands.txt`，按特性写明 `SERVICE`、`VERIFIER`、`STOP/CLEANUP` 三类命令。脚本自动生成的命令也要在页面打印；`scripts/common/online_feature_common.py` 的 `launch command` 输出可直接作为证据。

   - **单机**：后台 `nohup ... > log 2>&1 &`，**轮询就绪**：`curl http://<host>:<port>/health` 返回 200，或日志出现 `Application startup complete` / `Uvicorn running`；同时**监控日志错误**（`Traceback` / `ImportError` / `Failed to load the backend extension` 等）。
   - **多机**：复用 `serve_start.py` 拉起集群（`--worker-hosts`），脚本先完成所有节点 preflight，再等待 head DP RPC 就绪后启动 worker，并轮询 head `/health` 与 `/v1/models`；任一 worker 启动失败按报错定位并由 `serve_stop.py` 回滚已启动节点。
   - 拉起失败要能识别根因（常见：CANN 库缺失、端口/资源占用、参数不合法），先按 [references/failure-diagnosis.md](./references/failure-diagnosis.md) 收集证据、逐帧回溯并产出《问题定位报告》；不要“拉起失败却一直等”，也不要在诊断阶段自动修正后重跑。**若报显存不足（`Free memory < desired` / `Out of Memory during initialization` / `Memory_Allocation_Failure EL0004` 等）→ 不重试，按 [references/capacity-check.md](./references/capacity-check.md) 输出《容量报告》终止流程**（说明权重无法在当前卡数下放下，附实际日志数值；多机下可能指向「需增加节点数」）。

**阶段3：验证**

1. 按该特性的验证方法执行验证（功能验证 / 功能+精度验证），具体见 references/feature-matrix.md。**CP（DCP、PCP 分开）、SP、PP 均默认执行，不再询问是否增加这些验证项；用户明确排除或客观阻塞时记录原因和证据。**

   - **每次验证命令都要展示**：在页面输出实际执行的 verifier/torchrun/curl/AISBench 命令，并追加到 `logs/launch_commands.txt`；报告 JSON 尽可能写入 `launch_command`、`verification_command`、`service_url` 和日志路径字段。无法由脚本自动记录的 HCCL/IPC/Checkpoint/睡眠服务命令，由 agent 在执行前后补录，禁止用“同上”代替。
   - **权重更新项先做名称对齐预检**：HCCL/IPC verifier 启动后先执行 mapper/schema 握手，打印 source/target 数量和不匹配摘要；预检未通过时直接输出 `blocked`/`failed` 报告，不进入 level-2 sleep。不得用 `strict=False`、跳过参数或位置对应继续发送。

2. **若该特性验证失败（脚本退出码非 0 / 服务拉起失败 / 断言不通过）→ 先做失败诊断**：保存完整 traceback（如有）、服务日志、退出码/断言点、更新前后输出、实际命令、环境/资源状态和部署版本。普通失败按 [references/failure-diagnosis.md](./references/failure-diagnosis.md) 从最内层异常向外标注 L0～L5，并回到实际部署源码印证根因，产出《问题定位报告》。**HCCL、IPC 或 Checkpoint 更新前后结果不一致时，进入 [references/weight-update-consistency.md](./references/weight-update-consistency.md)**：依次排查遗漏权重、地址变化引起的脏数据/旧引用、图模式 capture/replay 地址变化，并补查异步传输、缓存和派生状态；可对已证实根因实施最小修复，包括必要的 `register_buffer`、原地更新及图缓存刷新，随后完整复验。其他失败默认只定位，不关闭 strict load、不跳过特性、不替换依赖、不反复运行同一失败命令；模型/版本级硬限制仍标注为“代码/版本级限制”。

3. **记录运行结果**（功能是否正常、输出、精度/性能指标、**失败时的《问题定位报告》**等关键证据），形成该特性的开箱记录；**精度问题由客户判定**。
4. 结束后**停止当前服务**，回到阶段2 处理下一特性。

> **失败诊断边界**：普通参数/环境错误按客户授权处理；其他普通失败形成《问题定位报告》后默认保留现场，不自动实施候选补丁。权重更新前后结果不一致按专项规则允许定位、最小修复和原配置复验，但首次失败仍保留为 `failed`，只有真实复验满足双接口 text/token/logprob 完全一致后才能记录修复后通过。初始化 OOM 仍按 [references/capacity-check.md](./references/capacity-check.md) 输出《容量报告》并终止。
> **验证手段**：以 **AISBench** 为主（精度/性能评估），curl 调 API 为辅（功能/服务可用性）。**所有特性必须同时验证两个接口**：`POST /v1/completions` 和 `POST /v1/chat/completions`，使用同一组 prompt、model、temperature、seed 和 max_tokens；分别记录 HTTP 状态、输出文本、token/logprob（接口支持时）和错误。不能只验证其中一个接口就声称该特性通过。**执行前需 `unset` 环境代理**（否则被代理拦截无法直连本地服务）。详细见 [references/benchmark-aisbench.md](./references/benchmark-aisbench.md)。
>
> **e2e 用例脚本验证（推荐，逐特性接入）**：优先采用目标 `vllm-ascend` checkout 中与当前 commit 匹配的 e2e 用例，并把 agent 生成的 baseline 拉起命令作为唯一服务入口。当前仓库没有假设这些专项脚本一定存在；调用前必须检查脚本路径、参数和版本，缺失时使用本 skill 的 curl/AISBench 验证合同，并在报告中标记专项脚本为 `not_run`。
>
> - routing replay：使用目标仓库对应 e2e 用例；若目标 checkout 没有可直接运行的 CLI，用本 skill 的 [scripts/routing/test_moe_routing_replay.py](./scripts/routing/test_moe_routing_replay.py) 执行 HTTP + 进程内双路径验证，服务由 `../vllm-ascend-serving/scripts/serve_start.py` 拉起；多机通过 `--worker-hosts` 委托服务脚本，进程内路径只作单机补验。
> - sleep mode + 权重更新：优先使用目标版本提供、与当前 commit 匹配的原生 e2e/API；没有共享 checkpoint 或其他真实权重数据面时不得伪造权重更新结论。纯 sleep 状态机可使用 [scripts/lifecycle/test_sleep_mode.py](./scripts/lifecycle/test_sleep_mode.py)。权重更新必须按 HCCL、IPC、Checkpoint 三个独立项分别执行：HCCL 使用 [scripts/weight-update/test_hccl_weight_update.py](./scripts/weight-update/test_hccl_weight_update.py)，IPC 使用 [scripts/weight-update/test_npu_ipc_weight_update.py](./scripts/weight-update/test_npu_ipc_weight_update.py)，Checkpoint 使用 [scripts/weight-update/test_reload_weights.py](./scripts/weight-update/test_reload_weights.py)。HCCL/IPC 验证必须把服务端 `--gpu-memory-utilization` 调低到明显低于普通推理 baseline 的值（可先从 `0.3` 起，根据容量预检、模型和拓扑调整），为最大单 checkpoint 张量、packed/IPC buffer 和 HCCL/运行时开销预留显存；不得直接沿用 baseline 的高利用率设置。脚本只连接已运行服务，服务启停仍由阶段2的 serving 入口管理；执行前确认容量、拓扑、checkpoint、后端和目标版本 API，缺少真实依赖时记录为 `not_run`/`blocked`。
> - CP / SP / PP：优先使用目标版本中与当前 commit 匹配的在线 HTTP e2e 用例；DCP、PCP、SP、PP 均默认执行。若目标 checkout 没有可直接复用的 CLI，可使用 [scripts/parallel/test_context_parallel.py](./scripts/parallel/test_context_parallel.py)、[scripts/parallel/test_sequence_parallel.py](./scripts/parallel/test_sequence_parallel.py) 或 [scripts/parallel/test_pipeline_parallel.py](./scripts/parallel/test_pipeline_parallel.py)；CP 使用默认 `--dim both` 分别执行 DCP、PCP，脚本会分别拉起 baseline 与特性服务，对同一组 `/v1/completions` 请求做在线对照；传入 `--worker-hosts` 时由 `serve_start.py` 启动多机服务。PP 额外要求先阅读 [references/pp-sparse-attention.md](./references/pp-sparse-attention.md)：稀疏 Indexer/Top-K 边界必须通过预检，多机 `--worker-hosts` 只代表 DP 编排，不代表跨节点 PP。脚本返回 `2` 表示模型/硬件/版本或拓扑不支持该特性，应保留证据并记录为 `blocked`。
>
> **多机适配概览**：跨节点的 HTTP 特性通过 `serve_start.py` 委托启动；但 `serve_start.py --worker-hosts` 当前是跨节点 DP 编排。CP/SP/PP 的在线脚本若使用该入口，只能证明对应的多机 DP 服务可用，不能证明跨节点 PP。跨节点 PP 必须以目标版本 e2e、原生 MP/Ray/专用 launcher、服务日志和实际拓扑为准；不支持时输出 `blocked`/`not_run`，不自动改并行策略。连接参数、headless 规则和 DP 拓扑见 `references/multi-node.md`，稀疏模型边界见 `references/pp-sparse-attention.md`。
>
> 运行前需 `unset` 环境代理；CP/SP/PP 使用在线 HTTP 脚本，其余特性使用 HTTP/开发端点或目标版本 e2e，详见各 reference 与脚本头注释。

### 验证顺序建议

- 优先：routing replay → **sleep mode + HCCL 权重更新 + IPC 权重更新 + Checkpoint 重载**；sleep、HCCL、IPC、Checkpoint 分别记录结论，不能合并为单一“权重更新通过”。
- 随后：CP（DCP、PCP 分别执行）、SP、PP，均为默认执行项。

## 完成条件与输出

容量预检通过时，任务只有在以下内容均已固化时才算完成：

1. `manifest.json`、`audit.md`、原始日志、`logs/launch_commands.txt` 和 `report.md` 已写入 `model_openbox/<model-slug>/`；
2. 基础配置经过容量预检，且服务健康检查与标准 curl 均有明确状态；
3. 每个必选特性都有 `passed`、`failed`、`not_run` 或 `blocked` 结论及对应命令/日志；
   DCP、PCP、SP、PP 必须分别列项；未执行项注明用户明确排除、环境不可用或前置阻塞等实际原因，不保留默认可选跳过的结论。
4. 多机任务记录节点数、每节点卡数、TP/DP/PP 推导和 head/worker 状态；
5. 失败项先完成根因定位并保留现场；权重更新一致性问题需记录根因证据、最小补丁及原配置复验结果，不以关闭 strict load、跳过特性、降低比较标准或反复重试制造通过结论。
6. sleep + 权重更新项必须分别记录 HCCL、IPC 和 Checkpoint 结论。服务先加载真实 checkpoint；更新后两个接口、所有 prompt 的 text/token/logprob 必须与更新前完全一致，否则不能通过。HCCL/IPC 必须归档 producer、服务完整日志，以及 checkpoint manifest checksum、版本或一致性证据；Checkpoint 必须归档客户端报告、服务完整日志、checkpoint 路径/版本及更新前后输出。服务恢复正常不等于权重切换已被证明。
7. HCCL/IPC 权重更新报告必须额外包含 checkpoint 格式、分片数、tensor 数、总字节数、最大单张量、manifest checksum 和 schema 预检结果；没有这些证据不得标记为“权重更新已验证”。
8. 在线功能验证报告必须包含 `endpoint_checks`、`baseline_outputs_by_endpoint` 和 `feature_outputs_by_endpoint`（权重更新项对应 `baseline_by_endpoint` / `post_update_by_endpoint`），并明确 `/v1/completions` 与 `/v1/chat/completions` 是否通过。权重更新项统一按接口保存更新前后输出，通过 `outputs_identical_by_endpoint` 记录每条提示词的一致性结果；不额外保存单接口输出副本或重复的变化诊断字段。双接口探针对 completions 请求设置 `logprobs=1`，对 chat 请求设置 `logprobs=true`、`top_logprobs=1`，并从 `logprobs.content` 提取生成 token 及其 logprob；两接口的 token 列表须非空，logprob 数量须与 token 一致且值为有限数值，缺失数据直接失败，不得把两侧空值相等作为验证通过依据。Profiling、硬件测算与数据集评测使用下述专项证据，不要求伪造双接口对照输出。
9. **最终必须生成 Markdown README**：使用 [scripts/reporting/generate_rl_case_guard_readme.py](./scripts/reporting/generate_rl_case_guard_readme.py) 汇总本目录下各特性 JSON、`logs/launch_commands.txt` 和推理输出，生成 `model_openbox/<model-slug>/README.md`。README 至少包含每个特性的 `passed`/`failed`/`blocked`/`not_run` 结论、逐项 checks、两个接口的检查结果、错误/阻塞原因、日志路径和实际拉起/验证命令；存在 baseline 与特性后推理结果时，按接口和 prompt 展示两者。README 生成失败时任务不能标记完成，应记录编码或输入证据错误。旧版 PDF 脚本仅作兼容保留，不再作为交付物。

10. `manifest.json` 和 `audit.md` 记录环境是否需要前置修改、含 PR 号/链接的修改清单及实际状态，以及用户对 baseline 特殊配置的答复与固化配置版本。每项特性记录所用配置版本、公共配置继承情况及差异；README 展示前置修改清单和 baseline 特殊配置。待应用的必要修复、待核实的必要前置条件或未解决配置冲突，应将依赖它的验证项标为 `blocked`，不得声称已在目标配置下通过。
11. Profiling、硬件测算与数据集评测按 [references/performance-and-evaluation.md](./references/performance-and-evaluation.md) 执行并分别记录状态。Profiling 归档实际输入/输出 token、真实多模态输入和中段 10 步证据；硬件测算归档单卡 CSV、筛选口径、经验值来源、JSON 与 Excel；评测归档用户选择、生成长度、数据集/配置版本和原始指标。减层测算标记 `functional_smoke`，不能作为全层模型性能结论；用户未选择的评测记录 `not_run`。

若容量预检失败，或服务初始化阶段确认发生显存不足，则任务以“容量阻塞/已终止”完成：必须保存固定格式《容量报告》、权重统计、逐节点 HBM/卡数、基础配置候选和原始错误证据；不要求继续执行健康检查、标准 curl 或后续特性验证，也不得为了满足常规完成条件而再次拉起服务。

报告必须分别列出 sleep mode 与权重更新结果；每项必须保存 server 完整日志到 `logs/`，并在报告中给出日志路径、错误分析/通过依据及关键报错日志摘录。报告必须确认并单独记录 DP、EP、TP 的实际拓扑、启动参数、验证结果和证据。报告必须区分“静态检查通过”“服务功能正常”“精度由客户判定”和“昇腾实机已验证”；没有真实 NPU 日志时不得声称后者。

## 参考资源

- `references/performance-and-evaluation.md` — 复用 Profiling 与 AISBench，单卡加权极致性能达成率、经验上限配置及 Excel 导出。
- `references/reduced-checkpoint.md` — 可选的 config-only 减层权重生成、结构保留、初始化限制与功能验证边界。
- `references/feature-matrix.md` — 特性清单各特性的开启方式与验证方法（核心参考）。
- `references/multi-node.md` — **多机场景专篇**：多节点探测、基础配置多机推导（TP 每节点 / 跨机 DP/PP）、复用 serve_start/status/stop 拉起集群、多机验证与注意事项（headless、网络接口、连接模式）。
- `references/capacity-check.md` — 容量预检（单机/多机通用）与《容量报告》格式。
- `references/benchmark-aisbench.md` — AISBench 为核心的精度/性能评估步骤（阶段3 验证手段）。
- `references/ssh-login-troubleshooting.md` — 阶段1-A 登录失败的通道切换和安全边界。
- `scripts/routing/` — routing replay 专项验证。
- `scripts/lifecycle/` — sleep/wake 与 pause/resume 生命周期验证。
- `scripts/weight-update/` — HCCL、NPU IPC、Checkpoint 三类独立权重更新验证。
- `scripts/parallel/` — CP、SP、PP 在线专项验证。
- `scripts/baseline/`、`scripts/reporting/`、`scripts/common/` — 基线生成、Markdown README 报告和公共实现。
- `references/execution-notes.md` — 环境前置修改及 PR 清单、baseline 特殊配置询问与继承、CANN 加载和就绪轮询。
- `references/failure-diagnosis.md` — 按调用栈和源码证据定位特性失败根因。
- `references/weight-name-alignment.md` — checkpoint 原始 key 合同、manifest 生成、vLLM `load_weights()` 边界和 HCCL/IPC 发送前校验。
- `references/weight-update-consistency.md` — 权重更新结果不一致的有序根因排查、buffer/地址/图模式修复及完整复验规则。
- `references/pp-sparse-attention.md` — 通用 PP 切层、稀疏 Indexer/Top-K 边界校验、模型级限制和多机 PP 启动判定。
- `../vllm-ascend-serving/scripts/serve_start.py`、`serve_status.py`、`serve_stop.py`、`serve_probe_npus.py` — 统一的服务和多机控制入口。
- `../vllm-ascend-serving/SKILL.md` — 连接模式、环境加载、端口、headless 和多机服务生命周期。
- 目标 `vllm-ascend` checkout 中的特性 e2e 用例 — 仅在确认版本和脚本存在后接入，不由本 skill 假设或伪造。
