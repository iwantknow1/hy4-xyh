---
name: ascend-profiling-collection
description: >-
  This skill should be used when the user asks to "采集 profiling",
  "torch profiler 跑一个 case", "采一份 profile", "采 profiling 给我分析",
  "collect torch profiler trace", or wants to collect an end-to-end Ascend
  torch-profiler case on a managed NPU environment through the serving skill.
version: 1.2.0
domain: inference-model
function: performance-optimization
tags:
  - vllm-ascend
  - profiling
  - torch-profiler
  - performance
author: Liccol
hooks:
  PreToolUse:
    - matcher: "Bash|PowerShell"
      prompt: |
        STOP. Before executing any command, you MUST:
        1. Resolve omitted optional parameters to the documented defaults and present the full parameter snapshot (including: target host+container/conda-env,
           CANN dir, model, served-model-name, TP/DP, mode, tag, profile-phase,
           random-input-len, random-output-len, profile-steps, ignore-frontend, workload-kind,
           multimodal token overhead and image dimensions (when applicable), num-prompts, max-concurrency,
           sampling parameters; mark omitted sampling parameters as "not passed / preserve runtime defaults", never as None,
           torch-profiler-dir, devices, extra-env).
        2. If the user has already explicitly requested automatic/unattended execution, proceed with that snapshot without asking for a second confirmation. Otherwise wait for explicit confirmation.
        Required parameters that cannot be resolved from the request or defaults must still be requested before execution.

    - tools: ["Bash"]
      matcher: "ssh.*\\s/(home|workspace|tmp|usr|vllm)"
      prompt: |
        ⚠️ Git Bash 会转换 Linux 路径。请用 **PowerShell** 执行 SSH 命令。

    - tools: ["Bash", "PowerShell"]
      matcher: "(compilation_config|additional_config|profiler.config)"
      prompt: |
        ⚠️ JSON 双引号必须转义：PowerShell 用 `\"`，Git Bash 用 `\\"`。
targets-repo:
  - vllm-ascend
---

# ascend-profiling-collection

## 功能概述

端到端采集一份 Ascend torch-profiler 案例：启动带 profiler 的服务 → 激活 profiling → 运行 workload → 停止服务 → 分析 trace 数据 → 输出 manifest。

- 支持三种 profiling 阶段：`head`（前段）、`middle`（中段，默认）、`tail`（尾端）
- 自动验证每个 rank 的 `kernel_details.csv` 是否生成
- 强失败策略：rank 缺失、benchmark 成功率不足时拒绝标记为成功
- 当前 checkout 的公共 serving 入口支持 `docker`、`conda`、`direct` 三种 SSH 连接模式；执行前以入口 `--help` 核对，不能在尚未支持本地执行的版本传入 `local`
- 本地状态保存在 `.workspace-local/ascend-profiling-collection/runs/`
- profiling 数据默认保存到环境的 `/home/workspace/`（绝对路径），`*_ascend_pt` 目录直接位于该路径下；传入相对路径则相对于 vLLM runtime_dir 解析

## 使用方式

- 直接调用：`/mindspeed-agent:ascend-profiling-collection`
- 描述 profiling 采集需求，由 Claude 自动匹配调用

## 操作步骤

### 步骤一：确认参数

| 参数 | 说明 | 必填 |
| ------ | ------ | ------ |
| 目标 | Docker、direct SSH 或 Conda 参数（`--host <ip> [--connection-mode docker] --container <name>`（Docker）、`--host <容器IP> --connection-mode direct --password <pwd>`（直连容器SSH）、`--host <ip> --connection-mode conda --conda-env <name>`（Conda）） | ✅ |
| 连接模式 | `--connection-mode {docker,conda,direct}`（默认 `docker`）：选择当前 serving 支持的连接拓扑 | ✅ |
| CANN 路径 | `--cann-dir <path>`（默认 `/usr/local/Ascend`），执行前必须确认 | ✅ |
| 模型 | `--model <remote-path>` | ✅ |
| 服务名 | `--served-model-name <name>` | ✅ |
| TP / DP | `--tp <N> --dp <N>` | ✅ |
| 图模式 | `--mode {enforce_eager\|full_decode_only\|piecewise_graph}` | ✅ |
| 标签 | `--tag <stable-id>`（用于 run 目录命名和 manifest） | ✅ |
| 采集阶段 | `--profile-phase {head\|middle\|tail}` | ⭐ |
| 输入/输出长度 | `--random-input-len <N> --random-output-len <N>` | ⭐ |
| 采集步数 | `--profile-steps <N>`（默认 20；RL 用例使用 10） | 🔧 |
| 工作负载 | `--workload-kind text\|multimodal`（默认 text） | 🔧 |
| 固定请求长度 | `--random-range-ratio 0`（省略则保留 benchmark 默认） | 🔧 |
| 多模态输入开销 | `--mm-input-token-overhead <N>`（图片展开与模板的实测 token；multimodal 必填） | 🔧 |
| 图片尺寸 | `--image-height <N> --image-width <N>`（默认 256×256，每请求 1 张） | 🔧 |
| 排除前端采集 | `--ignore-frontend`（按 worker 步数采集时使用） | 🔧 |
| 请求数 | `--num-prompts <N>`（典型：10-96） | ⭐ |
| 保存目录 | `--torch-profiler-dir <path>`（默认 `/home/workspace`） | 🔧 |
| 设备 | `--devices <ids>` | 🔧 |
| GPU 内存利用率 | `--gpu-memory-utilization <f>`（默认 0.85） | 🔧 |
| max-model-len | `--max-model-len <N>`（默认 4096） | 🔧 |
| 并发数 | `--max-concurrency <N>`（默认 = num-prompts） | 🔧 |
| 采样参数 | `--temperature`、`--top-p`、`--top-k`、`--min-p`、`--frequency-penalty`、`--presence-penalty`、`--repetition-penalty`；未配置时不传递对应参数，沿用 benchmark / vLLM 的原默认行为；仅透传显式提供的参数 | 🔧 |
| 额外 vllm 参数 | `--serve-args "<args>"`（透传给 vllm serve 的任意参数） | 🔧 |
| 多机 Worker | `--worker-hosts <csv>`（多机部署时必填） | 🔧 |
| 多机参数 | `--worker-password`、`--worker-container`、`--worker-conda-env` 等（可选） | 🔧 |

> `--serve-args` 示例：`--serve-args "--additional-config '{\"enable_cpu_binding\":true}' --compilation_config '{\"cudagraph_capture_sizes\":[4,8,16]}'"`。
> 用于传递 profiling 脚本内置参数不覆盖的 vllm 参数（additional-config、自定义 cudagraph_capture_sizes 等）。

#### Profiling 阶段选择

| 阶段 | delay_iterations | 说明 |
| ------ | ------------------- | ------ |
| `head` | 0 | 采集前 ~20 个迭代 |
| `middle` | `output_len // 2` | 中段（默认） |
| `tail` | `max(output_len - 100, 0)` | 采集最后 ~20 个迭代 |

迭代数由 `--profile-steps` 控制；输出长度必须覆盖延迟与采集窗口。`max-model-len` 至少覆盖输入加输出。多模态使用 `random-mm` 通过 chat 接口发送图片与文本，目标输入长度包含实测的图片/模板开销；脚本从目标值扣除 `--mm-input-token-overhead` 后生成文本。实测请求 token 和实际 trace 步数仍需核对，不能以参数替代证据。具体 RL 负载及测算流程见 [performance-and-evaluation.md](../../deployment/model-rl-case-guard/references/performance-and-evaluation.md)。

### 步骤二：执行采集

当前版本通过已配置的 SSH 通道执行；同机运行也需使用已授权的 SSH 目标。执行采集入口前取消大小写 HTTP/HTTPS 代理，避免 loopback 的 profiler 控制请求被代理转发。参考文档中提及的 `local` 直调需要底层 serving 提供相应支持，当前入口不接受该模式。

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode direct --host <target-host> \
  --model /home/vllm/weights/Qwen3-30B-A3B --served-model-name qwen3-30b-a3b \
  --tp 4 --dp 1 --mode enforce_eager --tag batch-invariance-phase1 \
  --devices 8,9,10,11 --profile-phase middle \
  --random-input-len 5 --random-output-len 20 --profile-steps 10 \
  --num-prompts 8 --max-concurrency 8 --max-num-seqs 8 \
  --extra-env VLLM_BATCH_INVARIANT=1 \
  --serve-args "--block-size 128 --no-enable-chunked-prefill --no-enable-prefix-caching --no-async-scheduling"
```

```bash
# docker 模式 + full_decode_only 图模式
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host <ip> --container <name> --key-file <path> \
  --model <remote-path> --served-model-name <name> \
  --tp 2 --dp 2 --mode full_decode_only --tag <tag> \
  --enable-expert-parallel --devices 8,9,10,11 \
  --profile-phase tail --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 96 --max-model-len 4096

# direct 模式 + piecewise_graph 图模式
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode direct --host <容器IP> --password <pwd> \
  --model <remote-path> --served-model-name <name> \
  --tp 2 --dp 2 --mode piecewise_graph --tag <tag> \
  --profile-phase tail --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 96 --max-model-len 4096

# conda 模式 + enforce_eager 图模式
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode conda --host <ip> --conda-env <env-name> --key-file <path> \
  --model <remote-path> --served-model-name <name> \
  --tp 2 --dp 2 --mode full_decode_only --tag <tag> \
  --enable-expert-parallel --devices 8,9,10,11 \
  --profile-phase tail --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 96 --max-model-len 4096

# ===== 多机部署 =====
# docker 模式 — 两节点
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B --served-model-name qwen2-72b \
  --tp 8 --mode full_decode_only --tag multi-node-test \
  --profile-phase middle --num-prompts 96

# direct 模式 — 两容器直连
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode direct --host 10.0.0.5 --password hdpwd \
  --worker-hosts 10.0.0.6 --worker-password wkpwd \
  --model /data/models/Qwen2-72B --served-model-name qwen2-72b \
  --tp 8 --mode piecewise_graph --tag multi-node-direct \
  --profile-phase tail --num-prompts 96

# conda 模式 — 两节点
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-hosts 192.168.1.101 --worker-conda-env vllm_env \
  --model /data/models/Qwen2-72B --served-model-name qwen2-72b \
  --tp 8 --mode enforce_eager --tag multi-node-conda \
  --profile-phase middle --num-prompts 96
```

> **多机 profiling 行为**：脚本先通过 serve_start 启动多机集群（head + workers，每节点均带 profiler config），然后对 head 发 `/start_profile`、运行 workload、`/stop_profile`，各节点的 `*_ascend_pt` 数据保存在各自容器内（需要分别分析）。Worker 参数（`--worker-*`）透传给底层 serving。

### 步骤三：验证结果

检查 manifest 状态：

- `status: ok` — 采集成功
- `analysis_status: ok` — 所有 rank 的 `kernel_details.csv` 和 `trace_view.json` 均已生成
- `rank_count` 应等于 `tp * (dp or 1)`

### 步骤四：生成本地报告

采集完成后，必须将结果整理为报告并保存在本地 `.workspace-local/ascend-profiling-collection/runs/` 对应 run 目录下。

**报告内容**：

1. **采集概要**：采集时间、目标机器、模型、TP/DP、图模式、采集阶段等
2. **执行过程**：服务启动时间、profiling 窗口参数、benchmark 吞吐量
3. **采集结果**：rank 数量、`*_ascend_pt` 目录列表、`kernel_details.csv` 状态
4. **下一步指引**：提示使用 `ascend-msprof-analyze-cli` 对采集数据进行深度分析

**报告文件命名**：`report.md`（保存在 run 目录中，与 `manifest.json` 同级）

**报告示例结构**：

````markdown
# Profiling 采集报告

> **采集时间**: 2026-07-14 08:30:00 UTC
> **目标机器**: 192.168.13.190 (Docker: my-container)
> **模型**: Qwen3-30B-A3B
> **配置**: TP=2, DP=2, full_decode_only, 4 ranks

## 采集参数

| 参数 | 值 |
|------|-----|
| 采集阶段 | tail (delay_iterations=924) |
| 请求数 | 96 |
| 输入/输出长度 | 1024/1024 |
| Profiling 目录 | /home/workspace/my-tag |

## 采集结果

| 指标 | 状态 |
|------|------|
| 服务启动 | ✅ ready |
| Profile 窗口 | ✅ 已执行 |
| Benchmark | ✅ 成功 (1060 tok/s) |
| Rank 数量 | ✅ 4/4 |
| kernel_details.csv | ✅ 全部生成 |
| trace_view.json | ✅ 全部生成 |

## 远端数据路径

```text

/home/workspace/my-tag/
  rank_0_ascend_pt/
  rank_1_ascend_pt/
  rank_2_ascend_pt/
  rank_3_ascend_pt/

```

## 下一步

使用 `/ascend-msprof-analyze-cli` 对采集数据进行分析：
- 集群分析：`msprof-analyze -d /home/workspace/my-tag --agent`
- 专家建议：`msprof-analyze advisor all -d /home/workspace/my-tag --agent`
````

## 注意事项

- Profiling 采集开销极大，benchmark 吞吐量会显著下降（通常 1-2 个数量级）
- **所有分析在容器或conda环境内执行**，不要将原始 `*_ascend_pt` 目录拉回本地
- 采集失败的可能原因：
  - 服务未就绪 → 检查远端日志
  - `/start_profile` 失败 → 检查 profiler control plane
  - benchmark 成功率不足 → 任务不是真实的推理流量
  - rank 数量不匹配 → `*_ascend_pt` 目录数 ≠ `tp * dp`
  - `kernel_details.csv` 缺失 → 重新采集
- `--profiler-config` 是 serving skill 唯一知道的 profiling 相关内容，profiling skill 的其他逻辑不侵入 serving
- **⚠️ 跨节点（多机）`--headless` 铁律**：`--headless` 由底层 `serve_start.py` 自动加在 Worker 侧，**Head 节点禁止 `--headless`**（Head 必须承担 API Server，headless 会导致 `/health` 就绪失败、DP Engine 握手超时崩溃）。即使多机 profiling 场景中脚本把带 profiler 的 compute 发给 head+workers，Head 也绝不以 headless 启动；此规则由 serving 自动处理，无需手动干预
- **⚠️ 多机禁用 `--api-server-count`**：多机（跨节点）部署时 Worker 以 `--headless` engine-only 模式运行，无法承载多个 API Server，因此 `--api-server-count` 与 `--headless` 冲突。采集脚本在多机场景会**自动忽略** `--api-server-count`（无论是否传入），仅保留单机时透传；多机必须使用单一 API Server。若确实需要排查多 API server 相关现象，请回到单机部署
- **⚠️ 跨节点 DP 就绪等待**：大模型（如 397B MoE）跨节点加载 + 图模式 `FULL_DECODE_ONLY` 编译可能远超 120s，底层 `serve_start.py` 默认 DP RPC 等待 `--dp-rpc-timeout 3600`、health 超时 1800s，避免 worker 未启动就超时
- **⚠️ A5 通信网卡**：A5（Ascend950DT）跨节点数据并行通信必须走专用网卡 `data0.3001`，`HCCL_IF_IP` 直接用本机管理 IP（`--host`/`--worker-hosts`），`GLOO/IP/HCCL_SOCKET_IFNAME=data0.3001`。**禁止**用 `ifconfig data0.3001 | grep inet | awk` 提取 IP。A3（Ascend910）不设此项。multi-node profiling 场景亦由底层 serving 自动注入，无需手动设置
- 跨平台：macOS/Linux/WSL 用 `python3`，Windows 用 **PowerShell** + `py -3`

## 参考信息

- [behavior.md](references/behavior.md) — 控制面时序、模式映射
- [command-recipes.md](references/command-recipes.md) — 不同 profiling 场景的调用模式
- [acceptance.md](references/acceptance.md) — 采集质量验收标准
- 依赖 skill：`vllm-ascend-serving`（服务启动/停止）
- 后续分析：`ascend-msprof-analyze-cli`（msprof-analyze 深度分析）
