---
name: vllm-ascend-serving
description: >-
  This skill should be used when the user asks to "启动服务", "拉服务",
  "在远端起个服务", "重启服务", "看服务状态", "停掉服务", "起一个 vllm 服务",
  "start vllm serve", or wants to manage the lifecycle of a single-node
  or multi-node vLLM Ascend online service on a managed remote container.
version: 1.0.0
domain: inference-model
function: deployment
tags:
  - vllm-ascend
  - serving
  - deployment
author: Liccol
targets-repo:
  - vllm-ascend
---

# vllm-ascend-serving

## 功能概述

管理单节点及多节点 vLLM Ascend 推理服务的全生命周期：启动、状态检查、停止和 NPU 探测。
所有远程操作通过 `scripts/` 下的脚本执行，不直接构造 SSH 命令。

- 目标机器：通过 `--mode` 统一选择连接拓扑（默认 `docker`）：
  - **`docker`**（`--host <宿主IP> --container <name>`）：SSH 到宿主机后 `docker exec` 进容器
  - **`conda`**（`--host <宿主IP> --conda-env <name>`）：SSH 到宿主机后激活指定 conda 环境
  - **`direct`**（`--host <容器IP> --password <pwd>`）：SSH 直连容器/虚拟机 IP，不做 docker exec 包装，纯密码认证
- 运行状态记录在 `.workspace-local/serving/<alias>.json`
- 脚本进度输出至 stderr，最终结果以 JSON 输出至 stdout

## 使用方式

- 直接调用：`/mindspeed-agent:vllm-ascend-serving`
- 描述服务管理需求，由 Claude 自动匹配调用

## 操作步骤

### 步骤一：确认参数

需确认的参数包括：

| 参数 | 说明 |
| ------ | ------ |
| 目标机器 | `--mode docker --host <宿主IP> --container <name>`（Docker exec） 或 `--mode direct --host <容器IP> --password <pwd>`（直连容器SSH） 或 `--mode conda --host <宿主IP> --conda-env <name>`（Conda） |
| 连接模式 | `--mode {docker,conda,direct}`（默认 `docker`）：统一选择连接拓扑 |
| 登录凭据 | `--password <pwd>`（密码，direct 模式推荐）、`--key-file <path>`（SSH 私钥）、`--user <name>`（默认 root） |
| CANN 路径 | `--cann-dir <path>`（默认 `/usr/local/Ascend`），执行前必须确认 |
| 模型路径 | `--model <remote-path>` |
| TP / DP | `--tp <N> --dp <N>` |
| 设备 | `--extra-env ASCEND_RT_VISIBLE_DEVICES=<ids>`（可选，不设置则所有设备可见） |
| 额外参数 | `--extra-env KEY=VALUE`、`--port <N>`、`--health-timeout <s>` 等 |
| vllm serve 参数 | `-- <vllm-args>`，如 `--max-model-len 4096 --compilation-config ...` |

### 步骤二：启动服务

```bash
# 模式 1: docker — SSH 到宿主机, docker exec 进容器（默认）
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host <宿主IP> --container <name> --key-file <path> \
  --model <remote-path> --tp <N> --dp <N> \
  -- --max-model-len <N> --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'

# 模式 2: direct — SSH 直连容器/虚拟机 IP
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --mode direct --host <容器IP> --password <密码> \
  --model <remote-path> --tp <N> --dp <N> \
  -- --max-model-len <N> --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'

# 模式 3: conda — SSH 到宿主机, 激活 conda 环境
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --mode conda --host <ip> --conda-env <env-name> --key-file <path> \
  --model <remote-path> --tp <N> --dp <N> \
  -- --max-model-len <N> --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'
```

> **Windows 用户**：推荐使用 **PowerShell** + `py -3`，因为 Git Bash 的 MSYS 运行时默认会将 Unix 路径（如 `/home/vllm/weights`）转换为 Windows 路径。
> 如果必须使用 Git Bash，请在命令前加上 `MSYS_NO_PATHCONV=1` 以禁用路径转换：
>
> ```bash
> MSYS_NO_PATHCONV=1 python3 serve_start.py --model /home/vllm/weights/...
> ```
>
> **三种模式对比**：
>
> | 模式 | SSH 目标 | 命令执行方式 | 密码认证 |
> | ------ | --------- | ------------- | --------- |
> | `docker` | 宿主机 IP | `docker exec -i <容器>` | 自动生成密钥对 |
> | `conda` | 宿主机 IP | `conda activate <env>` 后原生执行 | 自动生成密钥对 |
> | `direct` | 容器/虚拟机自身 IP | 原生执行，无包装 | 纯密码直连，不生成密钥 |

#### 启动流程

1. 解析目标机器（根据 `--mode` 选择 docker / conda / direct 连接拓扑）
2. 停止已有服务（如存在）
3. 验证模型路径存在，分配空闲端口
4. 通过 `nohup` + `disown` 启动后台进程
5. 轮询 `/health` 和 `/v1/models` 直至就绪（默认超时 1800s）

#### 重新启动

```bash
# 沿用上次配置
python3 .../serve_start.py --host <ip> --container <name> --relaunch

# direct 模式下 relaunch
python3 .../serve_start.py --mode direct --host <容器IP> --password <pwd> --relaunch

# conda 模式下 relaunch
python3 .../serve_start.py --mode conda --host <ip> --conda-env <name> --relaunch

# 覆盖部分参数
python3 .../serve_start.py --host <ip> --container <name> --relaunch --extra-env VLLM_LOGGING_LEVEL=DEBUG

# 移除参数
python3 .../serve_start.py --host <ip> --container <name> --relaunch --unset-args=--enforce-eager
```

### 多机部署（Multi-Node）

通过 `--worker-hosts` 参数启用多机模式。vLLM 使用 **Data Parallel (DP)** 模式实现跨节点推理：通过 `--data-parallel-size` / `--data-parallel-size-local` / `--data-parallel-start-rank` / `--data-parallel-address` / `--data-parallel-rpc-port` 编排多节点，Head 节点运行 API Server + Engine，Worker 节点运行 Engine（`--headless`），无需 Ray 依赖。

> **`--headless` 铁律**：**Head 节点绝对禁止携带 `--headless`**——Head 必须承担 API Server，只有 Worker 节点才以 engine-only（`--headless`）方式运行。启动脚本会在构建 Head 命令时**自动剥离**用户传入的 `--headless`（仅对 Head 生效，Worker 保持 `--headless`）。因此即使 `--serve-args` 里写了 `--headless`，Head 也绝不会 headless 启动。若不遵守此规则，Head 将无 API Server，`/health`、`/v1/models` 与 `bench` 打点全部失效。
>
> **A5 通信网卡**：A5 机型（`Ascend950DT`）跨节点数据并行通信必须走专用网卡 **`data0.3001`**，网卡与本机 IP 直接写死在 `_serve.sh` 中：`HCCL_IF_IP` = **本机 IP**（即 `--host` / `--worker-hosts` 传入的管理 IP，如 head `141.61.54.220`、worker `141.61.54.216`），`GLOO/IP/HCCL_SOCKET_IFNAME` 统一设为 **`data0.3001`**。**禁止**用 `ifconfig data0.3001 | grep inet | awk '{print $2}'` 这类命令去提取 IP——直接使用本机 IP 即可。A3 机型无此限制，走 `local_ip` 的 sibling-NIC 自动检测。

#### 多机参数

| 参数 | 说明 | 必填 |
| ------ | ------ | ------ |
| `--worker-hosts <csv>` | Worker 节点 IP 列表，逗号分隔 | 多机必填 |
| `--master-port <N>` | DP RPC 端口（默认自动分配空闲端口） | 可选 |
| `--dp-rpc-timeout <N>` | Head DP RPC 就绪等待秒数（默认 3600；大模型跨节点加载/编译可能远超 120s） | 可选 |
| `--worker-key-file <path>` | Worker 节点 SSH 私钥（默认同 head） | 可选 |
| `--worker-password <pwd>` | Worker 节点 SSH 密码（默认同 head） | 可选 |
| `--worker-container <name>` | Worker 节点 Docker 容器名（默认同 head） | 可选 |
| `--worker-conda-env <name>` | Worker 节点 Conda 环境名（默认同 head） | 可选 |
| `--worker-extra-env <KV>` | Worker 节点额外环境变量（repeatable） | 可选 |

> **连接模式继承**：Worker 节点自动继承 head 的 `--mode` 设置，多机场景下所有节点统一使用同一种连接拓扑。Worker 的连接凭据（`--worker-password`、`--worker-key-file` 等）可在需要时单独覆盖。

#### 多机拓扑场景

三种连接模式在多机场景下对应不同的物理拓扑，根据实际部署环境选择：

**场景 A：`docker` — 登录宿主机 + docker exec 进容器**

最常见的企业部署场景。每台昇腾服务器上运行一个 Docker 容器，容器内安装 vLLM + CANN。
脚本通过 SSH 登录到每台服务器，再 `docker exec` 进各自的容器执行命令。

```text
控制机 ──SSH──► 宿主机 A (192.168.1.100) ──docker exec──► 容器 A (vLLM head)
    │
    └──SSH──► 宿主机 B (192.168.1.101) ──docker exec──► 容器 B (vLLM worker)
```

特点：

- `--host` 填**宿主机 IP**，`--container` 填容器名（head/worker 可不同）
- SSH 认证走宿主机（`--key-file` 或自动生成密钥对）
- 容器无需暴露独立 IP，无需运行 SSH 服务端

**场景 B：`conda` — 登录宿主机 + 激活 conda 环境**

适用于裸金属部署或需要在宿主机上直接运行的场景。每台服务器上已配置好 conda 环境（含 vLLM + CANN 依赖）。
脚本 SSH 到每台服务器，`conda activate` 后直接执行命令，不涉及 Docker。

```text
控制机 ──SSH──► 宿主机 A (192.168.1.100) ──conda activate vllm_env──► vLLM head
    │
    └──SSH──► 宿主机 B (192.168.1.101) ──conda activate vllm_env──► vLLM worker
```

特点：

- `--host` 填**宿主机 IP**，`--conda-env` 填 conda 环境名
- 无 Docker 层，进程直接在宿主机上运行
- head/worker 可使用不同的 conda 环境（`--worker-conda-env`）

**场景 C：`direct` — SSH 直连容器/虚拟机**

适用于容器自身运行 SSH 服务端、拥有独立 IP 的场景（如 Kubernetes Pod、OpenStack 虚拟机、
或手动配置了 sshd 的 Docker 容器）。脚本直接 SSH 到每个容器/虚拟机的 IP，不做 `docker exec` 包装。

```text
控制机 ──SSH──► 容器/VM A (10.0.0.5, 自带sshd) ──► vLLM head
    │
    └──SSH──► 容器/VM B (10.0.0.6, 自带sshd) ──► vLLM worker
```

特点：

- `--host` 填**容器/虚拟机自身的 IP**
- 必须提供密码（`--password`），不做自动密钥注入
- head/worker 可使用不同密码（`--worker-password`）
- 容器内需安装并运行 SSH 服务端（sshd）
- 不依赖宿主机访问权限，适合云原生/多租户环境

#### 多机执行流程

启动多机集群时，脚本按以下顺序执行：

1. **解析 Worker 目标**：根据 head 的 `--mode` 和连接凭据，为每个 worker IP 构造 `ExecutionTarget`
2. **预检所有节点**：逐一验证 SSH 连通性、模型路径存在
3. **停止已有集群**：若存在残留的旧集群进程，先清理（SIGINT → SIGTERM → SIGKILL）
4. **NPU 探测与设备分配**：在每个节点上 probe NPU。若 probe 能获得 vLLM 可见逻辑设备数（direct 模式按 `/dev/davinci*` 计数，单物理卡常含 2 个 die），走 **die-aware** 分支：`dp_local = min(各节点 vllm设备数 // tp)`，且**不注入** `ASCEND_RT_VISIBLE_DEVICES`（vLLM 自动枚举逻辑设备）。否则回退到旧物理卡模型，按最大连续空闲块统一裁剪
5. **端口分配**：在 head 节点上分配 `service_port`（API）和 `dp_rpc_port`（DP 通信）
6. **计算 DP 配置**：`dp_size = dp_local_min × nnodes`，`dp_start_rank = dp_local_min × node_index`
7. **启动 Head**：构建启动脚本、SSH 远端执行 `nohup`、记录 PID
8. **等待 Head DP RPC 就绪**：轮询 head 的 `dp_rpc_port`（超时 `--dp-rpc-timeout`，默认 3600s；397B 等大模型跨节点加载+图模式编译常远超 120s），确保 worker 连接时端口已监听
9. **启动 Worker**：逐个启动 worker 节点（`--headless` 模式），任一 worker 启动失败则回滚已启动节点
10. **等待集群就绪**：轮询 head 的 `/health` 和 `/v1/models`
11. **持久化集群状态**：写入 `.workspace-local/serving/cluster/<head-alias>.json`

#### 两节点部署

```bash
# ===== docker 模式 =====
# Head (192.168.1.100) + Worker (192.168.1.101), 各 8 NPU
python3 serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B \
  --tp 8

# ===== direct 模式 =====
# 容器自带 IP 和 SSH，直连
python3 serve_start.py \
  --mode direct --host 10.0.0.5 --password hdpass \
  --worker-hosts 10.0.0.6 --worker-password wkpass \
  --model /data/models/Qwen2-72B \
  --tp 8

# ===== conda 模式 =====
python3 serve_start.py \
  --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-hosts 192.168.1.101 --worker-conda-env vllm_env \
  --model /data/models/Qwen2-72B \
  --tp 8
```

自动行为（以 docker 模式为例）：

- Head: `--data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-start-rank 0 --data-parallel-address 192.168.1.100 --data-parallel-rpc-port <auto>`
- Worker: `--data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-start-rank 1 --data-parallel-address 192.168.1.100 --data-parallel-rpc-port <auto> --headless`
- 各节点自动 probe NPU 并按最大连续空闲块分配设备

#### 三节点 + Pipeline Parallelism

```bash
# docker 模式
python3 serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101,192.168.1.102 \
  --model /data/models/DeepSeek-V3-685B \
  --tp 8 \
  -- --pipeline-parallel-size 3

# direct 模式 — 三容器直连
python3 serve_start.py \
  --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-hosts 10.0.0.6,10.0.0.7 --worker-password wkpwd \
  --model /data/models/DeepSeek-V3-685B \
  --tp 8 \
  -- --pipeline-parallel-size 3
```

#### 停止多机集群

```bash
# 在任意节点上执行（head 或 worker 均可），模式须与启动时一致
# docker 模式
python3 serve_stop.py --host 192.168.1.100 --container my-container

# direct 模式（worker 密码不同时用 --worker-password 覆盖）
python3 serve_stop.py --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-password wkpwd

# conda 模式
python3 serve_stop.py --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-conda-env vllm_env
```

停止流程：读取持久化的集群状态 → 遍历所有节点 → SIGINT（5s 等待）→ SIGTERM（5s 等待）→ `--force` 时 SIGKILL → 清理 NPU 孤儿进程。

> **Worker 凭据**：`serve_stop.py` 支持 `--worker-key-file/--worker-password/--worker-container/--worker-conda-env`，当 worker 使用与 head 不同的凭据时必须显式传入，否则默认继承 head 的凭据。

#### 查看多机集群状态

```bash
# docker 模式
python3 serve_status.py --host 192.168.1.100 --container my-container

# direct 模式（worker 密码不同时用 --worker-password 覆盖）
python3 serve_status.py --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-password wkpwd

# conda 模式
python3 serve_status.py --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-conda-env vllm_env
```

输出包含 `cluster_type: "multi-node"`、`head`（含 alive/health/models_ok）、`workers`（各节点 alive 状态）、`nnodes`、`dp_rpc_port`、`all_nodes_healthy` 等信息。每个节点的存活状态在**该节点上**检查（跨 PID namespace 安全）。

#### 多机 NPU 探测

```bash
# docker 模式
python3 serve_probe_npus.py --host 192.168.1.100 --container my-container \
    --worker-hosts 192.168.1.101,192.168.1.102

# direct 模式
python3 serve_probe_npus.py --mode direct --host 10.0.0.5 --password hdpwd \
    --worker-hosts 10.0.0.6,10.0.0.7

# conda 模式
python3 serve_probe_npus.py --mode conda --host 192.168.1.100 --conda-env vllm_env \
    --worker-hosts 192.168.1.101,192.168.1.102
```

返回所有节点 NPU 设备汇总，含 `nodes`（每个节点的 `total`/`free`/`busy`）和 `summary`（`total_devices`/`total_free`/`nodes_probed`）。

### 步骤三：查看状态

```bash
# docker 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_status.py --host <ip> --container <name>

# direct 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_status.py --mode direct --host <容器IP> --password <pwd>

# conda 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_status.py --mode conda --host <ip> --conda-env <name>
```

### 步骤四：停止服务

```bash
# docker 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --host <ip> --container <name> [--force]

# direct 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --mode direct --host <容器IP> --password <pwd> [--force]

# conda 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --mode conda --host <ip> --conda-env <name> [--force]
```

### 步骤五：探测 NPU

```bash
# docker 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --host <ip> --container <name>

# direct 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --mode direct --host <容器IP> --password <pwd>

# conda 模式
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --mode conda --host <ip> --conda-env <name>
```

返回空闲/占用设备列表（含 PID 和 HBM 详情）。

## 注意事项

- **不要手动构造 SSH 命令**，所有服务管理通过脚本执行
- **⚠️ `--mode` 连接模式**：通过 `--mode {docker,conda,direct}` 选择连接拓扑（默认 `docker`）。多机模式下所有节点统一使用同一种模式，worker 自动继承 head 的 `--mode`。`--mode direct` 时提供的 `--password` 直接用于 SSH 密码认证，不生成密钥对
- **⚠️ CANN 环境必须确认**：执行前需向用户确认远端 CANN 安装根目录（`--cann-dir`，默认 `/usr/local/Ascend`）。脚本优先 source `<cann-dir>/ascend-toolkit/set_env.sh`（新版 CANN 布局），其次 `<cann-dir>/cann/set_env.sh`，再依次 `<cann-dir>/nnal/atb/set_env.sh`；若全部不存在则回退为手动导出 `LD_LIBRARY_PATH`。若容器需专用 Python 目录，脚本会检测 `/usr/local/python3.12.13/bin` 并自动加入 `PATH`
- 进程启动后立即写入 `status=starting` 状态文件，确保即使健康检查失败也能被 `serve_stop.py` 清理
- 启动失败时，应先读取远端 `stdout.log` 和 `stderr.log`，确认根因后再调整参数
- NPU 设备：物理 NPU 常含 2 个逻辑设备（die，如 davinci14/15）。vLLM 的 `--tensor-parallel-size` 按**逻辑 die** 计数，不是物理卡。die-aware 模式（direct 且 probe 到 `/dev/davinci*`）下脚本**不注入** `ASCEND_RT_VISIBLE_DEVICES`，让 vLLM 自动枚举；如需手动指定用 `--extra-env ASCEND_RT_VISIBLE_DEVICES=<ids>`（die 逻辑 id，如 `0,1`）
- **⚠️ `--port` 必须作为脚本级参数**：`--port 8006` 放在脚本主参数中（`--` 之前），**不可**放在 `--` 之后作为 vllm 透传参数。脚本使用该端口进行健康检查和状态探测；放入 `--` 后会导致 vllm 绑定到用户指定端口，但脚本自动分配的端口不同，健康检查探测错误端口而超时
- **多机 TP 语义**：`--tp` 表示**每节点** vLLM 逻辑 TP 设备数（健康检查 fingerprint 形如 `vllm-0.23.0-tp2-dp2-ep-...` 表示 TP=2/DP=2/EP 已启用）。每节点 1 张物理卡（2 die）即可 `--tp 2`。多机下脚本自动计算 `dp_local = min(各节点 vllm设备数 // tp)`，`dp_size = dp_local × nnodes`（本机 die=2、tp=2 → dp_local=1、dp_size=2），并在 die-aware 分支下不注入设备环境变量。A3 机型每节点 8 物理 NPU / 16 die
- 跨平台：macOS/Linux 用 `python3`；Windows 推荐 **PowerShell** + `py -3`，或用 **Git Bash** + `MSYS_NO_PATHCONV=1 python3`

## 已知问题与修复

本小节记录实践（含多机 TP2/DP2）中修复的脚本问题与方案，供后续迭代参考：

- **⚠️ 启动成功后必须读取并发预算日志**：vLLM 启动后按**每个 DP rank** 打印一行
  `Maximum concurrency for <max-model-len> tokens per request: <N>x`（如
  `Maximum concurrency for 36,864 tokens per request: 33.38x`），`Nx` 是服务端估算的
  "打满 KV cache 时可支撑的最大并发数"，**总并发预算 ≈ DP 数 × 单 rank 预测**。
  服务就绪后应读取并记录（`grep "Maximum concurrency" <runtime>/stdout.log`），
  供 benchmark 选高并发档位、排查并发上界使用。详细方法见 `vllm-ascend-benchmark`
  的 `references/max-concurrency-tuning.md`。

- **Windows 密码认证 askpass 失败**（`_common.py: _ssh_askpass_script`）：工作区路径含非 ASCII 字符（如 `昇腾`）时，OpenSSH 的 askpass 经 `CreateProcessW` 无法执行裸 `.py` 且会乱码非 ASCII 路径。修复：Windows 下改写 `mindspeed-askpass.cmd`（`@echo off` + `echo %_WORKSPACE_SSH_PASSWORD%`）到 ASCII 系统临时目录；新增 `_atomic_write_text`（mkstemp + os.replace）保证并发安全。
- **`npu-smi` 单物理卡多 die 误判**（`_common.py: _parse_npu_smi`）：报头正则原为 `\|\s*(\d+)\s+\d*\w+\d+\w*\s+\|`，会把单卡 chip 行（如 `| 0 14 | 0000:...|`）误判为多设备。修复：第 2 字段必须为含字母的模型名（如 `Ascend910`），`^[A-Za-z][\w.-]*$`，正确识别 head=devices[7]、worker=[0]。
- **probe 时 `npu-smi` 报 `libc_sec.so` 错误**：`npu-smi` 依赖 CANN 运行时库。修复：`probe_npus` 前置 source `ascend-toolkit/set_env.sh` 与 `nnal/atb/set_env.sh`；direct 模式下额外用 `ls /dev/davinci*` 统计 vLLM 可见逻辑设备数（返回 dict 新增 `vllm_devices`/`vllm_device_count`）。
- **多机设备选择错用物理卡 id**（`serve_start.py: _run_multi_node`）：旧逻辑把 `npu-smi` 物理卡 id（如 `7`）注入 `ASCEND_RT_VISIBLE_DEVICES`，但 vLLM 期待 die 逻辑 id，且单卡含 2 die 时 `--tp 2` 会因“设备数不足”失败。修复：新增 **die-aware** 分支——每节点 `vllm_device_count >= tp` 时，`dp_local = min(vllm//tp)`、`dp_size = dp_local×nnodes`、`auto_visible_devices=True`（不注入设备环境变量，vLLM 自动枚举）。已验证 fingerprint `vllm-0.23.0-tp2-dp2-ep`。
- **CANN 布局/专用 Python 目录**：`serve_start.py: build_launch_script` 现优先 source `ascend-toolkit/set_env.sh`（新版 CANN），并检测 `/usr/local/python3.12.13/bin` 自动加入 `PATH`（管理容器常见场景）。
- **⚠️ MoE 并发上边界误判：`HCCL_BUFFSIZE` 不足导致 `MoeDistributeDispatch` 崩溃/进程被杀（不要归因于"并发已达上限"或"显存不够"）**：vLLM-Ascend 在 MoE dispatch/combine 执行前会检查 HCCL 通信缓冲容量，报错形如
  `HCCL_BUFFSIZE_EP is too SMALL, maxBs = 192, h = 2048, epWorldSize = 8, localMoeExpertNum = 32, sharedExpertNum = 0, tokenNeedSizeDispatch = 4608, tokenNeedSizeCombine = 4096, k = 8, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * epWorldSize * localMoeExpertNum) + (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) + epWorldSize * 1024 = 445MB, HCCL_BUFFSIZE=200MB, actual CCL_BUFFSIZE=400MB`，
  随后 `Check UB/window size or set workspace/hcomm cfg failed [FUNC:MoeDistributeDispatchA3TilingFuncImplPublic][FILE:moe_distribute_dispatch_v2_tiling.cpp]` → `aclnnMoeDistributeDispatchV4 failed, error code is 561002`。需求 `NEEDED_HCCL_BUFFSIZE` 随 `maxBs`（≈ `--max-num-seqs`）线性增长，超过默认 `HCCL_BUFFSIZE=200MB`（vLLM-Ascend 内部 ×2 = `actual CCL_BUFFSIZE=400MB` 用于 EP）即失败；部分情形引擎静默退出，bench 进程表现为 `SIGKILL (rc=137)` 且服务端无报错。
  **修复**：以 `--extra-env HCCL_BUFFSIZE=<MB>` 注入，取值 > 日志中 `NEEDED_HCCL_BUFFSIZE` 的一半即可（实测 Qwen3.5-35B-A3B：EP8/h2048/256e-per-card32/top8，maxBs=192 时 NEEDED=445MB → 设 600 稳过；该 env 每 rank 增加对应显存占用，接近显存上限时可先按 NEEDED/2 取整）。
  **排查顺序**：先 grep 服务端日志 `HCCL_BUFFSIZE` / `NEEDED_HCCL` / `Check UB/window` / `MoeDistributeDispatch`，命中即为该检查问题；`error code 561000` 是另一种 tiling/UB 失败，与 buffer 检查不同。
- **⚠️ HCCL_BUFFSIZE 调低实验：找最小够用值，释放显存提升 KV 预算**：HCCL_BUFFSIZE 设得越大越浪费显存
  （每 rank 占用对应设备显存），可能压缩 KV cache、降低 `Maximum concurrency` 并发预算。寻优时应单独做一次
  **调低实验**——去掉 `HCCL_BUFFSIZE` env（恢复 vLLM 默认 200MB）跑一档对应并发；若该并发超过默认缓冲能力，
  报错日志会打印 `NEEDED_HCCL_BUFFSIZE=xxxMB`（格式见上条），据此配一个**更大的临近值**（实测 NEEDED=445MB
  → 设 600 稳过；切忌一步设到 2400 这类大值）。每次调整后读回服务端 `Maximum concurrency` 验证 KV 预算是否
  改善；显存紧张时优先微调该值，再考虑降 gmu。

## 参考信息

- [behavior.md](references/behavior.md) — 重启、端口分配、环境变量合并和 NPU 探测的详细行为
- [command-recipes.md](references/command-recipes.md) — 常见调用模式
- [acceptance.md](references/acceptance.md) — 服务启动验收标准
- 依赖 skill：无（这是基础服务层）
- 被以下 skill 使用：`vllm-ascend-benchmark`、`ascend-profiling-collection`、`vllm-ascend-aisbench-accuracy`
