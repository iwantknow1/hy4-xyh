# 多机场景（Multi-Node）专篇

> 本文件是 `model-rl-case-guard` 的多机场景权威指引，与 [SKILL.md](../SKILL.md) 的阶段 1-B、阶段 1-D、基础配置和阶段 2 配合使用。

核心原则：**开箱 skill 不手工拼跨机命令**，而是复用 `vllm-ascend-serving` 已验证的探测、拉起、状态和停止脚本；本文件只负责把逐节点资源和模型约束转换为可审计的 TP/DP/PP 拓扑。

## 1. 统一脚本入口

统一调用同级 skill 的脚本：

| 用途 | 脚本 | 结果/约束 |
| --- | --- | --- |
| 逐节点 NPU 探测 | `serve_probe_npus.py` | 返回 `nodes` 的 `total/free/busy` 与 `summary` 的 `total_devices/total_free/nodes_probed` |
| 拉起集群 | `serve_start.py --worker-hosts ...` | head 运行 API + Engine，worker 仅 Engine/headless；启动前完成全节点 preflight |
| 查状态 | `serve_status.py` | 返回 `cluster_type`、head/workers 存活、`nnodes`、`dp_rpc_port`、`all_nodes_healthy` |
| 停止集群 | `serve_stop.py` | 遍历节点 SIGINT → SIGTERM → force 时 SIGKILL，并清理残留服务进程 |

`model-rl-case-guard` 的任何专项脚本若提供 `--worker-hosts`，应把多机启动委托给上述脚本；不重复实现 SSH、DP RPC 或回滚逻辑。

## 2. 节点连接模式

所有节点必须统一使用一种连接模式：

- `docker`（默认）：SSH 到宿主机，再 `docker exec` 进入容器；`--host` 是宿主机 IP，`--container` 是容器名。
- `conda`：SSH 到宿主机，激活指定 Conda 环境；`--conda-env` 是环境名。
- `direct`：SSH 直连自带 sshd 的容器/VM IP；不做 Docker/Conda 包装，不自动注入密钥。

worker 默认继承 head 的用户、模式、凭据、容器和 Conda 环境；不同节点必须显式传 `--worker-password`、`--worker-key-file`、`--worker-container` 或 `--worker-conda-env`。`direct` 模式禁止同时使用 `--container`/`--conda-env`。

## 3. 逐节点探测和自动推导

### 3.1 先探测，再规划

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --host <head-ip> --worker-hosts <worker1>,<worker2> \
  --mode <docker|conda|direct> [连接参数]
```

保存每节点 `total/free/busy`、NPU 型号、物理卡/逻辑 die 数和 HBM。多机总可用卡数是各节点可用卡之和；容量判定还必须按节点实际 HBM 汇总，不能套用单一型号容量。

### 3.2 推导规则

设节点数为 `K`，节点可用逻辑设备数为 `n_1 ... n_K`：

1. **TP（每节点）**：`TP ≤ min(n_i)`，并取模型和后端允许的合法约数。TP 的 all-reduce/all-gather 组默认在同一节点，禁止用跨机 TP 代替 DP/PP。
2. **DP（跨节点优先）**：`dp_local = min(n_i // TP)`，`dp_total = dp_local × K`。脚本负责注入 `--data-parallel-size`、`--data-parallel-size-local`、`--data-parallel-start-rank`、`--data-parallel-address` 和 `--data-parallel-rpc-port`。
3. **PP（默认专项验证）**：在已规划资源内生成合法 PP>1 配置并独立做容量预检，默认执行流水线并行验证；普通 baseline 是否依赖 PP 不影响专项执行。不得通过增大 TP 跨节点实现 PP。注意：本文件统一入口的 `serve_start.py --worker-hosts` 自动推导并注入的是跨节点 DP；它不是原生 PP launcher。跨节点 PP 使用目标版本原生入口，不满足条件时保留该项并记录 `blocked` 和原因。
4. **MoE/EP**：MoE 默认候选 `--enable-expert-parallel`，并核对 EP、TP、DP 的合法组合。
5. **端口**：service API port 和 DP RPC port 交给 `serve_start.py` 自动分配或显式记录；`--port` 是脚本级参数，不能塞入 `--` 后的 vLLM 透传参数。

示例：两个节点各有相同数量的逻辑设备，`TP` 取单节点设备数，`dp_local=1`、`dp_total=2`。head 为 rank 0 并提供 API，worker 为 rank 1 并使用 `--headless`。

### 3.3 与容量预检的关系

多机容量预检同时检查总量和单卡：

- 总可用 HBM 为各节点逐卡可用 HBM 之和；
- 单卡权重近似 `total_weight / TP`，且 TP 受最小节点卡数限制；
- 若 `total_weight / TP_max + KV/激活/编译开销` 仍超过任一节点单卡可用容量，输出《容量报告》，指出需要更多节点或更大 HBM；
- 详细统计、利用率和终止格式见 [capacity-check.md](capacity-check.md)。

## 4. 多机生命周期

严格顺序为：

1. 解析 head/worker 目标，校验 worker 不重复 head；
2. 对所有节点做 SSH、模型路径、容器/Conda、CANN、NPU 和端口 preflight；
3. 任一节点失败则停止后续动作，修复决策交给客户，不提前启动其他节点；
4. 停止旧集群并保留现场；
5. 探测每节点设备，计算 TP/DP/PP 与 rank；
6. 分配 API 与 DP RPC 端口，启动 head；
7. 等待 head DP RPC 就绪，再启动 worker；
8. 等待 head `/health`、`/v1/models` 以及所有 worker 存活；
9. 持久化 head/worker、拓扑、端口、PID 和日志路径；
10. 特性失败或 worker 失败时按失败诊断保留首个原始错误，并由 `serve_stop.py` 回滚已启动节点。

Head **绝对不能**带 `--headless`；只有 worker 使用 headless engine-only。不要手工在 head 上拼 worker 的 SSH 命令。

## 5. 多机特性支持边界

| 特性/脚本 | 多机策略 | 结论边界 |
| --- | --- | --- |
| routing replay | 通过 `serve_start.py --worker-hosts` 拉起；只跑 HTTP 验证 | 进程内 LLM 不跨节点 |
| sleep + weight update online | 以目标版本脚本的 `--node-size/--node-rank` 或服务脚本支持为准 | 需要 checkpoint-native HCCL/IPC 数据面；没有则只记状态机 |
| pause/resume 降级 | 可在多机 HTTP 端点执行；内存指标按节点记录 | 不证明权重已替换 |
| CP / SP / PP | CP/SP 可按目标版本使用对应在线脚本；PP 的跨节点场景必须使用目标版本支持的 MP/Ray/专用 PP launcher，或连接到已由该 launcher 启动的服务 | `serve_start.py --worker-hosts` 的多机结果是 DP 编排，不能作为 PP 通过证据；不满足原生 PP 启动条件时记 `blocked`/`not_run`，不把普通 DP 结果当作 PP 结果 |

专项脚本缺失或版本不匹配时记录 `not_run`/`blocked`，不要用普通 curl 结果替代。

## 6. 通信和网络铁律

- 若目标平台要求专用通信网卡，按目标服务脚本设置 `HCCL`/`GLOO`/`IP` 相关接口变量；`HCCL_IF_IP` 使用对应节点明确提供且可达的本机 IP。
- 不要用未经验证的 `ifconfig`/路由输出猜测通信 IP；管理面 SSH 可达不代表 DP RPC/HCCL 数据面可达。
- 不同芯片、驱动和网络拓扑的接口名称与绑定规则可能不同，必须以目标平台文档、服务脚本和实机探测结果为准。
- 管理 SSH 可达不代表 DP RPC 可达；检查防火墙、路由、端口监听和节点间网络策略。
- 测试客户端只连接 head API port；控制机不可达时先建立 SSH 隧道或使用可达代理。

## 7. 多机调用示例

```bash
# docker：head + worker，各 8 个逻辑设备，TP8、DP2（跨节点 DP；不是 PP）
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101 --model /data/models/<model> --tp 8

# 真正的跨节点 PP 不使用上面的 DP wrapper；按目标版本的 vLLM-Ascend
# pipeline_parallel.md 使用 MP/Ray 原生 launcher（例如 --nnodes/--node-rank
# 或 --distributed-executor-backend ray），再连接 head API 做验证。

# direct：head/worker 凭据不同
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --mode direct --host 10.0.0.5 --password <head-password> \
  --worker-hosts 10.0.0.6 --worker-password <worker-password> \
  --model /data/models/<model> --tp 8

# 状态与停止
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_status.py \
  --host 192.168.1.100 --container my-container --worker-hosts 192.168.1.101
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --host 192.168.1.100 --container my-container
```

`serve_stop.py` 从 head 的持久化集群状态读取 worker 列表，因此停止时不再传 `--worker-hosts`；若状态文件缺失，可先以 head 连接信息执行 `--model <model>` 的定向残留清理，并保留命令输出作为证据。

命令中的密码仅作占位符；真实凭据应通过受控 askpass/私钥传递，不写入日志。

## 8. 失败处理和报告

- 任一 worker preflight、DP RPC、CANN、NPU 或模型路径失败：保留 head/worker 原始日志，先报告首错，再停止已启动节点。
- 显存不足：转 [capacity-check.md](capacity-check.md)，输出《容量报告》并终止全流程。
- 权重更新前后结果不一致：按 [weight-update-consistency.md](weight-update-consistency.md) 检查所有相关 worker/rank 的权重覆盖、张量地址/状态与图模式引用，允许有依据的最小修复并在原多机拓扑下复验。
- 其他 traceback/断言：转 [failure-diagnosis.md](failure-diagnosis.md)，按 L0～L5 逐帧定位，默认只出《问题定位报告》，不自动修复或重复跑同一失败命令。
- 最终记录每节点角色、模式、卡数、HBM、TP/DP/PP、rank、API/RPC 端口、状态和日志路径。
