# 特性矩阵：开启方式与验证方法

> 来源：`vllm-project/vllm`、`vllm-project/vllm-ascend` 的文档、代码和测试。参数和行为随版本变化，执行前必须以实际部署 checkout、commit 和测试为准。

## 使用合同

本文件定义模型开箱的逐特性检查合同。服务由 `vllm-ascend-serving` 脚本统一拉起、查询和停止；每轮只改变当前特性开关，保留模型、权重、拓扑、端口、输入和容量预检结果不变。

- 先完成阶段 1 环境、逐节点 NPU 和容量预检，再开始任何特性拉起。
- 目标是记录特性是否可开启、服务是否工作以及输出/内存/指标证据；精度是否符合业务要求由客户判定。
- 专项脚本不存在、参数不匹配或目标版本不支持时，记录 `not_run` 或 `blocked` 及原因，不用无关 curl 结果替代专项结论。
- 文中 `scripts/{routing,lifecycle,weight-update,parallel}/*.py` 是推荐的专项脚本接口；执行前必须检查当前 skill/目标 checkout 中的文件、参数和版本是否存在。若当前工作区没有该脚本，记录 `not_run`，使用目标版本 e2e 或手动步骤时必须在报告中说明替代关系。
- 特性验证失败时先按 [failure-diagnosis.md](failure-diagnosis.md) 定位根因。普通失败默认只定位；HCCL、IPC、Checkpoint 更新前后结果不一致时，按 [weight-update-consistency.md](weight-update-consistency.md) 有序排查并允许最小修复和完整复验。所有场景均不绕过检查或无证据地重复运行同一失败命令。
- 命中明确初始化显存不足时，转按 [capacity-check.md](capacity-check.md) 输出《容量报告》并终止整个开箱流程。

## 通用 RL 主开关

多数 RL 协同特性使用：

```bash
vllm serve <model> \
  --additional-config '{"rl_config":{"enabled":true}}'
```

`rl_config` 的关键字段：

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| `enabled` | `false` | 主开关；启用后通常强制 `weight_nz_mode=0`、`VLLM_ASCEND_ENABLE_NZ=0`、`VLLM_SERVER_DEV_MODE=1`，并移除 `expandable_segments` |
| `sleep_mode_extra_cleanup` | `false` | 同卡 sleep 额外释放 HCCL 进程组和 ACL graph workspace；唤醒更慢 |
| `enable_training_consistency` | `false` | 启用 FA3 以保证训推一致性；需 `flash_attn_npu_v3` |
| `enable_batch_invariant` | `false` | 开启确定性计算；通常还需 `VLLM_BATCH_INVARIANT=1`、`HCCL_DETERMINISTIC=strict`、`LCCL_DETERMINISTIC=1` |

`enabled=false` 时其余字段全部忽略。sleep/wake/pause/resume/collective_rpc 属于开发端点，仅在可信开发环境使用。

## 特性 1：routing replay（Router Replay / R3）

### 开启

```bash
vllm serve <model> --enable-return-routed-experts
```

只对 MoE 模型有意义。若模型不是 MoE，记录 `not_applicable`，不要伪造 routed experts 结果。

### 推荐验证：HTTP + 进程内双路径

优先使用目标 checkout 对应的 e2e 用例；若目标 checkout 没有可直接运行的 CLI，使用本 skill 的 [scripts/routing/test_moe_routing_replay.py](../scripts/routing/test_moe_routing_replay.py)，服务命令必须从已验证的 baseline 配置派生：

```bash
python scripts/routing/test_moe_routing_replay.py \
  --baseline-cmd "vllm serve /path/to/<moe-model> \
    --tensor-parallel-size <T> --data-parallel-size <D> \
    --enable-expert-parallel \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
    --port <port>" \
  --verify-mode both
```

HTTP 路径轮询 `/health` 后发 `POST /v1/completions`；进程内路径用 `enable_return_routed_experts=True` 读取输出。验证记录：

1. `choices[].routed_experts` 非空且为 base64 编码数组；
2. 解码数组维度为 3，形状为 `(num_tokens, num_layers, num_experts_per_tok)`；
3. `num_tokens > 0`，层数/top-k 与模型配置一致，专家 id 在 `[0, num_experts)`；
4. 生成文本无乱码、无 traceback；温度 0 单请求作为可重复基线；
5. 捕获关闭或 forward 前中止时字段为 `null`，该情况要结合日志判断，不能简单判为通过。

记录服务端启动命令、原始响应和路由数组摘要（shape、dtype、最小/最大值、唯一值数）。

## 特性 2 + 3：sleep mode 与权重更新（合一验证）

开箱将 sleep 和权重更新放在同一个调度项中，但必须分别给出结论。HCCL/IPC 统一使用 checkpoint-native 数据面，不构造 Transformers 模型，不使用 FSDP。

- sleep(level=2)：使用目标版本原生 e2e，或按下文手动 HTTP 顺序验证。`scripts/lifecycle/test_sleep_mode.py` 只提供 level-1 辅助回归，不能替代 level-2 必测项。
- HCCL 权重更新：使用 [scripts/weight-update/test_hccl_weight_update.py](../scripts/weight-update/test_hccl_weight_update.py)，用 `python` 在独立物理 NPU 启动单 checkpoint producer；服务端降低 `--gpu-memory-utilization`，为最大单张量、packed buffer 和 HCCL 开销预留显存。
- NPU IPC 权重更新：使用 [scripts/weight-update/test_npu_ipc_weight_update.py](../scripts/weight-update/test_npu_ipc_weight_update.py)，通过 `torchrun` 为每个推理 worker 启动同卡 checkpoint producer；服务端为最大单张量、packed/IPC buffer 预留显存；仅在可信环境允许不安全序列化。
- Checkpoint 权重重载：使用 [scripts/weight-update/test_reload_weights.py](../scripts/weight-update/test_reload_weights.py)，无需独立 producer NPU 或 `--weight-transfer-config`；服务必须先以真实 checkpoint 启动，再通过 dev-mode `/collective_rpc` 调用 `reload_weights(is_checkpoint_format=true)` 在推理 worker 内就地重载同一 checkpoint。
- 仅需验证暂停/恢复控制面时，可使用 [scripts/lifecycle/test_pause_resume.py](../scripts/lifecycle/test_pause_resume.py)，但该结果不能证明权重已传输或改变。

### level-1 辅助回归脚本

`test_sleep_mode.py` 在当前服务上重新采集双接口基线；`--baseline-result` 只复用提示词，不比较其中旧的推理结果。每轮执行 `sleep(level=1) → 确认 sleeping → wake(weights) → 确认仍 sleeping → wake(kv_cache) → 等待 awake → /health → 双接口生成`，逐接口、逐 prompt 严格比较 text、tokens 和 token_logprobs，拒绝缺失、非有限数值及数量不符。该过程不丢弃权重，不需要插入权重更新。

显存检查从现有 `vllm-ascend-serving/scripts/serve_probe_npus.py` 获取远端 `npu-smi` 的实时 HBM 数据，不读取客户端 `torch.npu.memory_allocated()`。先核对推理 worker 的物理板卡映射，再通过可重复的 `--memory-device HOST/physical-NPU-ID` 列出全部推理设备；不要填容器内重映射的逻辑编号，也不要选 producer 或无关空闲卡。探测可以包含其他设备，但只比较显式选中的推理设备。

在本 skill 目录执行，按目标环境替换路径、连接参数和设备：

```bash
python scripts/lifecycle/test_sleep_mode.py \
  --server-url http://192.168.1.100:8000 --model policy \
  --baseline-result /evidence/baseline.json --report /evidence/sleep.json \
  --cycles 3 \
  --memory-probe-command 'python ../vllm-ascend-serving/scripts/serve_probe_npus.py --host 192.168.1.100 --mode direct --key-file /keys/worker-key' \
  --memory-device 192.168.1.100/0 --memory-device 192.168.1.100/1
```

多机时给 probe 命令加实际 `--worker-hosts` 与连接参数，并列出所有节点的推理设备。探测命令以 argv 执行，不经过 shell，不接受管道/重定向；stdout 须为 probe 的完整 `status=ok` JSON 对象，支持其默认缩进格式，进度信息走 stderr。采集器会在基线生成后、每次 sleep 后及唤醒后生成完成时重新运行；逐设备检查 sleep 后 HBM 下降、唤醒后 HBM 相对初始基线的最大增长不超过 5%，不以所有设备求和掩盖单卡增长。设备身份、容量变化或采集失败均不能判通过。HBM 为设备总用量，需避免同卡其他任务变化干扰；它不是逐进程泄漏归因。

不提供内存探测参数时，双接口生命周期通过的报告为 `passed_http_only`，退出码为 `0`；内存结论为 `not_run`、相关 checks 为 `null`，不能声称显存释放/无泄漏。配置探测且所有检查通过时才为 `passed`。任何状态、推理或已配置内存检查失败均返回 `failed`、退出码 `2`，写入错误、失败阶段、已完成轮次及部分输出。sleep 已尝试但失败时会再次查询状态，记录仍休眠或状态未知，不自动唤醒掩盖现场。README 使用正常的 `baseline_outputs_by_endpoint`/`feature_outputs_by_endpoint` 展示对照，并保留各轮详细证据。

多机时先通过 `vllm-ascend-serving` 启动完整集群。IPC 在每个节点启动与本地 worker 同卡的 producer，且各节点 checkpoint manifest checksum 必须一致。没有共享 checkpoint 或真实数据面时，将权重更新记录为 `not_run`/`blocked`。

```text
generate → sleep(level=2) → wake(weights) → update → wake(kv_cache) → generate → precision
```

### 开启与环境

先由 `vllm-ascend-serving` 的服务入口按实际单机/多机配置启动服务，并确保 head 与全部 worker 就绪。以下为服务端需包含的配置示意，不能作为验证脚本的参数：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve <model> --enable-sleep-mode \
  --additional-config '{"rl_config":{"enabled":true}}'
```

确认服务端已设置 `VLLM_SERVER_DEV_MODE=1`、`VLLM_ASCEND_ENABLE_NZ=0`，并按更新方法配置 HCCL/IPC `--weight-transfer-config`；HCCL/IPC 服务必须使用低于普通 baseline 的 `--gpu-memory-utilization`，先以 `0.3` 作为起点并结合容量预检调整，不能为了塞入推理权重而占满显存；同卡 IPC 后端还需要在可信环境中显式允许不安全序列化。验证脚本不设置服务端环境变量。服务必须以真实 checkpoint 启动并使用 `--load-format auto`；不再用 dummy 权重作为正式权重更新基线。三个 sleep level：

| level | 语义 | 开箱策略 |
| --- | --- | --- |
| 0 | 只暂停调度，不释放内存 | 仅排障对照 |
| 1 | 权重 offload 到 CPU，丢弃 KV cache | 仅排障对照 |
| 2 | 权重和 KV cache 全部丢弃 | **开箱唯一必测级别** |

### sleep 验证步骤

1. 记录 sleep 前后 NPU free bytes；level 2 应释放权重和 KV，并将实际变化量写入报告。
2. 调 `POST /sleep?level=2`，确认 `/is_sleeping=true`、metrics 为 `discard_all`。
3. 先 `POST /wake_up?tags=weights`，确认仍在 sleeping；再 `POST /wake_up?tags=kv_cache`，确认 `/is_sleeping=false` 且 `/health=200`。
4. 温度 0 的同一 prompt 在 sleep 前后输出一致；重复多次 sleep→wake，记录每次内存和输出，检查无明显增长或泄漏。
5. spawn worker 可能独立于主进程，停止后用 `npu-smi` 核对残留 `VLLMWorker`/`VLLMWorker_DP`，保留 PID 和日志证据；不要仅按 `vllm serve` 进程名判断清理完成。

手动 HTTP 步骤（没有整合脚本时使用）：

```bash
curl http://<host>:<port>/is_sleeping
curl -X POST "http://<host>:<port>/sleep?level=2"
curl http://<host>:<port>/is_sleeping
curl -X POST "http://<host>:<port>/wake_up?tags=weights"
curl http://<host>:<port>/is_sleeping       # 仍应为 true
curl -X POST "http://<host>:<port>/wake_up?tags=kv_cache"
curl http://<host>:<port>/is_sleeping       # false
curl http://<host>:<port>/health             # 200
```

无共享 checkpoint 时，只能验证 HTTP sleep/wake 和分阶段唤醒状态机，并明确不能证明权重已被替换。遇到显存、主机内存、HCCL 或端口错误时，先保留现场，再按失败诊断规则停止。

### 权重更新后端与边界

HCCL、IPC 和 Checkpoint 重载必须作为三个独立权重更新执行项登记。完成其中一个后，其他项仍保持 `not_run`，直到使用对应服务配置和专项脚本完成真实数据面验证。

跨设备优先 `--weight-transfer-config '{"backend":"hccl"}'`；同设备 IPC 使用 `{"backend":"npu_ipc"}`，并仅在可信环境启用不安全序列化。两者的生产端都是 checkpoint producer：

- `checkpoint`/`collective_rpc reload_weights` 的可用性取决于目标 vLLM/vLLM-Ascend 版本和 Worker 实现；HTTP 500、缺少 `reload_weights` 或其他 Worker 异常均不能当作更新成功；
- HCCL 使用 `--weight-transfer-config '{"backend":"hccl"}'` 与 `test_hccl_weight_update.py`；IPC 使用 `--weight-transfer-config '{"backend":"npu_ipc"}'` 与 `test_npu_ipc_weight_update.py`。两者必须分别保存服务端完整日志、producer 日志、更新前后输出和 checksum/版本证据；
- Checkpoint 重载使用 `test_reload_weights.py`；服务必须已用真实 checkpoint 启动，再调用 `reload_weights` 重载同一 checkpoint。必须保存服务端完整日志、更新前后输出和 checkpoint 路径/版本证据；服务恢复健康不单独构成重载成功证据；
- 完整流程应先以目标真实 checkpoint 启动并记录双接口基线，再初始化 weight-transfer engine，执行 `start_weight_update`、`update_weights`、`finish_weight_update`，最后唤醒 KV cache；更新后两个接口、所有 prompt 的 text/token/logprob 必须与基线完全一致，任何差异均失败；
- 无共享 checkpoint 或数据面时，只能验证 pause/resume 状态机，记录完整热更新为 `not_run`/`blocked`，不伪造替换结论；
- 未初始化引擎就调用 `/update_weights` 可能触发 `EngineDeadError`，应按失败诊断记录并停止重试。

**结果不一致后的修复流程**：保留原始失败和更新前后双接口输出，按 [weight-update-consistency.md](weight-update-consistency.md) 依次检查遗漏权重、地址变化导致的脏数据/旧引用、图模式的捕获与重放地址变化。对证实根因可使用 `register_buffer` 管理非参数状态、预分配与原地更新、引用刷新或目标版本支持的图缓存失效/重新捕获，并在相同 checkpoint、拓扑和原图模式下重新生成基线、完整验证。仅有 eager 对照、补丁或静态检查时不得标记通过。

手动 weight-transfer 顺序（仅用于 checkpoint producer/HCCL 调试）：

```bash
curl -X POST http://<host>:<port>/init_weight_transfer_engine \
  -H "Content-Type: application/json" \
  -d '{"init_info":{"master_address":"<producer-ip>","master_port":<port>,"rank_offset":1,"world_size":<N>}}'
curl -X POST http://<host>:<port>/start_weight_update \
  -H "Content-Type: application/json" \
  -d '{"is_checkpoint_format":true}'
curl -X POST http://<host>:<port>/update_weights \
  -H "Content-Type: application/json" \
  -d '{"update_info":{"names":[...],"dtype_names":[...],"shapes":[[...] ],"packed":false}}'
curl -X POST http://<host>:<port>/finish_weight_update
```

先 pause、再更新、再 resume；未执行 `init → start → update → finish` 的调用链不算成功。若 `collective_rpc reload_weights` 在目标版本不可用，应记录实际异常并停止重复调用。

### 推荐入口

先阅读脚本头部的环境、拓扑和显存约束。HCCL 使用单 `python` producer，IPC 按 worker 数使用 `python` 或 `torchrun`：

```bash
python scripts/weight-update/test_hccl_weight_update.py --help
python scripts/weight-update/test_npu_ipc_weight_update.py --help
python scripts/weight-update/test_reload_weights.py --help
python scripts/lifecycle/test_pause_resume.py --help
```

完整权重更新报告必须归档 driver 输出、服务日志、更新前后生成结果，以及实际 `--gpu-memory-utilization`、预留显存依据、checksum、checkpoint 版本或精度证据。服务重新健康只说明生命周期恢复，不单独构成权重切换证据。

## 特性 4：CP（上下文并行）

CP 默认执行 DCP、PCP 两个独立验证项，不需要用户额外指定。用户明确排除某项时记录排除原因；模型、版本或资源不满足时记录 `blocked` 和实际证据。

新版 vLLM 将 CP 拆成 prefill 和 decode 两个维度：

```bash
vllm serve <model> \
  --prefill-context-parallel-size <P> \
  --decode-context-parallel-size <D>
```

总 CP 规模为 `P × D`。DCP 选值必须满足目标版本的 TP、KV 头数、attention backend 和 token 整除约束；部分模型或 attention backend 可能要求 `dcp_size == tensor_parallel_size`。PCP 与 DP、Model Runner V2 也可能存在互斥或不支持关系。

常见参数检查顺序：

1. `tensor_parallel_size >= dcp_size` 且 `tensor_parallel_size % dcp_size == 0`；
2. `tensor_parallel_size / num_key_value_heads` 能被 `dcp_size` 整除；
3. 若目标模型或 attention backend 有特殊约束，按实际代码校验 `dcp_size` 与 `tensor_parallel_size` 的关系；
4. 长请求的 `num_tokens` 满足 attention/context-parallel 的整除要求；
5. 先分别验证 DCP、PCP，再考虑组合，避免把两个维度的失败混在一起。

### DCP/PCP 分开验证

- **DCP 默认执行**：保持 PCP=1，开启 `decode-context-parallel-size > 1`，用长 decode 请求与 CP=1 baseline 对比输出/日志。
- **PCP 默认执行**：保持 DCP=1，开启 `prefill-context-parallel-size > 1`，用长 prefill 请求与 baseline 对比，单独记录结果。
- 可选做 PCP×DCP 组合，但不得把组合结果替代单维度结果；DSA 后端下 PCP 与 DSA-CP 互斥。

本 skill 已提供通用在线脚本 [scripts/parallel/test_context_parallel.py](../scripts/parallel/test_context_parallel.py)，目标 checkout 若有版本匹配的 HTTP e2e 用例应优先使用目标用例；否则使用该脚本从已验证的 baseline 命令启动 baseline 与 CP variant 两个服务，对同一组 `/v1/completions` 请求做对照。默认检查两边均能返回非空、无乱码结果；只有客户要求逐字一致时才加 `--strict-output-match`。传入 `--worker-hosts` 时，多机生命周期委托给 `serve_start.py`/`serve_stop.py`，客户端只连接 head API：

```bash
python scripts/parallel/test_context_parallel.py \
  --baseline-cmd "vllm serve /path/to/<model> --tensor-parallel-size <T> \
    --data-parallel-size 1 --port <port>" \
  --dim both --dcp-size <D> --pcp-size <P>
```

多机时在命令中补充 `--host <head-ip> --worker-hosts <worker1,worker2>` 及连接模式/凭据参数；`serve_start.py` 负责节点 preflight、head/worker 启停和就绪检查。跨节点 CP 是否受目标版本、模型和 attention backend 支持，必须以实际日志和源码为准，失败记录为 `blocked`。

脚本默认 `--dim both`，分别生成 DCP 和 PCP 的结果；`--dim dcp` / `--dim pcp` 用于用户明确指定单项或失败定位，不能据此省略另一默认项。TP=1 或模型不支持时记录退出码 2/`blocked`。

### 特殊注意力模型的 CP 候选

具有稀疏注意力或专用索引器的模型，不要在通用 DCP 失败后无限重试。目标版本支持专用 CP 路径时，按实际代码和 e2e 用例生成候选配置，例如：

```bash
vllm serve <model> --tensor-parallel-size <N> \
  --block-size <B> \
  --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true}'
```

确认专用 CP 开关对模型生效、未与 PCP/DP 冲突，并记录 attention backend、prefix caching、chunked prefill、speculative decoding 等兼容项。若目标版本报不支持该模型、attention backend 或 CP 组合，按源码定位为模型或版本级限制，不强拉通用 DCP。

平台、模型和 attention backend 的差异必须以目标版本源码和实机结果为准，不能把一台机器或一个模型的 CP 结果推广到其他环境。

## 特性 5：SP（序列并行 / MoE sequence parallel，默认执行）

SP 通常由 `use_sequence_parallel_moe` 推导，不是独立 CLI 开关。vLLM 常见触发条件全部满足：

- `--all2all-backend` 属于目标版本支持的集合；
- `--enable-expert-parallel`；
- `--tensor-parallel-size > 1`；
- `--data-parallel-size > 1`。

SP 配置示例：

```bash
vllm serve <moe-model> \
  --data-parallel-size <D> --tensor-parallel-size <T> \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --distributed-executor-backend mp
```

不要把 `compilation_config.pass_config.enable_sp`（FlashComm/restore 相关）与 `use_sequence_parallel_moe` 混为一谈。SP 默认执行；需 DP>1、TP>1 和 MoE 前置，按可用资源与模型约束生成并预检专项配置。

在线验证先以 SP off（默认 `--all2all-backend nop`）的服务生成 baseline，再启动指定 all2all backend 的 SP 服务，对同一组 `/v1/completions` 输出做功能对照并检查乱码；默认不要求逐字一致，客户提供严格精度标准时再加 `--strict-output-match` 或使用目标 e2e/logprobs。若前置不满足，记录 `blocked` 和原因，不按可选项跳过。传入 `--worker-hosts` 时由共享服务脚本启动多机服务；跨节点 SP 支持仍须以目标版本和通信拓扑为准。

本 skill 已提供通用在线脚本 [scripts/parallel/test_sequence_parallel.py](../scripts/parallel/test_sequence_parallel.py)；目标 checkout 若有版本匹配的 SP HTTP e2e 用例应优先使用目标用例，否则可用该脚本以 baseline 命令派生 SP off/on 的在线服务对照：

```bash
python scripts/parallel/test_sequence_parallel.py \
  --baseline-cmd "vllm serve /path/to/<moe-model> \
    --tensor-parallel-size <T> --data-parallel-size <D> \
    --enable-expert-parallel --port <port>"
```

如需数值精度验收，使用目标模型和客户提供的阈值；本 skill 只记录对照输出和原始指标，不预设跨模型固定阈值。

## 特性 6：PP（流水线并行，默认执行）

```bash
vllm serve <model> --pipeline-parallel-size <N>
```

先检查 `is_pp_supported_model`、`SupportsPP`、stage 数和分布式 executor。若模型含 `use_index_cache` 或 `indexer_types`，还要按 [pp-sparse-attention.md](pp-sparse-attention.md) 检查每个 PP stage 的 Top-K/Indexer 边界。脚本使用 PP=1 在线服务生成 baseline，再启动 PP=N 服务，对同一组 `/v1/completions` 请求做功能对照与乱码检查；默认不要求逐字一致，客户要求时加 `--strict-output-match`。PP 默认执行，即使普通 baseline 不依赖 PP 也要单独验证；传入 `--worker-hosts` 时调用的是跨节点 DP serving wrapper，多机 DP 不等同于 PP，跨节点 PP 必须由目标版本实际支持的 MP/Ray/专用 launcher 启动。

若模型不支持 PP、目标版本 profile 阶段崩溃或出现 `IntermediateTensors`/`NoneType ... items` 等 PP 特有错误，记录 `blocked` 和实际调用栈，按 [failure-diagnosis.md](failure-diagnosis.md) 标为代码/版本级限制，不硬失败整个开箱。

本 skill 已提供通用在线脚本 [scripts/parallel/test_pipeline_parallel.py](../scripts/parallel/test_pipeline_parallel.py)；目标 checkout 若有版本匹配的 PP HTTP e2e 用例应优先使用目标用例，否则可用该脚本以 baseline 命令派生 PP=1/PP=N 在线服务对照：

```bash
python scripts/parallel/test_pipeline_parallel.py \
  --baseline-cmd "vllm serve /path/to/<model> \
    --tensor-parallel-size 1 --port <port>" \
  --pp-size 2
```

多机 PP 不要把 `--host <head-ip> --worker-hosts <worker1,worker2>` 直接当作跨节点 PP 启动方式：该入口当前是 DP 编排。应使用目标版本的原生 MP/Ray/专用 PP launcher，再让脚本连接 head API；若只能使用 serving wrapper，PP 记为 `blocked`/`not_run`。若目标版本在 profile、stage 间通信或 `IntermediateTensors` 路径失败，记录 `blocked` 并按 [failure-diagnosis.md](failure-diagnosis.md) 定位，不通过重复启动制造通过。

若 PP 在 profile 阶段失败或出现 `IntermediateTensors`/`NoneType ... items` 等错误，记录 `blocked` 和实际调用栈，按 [failure-diagnosis.md](failure-diagnosis.md) 标为代码/版本级限制，不通过改参数或重复运行制造通过。

## 每轮执行和结果记录

每个特性都遵循：

1. 复用已通过容量预检的 baseline 配置，只叠加当前特性开关；
2. 按单机或多机模式调用服务脚本，等待 `/health`、`/v1/models` 和必要的开发端点；
3. 保存启动命令、环境变量、服务端日志、客户端响应、内存/NPU 指标和版本 commit；
4. 结果标记为 `passed`、`failed`、`not_run`、`not_applicable` 或 `blocked`，并写明证据和客户判定边界；
5. 失败先做根因定位，结束后停止当前服务，再进入下一特性。容量失败除外：直接输出《容量报告》并终止全流程。

## 相关文档

| 文档 | 主题 |
| --- | --- |
| [multi-node.md](multi-node.md) | 多机节点、DP/PP、head/worker 约束 |
| [capacity-check.md](capacity-check.md) | 容量硬拦截与《容量报告》 |
| [failure-diagnosis.md](failure-diagnosis.md) | traceback 分层、源码印证和《问题定位报告》 |
