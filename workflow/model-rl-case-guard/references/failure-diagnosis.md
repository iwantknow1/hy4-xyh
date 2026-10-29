# 特性失败诊断：按调用栈定位根因

本参考用于模型开箱阶段的特性验证脚本失败、服务拉起失败或特性断言失败。目标是依据完整 traceback、服务日志和实际部署版本源码，定位可验证的根因并产出《问题定位报告》。**普通失败默认只定位；权重更新前后结果不一致允许按专项规则定位、最小修复并复验**，具体见 [weight-update-consistency.md](weight-update-consistency.md)。

适用特性包括 routing replay、sleep mode/权重更新、CP、SP、PP，以及基础服务拉起。若问题明确属于容量不足，优先按 [capacity-check.md](capacity-check.md) 输出《容量报告》；容量报告完成后不进入本流程的重试。

## 1. 触发条件

任一条件满足即触发本方法：

- 阶段 3 特性验证脚本退出码非 0（HTTP、进程内 LLM 或端点断言失败）；
- 服务 `/health` 未返回 200，或日志出现 `Traceback`、`ImportError`、`Failed to load the backend extension`、`EngineDeadError` 等；
- 特性断言失败但服务仍存活，例如输出乱码、输出与 baseline 不一致、sleep/wake 状态机不满足；
- worker 启动失败或多机集群未就绪，且需要区分 worker 原始错误与 head/HTTP 级联错误。

当 HCCL、IPC 或 Checkpoint 更新后的 text/token/logprob 与同一真实 checkpoint 的基线不一致时，优先进入 [权重更新一致性专项流程](weight-update-consistency.md)，按“遗漏权重 → 地址变化与脏数据 → 图模式地址变化”的顺序排查。没有服务 traceback 时，记录客户端断言和输出首差异，以源码和数据证据定位，不编造异常栈。

## 2. 第 0 步：先收集完整证据

**缺证据不开始定位。** 将原始文件保存到 `model_openbox/<model-slug>/logs/`，不要只复制最后一行：

| 证据 | 最低要求 |
| --- | --- |
| 完整 traceback | 从 `Traceback (most recent call last)` 到最后的 `ExceptionType: message`，包括所有 `File ... line N, in ...` 帧 |
| 服务日志 | `nohup`/脚本生成的 stdout、stderr 尾部至少 100 行；多机包含 head 和每个 worker |
| 退出码和断言点 | 脚本退出码、失败的测试函数/断言、HTTP 方法和端点 |
| 命令与参数 | 实际执行命令、模型路径、TP/DP/EP/PP/CP、图模式、端口；密码和私钥内容必须脱敏 |
| 环境上下文 | vLLM/vllm-ascend/CANN 版本或 commit、Python/torch/torch_npu、单机/多机、docker/conda/direct |
| 资源状态 | 相关节点的 `npu-smi info`、进程和显存摘要；若疑似缺库，附 `ldd` 或动态库错误 |
| 源码版本 | 实际部署 checkout/安装包对应的源码和 commit，不能用另一版本的行号替代 |

若日志包含多段 traceback，先按异常链 `__cause__`/`__context__` 区分，优先从最后一次抛出的、最深层的异常开始；同时保留外层异常，因为它可能揭示哪个服务阶段把底层错误包装成了通用错误。

## 3. 第 1 步：逐帧分层

将 traceback 从内向外展开，为每一帧标记层级。层级只用于决定搜索方向，不取代对异常条件的源码核实。

| 层 | 典型模块/特征 | 首要检查方向 |
| --- | --- | --- |
| **L0 测试/脚本层** | `skill/scripts/*.py`、`test_*.py`、HTTP 客户端 | 参数、断言、端口/host、代理、超时、测试夹具 |
| **L1 vLLM 入口/服务层** | `vllm/entrypoints/...`、`api_server`、`engine_core`、`LLM` | serve 参数解析、启动阶段、采样参数、进程间错误包装 |
| **L2 vLLM core 推理层** | `vllm/model_executor/...`、`attention`、`worker`、`model_runner`、`sampler` | 模型结构、形状/维度、KV cache、profile、graph capture |
| **L3 vllm-ascend 适配层** | `vllm_ascend/...`、`patch/`、`worker/`、`models/`、`ops/` | NPU 后端、CP/SP/PP、sleep/权重更新、NZ/图模式、通信后端 |
| **L4 框架/算子层** | `torch_npu`、`torch`、`msadapter` | 算子支持、device 上下文、同步、dtype/shape、算子输入约束 |
| **L5 栈/芯片层** | `_C`、`.so`、`libhccl`、`aclrt*`、CANN | 缺库、显存、HCCL 组、驱动、异步设备错误 |

### 根因与症状的关系

最内层通常给出直接异常（例如 `AttributeError`、算子失败或设备错误），但触发条件可能由外层 L1/L2/L3 分支决定。报告必须同时记录：

1. 最内层异常是什么；
2. 哪个上层调用把什么参数传入了该帧；
3. 哪个配置、模型属性或运行阶段使该分支被执行。

## 4. 第 2 步：沿调用栈回溯

从最内层帧向外逐帧回答三个问题：

1. **这一帧的异常是什么？** 记录 `ExceptionType`、完整消息、断言表达式和关键变量值。
2. **谁把它带到这里？** 阅读上一层 `File ... line N, in <function>`，说明调用方正在做 profile、前向、采样、权重传输、sleep/wake、HTTP 处理还是 worker 启动。
3. **它是不是根因？**
   - 直接缺失/断言是根因候选，例如 `'NPUWorker' object has no attribute 'reload_weights'`、`NoneType ... items()` 或 `Free memory < desired`；
   - generic wrapper、worker died 或 HTTP 500 多半是症状，需要继续向内找首个真实异常；
   - 异步 NPU 错误可能在后续帧才显现，要结合最早的设备调用、`npu-smi` 和 CANN 日志判断。

建议在报告中使用“内层异常 → 调用方 → 配置触发条件”的顺序列出 2～5 个关键帧，而不是粘贴无法阅读的完整堆栈。

## 5. 第 3 步：回到实际源码印证

不能只凭报错字符串猜根因。对根因候选帧，必须在实际部署版本源码中完成以下检查：

1. **搜索唯一错误片段**：在实际部署版本的 `vllm_ascend/`、`vllm/` 和相关测试目录中 grep 报错消息，定位抛出异常的分支。
2. **读取抛出异常的 if/branch**：记录判断条件、默认值、异常前的配置属性和调用阶段；不要只记录行号。
3. **核对配置与模型属性**：确认 TP/DP/EP/PP/CP、`enable_sp()` 条件、`is_pp_supported_model`、`VLLM_SERVER_DEV_MODE`、权重传输配置、图模式等实际值。
4. **核对版本**：用实际部署 commit 的文件和行号；若源码已变更，报告“行号可能漂移”并给出 commit。
5. **区分硬限制与环境问题**：将结论归类为参数错误、环境/依赖、模型实现、代码/版本级限制或设备/资源问题。

若报错字符串只存在于底层 `.so` 或完全搜索不到：

- 收集 `npu-smi info`（显存/进程）、`ldd`（缺库）、CANN `plog`/运行时日志和必要的 `dmesg` 片段；
- 根据 L5 证据给出“最可能方向”，标记为未能在 Python 源码中印证；
- 不把推测写成确定根因，也不借此自动修改环境。

## 6. 特性矩阵交叉核对

先查 [feature-matrix.md](feature-matrix.md) 对应特性小节，再开始长时间源码追踪：

- PCP 与 DP 冲突、专用 CP 未支持、PP profile 崩溃、`collective_rpc` 未实现等若已被目标版本源码确认，报告中标注“代码/版本级限制”；
- CP 的 DCP/PCP、SP、PP 的单机/多机边界必须与矩阵和实际版本一致；
- routing replay、sleep mode/权重更新等专用 e2e 脚本缺失或版本不匹配时，记录 `not_run`，不要用无关 curl 结果代替专项结论。

## 7. 固定《问题定位报告》格式

将报告写入 `model_openbox/<model-slug>/report.md` 或该特性的独立记录，并保留原始证据路径：

```markdown
## 问题定位报告
- 特性：<routing replay / sleep mode / weight update / CP / SP / PP / 服务拉起>
- 症状：<脚本退出码 + 一句话现象>
- 直接异常：<ExceptionType: message>（最内层）
- 根因结论：<一句话——层级 + 触发条件/缺失项>
- 根因分类：<参数 / 环境依赖 / 模型实现 / 代码或版本级限制 / 设备资源>
- 关键调用栈帧：
  1. <文件:行号, 函数> — 这一帧做什么
  2. <文件:行号, 函数> — 谁调用的、传了什么
  3. <文件:行号, 函数> — 触发了哪个条件
- 源码印证：<grep 结果、if/branch 条件、实际 commit>
- 证据：<日志片段、配置属性、资源状态、原始文件路径>
- 可能方向（不解决，供参考）：<可选补丁点 / 需等待版本 / 需客户决策>
- 状态：**<仅定位未修复 / 已修复待实机验证 / 修复后通过 / 修复后仍失败>**
```

未实施的“可能方向”不能写成已修复。权重更新一致性问题实施修复时，按专项流程追加权重覆盖、地址/状态、图模式检查、补丁与依据及原配置复验结果；没有真实复验日志时不记录实机通过。

## 8. 边界与停止条件

- **普通失败默认只定位**：除用户另行授权或权重更新一致性专项场景外，不自动修改上游 vLLM/vllm-ascend 代码。权重更新一致性问题可对证实根因实施最小修复并完整复验；所有场景均保留 strict load 和验证断言，不以替换环境依赖或降低验收标准制造通过。
- **不反复重试**：普通失败完成定位报告后不无意义地重跑同一命令，最多允许一次只读复现以确认触发条件。权重更新一致性问题按专项流程保留每轮证据，有已证实根因和实际补丁或新证据时允许完整复验；没有新证据时停止重复尝试。
- **先判断容量**：命中明确初始化 OOM 时按 [capacity-check.md](capacity-check.md) 终止并输出《容量报告》，不要把容量问题写成普通特性失败后继续循环。
- **区分首错和级联错**：多机优先记录最早失败的 worker/设备异常，再解释 head、RPC 或 HTTP 的级联症状。
- **报告可复现**：保留脱敏后的实际命令、版本 commit、关键帧和日志路径，使修复者无需重新猜测现场。
