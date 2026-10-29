# 权重更新结果不一致：定位、修复与复验

适用于 HCCL、NPU IPC 或 Checkpoint 重载完成后，同一真实 checkpoint 的更新前后推理结果不一致。此场景允许 agent 在已授权的代码仓、节点和模型范围内，定位根因、实施最小代码修复并重新验证；不受普通失败“只定位、不修复”的默认规则限制。初始化 OOM、缺少真实 checkpoint、后端不支持等问题仍按原有容量或阻塞规则处理。

## 进入流程前保留现场

- 保存原始基线、更新后输出、请求参数、checkpoint manifest、部署 commit、服务和 producer 日志。按接口与 prompt 定位第一个不同的 text、token 或 logprob，记录 token 位置及两侧数值；completions 和 chat 分别比较各自更新前后的结果。
- 确认 checkpoint 内容、模型名、tokenizer/chat template、prompt、seed、采样参数、生成长度、dtype、量化格式与并行拓扑一致。缺失或无效 logprob 属于证据不完整，不是数值不一致，先补齐采集。
- 无服务 traceback 时明确记录“无服务异常栈”，使用客户端断言、输出差异和源码路径定位，不编造调用栈。
- 在权重传输完整结束且 KV cache 成功唤醒后进行比较。服务仍休眠、传输中断或权重不完整时，保留失败现场，通过 serving 入口重启服务，不继续复用该状态采样。

## 按顺序定位三个主要根因

### 1. 检查是否遗漏权重更新

1. 对照 checkpoint manifest、producer 实际发送清单、worker loader 消费清单，列出遗漏、重复、未识别和被跳过的 checkpoint key。
2. 按目标模型的 `load_weights()` 核对 fused QKV、gate/up、MoE expert/router、shared expert、量化 scale/zero-point，以及 TP/PP/EP 的本地切片。权重融合、共享参数和分片会改变名称或数量，不能直接要求 checkpoint key 与 worker 内部参数一一同名。
3. 在每个相关 worker/rank 上，对照同一 checkpoint 直接加载的预期本地张量，检查形状、dtype、切片位置和内容；保存校验摘要，必要时逐层找第一个发生变化的参数或前向输出。
4. 若确认漏更新，修复实际 loader/传输中的名称映射、分片、过滤或状态刷新逻辑。保留 strict load 和完整性校验，不通过跳过权重、关闭断言或降低比较标准制造通过。

### 2. 检查地址变化是否导致脏数据或旧引用

1. 审计 worker 内的参数、`named_buffers()`、普通张量属性、view/alias，以及缓存或闭包持有的张量引用。普通属性中的张量可能没有进入框架的状态管理。
2. 在基线、sleep 前、wake(weights) 后、更新完成和再次推理时，记录相关张量的 `data_ptr()`、底层 storage 地址、shape、stride、storage offset、dtype 和 device，并核对实际内容。
3. 检查是否用新张量替换了原属性，但下游仍读取旧 view/缓存引用；是否复用了未经完整写入的缓冲区；是否遗漏了索引、长度、mask、计数器、路由或量化派生状态的刷新。
4. 地址发生变化本身不是错误证据。必须确认有消费者继续使用旧地址、缓冲区内容未初始化或更新后可见数据不正确，再决定修复。

### 3. 检查图模式 capture/replay 的地址变化

1. 核对实际图模式与目标部署版本的 graph capture/replay、内存池、sleep/wake 和权重更新实现，比较图捕获时的地址与更新后实际使用的地址。
2. 检查图输入、权重、运行时 buffer、attention 元数据及输出张量是否重新分配；图缓存、算子参数或工作区是否仍持有捕获时的旧引用。
3. 可以在同一 checkpoint、请求和拓扑下临时使用 eager 模式作诊断对照。eager 正常而图模式异常只说明应优先检查图路径，不能单凭该现象认定地址变化就是根因。
4. 根因确认后，优先保持捕获涉及的张量生命周期和原地更新；若目标实现允许地址改变，按该版本支持的方法刷新引用、失效图缓存并重新 capture。复验必须恢复原图模式，不能永久关闭图模式作为通过条件。

## 修复方法：注册 buffer 与原地更新

对于确认需要参与模块状态管理的非参数常量或运行时张量，可使用 `register_buffer` 管理其生命周期。在 device/dtype 和 shape 确定后预分配，在捕获前初始化，并通过 `copy_()`、`fill_()` 或 `zero_()` 原地更新已经分配的存储；重置哪些值必须依据算法和源码证据。

以下示例只展示非参数运行时状态的管理方式，名称、形状、初值和刷新时机应按实际模型实现确定：

```python
import torch


class GraphRuntimeState(torch.nn.Module):
    def __init__(self, state_shape, device, dtype):
        super().__init__()
        self.register_buffer(
            "runtime_values",
            torch.zeros(state_shape, device=device, dtype=dtype),
            persistent=False,
        )

    @torch.no_grad()
    def update_runtime_values(self, current_values):
        if current_values.shape != self.runtime_values.shape:
            raise ValueError("运行时状态形状变化，需要重建状态并重新捕获图")
        if current_values.device != self.runtime_values.device:
            raise ValueError("运行时状态 device 不一致")
        if current_values.dtype != self.runtime_values.dtype:
            raise ValueError("运行时状态 dtype 不一致")
        self.runtime_values.copy_(current_values)
```

- 不将应参与权重加载的 `Parameter` 随意改为 buffer，也不凭空添加 checkpoint key。按实际 checkpoint 合同选择 `persistent`；它控制是否进入 `state_dict`，不控制地址固定或 NPU 休眠恢复。
- `register_buffer` 本身不保证地址不变；重新赋值、设备或 dtype 转换、sleep/wake 中的重新分配都需要核实。对于同一已捕获图的重放，避免用新张量替换被捕获的存储；必须重新分配时同步处理引用和图缓存。
- 定位非参数状态在目标版本 sleep/wake 中的保存、恢复与初始化路径，核实新 buffer 是否确实被这些路径管理，不以“已经注册”替代生命周期验证。
- 调试时记录修复涉及的张量在关键阶段的地址与内容，核对传输/计算流依赖。地址稳定但仍读到旧值时，继续检查写入顺序与可见性。

## 补充异常排查规则

| 方向 | 需要的证据与处理 |
| --- | --- |
| 异步传输或计算未完成 | 检查 HCCL/IPC 发送、接收、加载、finish 和首次推理之间的事件或流依赖。临时同步可帮助确认竞态；修复应建立正确的依赖关系，不能用固定等待秒数掩盖错误。 |
| KV/前缀缓存或派生状态残留 | 检查更新生命周期是否失效旧 KV、前缀缓存、量化预处理结果和权重派生缓存；只刷新已确认需要重建的状态，并验证下一次请求读取了正确内容。 |
| 请求或批次不确定性 | 核对请求参数、并发请求、batch 调度和随机状态。固定输入和串行对照用于定位；单纯增大 logprob 容差或只比较 text 不构成修复。 |
| 单个 rank 或分片异常 | 保存各 worker 的加载清单、切片和首个异常，不用 rank 0 正常或 HTTP 成功代替全部 worker 更新成功。修复对应的 rank、设备映射或 loader 分支。 |
| 权重传输后布局或派生值不一致 | 对照直接加载路径的 dtype、量化、NZ/packed 布局及后处理，检查热更新是否遗漏必要的转换或刷新。不要重复执行已经由模型 loader 完成的融合/切片。 |

这些方向是排查候选，不能直接写成根因。每次修改需同时提供源码依据、观测证据和可解释的修复点。

## 修复后复验和停止条件

1. 保存最小补丁、修复前后版本及诊断日志，不覆盖首次失败的基线和报告。每次修复与复验使用独立记录目录。
2. 在同一真实 checkpoint、模型、拓扑和原图模式下重新启动干净服务，生成双接口基线，再完整执行 `sleep(level=2) → wake(weights) → update/reload → wake(kv_cache) → 双接口推理比较`。
3. 两个接口、全部 prompt 的 text、tokens 和有效 token_logprobs 必须完全一致。对受影响的 HCCL、IPC、Checkpoint 后端分别验证，不能用一个后端的通过结论替代另一个。
4. 补充《问题定位报告》中的“权重覆盖检查”“地址/状态检查”“图模式检查”“补丁与依据”“修复后原配置复验结果”。状态明确为 `修复后通过`、`已修复待实机验证` 或 `修复后仍失败`；仅有补丁、eager 对照或静态检查时，不标记实机验收通过。
5. 仅在有已证实根因和实际补丁或新证据时进行下一轮复验。同一失败没有新证据时停止无意义重试；缺少实机、目标源码或所需数据时记录阻塞原因。容量 OOM 仍按容量报告规则终止，不扩大为环境替换、依赖升级或无关代码修改。

## 技术依据

- [PyTorch Module.register_buffer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)：非参数 buffer 的注册与 `persistent` 的含义。
- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)：图重放的地址与长期张量引用要求。此处作为排查原理参考；昇腾 ACL graph 的具体地址、同步和重捕获行为，以实际部署的 vLLM-Ascend/torch_npu/CANN 源码与日志为准。
