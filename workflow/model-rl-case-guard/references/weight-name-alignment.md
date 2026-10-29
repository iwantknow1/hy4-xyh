# Checkpoint-native 权重名称与布局合同

本流程只用于 checkpoint-native HCCL、IPC 和其他 `send_weights` 验证。生产端
不得构造 Transformers 模型或 FSDP 分片，也不得从 `named_parameters()` 取得
权重。它直接发布 checkpoint 中的原始名称与张量；接收端 vLLM 模型的
`load_weights()` 是唯一负责名称映射、融合、量化布局和并行切分的实现。

## 为什么发送 checkpoint 原名

vLLM 模型 loader 通常以 Hugging Face checkpoint key 为输入，并在 worker 内完成：

- Q/K/V 到 fused QKV 的合并；
- gate/up 到 fused gate-up 的合并；
- MoE expert、shared expert 与 router 的模型专用映射；
- TP、PP、EP 的本地参数选择和切片；
- 量化权重、scale、zero-point 和特殊布局处理。

若 producer 预先把名称转换为 worker 内部参数名，会绕过或重复上述 loader
逻辑，尤其容易破坏刚适配的新模型。因此 producer 只允许发送 checkpoint key。

## 强制预检

在 `sleep(level=2)` 前完成以下检查：

1. 解析 `model.safetensors.index.json`、`pytorch_model.bin.index.json` 或目录内
   权重分片，确认索引引用的每个文件存在；
2. 从 safetensors header 或 PyTorch state dict 记录每个 key 的 dtype、shape、
   numel、来源分片；
3. 拒绝重复 key、空 checkpoint、不支持的 dtype、索引与分片内容不一致；
4. 使用确定性顺序生成 manifest，并计算 `manifest_sha256`；
5. 计算最大单张量字节数，确认 producer 临时显存和 packed buffer 能容纳；
6. 静态确认目标 vLLM/vLLM-Ascend checkout 已注册该模型，且模型实现提供
   `load_weights()` 或等价 checkpoint loader；服务无法从同一 checkpoint 正常
   启动时，不得继续热更新验证；
7. HCCL producer 使用独立物理 NPU；IPC producer 数量、节点和物理 NPU 映射
   必须与推理 worker 一一对应。

预检失败时不得先执行 level-2 sleep，因为它会丢弃当前服务权重。

## 流式内存合同

- safetensors 使用 mmap/header 扫描与单张量读取，不得把完整 state dict 读入内存；
- PyTorch `.bin/.pt/.pth` 无法保证真正的单张量随机读取，允许一次加载一个
  checkpoint shard，但必须在切换分片前释放上一分片；报告需标明主机内存峰值
  至少可能达到最大分片大小；
- NPU 常驻内存只允许包含当前张量或当前 packed chunk、传输 buffer 和后端开销；
- 不得在列表、cache 或闭包中保留已经发送的 NPU tensor；
- metadata 与实际迭代必须逐项具有相同名称、dtype、shape 和顺序。

## 多 rank 合同

- HCCL 模式只有一个 checkpoint producer rank，向所有推理 worker 广播完整的
  checkpoint stream；worker 自行选择其本地参数和分片；
- IPC 模式每个 worker 对应一个同卡 producer。所有 producer 读取相同 manifest，
  以完全相同的顺序导出 handle；`WORLD_SIZE` 必须等于推理 worker 数；
- 多机 IPC 要求每个节点可从相同路径读取同一 checkpoint，或各节点路径内容具有
  相同 `manifest_sha256`；仅路径字符串相同不构成一致性证据。

## 报告要求

HCCL 和 IPC 报告至少保存：

```json
{
  "checkpoint": {
    "path": "/models/new-model",
    "format": "safetensors",
    "file_count": 8,
    "tensor_count": 1234,
    "total_tensor_bytes": 0,
    "max_tensor_name": "model.layers.0.mlp.down_proj.weight",
    "max_tensor_bytes": 0,
    "manifest_sha256": "..."
  }
}
```

同时归档 producer 日志、服务完整日志、实际命令、更新前后双接口输出和
`start/update/finish` 生命周期证据。服务恢复健康但缺少完整权重流证据时，不能
判定权重更新通过。

## 精度判定

- 用户选择 [减层随机权重功能验证](reduced-checkpoint.md) 时，可以用生成的同一 checkpoint 复用下述幂等检查；报告必须标记 `functional_smoke`，不能作为真实权重验收。离线生成模型与 producer 流式发送是两个阶段，producer 的 checkpoint-native 合同保持不变。
- 权重更新正式验收必须先以目标真实 checkpoint 启动服务，再执行
  `sleep(level=2) -> wake(weights) -> update/reload -> wake(kv_cache)`；两个接口、
  所有 prompt 的 text、token 和 logprob 必须与更新前基线一致；
- dummy/旧 checkpoint 更新到目标 checkpoint 只能作为数据面调试或加载生效检查，
  不得作为正式权重更新通过标准；
- 最可靠的对照是另起一个直接从目标 checkpoint 启动的 vLLM 服务，对相同 prompt
  比较 token 与 logprob。该对照同样不依赖 Transformers。

更新前后结果不一致时，按 [weight-update-consistency.md](weight-update-consistency.md) 先核实 producer 发送与 worker loader 消费的完整性，再检查非参数状态、内存地址和图模式引用。确认根因后允许最小修复并完整复验；不改变本文件的原名发送、模型 loader 负责映射和双接口严格一致性合同。
