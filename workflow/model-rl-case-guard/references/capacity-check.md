# 容量预检：显存不足时硬拦截

容量预检是模型开箱流程的硬闸门，目标是判断当前节点、当前可用卡数和当前并行配置是否能装下模型权重，不是做精确性能建模。预检不通过时，立即终止开箱，不拉起服务、不继续特性循环、不盲目降低并行度重试，并输出《容量报告》。

## 1. 触发时机

### 1.1 拉起前主拦截

阶段 1-B 完成逐节点卡和 HBM 探测后，agent 先在内存中生成基础配置候选（TP/DP/EP/PP、图模式、`max_model_len` 等），再执行容量预检。预检通过前不得调用 `serve_start.py`、`vllm serve` 或任何会加载权重的命令。

### 1.2 拉起失败兜底

若预检通过但初始化日志出现以下显存不足证据，也按同一报告格式终止，不反复重试：

- `Free memory < desired`
- `Out of Memory during initialization`
- `aclrtMalloc` 失败
- `Memory_Allocation_Failure EL0004`
- `Out of memory` 或 `device memory allocation failed`

读取原始 stdout/stderr，保留 `g=X`、`desired=Y`、设备号和时间戳。若更像缺 CANN 库、端口冲突或参数错误，应转到 [failure-diagnosis.md](failure-diagnosis.md)，不要误报为容量不足。

## 2. 统计权重大小

遍历权重目录并记录 `config.json`、tokenizer、权重索引和自定义代码；`weight_bytes_loaded` 只统计实际会加载到设备的权重分片（通常是 `*.safetensors`、`*.bin` 及其分片），不把配置和 tokenizer 元数据计入显存：

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("<权重目录>")
weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
total = 0
for path in sorted(root.rglob("*")):
    if path.is_file() and path.suffix.lower() in weight_suffixes:
        size = path.stat().st_size
        total += size
        print(f"{size:>14}  {path.relative_to(root)}")
print(f"weight_bytes_loaded={total}")
print(f"weight_GiB={total / 1024**3:.3f}")
PY
```

Windows PowerShell 使用 `py -3` 替换 `python3`。若模型通过权重索引只加载部分分片，按索引和实际 `--load-format` 修正统计；不要把缓存、下载临时文件或重复副本算入权重。

### dtype 和量化修正

- bf16/fp16 通常按每元素 2 字节估算；fp32 按 4 字节，int8 按 1 字节。
- W4A8、int4 等 bit-packed 格式应按实际存储布局估算，不能把逻辑元素数当成字节数。
- 若加载过程会把权重从 fp16 转成 int8，按设备侧实际 dtype 计算；若无法确认，采用较保守的较大值并在报告中标注“估算”。
- 对 MoE 记录专家权重占比 `expert_ratio`；若无法可靠取得，先按不分 EP 的总权重做保守预检。

## 3. 确定可用 HBM 和卡数

### 3.1 逐节点探测

单机直接执行 `npu-smi info`。多机必须复用 `vllm-ascend-serving/scripts/serve_probe_npus.py`（或其 multi-node probe），先逐节点探测，再汇总：

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --host <head-ip> --worker-hosts <worker1>,<worker2> \
  --mode <docker|conda|direct> [连接参数]
```

记录每个节点的 `total`、`free`、`busy`、卡型号、每卡 HBM 总量和当前空闲 HBM。多机总可用卡数是各节点 `free` 卡数之和；总 HBM 是各节点实际 HBM 之和，不能用“卡数 × 单一型号容量”代替异构节点明细。

### 3.2 可分配容量

对节点 `i` 的每张卡使用：

```text
usable_hbm_i = min(current_free_hbm_i, total_hbm_i × utilization)
```

其中 `utilization` 默认取 0.70～0.85 的开箱保守区间；不得默认使用 0.95 以上。若只有卡级总 HBM，则按 `total_hbm × utilization` 估算，并明确这是上限估计。已有其他进程占卡时，还要扣除其占用和运行时碎片余量。

## 4. 并行分摊和多机约束

### 4.1 单卡需求

基础近似（非 MoE、PP=1）：

```text
per_card_weight ≈ weight_bytes_loaded / TP
per_card_need    ≈ per_card_weight + KV + activation + graph_compile_runtime
```

更细的 MoE/PP 估算：

```text
per_card_weight ≈
  non_expert_bytes / (TP × PP)
  + expert_bytes / (TP × PP × EP)
```

DP 不减少单卡模型权重；每个 DP 副本独立占一份模型，因此总需求可按 `per_card_need × DP` 核对总 HBM，但每张参与计算的卡仍必须通过单卡检查。

### 4.2 额外开销

至少预留以下项目，不能只比较“权重大小 ÷ 卡数”和 HBM：

- KV cache：受 `max_model_len`、KV 头、层数和 block 配置影响；
- 激活和中间张量：受 `max_num_batched_tokens`、batch 和并发影响；
- 图模式、编译和运行时工作区：`FULL_DECODE_ONLY` 等图模式需要额外空间；
- HCCL、通信和内存碎片：多机、EP、DP 或大 batch 时可能增加；
- NPU runtime、驱动和框架常驻占用。

没有目标版本测量数据时，按 `KV + activation + graph_compile_runtime` 的保守估算预留，并在报告中逐项列出估算与依据。容量预检是拦截判定，不是精确模拟。

### 4.3 TP 上限和多机扩容

- 单机：`TP ≤ 该节点可用卡数`，并优先选择模型允许的合法并行度。
- 多机：`TP ≤ min(各节点可用卡数)`，且通常取能在每个节点内组成一致 TP 组的合法约数。TP 组通信默认限制在同一节点，不要把 TP 扩到跨机。
- 多机总可用卡数用于总量核对，但不能突破上述 TP 上限。普通 baseline 的 PP 可为 1；默认执行的 PP 专项需独立规划 PP>1 配置，并在服务启动前重新预检容量与拓扑。专项验证不代表允许容量失败后自动扩容或改并行重试。
- 若 `weight/TP_max + overhead` 仍超过单卡可用容量，结论应为“当前单卡仍放不下，需要增加节点或换更大 HBM”，而不是继续增大 TP。

## 5. 判定顺序

按以下顺序执行，任一硬失败立即进入报告并停止：

1. 统计实际加载权重和 dtype/量化方式。
2. 汇总每节点可用卡、每卡 HBM、当前 free HBM 和利用率阈值。
3. 从 `config.json` 和节点拓扑生成候选 TP/DP/EP/PP；多机锁定 TP 上限。
4. 快速总量检查：若 `weight_total × DP > total_hbm × 0.9`，通常已无足够余量，直接判定放不下；若未触发，再用 `weight_total × DP > total_usable_hbm` 做严格总量检查。
5. 逐节点、逐卡检查 `per_card_need ≤ usable_hbm_i`，并检查 DP/PP 分配是否能落在实际节点卡数上。
6. 通过后才允许生成最终服务命令和进入阶段 2；失败时不拉起、不调整参数重试、不进入下一特性。

判定核心为：

```text
per_card_need > usable_hbm_i  => 终止
```

总量通过不能抵消单卡失败；单卡通过也不能忽略异构节点或 DP/PP 拓扑无法落位。

## 6. 固定《容量报告》格式

预检失败或初始化显存兜底失败时，输出以下固定结构到任务 `report.md`，并在最终消息中给出结论摘要：

```markdown
## 容量报告：权重无法在当前卡数下放下
- 模型权重路径：<path>
- 权重总大小：<X GiB>（加载 dtype/量化：<...>）
- 可用卡数：<N> 张（总）；单机卡数：<n>（TP_max=<n>）；节点数：<K>（如 K×n）
- 每卡 HBM：<H GB>；总可用显存（×utilization）：<Y GiB>
- 基础配置（预检时）：TP=<T>, DP=<D>, EP=<E>, PP=<P>
- 预估单卡需权重：<W GiB>（+KV/激活/编译预估 <K> GiB ⇒ 合计 <M> GiB）
- 单卡可用容量：<H × utilization> GiB
- **结论：放不下**（<M> GiB > 单卡可用 <...> GiB）⇒ **终止开箱流程**
- 可选方向（供客户决策，不自行解决）：
  - 增加可用卡数 / 提高单卡 HBM（换更大显存机型）
  - **需要更多节点**（TP 已达单机卡数上限、单卡仍放不下）——用 PP/DP 跨机扩展
  - 开启量化降精度（如 fp16→int8/W4A8）减小权重
  - 调低 `--max-model-len` / `--max-num-batched-tokens` 降低 KV/激活占用
  - 校验是否误用了过大的 `max_model_len` 或 `gpu-memory-utilization`
- 实际日志证据（仅兜底）：<原始错误、设备号、g=X、desired=Y>
- 状态：**已终止（未做后续特性验证）**
```

报告必须明确“权重无法在当前卡数情况下放下”，不能只写“服务启动失败”。可选方向只供客户决策，agent 不得在同一轮自动扩容、换精度或改变并行后继续开箱。

## 7. 结束条件

- **预检通过**：保存权重统计、逐节点 HBM/卡数、候选基础配置和判定依据，然后才进入服务拉起。
- **预检失败**：保存《容量报告》，将任务标为容量阻塞/终止，停止整个开箱流程。
- **实际 OOM**：保存服务原始日志和同格式《容量报告》，标记“预检低估”，停止整个开箱流程，不做同一配置的第三次尝试。
- 若客户要求进一步确认是否误报，只做只读核对（权重索引、dtype、CANN/运行时日志和 `npu-smi`），并引用 [failure-diagnosis.md](failure-diagnosis.md)；不得用绕过容量检查的方式制造通过结论。
