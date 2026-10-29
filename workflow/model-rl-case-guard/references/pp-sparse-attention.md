# PP 与稀疏注意力模型

本参考用于 PP（Pipeline Parallelism）开箱前预检。规则以当前实际 checkout 的
`vllm` 与 `vllm-ascend` 源码为准；不要把某个模型或某次实测结论推广为所有稀疏模型的硬规则。

## 1. 通用 PP 切层

vLLM 仍按隐藏层的连续区间切分，不会因为模型使用 sparse attention 而自动换一套切分算法。
源码入口是 `vllm/distributed/utils.py:get_pp_indices()`：

- 默认尽量均匀分配隐藏层；余数从倒数第二个 stage 向前放，避免把最后 stage 的
  `norm`、`lm_head`、logits 和 sampling 一并压满。
- `VLLM_PP_LAYER_PARTITION` 可以显式指定每个 PP rank 的层数；元素个数必须等于
  PP size，元素之和必须等于隐藏层数。
- `make_layers()` 对不属于本 rank 区间的层使用 `PPMissingLayer`。embedding、最终
  norm 和 LM head 仍按首/末 stage 放置。
- 模型还必须通过 `SupportsPP` / `IntermediateTensors` 接口检查；仅有连续层并不代表
  该模型一定支持 PP。

因此，`base_model_pp_plan` 主要描述模块路径和 stage 间输入输出，不是 sparse-aware
的负载均衡器。

## 2. vLLM-Ascend 的稀疏 Indexer 边界校验

`vllm_ascend/platform.py:_validate_indexer_pp_config()` 在服务初始化早期调用。仅当
`pipeline_parallel_size > 1` 且模型配置包含 `indexer_types` 或启用 `use_index_cache`
时生效。它先使用 vLLM 的通用区间，再逐个 PP stage 检查稀疏依赖：

### IndexCache / Top-K

一个 PP stage 不能从跳过 Top-K 的层开始。`index_topk_pattern` 中边界层为 `S`，或按
`index_topk_freq` / `index_skip_topk_offset` 推导出跳过 Top-K 时，会报：

```text
Index cache dependency crosses a pipeline-parallel stage boundary
Cross-PP Top-K index propagation is not supported.
```

含义是每个 stage 的起始位置必须能在本 stage 内重新计算 Top-K，不能依赖前一 stage 的
索引缓存。

### `full` / `shared` Indexer

如果配置了 `indexer_types`，每个 stage 内必须先出现 `full`，再出现 `shared`。stage
第一层就是 `shared` 时，会报：

```text
IndexShare group crosses a pipeline-parallel stage boundary
Cross-PP Top-K index propagation is not supported.
```

这是当前 Ascend 后端的代码级限制，不是普通显存或通信故障。

### 自定义边界

可以用 `VLLM_PP_LAYER_PARTITION` 把边界移到可重新计算 Top-K 或 `full` Indexer 的层，
但仓库没有自动搜索“最优稀疏边界”的 partitioner。自定义后仍必须让所有 worker 使用
同一值，并重新启动服务。

## 3. 模型级例外

- **DeepSeek V4 / DSA 类模型**：模型实现使用通用 `make_layers()`，但其
  `use_index_cache`、`index_topk_pattern`、`index_topk_freq` 会触发上述边界校验；PP
  预检必须读取实际 `config.json` / `hf_text_config`。
- **MiniMax M3 Sparse**：稀疏层由 `sparse_attention_config` 指定，模型仍使用通用
  `make_layers()`；不要假设它有独立的 sparse PP 切层策略。
- **Kimi K3**：当前 Ascend 实现把其 sequence-parallel 路径限制为
  `pipeline_parallel_size == 1`，并在 PP 中对 SP 路径做断言。PP 与该模型 SP 路径
  不能同时开启。
- **Sparse KV offload**：当前配置直接拒绝 `pipeline_parallel_size > 1`，错误为
  `Sparse KV offload don't support pipeline parallel now.`
- **GLM5Next / GLM-5.3-Flash**：源码注释显示 text model 的
  `make_empty_intermediate_tensors` 能力存在 PP gating；必须按当前版本 registry、
  接口检查和支持矩阵确认，不能只看类名或 `SupportsPP` 继承关系。

## 4. 与 CP、图模式的组合限制

以下不是通用 PP 切层规则，但在稀疏模型开箱时必须一起预检：

- MRV2 PCP 与 PP 当前不支持；
- PCP 与 DCP 不能同时启用；
- DSA backend 下 PCP 与 DSA-CP 不能同时启用；
- vLLM-Ascend Xlite graph mode 与 PP 不兼容。

命中这些硬限制时记录为 `blocked`，进入 `failure-diagnosis.md` 的代码/版本级限制定位，
不要反复更换 PP 数值重试。

## 5. 多机 PP 的真实启动语义

当前 vLLM-Ascend 的 PP 文档支持：

```text
节点内：TP
节点间：PP
```

可使用目标版本提供的 MP 或 Ray 原生分布式启动。例如 MP 两节点、每节点 TP8、PP2
需要在两节点使用一致的 `--nnodes 2`、`--node-rank`、`--master-addr`、
`--master-port`、`--tensor-parallel-size 8` 和 `--pipeline-parallel-size 2`；worker
使用 `--headless`，head 提供 API。

注意：本 skill 复用的 `vllm-ascend-serving/scripts/serve_start.py --worker-hosts`
当前负责的是跨节点 DP 编排（自动注入 `data-parallel-*` 参数）。即使把
`--pipeline-parallel-size` 透传给它，也不能仅凭 `cluster_type=multi-node` 或 HTTP
成功就声称完成了跨节点 PP 验证。需要：

1. 使用目标版本真正支持 PP 的 MP/Ray/专用 launcher；或连接到已经由该 launcher
   拉起的 PP 服务进行 HTTP 对照；
2. 记录 TP、PP、节点数、rank、head/worker 日志和实际启动后端；
3. 若只能使用 `serve_start.py --worker-hosts`，结论只能是多机 DP 服务验证，PP 项
   目标记为 `blocked` 或 `not_run`，不能冒充 PP 通过。

## 6. 开箱预检顺序

1. 读取模型配置和实际源码，确认 `SupportsPP`、`num_hidden_layers`、稀疏 Indexer
   字段及 PP/SP/CP/offload/图模式组合；
2. 用通用 `get_pp_indices()` 或明确的 `VLLM_PP_LAYER_PARTITION` 得到各 stage 区间；
3. 对 `use_index_cache` 检查每个 stage 起始层是否会跳过 Top-K；对 `indexer_types` 检查
   每个 stage 是否先有 `full` 再有 `shared`；
4. 检查模型级硬限制；
5. 再决定单机 PP、原生多机 PP，或记录 `blocked` / `not_run`；
6. PP 失败时保留完整 traceback 和服务日志，按 `failure-diagnosis.md` 定位，不关闭
   校验、不盲目调大 PP、不重复运行同一失败命令。
