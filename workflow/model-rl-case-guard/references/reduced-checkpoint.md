# 可选：从模型配置生成减层权重

当用户没有真实权重、已有模型 `config.json`，希望先快速验证服务拉起、sleep/wake、pause/resume、权重更新、路由或并行功能时，使用此流程。已有真实权重或需要正式精度、性能验收时继续使用原流程。减层 checkpoint 是随机初始化的模型资产，不是从真实模型裁剪得到的训练权重。

## 保留结构与确定层数

先读取配置及目标版本建模代码，确认解码器路径、层类型、checkpoint loader 和模型专用初始化。仅减少解码层数及其关联配置；保留模型架构、词表、embedding/lm_head、hidden/intermediate size、注意力/KV 头、head dim、RoPE、专家数/每 token 专家数、共享专家、视觉/音频编码器及投影等基础结构。

减层数不固定为 1 或 2。优先选能覆盖原模型全部结构层类型的最短合法前缀：例如同时保留线性/全注意力、dense/MoE、滑窗/全注意力等。保留相应顺序与周期规则；PP 至少保证每个 stage 有有效层，稀疏注意力还需满足 Indexer/Top-K 边界。MTP、跨层共享、特殊首尾层和额外预测层必须按实际建模代码审查。无法用短前缀覆盖时增加层数，不以删除模块制造“保留结构”的结论。

通用脚本支持根配置或 `text_config.num_hidden_layers`，同步截取显式 `layer_types`，并检查层类型覆盖；对已识别的 Qwen MoE 分层规则同步处理 `mlp_only_layers`、检查 dense/MoE 覆盖。未知逐层列表、特殊分层字段、量化配置或缺少模型实现时明确退出。**脚本不是所有新模型的通用适配器**：对 DeepSeek 特殊分层、混合注意力周期、MTP、量化和定制模型，应先补充有源码依据的模型专用减层/初始化规则，再生成；不能删除报错字段、改模型类型或关闭 loader 校验绕过限制。

## 生成命令

在安装了目标模型对应 `torch`、`transformers`、`safetensors` 的环境运行。使用已安装的模型实现，脚本不下载权重、不执行远程建模代码。仅提供 config 并不意味着当前 Transformers 已支持该架构；目标引擎独有模型需要适配与其 checkpoint loader 匹配的生成器。

以下命令中的路径和层数按实际环境替换，输出目录必须尚不存在。脚本位于本 skill 目录下：

```bash
# 先检查减层计划；此步骤不需要模型依赖，不分配权重。
python scripts/model-preparation/generate_reduced_checkpoint.py \
  --config /models/source/config.json \
  --layers 4 --output /models/reduced-functional --dry-run

# 审查配置差异及层类型覆盖后生成；固定 seed 便于复现。
python scripts/model-preparation/generate_reduced_checkpoint.py \
  --config /models/source/config.json \
  --layers 4 --output /models/reduced-functional \
  --seed 42 --max-weight-gib 8 --max-shard-size 1GB
```

脚本先在 meta device 构造张量结构，检查权重大小上限，再在 CPU 上调用模型原生初始化并导出分片；需要足够的主机内存和磁盘，主机峰值可能高于权重大小，`--max-weight-gib` 不是进程内存硬限制。保留完整词表、专家数和多模态分支后，减层仍可能较大；超限时报告容量阻塞，不能自动缩小基础维度。生成失败不发布半成品目录，也不覆盖原始模型。

输出包含：

- `config.json`：仅修改已审查的层数/逐层字段；
- `model*.safetensors` 及按需生成的权重索引：模型自身初始化、保存的参数与持久 buffer；
- `source_config.json`：原配置副本；
- `reduced_checkpoint.json`：随机权重标识、原/减层数、保留层索引、配置差异、seed、dtype、依赖版本、原/生成配置及权重文件校验和。

浮点 dtype 沿用配置中的 `dtype`/`torch_dtype`，缺省为 FP32。可复现性以同一配置、seed、生成代码及依赖版本为前提。权重初始化使用模型实现，不能把所有浮点/整数张量统一填随机数或零；检查有限值，并审查模型专用的有效范围与布局。例如 `tid2eid` 必须为合法专家索引，且每行不得重复，不能仅以形状匹配认定有效。脚本检查已知路由表约束，不能替代新模型的初始化审计。

## 接入现有特性用例

1. 将生成目录、元数据和实际生成命令记录到任务 `manifest.json`、`audit.md` 和日志。确认目标 vLLM/vLLM-Ascend 支持该模型，并核对 checkpoint 键、形状、dtype、持久 buffer 与接收端 `load_weights()` 的合同；不把运行时非持久 buffer 当作 checkpoint 权重。先做容量预检，再正常从生成目录加载，使用 `--load-format auto`，不使用 `dummy`。
2. tokenizer、特殊 token、chat template、processor 从同一模型对应的已有资产提供，不由随机权重脚本生成。只有 config 时可以完成 checkpoint 生成；缺少 tokenizer 则双接口生成验证记录为 `blocked`。服务可通过 `--tokenizer <对应本地目录>` 指向已有资产，多模态请求还需对应 processor 和输入。
3. 基线、特性服务、HCCL/IPC producer、Checkpoint 重载全部使用同一减层配置和 checkpoint。仍按原用例验证两个接口、全部 prompt 的 text/token/logprob，缺失 token/logprob、NaN/Inf 或更新前后不一致仍为失败。随机输出语义不正常不能直接作为失败依据，也不能因此降低一致性标准。
4. HCCL/IPC producer 继续只读取 checkpoint 原始张量；离线生成阶段构造模型不改变生产端禁止构造 Transformers/FSDP 模型的约束。打包/传输、sleep/wake、图模式、错误定位和复验流程均复用现有用例。
5. 对每个特性记录实际触发的层类型、并行拓扑及执行路径。减层后无法满足的 PP/稀疏边界，或随机路由未实际触发的专家路径，记录覆盖缺口；不能宣称已验证所有专家、所有功能组合或原始完整模型。

HCCL、IPC、Checkpoint 的 JSON 报告在本机可读 checkpoint 元数据时自动写入 `checkpoint_provenance`、`validation_scope: functional_smoke`，并把一致性验收标签设为 `synthetic_reduced_checkpoint_outputs_identical`。远端路径在报告机不可读时，将元数据随日志归档，并在生成 README 时显式传入：

```bash
python scripts/reporting/generate_rl_case_guard_readme.py \
  --report-dir model_openbox/example \
  --model /models/reduced-functional \
  --checkpoint-metadata /evidence/reduced_checkpoint.json \
  --output model_openbox/example/README.md
```

README 会单独展示减层来源和结论边界，checkpoint 内的配置、索引和元数据不会作为特性报告计数。报告必须写明“减层随机权重功能验证”；不替代真实权重精度验收、完整模型显存验证或性能摸高，后续取得真实权重后按原流程复验。
