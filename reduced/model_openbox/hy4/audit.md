# 结构与实现审计

实际 config、权重索引、视觉配置与 SHA256 见 model-audit.json。

HY4 文本由原生 HYV4 模型与 Ascend attention/HC/专家适配执行；VL wrapper 为 vllm_ascend.patch.hy4_vl.model.HYV4VLForConditionalGeneration，包含视觉塔、MRoPE 及 SupportsPP。2L 保留 dense 与 MoE，6L 包含 full/shared Indexer；2L 的 PP2 边界为第二个 full Indexer，不跨共享组。6L 默认均分 PP2 会从 shared Indexer 开始，因此不能把本次 2L PP2 通过外推成任意 6L partition 通过。

DCP/PCP 受 patch_hy_v4.py:205–206 的 HY4 sink guard 阻塞，独立原条件执行记录见 dcp-audit.json / pcp-audit.json。MTP 只链接现有 27 个 MTP 张量，先核对隐藏维、头数、MoE 与词表，再以独立 2L+MTP 模型验证；不加载 32L 骨干做全量验收。

严格完整性检查保留。CPU staging 是显式选项，增加主机内存和 CPU copy 成本，不能作为全量模型的性能结论。
