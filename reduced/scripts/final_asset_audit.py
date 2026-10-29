import hashlib,json,pathlib
root=pathlib.Path('/codex');audit={}
for layers in (2,6):
    path=pathlib.Path(f'/models/Hy4-{layers}L');cfg=json.loads((path/'config.json').read_text());index=json.loads((path/'model.safetensors.index.json').read_text())
    fields=['architectures','model_type','num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim','q_lora_rank','kv_lora_rank','index_n_heads','index_topk','indexer_types','layer_types','mlp_layer_types','n_routed_experts','num_experts_per_tok','num_nextn_predict_layers','learnable_sink','gated_mla','enable_ihc','rope_parameters','vocab_size']
    audit[str(layers)+'L']={'path':str(path),'config':{k:cfg.get(k) for k in fields},'tensor_count':len(index['weight_map']),'shard_count':len(set(index['weight_map'].values())),'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),'index_sha256':hashlib.sha256((path/'model.safetensors.index.json').read_bytes()).hexdigest(),'vision_config':cfg.get('vision_config')}
audit['2L+MTP']=json.loads((root/'models/Hy4-2L-MTP/reduced_checkpoint.json').read_text())
(root/'model_openbox/hy4/model-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
p=root/'model_openbox/hy4/audit.md';p.write_text('# 结构与实现审计\n\n实际 config、权重索引、视觉配置与 SHA256 见 model-audit.json。\n\nHY4 文本由原生 HYV4 模型与 Ascend attention/HC/专家适配执行；VL wrapper 为 vllm_ascend.patch.hy4_vl.model.HYV4VLForConditionalGeneration，包含视觉塔、MRoPE 及 SupportsPP。2L 保留 dense 与 MoE，6L 包含 full/shared Indexer；2L 的 PP2 边界为第二个 full Indexer，不跨共享组。6L 默认均分 PP2 会从 shared Indexer 开始，因此不能把本次 2L PP2 通过外推成任意 6L partition 通过。\n\nDCP/PCP 受 patch_hy_v4.py:205–206 的 HY4 sink guard 阻塞，独立原条件执行记录见 dcp-audit.json / pcp-audit.json。MTP 只链接现有 27 个 MTP 张量，先核对隐藏维、头数、MoE 与词表，再以独立 2L+MTP 模型验证；不加载 32L 骨干做全量验收。\n\n严格完整性检查保留。CPU staging 是显式选项，增加主机内存和 CPU copy 成本，不能作为全量模型的性能结论。\n')
manifest=json.loads((root/'model_openbox/hy4/manifest.json').read_text())
snapshot={'base_image_id':manifest['environment_prerequisites']['base_image_id'],'source_files':manifest['source_files'],'environment':manifest['environment_prerequisites'],'validation_scope':'functional_smoke'}
(root/'image/source-manifest.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
print({name:(value.get('tensor_count'),value.get('shard_count')) for name,value in audit.items()})
