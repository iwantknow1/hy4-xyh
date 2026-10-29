import hashlib,json,pathlib
base=pathlib.Path('/models/Hy4-2L');source=pathlib.Path('/models/Hy4-32L-MTP');out=pathlib.Path('/codex/models/Hy4-2L-MTP');out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((base/'config.json').read_text());other=json.loads((source/'config.json').read_text())
for key in ['hidden_size','num_attention_heads','num_key_value_heads','head_dim','kv_lora_rank','q_lora_rank','qk_nope_head_dim','qk_rope_head_dim','n_routed_experts','moe_intermediate_size','vocab_size']:
    assert cfg.get(key)==other.get(key),(key,cfg.get(key),other.get(key))
index=json.loads((base/'model.safetensors.index.json').read_text());src_index=json.loads((source/'model.safetensors.index.json').read_text())
mtp={k:v for k,v in src_index['weight_map'].items() if k.startswith('model.mtp_layers.0.')}
assert mtp and any('eh_proj' in k for k in mtp) and any('experts.gate_up_proj' in k for k in mtp)
for p in base.iterdir():
    if p.name in ('config.json','model.safetensors.index.json','reduced_checkpoint.json'):continue
    dest=out/p.name
    if not dest.exists():dest.symlink_to(p)
for name in set(mtp.values()):
    assert (source/name).is_file(),name
    dest=out/name
    if not dest.exists():dest.symlink_to(source/name)
index['weight_map'].update(mtp)
# Compute indexed tensor byte count from safetensors headers, without copying weights.
import struct
headers={};total=0
for name in set(index['weight_map'].values()):
    with (out/name).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];headers[name]=json.loads(f.read(size))
for key,name in index['weight_map'].items():
    offsets=headers[name][key]['data_offsets'];total+=offsets[1]-offsets[0]
index['metadata']={'total_size':total};cfg['num_nextn_predict_layers']=1
(out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
(out/'model.safetensors.index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
provenance={'validation_scope':'functional_smoke','backbone':str(base),'mtp_source':str(source),'mtp_tensor_count':len(mtp),'mtp_shards':sorted(set(mtp.values())),'total_tensor_bytes':total,'backbone_config_sha256':hashlib.sha256((base/'config.json').read_bytes()).hexdigest(),'mtp_source_index_sha256':hashlib.sha256((source/'model.safetensors.index.json').read_bytes()).hexdigest(),'output_index_sha256':hashlib.sha256((out/'model.safetensors.index.json').read_bytes()).hexdigest(),'note':'Independent reduced backbone plus existing real MTP weights; original assets unchanged.'}
(out/'reduced_checkpoint.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))
