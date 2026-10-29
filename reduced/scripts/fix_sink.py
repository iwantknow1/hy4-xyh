from pathlib import Path
p = Path('/vllm-workspace/vllm/vllm/models/hy_v4/nvidia/model.py')
s = p.read_text()
old = '''                    n = narrow_weight.shape[0]
                    with torch.no_grad():
                        params_dict[name][:n].copy_(narrow_weight)'''
new = '''                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, narrow_weight)'''
assert s.count(old) == 1
p.with_suffix('.py.codex_before').write_text(s)
p.write_text(s.replace(old, new))
