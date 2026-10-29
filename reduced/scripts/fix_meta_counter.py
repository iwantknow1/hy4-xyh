from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_hy4_reload.py')
s = p.read_text()
old = '''        if isinstance(dst, torch.Tensor) and dst.is_meta:
            return None'''
new = '''        if isinstance(dst, torch.Tensor) and dst.is_meta:
            return fn(self, *args, **kwargs)'''
assert s.count(old) == 1
p.with_suffix('.py.codex_before').write_text(s)
p.write_text(s.replace(old, new))
