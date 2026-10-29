from pathlib import Path
p=Path('/codex/src/vllm-ascend/vllm_ascend/patch/worker/patch_hy4_reload.py')
s=p.read_text(); start=s.index('"""'); end=s.index('"""',start+3)+3
s=s[:start]+'''"""Cache expert maps on CPU and materialize expert copy sources for HY4.

Meta-device copies must execute so layerwise reload can count loaded elements.
Sink parameters use the native HYV4Model weight loader.
"""'''+s[end:]
p.write_text(s)
