from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_hy4_reload.py')
s = p.read_text()
start = s.index('        # _skip_meta_copy:')
end = s.index('        dst = ', start)
s = s[:start] + '        # Meta copies must still contribute to layerwise loading counts.\n' + s[end:]
start = s.index('def _patch_hy4_sink_loader()')
end = s.index('\ndef apply()', start)
s = s[:start] + s[end:]
s = s.replace('    _patch_hy4_sink_loader()\n', '')
s = s.replace('# 3. Attention sinks: load through `weight_loader` so layerwise can replay them.', '# Sink loading is handled by the native HYV4Model weight loader.')
p.write_text(s)
