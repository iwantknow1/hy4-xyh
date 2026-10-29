import hashlib
import json
import math
import mmap
import struct
import subprocess
from pathlib import Path
import torch

source = Path('/models/Hy4-2L')
root = Path('/codex/variants')
index = json.loads((source / 'model.safetensors.index.json').read_text())
mapping = index['weight_map']
sink_names = [name for name in mapping if 'learnable_sink_param' in name]
expert_names = [name for name in mapping if '.layers.1.' in name and 'experts.gate_up_proj' in name]
assert sink_names and len(expert_names) == 1, (sink_names, expert_names)
def build(label, names, sinks):
    target = root / label
    target.mkdir(parents=True)
    changed = set(mapping[name] for name in names)
    for path in source.iterdir():
        if not path.is_file():
            continue
        dst = target / path.name
        if path.name in changed:
            subprocess.run(['cp', '--reflink=auto', '--', str(path), str(dst)], check=True)
            assert dst.stat().st_ino != path.stat().st_ino
        else:
            dst.symlink_to(path)
    changes = []
    for name in names:
        path = target / mapping[name]
        with path.open('r+b') as file:
            header_size = struct.unpack('<Q', file.read(8))[0]
            header = json.loads(file.read(header_size))
            meta = header[name]
            offset = 8 + header_size + meta['data_offsets'][0]
            with mmap.mmap(file.fileno(), 0) as mm:
                if sinks:
                    dtype = {'F32':torch.float32, 'BF16':torch.bfloat16, 'F16':torch.float16}[meta['dtype']]
                    values = torch.frombuffer(mm, dtype=dtype, count=math.prod(meta['shape']), offset=offset)
                    before = values.clone().tolist()
                    values.add_(5)
                    after = values.clone().tolist()
                    del values
                else:
                    experts, rows, cols = meta['shape']
                    assert meta['dtype'] == 'BF16' and rows > 16
                    chunk_size = 16 * cols * 2
                    before_hash = hashlib.sha256()
                    for expert in range(experts):
                        begin = offset + expert * rows * cols * 2
                        before_hash.update(mm[begin:begin + chunk_size])
                        mm[begin:begin + chunk_size] = bytes(chunk_size)
                    before = before_hash.hexdigest()
                    after = 'zeroed first 16 gate rows in every expert'
                mm.flush()
            changes.append({'name':name, 'shape':meta['shape'], 'before':before, 'after':after})
    (target / 'variant-manifest.json').write_text(json.dumps(changes, indent=2))
    print(label, names, flush=True)
build('sink-B', sink_names, True)
build('expert-C', expert_names, False)
