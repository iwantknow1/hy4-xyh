from pathlib import Path
p=Path('/codex/scripts/verify_image_hashes.py');s=p.read_text();s=s.replace("assert versions['torch']=='2.10.0+cpu',versions", "import torch\nversions['torch_runtime']=str(torch.__version__)\nassert versions['torch']=='2.10.0' and versions['torch_runtime']=='2.10.0+cpu',versions");p.write_text(s)
