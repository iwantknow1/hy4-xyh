from pathlib import Path
from textwrap import indent
p=Path('/codex/scripts/r3_inproc.py');s=p.read_text()
if not s.startswith('def main():'):p.write_text('def main():\n'+indent(s,'    ')+"\nif __name__ == '__main__':\n    main()\n")
p=Path('/codex/scripts/run_r3_inproc.sh');s=p.read_text();s=s.replace('export VLLM_USE_V2_MODEL_RUNNER=1','export VLLM_USE_V2_MODEL_RUNNER=1\nexport VLLM_WORKER_MULTIPROC_METHOD=spawn');p.write_text(s)
