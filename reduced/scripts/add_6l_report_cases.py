from pathlib import Path
p=Path('/codex/scripts/report_oldhost_exclusive.py')
s=p.read_text()
if "'Checkpoint 6L" not in s:
    s=s.replace('cases={', "cases={\n 'Checkpoint 6L TP2 graph sleep/reload, 3 rounds':'checkpoint-6l-graph.json',\n 'IPC 6L TP4 graph packed':'ipc-6l-tp4-graph.json',")
s=s.replace("'evidence':'logs/oldhost-unit.log'", "'evidence':'logs/resume-exclusive/prefinalize-final-unit.log'")
p.write_text(s)
