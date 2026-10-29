from pathlib import Path
p=Path('/codex/scripts/report_oldhost_exclusive.py')
s=p.read_text().replace(" 'HCCL 6L TP2 graph packed':'hccl-6l-graph.json',", " 'HCCL 6L TP2 eager packed':'hccl-6l-eager.json',\n 'HCCL 6L TP2 graph packed':'hccl-6l-graph.json',")
s=s.replace(" 'preflight':preflight,", " 'authorized_devices':[2,3,4,5,6,7], 'reserved_for_colleagues':[0,1],\n 'models':{str(n)+'L':{'path':f'/models/Hy4-{n}L','identity':{name:hashlib.sha256((Path(f'/models/Hy4-{n}L')/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors.index.json']}} for n in [2,6]},\n 'preflight':preflight,")
s=s.replace("'TP8 high-utilization sleep'", "'TP8 original scenario: blocked by six-device allocation; 0/1 reserved for colleagues'")
p.write_text(s)
