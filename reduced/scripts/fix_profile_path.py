from pathlib import Path
p=Path('/codex/scripts/run_profile.py');s=p.read_text();old="'PATH':'/codex/.venv/bin:/codex/tools:/usr/local/bin:/usr/bin:/bin'";assert old in s;s=s.replace(old,"'PATH':os.environ['PATH']");p.write_text(s)
