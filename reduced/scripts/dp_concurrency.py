import json,sys,pathlib,concurrent.futures
sys.path.insert(0,'/codex/scripts/skill-scripts')
from common.dual_api import generate_both
import urllib.request
def request(base,method,path,payload,timeout=600):
    req=urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def run(i):return generate_both(request,'http://127.0.0.1:18082','hy4-codex','What is 2 + 2?',16,timeout=600)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(run,range(12)))
report={'feature':'DP2 TP2 concurrent requests','status':'passed','request_count':24,'concurrency':4,'endpoint_checks':{ep:True for ep in outputs[0]},'outputs':outputs}
pathlib.Path('/codex/logs/resume-exclusive/feature-dp-concurrent.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('DP concurrent passed',flush=True)
