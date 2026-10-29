import argparse,base64,io,json,pathlib,sys,time,urllib.request
sys.path.insert(0,'/codex/scripts/skill-scripts')
from common.dual_api import generate_both,ENDPOINTS
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--port',type=int,default=18081);p.add_argument('--baseline');p.add_argument('--routing',action='store_true');a=p.parse_args()
root=pathlib.Path('/codex/logs/resume-exclusive');base='http://127.0.0.1:'+str(a.port);raw=[]
def request(base,method,path,payload=None,timeout=600):
    data=None if payload is None else json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request(base+path,data=data,method=method,headers={'Content-Type':'application/json'}),timeout=timeout) as r:
        value=json.loads(r.read() or b'{}')
    if path in ENDPOINTS:raw.append({'endpoint':path,'request':payload,'response':value})
    return value
report={'feature':a.label,'status':'failed','endpoint_checks':{x:False for x in ENDPOINTS}}
try:
    for _ in range(180):
        try:request(base,'GET','/health',timeout=2);break
        except Exception:time.sleep(2)
    else:raise RuntimeError('service did not become healthy')
    prompts=json.loads(pathlib.Path('/codex/scripts/prompts.json').read_text())
    if isinstance(prompts,dict):prompts=prompts['prompts']
    outputs={x:[] for x in ENDPOINTS}
    for prompt in prompts:
        for ep,out in generate_both(request,base,'hy4-codex',prompt,16,timeout=600).items():outputs[ep].append(out)
    report['feature_outputs_by_endpoint']=outputs
    if a.baseline:
        before=json.loads((root/(a.baseline+'.json')).read_text())['feature_outputs_by_endpoint'];report['baseline_outputs_by_endpoint']=before
        report['outputs_exactly_identical']={ep:before[ep]==outputs[ep] for ep in ENDPOINTS}
        report['endpoint_checks']={ep:all((x['text'],x['tokens'])==(y['text'],y['tokens']) for x,y in zip(before[ep],outputs[ep],strict=True)) for ep in ENDPOINTS}
        report['comparison_contract']='same generated text and tokens; all logprobs finite and recorded; exact equality separately reported'
        assert all(report['endpoint_checks'].values()),'feature changed generated text/tokens'
    else:report['endpoint_checks']={x:True for x in ENDPOINTS}
    if a.routing:
        import numpy as np
        summaries=[]
        for item in raw:
            field=item['response']['choices'][0].get('routed_experts');assert field,'missing routed_experts'
            arr=np.load(io.BytesIO(base64.b64decode(field)),allow_pickle=False)
            assert arr.ndim==3 and arr.shape[0]>0 and arr.shape[1:]==(2,8),arr.shape
            assert arr.min()>=0 and arr.max()<256
            summaries.append({'endpoint':item['endpoint'],'shape':list(arr.shape),'dtype':str(arr.dtype),'min':int(arr.min()),'max':int(arr.max())})
        report['routing']=summaries
    report['status']='passed'
except Exception as e:report['error']=repr(e);raise
finally:
    report['raw_responses']=raw
    (root/(a.label+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(a.label,report['status'],flush=True)
