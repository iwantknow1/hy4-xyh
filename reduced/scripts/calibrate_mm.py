import base64,io,json,pathlib,urllib.request
from PIL import Image
image=Image.new('RGB',(256,256),(127,63,191));buf=io.BytesIO();image.save(buf,format='PNG');url='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
results=[]
for text in ['', 'a']:
    payload={'model':'hy4-codex','messages':[{'role':'user','content':[{'type':'text','text':text},{'type':'image_url','image_url':{'url':url}}]}],'max_tokens':1,'temperature':0}
    req=urllib.request.Request('http://127.0.0.1:18081/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:response=json.load(r)
    results.append({'text':text,'response':response})
report={'feature':'multimodal input calibration','status':'passed','image_size':[256,256],'image_format':'PNG','results':results,'image_template_overhead_empty_text':results[0]['response']['usage']['prompt_tokens']}
pathlib.Path('/codex/logs/resume-exclusive/mm-input-calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
