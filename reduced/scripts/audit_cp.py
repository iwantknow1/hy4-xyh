import ast,hashlib,json,pathlib,types
p=pathlib.Path('/codex/src/vllm-ascend/vllm_ascend/patch/worker/patch_hy_v4.py');src=p.read_text();tree=ast.parse(src)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AscendHYV4MLAAttention')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
guard=next(x for x in fn.body if isinstance(x,ast.If) and 'decode_context_parallel_size' in ast.unparse(x.test))
code=compile(ast.Module(body=[guard],type_ignores=[]),str(p),'exec');checks=[]
for dcp,pcp in [(1,1),(2,1),(1,2)]:
    error=None
    try:exec(code,{'parallel':types.SimpleNamespace(decode_context_parallel_size=dcp,prefill_context_parallel_size=pcp)})
    except NotImplementedError as e:error=str(e)
    assert (error is not None)==(dcp>1 or pcp>1)
    checks.append({'dcp':dcp,'pcp':pcp,'error':error})
for feature in ['DCP','PCP']:
    report={'feature':feature,'status':'blocked','validation_level':'source audit and isolated execution of the unchanged constructor guard; no CP NPU service claimed','reason':'Current Ascend HY4 attention sink adapter explicitly requires PCP=1 and DCP=1. Generic DSA CP support does not remove this model-specific guard.','source':str(p),'line':guard.lineno,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'guard':ast.get_source_segment(src,guard),'checks':checks,'endpoint_checks':{'/v1/completions':None,'/v1/chat/completions':None}}
    pathlib.Path('/codex/logs/resume-exclusive/'+feature.lower()+'-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('DCP and PCP model-specific restrictions verified',checks)
