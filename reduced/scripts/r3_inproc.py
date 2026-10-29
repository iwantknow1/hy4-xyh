def main():
    import json,math,pathlib,numpy as np
    from vllm import LLM,SamplingParams
    report={'feature':'R3 in-process','status':'failed','model':'/models/Hy4-2L','tp':2,'devices':[2,3],'outputs':[]}
    try:
        llm=LLM(model='/models/Hy4-2L',trust_remote_code=True,tensor_parallel_size=2,enable_expert_parallel=True,enable_return_routed_experts=True,additional_config={'rl_config':{'enabled':True}},gpu_memory_utilization=.85,kv_cache_memory_bytes=8589934592,max_model_len=2048,max_num_seqs=8,compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8]})
        for prompt in ['What is 2 + 2?','Write one word about the sky.']:
            item=llm.generate([prompt],SamplingParams(temperature=0,seed=0,max_tokens=16,logprobs=1),use_tqdm=False)[0].outputs[0]
            arr=np.asarray(item.routed_experts)
            assert arr.ndim==3 and arr.shape[0]>0 and arr.shape[1:]==(2,8),arr.shape
            assert arr.min()>=0 and arr.max()<256
            probs=[entry[token].logprob for entry,token in zip(item.logprobs,item.token_ids,strict=True)]
            assert probs and all(math.isfinite(x) for x in probs)
            report['outputs'].append({'prompt':prompt,'text':item.text,'token_ids':list(item.token_ids),'token_logprobs':probs,'route_shape':list(arr.shape),'route_dtype':str(arr.dtype),'route_min':int(arr.min()),'route_max':int(arr.max())})
        report['status']='passed'
    except Exception as e:report['error']=repr(e);raise
    finally:pathlib.Path('/codex/logs/resume-exclusive/feature-r3-inproc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('R3 in-process passed',flush=True)

if __name__ == '__main__':
    main()
