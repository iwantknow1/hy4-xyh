import csv,json,pathlib,re
root=pathlib.Path('/codex');profiles=root/'profiles/hy4-2l-middle/hy4-2l-middle';out=[]
for folder in sorted(profiles.glob('*_ascend_pt')):
    csv_path=folder/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';trace_path=folder/'ASCEND_PROFILER_OUTPUT/trace_view.json'
    trace=json.loads(trace_path.read_text());events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    steps=[x for x in events if 'ProfilerStep' in x.get('name','') or x.get('name')=='MODEL_EXECUTE']
    with csv_path.open() as f:
        reader=csv.DictReader(f);header=reader.fieldnames;rows=list(reader)
    out.append({'folder':str(folder),'kernel_details':str(csv_path),'trace':str(trace_path),'step_events':[{'name':x.get('name'),'ts':x.get('ts'),'dur':x.get('dur')} for x in steps],'kernel_rows':len(rows),'columns':header,'task_types':sorted(set(x.get('Task Type',x.get('Accelerator Core','')) for x in rows))})
pathlib.Path('/codex/logs/resume-exclusive/profile-output-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for item in out:print(json.dumps({'folder':item['folder'],'step_events':item['step_events'],'kernel_rows':item['kernel_rows'],'task_types':item['task_types'],'columns':item['columns']},ensure_ascii=False))
