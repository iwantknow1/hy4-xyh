import csv,json,pathlib,collections
root=pathlib.Path('/codex/profiles/hy4-2l-middle/hy4-2l-middle')
for folder in sorted(root.glob('*_ascend_pt')):
    out=folder/'ASCEND_PROFILER_OUTPUT'
    for name in ['step_trace_time.csv','kernel_details.csv']:
        with (out/name).open() as f:rows=list(csv.DictReader(f))
        print(folder.name,name,'rows',len(rows))
        if name.startswith('step'):print(json.dumps(rows[:12],ensure_ascii=False))
        else:print('accelerator_core',collections.Counter(x['Accelerator Core'] for x in rows),'type',collections.Counter(x['Type'] for x in rows).most_common(12),'devices',collections.Counter(x['Device_id'] for x in rows))
    trace=json.loads((out/'trace_view.json').read_text());ev=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    print('trace_categories',collections.Counter(x.get('cat','') for x in ev).most_common(12));print('step-like events',[(x.get('name'),x.get('cat')) for x in ev if any(s in x.get('name','').lower() for s in ['step','forward','model_execute'])][:25])
