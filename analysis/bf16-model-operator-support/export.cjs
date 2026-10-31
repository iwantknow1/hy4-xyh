const fs=require('fs'),path=require('path');
const X=require(process.env.HY4_XLSX_MODULE || 'xlsx');
const root=__dirname,r=JSON.parse(fs.readFileSync(root+'/report.json','utf8')),a=r.aggregate;
const wb=X.utils.book_new();
function sheet(name,rows,widths){let s=X.utils.aoa_to_sheet(rows);s['!cols']=widths.map(w=>({wch:w}));s['!autofilter']={ref:s['!ref']};X.utils.book_append_sheet(wb,s,name);}
const functional='已观测执行（成功 Profiling 轮次）';
const E={bf16_io_observed:'I/O 元数据包含 BF16',dtype_unreported:'I/O dtype 未报告；模型路径已执行',no_bf16_io_observed:'I/O 元数据为其他 dtype；模型路径已执行'};
sheet('支持口径',[
['项目','本次采用的口径'],
['报告对象','BF16权重模型的全部已采集算子/任务；不是BF16 dtype专属支持目录。'],
['已观测执行记录',a.calls],
['什么叫功能支持','当前 dtype/shape 下实际执行成功，计为功能支持；不要求必须运行在 AI Core。'],
['AI_CPU','NPU 设备侧 CPU。执行成功即计入模型功能支持，与 I/O dtype 无关。'],
['AI Core/Vector/Mixed','执行引擎信息；它们与功能支持结论分别列示。'],
['BF16 已实测支持调用',a.evidence_calls.bf16_io_observed],
['其中 AI_CPU BF16 Index',a.bf16_io_device_cpu_calls],
['其中 Core/Vector/Mixed',a.bf16_io_accelerator_calls],
['dtype 未报告','模型路径已执行，功能证据有效；仅类型元数据未知。'],
['仅观察到其他 dtype','属于BF16权重模型的正常执行路径，同样计入模型功能支持。'],
['没有采到的算子/路径','未覆盖，无法判断；不标成不支持。'],
['本数据能直接确认不支持的证据','无；本报告未做不支持条件的负向测试。'],
['适用范围','减层 2L、TP2、两卡、中段 decode 10 步；不是全量所有 shape 的认证。'],
['数值正确性边界','实际执行不等于逐算子高精度参考误差认证。'],
['工作表','Modules 看主要模块；Kernels 看逐 kernel；Dtype Shapes 看具体 dtype/shape；Raw Evidence 可追溯全部 4760 行。']],[38,140]);
sheet('Summary',[
['Metric','Value','Interpretation'],
['Scope','HY4 2L / TP2 / 2 ranks / middle decode 10 steps','Not full-model or universal operator certification'],
['Captured kernel calls',a.calls,'Both ranks combined'],
['Unique kernel names',a.unique_kernel_names,'Names are not source operator count'],
['Unique CANN/custom Type',a.unique_cann_or_custom_types,'Contains integer, control, communication and other paths'],
['BF16 I/O calls',a.evidence_calls.bf16_io_observed,'Input OR output has explicit BF16 metadata'],
['BF16 I/O on Core/Vector/Mixed',a.bf16_io_accelerator_calls,'Does not establish internal accumulation dtype'],
['BF16 I/O on AI_CPU',a.bf16_io_device_cpu_calls,'Counts as observed functional BF16 support; engine remains AI_CPU'],
['Unreported compute dtype calls',a.unreported_compute_calls,'iHC and Triton RoPE; do not infer unsupported'],
['Other recorded dtype calls',a.evidence_calls.no_bf16_io_observed,'FP32, integer, bool and other types; not support failures'],
['All unreported dtype calls',a.evidence_calls.dtype_unreported,'Also includes communication and HCCL launch records'],
['Universal model operator support percentage','Not calculated','No valid all-model/all-shape denominator'],
['Summed kernel duration us',a.summed_kernel_duration_us,'Sum across ranks; not critical path or wall-clock'],
['New NPU/server/bench tests',0,'Offline analysis only']],[43,62,95]);
sheet('Modules',[
['Module','Calls','BF16 I/O calls','Engines','dtype 元数据（不作为支持过滤条件）','Kernel names','Summed duration us'],
...r.module_summary.map(v=>[v.module,v.calls,v.bf16_io_calls,v.engines,'模型功能路径已观测执行；dtype 另列',v.kernels,v.summed_duration_us])],[33,12,20,35,65,90,25]);
sheet('Kernels',[
['Kernel','Type','Engine','当前记录路径的执行状态','dtype 元数据（不作为支持过滤条件）','Family','Calls','Rank0 calls','Rank1 calls','Sum us','Mean us','Min us','Max us','Input dtype signatures','Output dtype signatures','Input shape variants','Output shape variants','First source reference'],
...r.operator_summary.map(v=>[v.kernel_name,v.cann_or_custom_type,v.engine,functional,E[v.evidence],v.module_family,v.calls,v.calls_rank0,v.calls_rank1,v.summed_duration_us,v.mean_duration_us,v.min_duration_us,v.max_duration_us,v.input_dtype_signatures.join('\n'),v.output_dtype_signatures.join('\n'),v.input_shape_signatures.length,v.output_shape_signatures.length,v.raw_references[0]])],[70,28,22,43,60,37,10,12,12,17,17,17,17,100,70,18,18,110]);
sheet('Dtype Shapes',[
['Rank','Kernel','Type','Engine','当前记录路径的执行状态','Input types','Output types','Input shapes','Output shapes','Evidence','Calls','Sum us','Source rows','Source CSV'],
...r.dtype_shape_signatures.map(v=>[v.rank,v.kernel_name,v.type,v.engine,functional,v.input_types,v.output_types,v.input_shapes,v.output_shapes,E[v.evidence],v.calls,v.summed_duration_us,v.source_lines.join(','),v.source_csv])],[10,70,28,20,43,100,80,100,100,60,10,17,60,110]);
const csvRows=X.utils.sheet_to_json(X.readFile(root+'/kernel-evidence.csv',{type:'file'}).Sheets.Sheet1,{header:1});
sheet('Raw Evidence',csvRows,[10,10,70,28,23,18,35,40,100,75,100,100,110,12]);
sheet('Engines',[
['Engine','Calls','Summed kernel duration us','Note'],
...Object.entries(a.engine_calls).map(([k,v])=>[k,v,a.engine_summed_duration_us[k],k==='AI_CPU'?'Device-side CPU; includes BF16 Index and HCCL launch':k==='COMMUNICATION'?'Communication, not BF16 arithmetic certification':'Accelerator execution, not internal-dtype proof'])],[25,15,32,100]);
sheet('Sources',[
['Rank','Rows','SHA256','Source path'],...r.scope.sources.map(v=>[v.rank,v.rows,v.sha256,v.path])],[10,12,70,115]);
sheet('Limits',[
['Item','Explanation'],...Object.entries(r.method).map(([k,v])=>[k,v]),...r.limits.map((v,i)=>['Limit '+(i+1),v]),...r.recommendations.map((v,i)=>['Next evidence '+(i+1),v])],[40,150]);
X.writeFile(wb,root+'/bf16-operator-coverage.xlsx',{compression:true});
const re=X.readFile(root+'/bf16-operator-coverage.xlsx');if(re.Sheets.Kernels['!ref']!=='A1:R99')throw Error('Kernel sheet mismatch');if(X.utils.sheet_to_json(re.Sheets['Raw Evidence']).length!==4760)throw Error('Raw evidence count mismatch');console.log(JSON.stringify({sheets:re.SheetNames,kernel_groups:r.operator_summary.length,raw_rows:4760,excel_bytes:fs.statSync(root+'/bf16-operator-coverage.xlsx').size}));
