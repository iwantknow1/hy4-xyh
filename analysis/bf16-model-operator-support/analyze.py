"""Build a BF16 observed-kernel coverage report from archived HY4 profiles."""
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import os
BASE = Path(os.environ.get('HY4_ARCHIVE_ROOT', Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent
ACCEL = {'AI_CORE', 'AI_VECTOR_CORE', 'AI_VECTOR', 'MIX_AIC', 'MIX_AIV', 'MIX_AIC_AIV', 'MIX_AIV_AIC'}
MISSING = {'', 'N/A', 'NA', 'NONE', '-'}

def number(value):
    result = float(value)
    assert result >= 0
    return result

def classify(row):
    types = row['Input Data Types'] + ';' + row['Output Data Types']
    if re.search(r'\b(?:DT_)?BF16\b|\bBFLOAT16\b', types, re.I):
        return 'bf16_io_observed'
    if row['Input Data Types'].strip().upper() in MISSING and row['Output Data Types'].strip().upper() in MISSING:
        return 'dtype_unreported'
    return 'no_bf16_io_observed'

def category(name, core):
    if core == 'COMMUNICATION' or re.search(r'hcom_|HcclLaunch', name):
        return 'communication_or_launch'
    if '_ihc_' in name:
        return 'ihc_triton'
    if '_triton_rope' in name:
        return 'rope_triton'
    if 'slot_mapping' in name:
        return 'slot_mapping'
    if 'GroupedMatmul' in name:
        return 'moe_grouped_matmul'
    if 'TransposeBatchMatMul' in name:
        return 'batch_matmul_or_layout'
    if 'MatMulV3' in name:
        return 'projection_matmul'
    if 'SparseFlashAttention' in name:
        return 'sparse_attention'
    if 'LightningIndexer' in name:
        return 'sparse_indexer'
    if 'MoeGating' in name:
        return 'moe_router'
    if 'MoeDistribute' in name:
        return 'moe_dispatch_or_combine'
    if 'KvRmsNormRopeCache' in name or 'ScatterPaCache' in name:
        return 'kv_cache_or_fused_rope'
    if 'RmsNorm' in name:
        return 'rmsnorm'
    if 'LayerNorm' in name:
        return 'layernorm'
    if 'RotaryPositionEmbedding' in name:
        return 'rope'
    if 'SwiGlu' in name:
        return 'swiglu'
    if 'Embedding' in name:
        return 'embedding'
    if 'IndexAiCpu' in name:
        return 'bf16_index_device_cpu'
    if 'Cast' in name:
        return 'cast_or_copy_conversion'
    return 'elementwise_index_layout_or_sampling'

rows = []
sources = []
for file in sorted((BASE / 'reduced/profiles').rglob('kernel_details.csv')):
    rank = int(re.search(r'_rank(\d+)_', str(file)).group(1))
    data = list(csv.DictReader(file.open(encoding='utf-8-sig', newline='')))
    assert len(data) == 2380
    sources.append({'rank': rank, 'path': str(file.relative_to(BASE)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'rows': len(data)})
    for line, row in enumerate(data, 2):
        duration = number(row['Duration(us)'])
        start = number(row['Start Time(us)'])
        row = dict(row, rank=rank, source_csv=str(file.relative_to(BASE)), source_line=line,
                   duration_us=duration, start_us=start, evidence=classify(row),
                   module_family=category(row['Name'], row['Accelerator Core']))
        row['model_functional_status'] = '已观测执行（成功的BF16权重模型轮次）'
        rows.append(row)
assert len(sources) == 2 and len(rows) == 4760

def stats(data):
    counts = Counter(r['evidence'] for r in data)
    engines = Counter(r['Accelerator Core'] for r in data)
    duration = sum(r['duration_us'] for r in data)
    return {'calls': len(data), 'unique_kernel_names': len({r['Name'] for r in data}),
            'unique_cann_or_custom_types': len({r['Type'] for r in data}),
            'summed_kernel_duration_us': duration, 'evidence_calls': dict(counts),
            'engine_calls': dict(engines),
            'engine_summed_duration_us': {k: sum(r['duration_us'] for r in data if r['Accelerator Core'] == k) for k in engines},
            'bf16_io_accelerator_calls': sum(r['evidence'] == 'bf16_io_observed' and r['Accelerator Core'] in ACCEL for r in data),
            'bf16_io_device_cpu_calls': sum(r['evidence'] == 'bf16_io_observed' and r['Accelerator Core'] == 'AI_CPU' for r in data),
            'unreported_compute_calls': sum(r['evidence'] == 'dtype_unreported' and r['Accelerator Core'] in ACCEL for r in data)}

groups = defaultdict(list)
for row in rows:
    groups[(row['Name'], row['Type'], row['Accelerator Core'], row['evidence'])].append(row)
operator_rows = []
for (name, typ, core, evidence), data in groups.items():
    duration = sum(r['duration_us'] for r in data)
    operator_rows.append({'kernel_name': name, 'cann_or_custom_type': typ, 'engine': core,
        'model_functional_status': '已观测执行（成功的BF16权重模型轮次）', 'evidence': evidence, 'module_family': data[0]['module_family'], 'calls': len(data),
        'calls_rank0': sum(r['rank'] == 0 for r in data), 'calls_rank1': sum(r['rank'] == 1 for r in data),
        'summed_duration_us': duration, 'mean_duration_us': duration / len(data),
        'min_duration_us': min(r['duration_us'] for r in data), 'max_duration_us': max(r['duration_us'] for r in data),
        'input_dtype_signatures': sorted({r['Input Data Types'] for r in data}),
        'output_dtype_signatures': sorted({r['Output Data Types'] for r in data}),
        'input_shape_signatures': sorted({r['Input Shapes'] for r in data}),
        'output_shape_signatures': sorted({r['Output Shapes'] for r in data}),
        'raw_references': [f"{r['source_csv']}:{r['source_line']}" for r in data]})
operator_rows.sort(key=lambda r: r['summed_duration_us'], reverse=True)
assert sum(r['calls'] for r in operator_rows) == 4760
# Different input/output type signatures must not be merged into one BF16 claim.
signatures = defaultdict(list)
for r in rows:
    key = (r['rank'], r['Name'], r['Type'], r['Accelerator Core'], r['Input Data Types'], r['Output Data Types'], r['Input Shapes'], r['Output Shapes'])
    signatures[key].append(r)
signature_rows = []
for key, data in signatures.items():
    rank, name, typ, core, it, ot, ishape, oshape = key
    signature_rows.append({'rank': rank, 'kernel_name': name, 'type': typ, 'engine': core,
        'input_types': it, 'output_types': ot, 'input_shapes': ishape, 'output_shapes': oshape,
        'evidence': data[0]['evidence'], 'calls': len(data),
        'summed_duration_us': sum(r['duration_us'] for r in data),
        'source_lines': [r['source_line'] for r in data], 'source_csv': data[0]['source_csv']})

rank_stats = {str(rank): stats([r for r in rows if r['rank'] == rank]) for rank in [0, 1]}
assert rank_stats['0']['engine_calls'] == rank_stats['1']['engine_calls']
assert rank_stats['0']['evidence_calls'] == rank_stats['1']['evidence_calls']
all_stats = stats(rows)
bf16_types = sorted({r['Type'] for r in rows if r['evidence'] == 'bf16_io_observed'})
unknown = [r for r in operator_rows if r['evidence'] == 'dtype_unreported']
report = {'title': 'HY4 BF16权重模型算子功能支持分析', 'generated_utc': datetime.now(timezone.utc).isoformat(),
    'validation_scope': 'bf16_weight_model_reduced_2layer_middle_decode_observed_tasks',
    'scope': {'model': 'HY4 2L reduced BF16', 'tp': 2, 'dp': 1, 'physical_devices': [2, 3],
        'input_tokens': 32768, 'output_tokens': 4096, 'captured_decode_steps_per_rank': 10,
        'phase': 'middle decode after 2048 worker steps', 'mtp_in_profile': False,
        'sources': sources},
    'aggregate': all_stats, 'ranks': rank_stats, 'observed_bf16_operator_types': bf16_types,
    'operator_summary': operator_rows, 'dtype_shape_signatures': signature_rows,
    'method': {'bf16_io_observed': 'BF16 appears in recorded input or output dtype. This does not establish internal arithmetic or accumulator dtype.',
        'dtype_unreported': 'Both recorded input and output dtype are missing/N/A. Do not infer unsupported or certify BF16 from model dtype.',
        'no_bf16_io_observed': 'A different dtype is recorded. FP32 math and integer/bool indexing are not BF16 failures.',
        'AI_CPU': 'Device-side CPU engine. Successful observed execution counts as model operator functional support regardless of I/O dtype; engine placement is separate from support.',
        'durations': 'Sums of recorded kernel durations, across two devices; not wall-clock time, critical-path time, or tokens/s.',
        'module_family': 'Heuristic categorization from kernel names, not a proven one-to-one source operator mapping.'},
    'limits': ['Observed workload/shape coverage only; no universal operator-support percentage.',
        'No dtype-specific reference numerical validation per kernel; successful execution is not a numerical precision certification.',
        'Vision/prefill kernels are not covered by the middle-decode window; MTP is not enabled in this profile.',
        'Full 78-layer multi-node model completed accuracy evaluation, but no full-model kernel profiling was collected.',
        'Shared-indexer layers, multi-node expert communication, other shapes, concurrency, CP, reload and transfer are not certified by this profile.',
        'Optional DT_UNDEFINED arguments do not alone mean an unsupported datatype.'],
    'functional_support_definition': 'All 4760 task records are evidence of execution in the successful BF16-weight model profiling run. FP32/integer/bool, dtype-unreported and AI_CPU paths are included. Dtype classification is metadata, not a support filter.',
    'recommendations': ['Count the BF16 Index AI_CPU path as observed functional support; keep engine information separate. Any performance optimization requires its own evidence.',
        'Missing dtype metadata does not negate observed execution; recover it only if datatype-specific claims are needed.',
        'If a full-model BF16 support certificate is required, separately profile representative prefill, decode, vision, MoE/MTP and topology shapes, then compare numerical references. No such new run is performed here.']}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

def csv_write(name, data, fields=None):
    fields = fields or list(data[0])
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(dict,list)) else row[k] for k in fields})
csv_write('operator-summary.csv',operator_rows)
csv_write('dtype-shape-matrix.csv',signature_rows)
raw_fields=['model_functional_status','rank','Device_id','Name','Type','Accelerator Core','duration_us','evidence','module_family','Input Data Types','Output Data Types','Input Shapes','Output Shapes','source_csv','source_line']
csv_write('kernel-evidence.csv',rows,raw_fields)

families=[('投影 GEMM','projection_matmul'),('MoE Grouped GEMM','moe_grouped_matmul'),('RMSNorm','rmsnorm'),('稀疏注意力','sparse_attention'),('Lightning Indexer','sparse_indexer'),('RoPE / KV cache','kv_cache_or_fused_rope'),('RoPE API','rope'),('SwiGLU','swiglu'),('MoE Dispatch / Combine','moe_dispatch_or_combine'),('Embedding','embedding'),('Index（AI_CPU，功能支持）','bf16_index_device_cpu'),('iHC Triton（dtype 未报告）','ihc_triton'),('RoPE Triton（dtype 未报告）','rope_triton'),('MoE Router（FP32 路径）','moe_router')]
module_rows=[]
for label, family in families:
    data=[r for r in rows if r['module_family']==family]
    module_rows.append({'module':label, 'model_functional_status':'已观测执行（成功的BF16权重模型轮次）', 'calls':len(data), 'bf16_io_calls':sum(r['evidence']=='bf16_io_observed' for r in data),
        'engines':', '.join(sorted({r['Accelerator Core'] for r in data})),
        'evidence':', '.join(sorted({r['evidence'] for r in data})),
        'kernels':'; '.join(sorted({r['Name'] for r in data})),
        'summed_duration_us':sum(r['duration_us'] for r in data)})
csv_write('module-matrix.csv',module_rows)
report['module_summary']=module_rows
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

def table(headers, data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
summary_table=table(['指标','结果'],[
    ['范围','减层 2L、TP2、两张 A5；每 rank 中段 decode 10 步'],
    ['采集调用 / kernel 名称',f"{all_stats['calls']} 次 / {all_stats['unique_kernel_names']} 个名称（不等于前端算子数）"],
    ['直接记录到 BF16 I/O',f"{all_stats['evidence_calls'].get('bf16_io_observed',0)} 次；{len(bf16_types)} 种 CANN/自定义 Type"],
    ['BF16 I/O 且在 AI Core/Vector/Mixed',all_stats['bf16_io_accelerator_calls']],
    ['BF16 I/O 且在 NPU 侧 AI_CPU',all_stats['bf16_io_device_cpu_calls']],
    ['未报告 I/O dtype 的计算调用',all_stats['unreported_compute_calls']],
    ['全量 kernel 级支持率','不可由当前采样计算，未给出 100% 支持结论']])
module_table=table(['模块/路径（均已观测执行）','调用数（两卡合计）','BF16 I/O 元数据调用','执行引擎','dtype 元数据分类'],[[r['module'],r['calls'],r['bf16_io_calls'],r['engines'],r['evidence']] for r in module_rows])
engine_table=table(['引擎','调用数','累计 kernel 耗时 μs（跨两卡相加）'],[[k,v,round(all_stats['engine_summed_duration_us'][k],3)] for k,v in all_stats['engine_calls'].items()])
operator_table=table(['kernel','Type','引擎','dtype 证据','调用数','累计 μs'],[[r['kernel_name'],r['cann_or_custom_type'],r['engine'],r['evidence'],r['calls'],round(r['summed_duration_us'],3)] for r in operator_rows])
unknown_names=', '.join(sorted({r['kernel_name'] for r in unknown if r['engine'] in ACCEL}))
body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>HY4 BF16权重模型算子功能支持分析</title><style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#172433}}h1{{font-size:32px}}h2{{margin-top:36px;color:#195b75}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}}th,td{{border:1px solid #d7e0e8;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#eaf2f7}}.note{{background:#fff5df;padding:16px 22px;border-left:5px solid #bc831c}}.ok{{background:#eaf5ef;padding:16px 22px;border-left:5px solid #337357}}code{{background:#eef1f5;padding:2px 5px}}a{{color:#176d9b}}@page{{size:A4;margin:14mm}}@media print{{body{{max-width:none;margin:0;font-size:11px}}h1{{font-size:23px}}h2{{font-size:17px;break-after:avoid}}table{{font-size:10px}}th,td{{padding:5px}}tr{{break-inside:avoid}}.note,.ok{{padding:10px 12px}}details{{display:none}}}}</style></head><body>
<h1>HY4 BF16权重模型算子功能支持分析</h1><p>依据既有减层 Profiling 离线生成；没有新启动模型、bench、Profiler 或裁判 API。</p>
<div class="ok"><b>结论：</b>本报告分析 BF16 权重模型所使用的全部已采集算子/任务，不限定其 I/O 必须为 BF16。4760 条任务均来自成功轮次，FP32、整数、BOOL、AI_CPU 及 dtype 未报告的执行记录均纳入模型功能支持证据。已观察到 BF16 投影 GEMM、Grouped GEMM、RMSNorm、稀疏 Attention、Indexer、SwiGLU、KV/cache 等路径在当前形状下实际执行。它们的 I/O dtype 与引擎有直接记录；这不等于全量模型全部算子支持认证，也不证明内部累加全部为 BF16。</div>
<div class="note"><b>本报告的支持口径：</b>算子在当前记录的 dtype/shape 下实际执行成功即计为功能支持，AI_CPU 与 AI Core/Vector 一样计入功能支持；引擎单独列示。未采到的路径为未覆盖；dtype 缺失仅表示类型元数据未知，已有执行记录仍计为模型功能支持证据。当前没有观察到能直接判为 BF16 不支持的失败证据。</div><h2>1. 范围与证据统计</h2>{summary_table}
<p>模型为 HY4 2L（1 Dense + 1 MoE），物理卡 2、3，TP2/DP1；请求实际输入 32768、输出 4096 token。每 rank 在延迟 2048 个 worker 步后记录中段 10 步。<b>这是 decode 窗口，不能把整个请求成功等同于视觉/prefill 的逐算子 dtype 已采到。</b></p>
<h2>2. 主要模块支持证据</h2>{module_table}
<p><code>bf16_io_observed</code>：输入或输出字段明确包含 BF16。<code>dtype_unreported</code>：输入与输出类型均为空/N/A。<code>no_bf16_io_observed</code>：记录了其他类型；不代表 BF16 不支持。表内模块归类基于 kernel 名称，并非完整调用栈映射。</p>
<h2>3. 需要单列的结果</h2>
<ul><li><b>BF16 Index 在 AI_CPU 上执行：</b>两卡合计 {all_stats['bf16_io_device_cpu_calls']} 次，kernel 为 <code>aclnnIndex_IndexAiCpu_Index</code>，明确记录 BF16 输入/输出和整数索引。该路径能执行，但不是 AI Core/Vector 路径。AI_CPU 是设备侧 CPU，<b>这 80 次执行按功能支持口径明确计为已实测支持</b>；不能写成宿主机回退或“不支持 BF16”。是否需要性能优化是另一个问题。</li>
<li><b>自定义 Triton 路径已执行，类型元数据未报告：</b>{html.escape(unknown_names)}。这些 kernel 在设备上有执行记录，但 dtype 字段缺失，因此计为已观测执行；仅不宣称其具体 I/O/内部算术类型。</li>
<li><b>FP32/整数路径并非错误：</b>Router、部分归一化/采样和索引包含 FLOAT、INT32/INT64、BOOL。混合 dtype 不能算作 BF16 支持失败。DT_UNDEFINED 也可能是未提供的可选输入。</li>
<li><b>执行成功不等于逐算子数值精度认证：</b>当前没有逐 kernel 与高精度参考的数值误差对照；全量数据集精度结果属于模型级证据。</li></ul>
<h2>4. 执行引擎统计</h2>{engine_table}
<div class="note">所有耗时都是记录的 kernel duration 之和，可能有重叠，也跨两个设备，不能当作墙钟耗时、batch 时延或端到端吞吐。数据中的通信与调度记录独立列出，不用它们推断 BF16 算术支持率。此前 weighted_attainment=1.0972 是另一套经验效率指标，不是本报告的支持率。</div>
<h2>5. 本次未覆盖的范围</h2><ul>'''+''.join('<li>'+html.escape(x)+'</li>' for x in report['limits'])+f'''</ul>
<p>四机全量 BF16 的 GPQA 185/198、Hallusion aAcc 79.1851% 已有独立报告，但没有全量 kernel 级 Profiling，因此不能将本报告中的次数和耗时乘 39 外推 78 层。MTP、共享 Indexer、跨机专家通信以及视觉/prefill 等需要相应窗口的真实采集。</p>
<h2>6. 逐 kernel 明细</h2><details><summary>展开 {len(operator_rows)} 个 kernel/引擎/dtype 证据组合</summary>{operator_table}</details>
<h2>7. 下载与可复核来源</h2><ul><li><a href="operator-summary.csv">逐 kernel 汇总 CSV</a></li><li><a href="dtype-shape-matrix.csv">dtype/shape 明细 CSV</a></li><li><a href="kernel-evidence.csv">4760 次原始证据行 CSV</a></li><li><a href="module-matrix.csv">模块矩阵 CSV</a></li><li><a href="report.json">完整机器可读 JSON（含来源 SHA256 和行号）</a></li><li><a href="bf16-operator-coverage.xlsx">Excel 汇总与逐项明细</a></li><li><a href="https://github.com/iwantknow1/hy4-xyh/tree/main/reduced/profiles">GitHub 原始 Profiling</a></li></ul>
<p>生成时间：{html.escape(report['generated_utc'])}。这是一份观测覆盖报告，结论限定于记录到的 workload / shape / 环境。</p></body></html>'''
(OUT/'report.html').write_text(body)
print(json.dumps({'aggregate':all_stats,'per_rank':rank_stats,'bf16_types':bf16_types,'dtype_shape_rows':len(signature_rows),'operator_summary_rows':len(operator_rows)},ensure_ascii=False,indent=2))
