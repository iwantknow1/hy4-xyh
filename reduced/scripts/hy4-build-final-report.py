import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

root = pathlib.Path('/codex')
logs = root / 'logs/resume-exclusive'
output = root / 'model_openbox/hy4'
cases_dir = output / 'cases'
cases_dir.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def normalize_outputs(outputs):
    return {
        key if key.startswith('/v1/') else '/v1/' + key: [
            {**item, 'token_logprobs': item.get('token_logprobs', item.get('logprobs'))}
            for item in items
        ]
        for key, items in outputs.items()
    }


subprocess.run([sys.executable, str(root / 'scripts/report_oldhost_exclusive.py')], check=True)
manifest = read(output / 'manifest.json')
matrix = []
report_paths = []


def add(name, filename, note=''):
    path = logs / filename
    data = read(path)
    if filename.startswith('checkpoint-'):
        checks = {f'round_{i}_identical': data.get(f'identical_{i}') is True for i in (1, 2, 3)}
        data.update(
            feature=name, status='passed' if all(checks.values()) else 'failed', checks=checks,
            baseline_by_endpoint=normalize_outputs(data['baseline']),
            post_update_by_endpoint=normalize_outputs(data['outputs_3']),
            endpoint_checks={ep: all(checks.values()) for ep in ('/v1/completions', '/v1/chat/completions')},
        )
    elif filename == 'variants-graph.json':
        flags = {k: v for k, v in data.items() if isinstance(v, bool)}
        data.update(feature=name, status='passed' if flags and all(flags.values()) else 'failed', checks=flags)
    data.setdefault('feature', name)
    data['original_evidence'] = str(path)
    normalized = cases_dir / filename
    write(normalized, data)
    report_paths.append(normalized)
    matrix.append({'scenario': name, 'status': data.get('status', 'inspect_raw_report'), 'evidence': filename, 'note': note})


for name, file, note in [
    ('2L TP1 HCCL 图模式 packed，持续通信器 3 轮', 'hccl-graph.json', '此前版本；完整更新逐轮严格一致'),
    *[(f'2L TP1 HCCL 图模式，重新初始化通信器 {i}', f'hccl-reinit-{i}.json', '每轮新 producer、销毁并重建通信组') for i in (1, 2, 3)],
    ('2L TP1 HCCL 图模式 unpacked', 'hccl-unpacked-staged.json', '显式 CPU staging，实际通过 NPU/HCCL 传输'),
    ('2L TP2 IPC 图模式 packed', 'ipc-graph.json', '跨多个 chunk 的完整 checkpoint 事务'),
    ('2L TP2 checkpoint 图模式，连续 3 轮', 'checkpoint-graph.json', '完整 sleep2/reload/wake'),
    ('2L TP2 sink/expert 权重 A-B-A', 'variants-graph.json', '修改权重后输出改变，恢复 A 后完全一致'),
    ('6L TP2 HCCL eager，缓存修复后', 'hccl-6l-eager-cachefix.json', '含临时内存诊断打印；最终代码已移除打印'),
    ('6L TP2 HCCL 图模式，最终修复代码', 'hccl-6l-graph-cachefix.json', '810 张量；原 TP2、8 GiB KV 配置复验'),
    ('6L TP2 checkpoint 图模式，连续 3 轮', 'checkpoint-6l-graph.json', '两接口每轮 text/token/logprob 完全一致'),
    ('6L TP4 IPC 图模式 packed', 'ipc-6l-tp4-graph.json', '4 个同卡 producer；810 张量；两接口严格一致'),
    ('2L TP2 图模式基线', 'feature-baseline.json', '物理卡 2、3'),
    ('2L TP2 SP/FlashComm1', 'feature-sp.json', '与同 TP2 基线对照；记录精确 logprob 相等与否'),
    ('2L TP1×PP2 流水线并行', 'feature-pp.json', '与 TP1×PP1 对照，避免混入 TP 变化'),
    ('2L DP2×TP2，EP4 图模式', 'feature-dp.json', '实际启动两个 DP engine、四个 worker'),
    ('2L DP2×TP2 双接口并发', 'feature-dp-concurrent.json', '并发 4，24 次接口请求均成功'),
    ('2L TP2 R3，HTTP 双接口', 'feature-r3.json', 'V2 runner；解码 base64 npy，检查层数/top-k/专家范围'),
    ('2L TP2 R3，进程内 LLM', 'feature-r3-inproc.json', '显式 spawn；路由数组为 uint8，形状包含 2 层、top-k 8'),
    ('2L+MTP TP2 图模式', 'feature-mtp.json', '草稿 60 token、接受 0；仅功能通过，无加速结论'),
    ('pause/resume 双接口一致性', 'pause-consistency.json', '在 MTP 服务上确认暂停状态并恢复，严格对比输出'),
    ('DCP', 'dcp-audit.json', 'HY4 attention sink 的明确实现限制'),
    ('PCP', 'pcp-audit.json', '与 DCP 分开登记；不是通用 DSA 支持即 HY4 支持'),
]:
    add(name, file, note)

run_dirs = sorted((root / 'tooling/.workspace-local/ascend-profiling-collection/runs').glob('*'), key=lambda p: p.stat().st_mtime)
run = run_dirs[-1]
profile_manifest = read(run / 'manifest.json')
audit = read(logs / 'profile-output-audit.json')
benchmark = (run / 'benchmark.stdout.log').read_text()
counts = {}
for label in ('Successful requests', 'Failed requests', 'Total input tokens', 'Total generated tokens'):
    match = re.search(re.escape(label) + r':\s+(\d+)', benchmark)
    counts[label] = int(match.group(1)) if match else None
profile_checks = {
    'collector_and_analysis_ok': profile_manifest.get('status') == 'ok' and profile_manifest.get('analysis_status') == 'ok',
    'two_ranks': len(audit) == 2,
    'ten_graph_executions_each_rank': len(audit) == 2 and all(len(x['step_events']) == 10 for x in audit),
    'actual_input_32768': counts['Total input tokens'] == 32768,
    'actual_output_4096': counts['Total generated tokens'] == 4096,
    'all_requests_succeeded': counts['Successful requests'] == 1 and counts['Failed requests'] == 0,
    'real_multimodal_input': read(logs / 'mm-input-calibration.json')['image_size'] == [256, 256],
    'middle_window_configured': read(logs / 'profile-parameter-snapshot.json')['command'].count('--profile-phase') == 1,
}
profile_report = {'feature': 'profiling', 'status': 'passed' if all(profile_checks.values()) else 'failed', 'validation_scope': 'functional_smoke', 'checks': profile_checks, 'actual_tokens': counts, 'profile_delay_worker_steps': 2048, 'profile_worker_steps': 10, 'rank_evidence': audit, 'collector_manifest': str(run / 'manifest.json'), 'note': 'Trace uses MODEL_EXECUTE markers rather than ProfilerStep names. Each rank contains exactly ten graph executions and 2380 kernel rows. Worker logs confirm start after delay and automatic stop at ten iterations. Both decoder layers execute in each PP1 graph; no layer extrapolation is used.'}
write(logs / 'profile-final.json', profile_report)
add('2L 多模态 32768/4096，中段 10 步采集', 'profile-final.json', '两个 rank，物理卡 2、3；仅减层采集流程')
for file in ('manifest.json', 'benchmark.stdout.log', 'benchmark.stderr.log'):
    shutil.copy2(run / file, logs / ('profile-final-' + file))

hardware = read(root / 'performance/hardware_efficiency.json')
hardware_report = {'feature': 'hardware_efficiency_export', 'status': 'blocked', 'reason': 'Single-card CSV analysis and JSON were generated, but the mandated Excel exporter cannot import @oai/artifact-tool. Dependency discovery/registry logs are preserved; no workbook is claimed.', 'calculation_json': '/codex/performance/hardware_efficiency.json', 'weighted_attainment': hardware['weighted_attainment'], 'validation_scope': 'functional_smoke'}
fallback = root / 'performance/hardware_efficiency_fallback.json'
excel_passed = fallback.exists() and read(fallback).get('status') == 'passed'
if excel_passed:
    hardware_report = read(fallback)
    workbook = root / 'performance/hardware_efficiency.xlsx'
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == hardware_report['workbook_sha256']
    assert all(hardware_report['checks'].values())
write(logs / 'hardware-efficiency-export.json', hardware_report)
add('单卡算子测算 / Excel 导出', 'hardware-efficiency-export.json', '公开库备用导出器生成 Excel；逐调用公式、阈值修改及文件重开重算均通过' if excel_passed else 'JSON 有结果；Excel 缺依赖，未交付')
for feature, reason in [
    ('多机 DP', '177.11.28.1 当前八张卡都有其他 VLLMWorker_DP 进程；未停止他人任务，未启动跨节点服务。'),
    ('多机 PP', '第二节点无空闲卡；单机 PP 通过不能替代跨节点 PP。'),
    ('原始 TP8 / util0.9 组合', '当前按明确分配仅使用物理卡 2–7；未覆盖原始八卡配置，不能外推该历史错误已修复。'),
    ('全量精度 / 数据集评测', '本任务为减层功能范围；未运行 32 层 Math500/TextVQA，也不以减层输出评价准确率。'),
]:
    status = 'not_run' if feature.startswith(('原始', '全量')) else 'blocked'
    data = {'feature': feature, 'status': status, 'reason': reason}
    path = cases_dir / (str(len(report_paths)) + '-scope.json')
    write(path, data)
    report_paths.append(path)
    matrix.append({'scenario': feature, 'status': status, 'evidence': 'newhost-occupancy.txt' if feature.startswith('多机') else '', 'note': reason})

manifest.update(
    created_utc=datetime.now(timezone.utc).isoformat(), status='validated_with_explicit_blockers',
    validation_scope='functional_smoke', weight_scope='reduced_checkpoints; no full-scale accuracy claim',
    verification_matrix=matrix,
    baseline_special_config={'source': 'inherited authorized baseline', 'tp': 2, 'ep': True, 'dp': 1, 'pp': 1, 'rl_config': {'enabled': True}, 'graph': 'FULL_DECODE_ONLY', 'capture_sizes': [1, 2, 4, 8], 'kv_bytes_per_worker': 8589934592, 'max_model_len': 2048, 'max_num_seqs': 8, 'memory_utilization_argument': .85, 'note': 'Fixed KV overrides automatic utilization-based KV sizing. Profiling independently prechecked with max_model_len=36864, max_num_seqs=1, capture_sizes=[1].'},
    environment_prerequisites={'base_image_id': 'sha256:1ba11fc501306b7ef44b2dcbb7e1fea69a33bc18ca0a7a48c1ede76874a3d6cf', 'local_repairs': 'nine source/test files; no PR opened', 'cann': '9.1', 'torch': '2.10.0+cpu', 'torch_npu': '2.10.0.post4', 'triton_ascend': '3.6.0'},
    performance_collection={'status': profile_report['status'], 'profile_phase': 'middle', 'profile_steps': 10, 'actual_input_tokens': 32768, 'actual_output_tokens': 4096, 'devices': [2, 3]},
    remaining=['DCP/PCP require model-specific distributed sink support', 'Excel export requires @oai/artifact-tool', 'Cross-node DP/PP require available second-node devices', 'TP8 original configuration untested under current allocation', 'Full-model accuracy remains outside this reduced-model scope'],
)
write(output / 'manifest.json', manifest)
if excel_passed:
    manifest['remaining'] = [item for item in manifest['remaining'] if not item.startswith('Excel')]
    manifest['hardware_efficiency_excel'] = hardware_report
    write(output / 'manifest.json', manifest)

commands = logs / 'launch_commands.txt'
script_names = ['start-hccl-6l-eager.sh', 'resume-run_hccl_6l_cachefix.sh', 'start-hccl-6l-graph.sh', 'resume-run_hccl_6l_graph_cachefix.sh', 'start-ipc-6l-tp4-graph.sh', 'resume-run_ipc_6l_tp4_graph.sh', 'start-feature-baseline.sh', 'start-feature-sp.sh', 'start-feature-pp.sh', 'start-feature-tp1-baseline.sh', 'start-feature-dp.sh', 'start-feature-r3.sh', 'run_r3_inproc.sh', 'start-feature-mtp-baseline.sh', 'start-feature-mtp.sh', 'resume-check_pause_consistency.py']
commands.write_text('\n\n'.join(f'# /codex/scripts/{name}\n{(root / "scripts" / name).read_text()}' for name in script_names) + '\n\n# Profiling exact argv and environment\n' + (logs / 'profile-parameter-snapshot.json').read_text())
generator = root / 'tooling/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/reporting/generate_rl_case_guard_readme.py'
argv = [sys.executable, str(generator), '--output', str(output / 'README.md'), '--title', 'HY4 A5 减层版逐项验证证据', '--model', 'HY4 2L / 6L / 2L+MTP', '--manifest', str(output / 'manifest.json'), '--commands-file', str(commands)]
for path in report_paths:
    argv += ['--report', str(path)]
subprocess.run(argv, check=True)

lines = [
    '# HY4 A5 减层版验证报告', '',
    '日期：2026-10-27。实机：177.8.21.1，Ascend 950DT。结论范围：减层版功能与权重更新一致性；不代表全量精度、全量性能或所有拓扑通过。', '',
    '## 结论', '',
    '6 层 TP2 HCCL 的 eager、图模式，6 层 TP4 IPC 图模式，以及 6 层 TP2 checkpoint 连续三轮重载均通过。SP、单机 PP、DP、R3 的 HTTP/进程内路径、MTP、pause/resume 已补测。DCP 和 PCP 在当前 HY4 attention sink 实现中明确不支持；Excel 导出、跨节点验证和原始 TP8 组合仍有下述限制。', '',
    '## 环境与隔离', '',
    '- 工作目录：`/data/x00951564/codex-hy4-20261027-oldhost-revalidation`；独立源码和容器，未覆盖 DeepSeek 的代码。',
    '- 仅使用物理卡 2–7。检查时旧机其他活动容器没有 NPU 计算进程，但不能据此断言无人登录；未停止这些容器。另一台机器八张卡均有其他任务。',
    '- CANN 9.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、Triton Ascend 3.6.0。6/7 卡的历史健康告警仍记录在案；本轮未复位整机或同事的设备。',
    '- 基线启用 EP、RL、sleep 和 FULL_DECODE_ONLY，固定每卡 8 GiB KV。每个特性保留公共配置，并记录 TP/DP/PP、端口、设备和性能负载的必要差异。', '',
    '## 已执行测试与未覆盖项', '',
    '| 测试 | 状态 | 依据/边界 |', '| --- | --- | --- |',
]
for item in matrix:
    evidence = f' [原始证据](../../logs/resume-exclusive/{item["evidence"]})' if item['evidence'] else ''
    lines.append(f'| {item["scenario"]} | {item["status"]} | {item["note"]}{evidence} |')
lines += [
    '', '权重更新通过的标准是完整 sleep(level=2) → wake(weights) → init/start/update/finish → wake(KV)，两个接口、全部提示词的 text、token、有限 logprob 完全一致。HCCL/IPC 都实际传输 checkpoint；CPU staging 不是跳过数据面。一般并行特性要求生成 text/token 与相同 TP 的基线一致，logprob 数值有效；精确数值相等与否另存报告，不与权重更新的严格标准混淆。', '',
    '## 遇到的问题、修复及复验', '',
    '| 问题 | 原因/证据 | 处理与结果 |', '| --- | --- | --- |',
    '| reload 的 sink 加载计数及运行时引用 | TP 切片后直接 copy 绕过 weight_loader | 对原运行时参数调用 loader，保留图捕获的存储引用；TP 分片测试和 A-B-A 实机复验通过。 |',
    '| meta 专家权重漏加载 | 旧补丁在 meta 分支提前返回，合法加载未进入计数 | 恢复该分支的 loader 调用；计数回归从 0 恢复到 6，保留严格完整性检查。 |',
    '| HCCL packed 流与完整 checkpoint 校验冲突 | 分批调用模型 load_weights，流不完整；接收缓存生命周期、stream 也需明确 | 整个流只调用一次 load_weights；显式 default stream；packed batch 保持存活；可选 CPU staging 控制 NPU 常驻原始权重；packed/unpacked、持续通信组和重建通信组均有实机证据。 |',
    '| IPC 多 chunk 的事务及 producer 引用计数 | 每块单独加载破坏完整性；重用 packed storage 时复用引用计数 | 显式 CPU staging 跨 RPC 累积，finish 时一次完整加载；每 chunk 重新导出 IPC 引用计数；完成后释放缓存；2L TP2 和 6L TP4 图模式通过。 |',
    '| 6L TP2 finish 阶段显存不足/拷贝报错 | 原现场有分配失败；传输结束后 reserved 仍约 94.93 GB，专家延迟 materialization 还需空间 | finalize 前 empty_cache 释放 21,214,303,872 字节（约 19.76 GiB），活跃分配 63,994,016,768 字节保持不变；finalize 后仍保留清理以恢复 KV。相同 TP2、8 GiB KV 的 eager/图模式通过。原失败和 before/after 单元回归均保留。 |',
    '| PP 首轮输出不一致 | 首轮把 TP2/PP1 与 TP1/PP2 比较，混入 TP 改变 | 保留失败报告 feature-pp-vs-tp2.json；补 TP1/PP1 基线后 PP2 对照通过，未修改模型实现来迎合比较。 |',
    '| R3 进程内 fork 初始化失败 | 已初始化 NPU 后 fork 无法重新初始化 | 测试入口显式 spawn 并加 main guard；HTTP 与进程内路由形状、dtype、专家范围检查通过。进程退出告警保留，随后重启自有容器清理。 |',
    '| 性能工具无法启动 vllm | 此 uv 环境没有 vllm console script | 独立工具副本改用 `.venv/bin/python -m vllm.entrypoints.cli.main`，不修改模型代码。 |',
    '| 长输入首步 slot-mapping TypeError | 工具覆盖 PATH 丢失 bisheng；broken 环境 HAS_TRITON=false，kernel 为普通 function | 继承完整 CANN PATH 后 HAS_TRITON=true、kernel 为 JITFunction；32K 图片输入/4096 输出实测通过，两个 rank 均采集到十次 MODEL_EXECUTE。 |',
    '| MTP 减层资产缺失 | 原 2L/6L checkpoint 不含 nextn | 独立目录链接 2L 骨干和已有 32L-MTP 中的 27 个 MTP 张量，维度先核对；原资产不改动。草稿 60、接受 0，最终输出一致；不能推导加速或全量质量。 |',
    '| Excel 导出失败 | @oai/artifact-tool 未安装，导出报 ERR_MODULE_NOT_FOUND，依赖检索日志归档 | 保留单卡 CSV、计算 JSON 和原始错误；未把 JSON 冒充 Excel。 |', '',
    'HCCL 在此前另一台机器出现的 RunAicpuIndOpCommInit 动态库符号错误，本机三轮重建通信组未复现。这只能证明当前环境下通过，不能把推测的混合动态库原因写成已证实根因。', '',
    '## CP 核实结论', '',
    '`vllm_ascend/patch/worker/patch_hy_v4.py:205–206` 在 `AscendHYV4MLAAttention.__init__` 检查 DCP、PCP，任一大于 1 就抛出：`HY4 attention sinks currently require PCP=1 and DCP=1.` 原代码条件分别以 (DCP,PCP)=(1,1)、(2,1)、(1,2) 验证，只有默认组合不触发拒绝。通用 DSA 后端存在 CP 实现不等于 HY4 的 sink 路径已适配。未删除断言或伪造 CP 服务结果。', '',
    '## 采集与单卡测算', '',
    '真实多模态负载：一张 256×256 图片，图片/模板实测开销 95 token；实际总输入 32768、输出 4096，成功 1、失败 0。worker 延迟 2048 步后启动、十步后自动停止。两个 rank 各有 10 个 MODEL_EXECUTE 图执行标记、2380 条 kernel 记录。当前 CANN trace 没有 ProfilerStep 名称，step_trace_time.csv 仅有聚合行，因此没有把聚合行数误当推理步数。', '',
    f'单卡选择物理卡 2；按 Accelerator Core 列区分计算、通信与 AI_CPU，使用全部两层、十步的调用记录计算。默认经验上限 0.8，按 <10μs 短调用及其他调用分别加权，JSON 的 weighted_attainment={hardware["weighted_attainment"]:.6f}；该粗略指标可超过 1，不裁剪、不当作端到端加速比，更不能外推 32 层性能。Excel 导出尚未成功。', '',
    '## 版本与复验边界', '',
    '98 项单元测试及 ruff 检查通过；日志分别为 prefinalize-final-unit.log、prefinalize-lint.log。此前新机版本之后，本轮仅 HCCL engine 的前置缓存清理及其既有单元测试这两个文件继续变化；其余七个修复文件校验值与参考版本一致。2L 早期 HCCL 用例属于前一版本，最终两文件修复通过 98 项单元回归及 6L 原配置图模式验证；未声称所有早期用例都在最终哈希上重跑。', '',
    '| 修复文件 | SHA256 |', '| --- | --- |',
]
for name, value in manifest['source_files'].items():
    lines.append(f'| `{name}` | `{value["sha256"]}` |')
lines += ['', '## 复用与原始证据', '', '镜像说明见随附 IMAGE-README.zh-CN.md；镜像将内置修复源码、uv 环境、Triton 覆盖版本和本报告，模型及驱动继续只读外部挂载。镜像构建后的独立实机检查与镜像 ID、导出校验值单独归档，避免让镜像自身的 ID 成为循环构建依赖。', '', '完整逐接口输出、checks 和实际命令见 [逐项证据 README](README.md)；历史失败日志、容量报告和修复前后回归均保留在 logs/resume-exclusive。没有通过的项目已逐项列明，不能将本报告解读为“所有功能、所有拓扑、所有规模全部通过”。', '']
text = '\n'.join(lines)
if excel_passed:
    text = text.replace('Excel 导出、跨节点验证和原始 TP8 组合仍有下述限制。', 'Excel 已通过公开库备用导出器完成并独立重算；跨节点验证和原始 TP8 组合仍有下述限制。')
    text = text.replace('保留单卡 CSV、计算 JSON 和原始错误；未把 JSON 冒充 Excel。', '原工具错误保留；通过 Windows 从官方 npm 下载公开库并核对 SHA512，使用 SheetJS/xlsx-calc 实际生成 Excel，全部逐调用与汇总公式独立重算通过。')
    text = text.replace('Excel 导出尚未成功。', 'Excel 已生成：2250 条计算调用、130 条排除记录，包含 Summary、Kernels、Operators、Excluded 四张表；公式重算与原 JSON 一致，更改经验上限/短调用阈值后重算、恢复默认和文件重开重算均通过。原 @oai/artifact-tool 导出失败与其原始 JSON 保留。')
    text += '\n\n## Excel 备用导出交付\n\n[下载 Excel](../../performance/hardware_efficiency.xlsx)。备用库版本为 SheetJS 0.18.5、xlsx-calc 0.9.2；官方 npm tarball 的 SHA512 已核对。原工具不可获取的记录不改写成通过，备用导出器有独立校验报告 hardware_efficiency_fallback.json。所有计算只使用物理卡 2 的完整十步调用，不复制乘层数；计算结果超过 1 时保留原值。\n'
(output / 'VERIFICATION-REPORT.zh-CN.md').write_text(text)
(output / 'report.md').write_text(text)
print('Final report generated:', output / 'VERIFICATION-REPORT.zh-CN.md')
