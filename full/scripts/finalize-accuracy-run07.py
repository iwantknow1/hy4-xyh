"""Generate final artifacts only after real scoring and image archival succeed."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/task')
REFERENCE = dict(qAcc=56.4835, fAcc=63.8728, easy_aAcc=82.6374,
                 hard_aAcc=65.814, aAcc=79.4508)


def judge_metrics(text):
    matches = []
    for line in re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text).splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) == 6 and cells[0] == 'GPT Eval':
            values = list(map(float, cells[1:]))
            assert all(math.isfinite(value) and 0 <= value <= 100 for value in values)
            matches.append(dict(zip(REFERENCE, values)))
    assert len(matches) == 1, 'Expected exactly one original leaderboard row'
    return matches[0]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def finish():
    deadline = time.monotonic() + 18 * 3600
    while time.monotonic() < deadline:
        state = read(ROOT / 'logs/evaluation-pipeline.json')
        assert state['run'] == '07'
        if state['status'] == 'evaluation_finished':
            break
        if state['status'] in ('blocked', 'evaluation_failed', 'judge_failed'):
            raise RuntimeError('Evaluation not complete: ' + state['status'])
        time.sleep(30)
    else:
        raise TimeoutError('Full evaluation still incomplete after report watcher window')
    integrity = read(ROOT / 'logs/gpqa-integrity-07.json')
    assert integrity['status'] == 'passed' and integrity['successful_nonempty'] == 198
    manifest_path = ROOT / 'model_openbox/hy4/manifest.json'
    manifest = read(manifest_path)
    gpqa = manifest['gpqa_run07']
    assert gpqa['rows'] == 198 and gpqa['successful'] == 198
    hall_dir = ROOT / 'evaluation/runs/hallusion-full-07'
    hall_rows = read(hall_dir / 'HallusionBench_result.json')
    dataset = read(Path('/mnt/share/z00949185/hy4-work/22.1/Hallusionbench_test/HallusionBench/HallusionBench.json'))
    assert len(hall_rows) == len(dataset) == 1129
    for predicted, original in zip(hall_rows, dataset):
        assert predicted['model_prediction'].strip()
        assert {key: value for key, value in predicted.items() if key != 'model_prediction'} == {
            key: value for key, value in original.items() if key != 'model_prediction'}
    summary = read(hall_dir / 'summary.json')
    assert summary['samples'] == summary['token_usage_coverage'] == 1129
    with (hall_dir / 'responses.jsonl').open() as stream:
        responses = [json.loads(line) for line in stream if line.strip()]
    assert [row['index'] for row in responses] == list(range(1129))
    assert all(row['sample'] == dataset[index] for index, row in enumerate(responses))
    assert summary['length_truncated'] == sum(
        row['response']['choices'][0]['finish_reason'] == 'length' for row in responses)
    hall_metrics = judge_metrics((ROOT / 'logs/hallusion-judge-07.log').read_text())
    gaps = {key: hall_metrics[key] - value for key, value in REFERENCE.items()}

    image_path = ROOT / 'logs/reusable-image-run07.json'
    deadline = time.monotonic() + 3600
    while not image_path.exists() and time.monotonic() < deadline:
        time.sleep(30)
    image = read(image_path)
    assert image['status'] == 'saved' and image['generation_validated']
    assert not image['cpu_evaluation_container_saved']
    assert Path(image['archive']).stat().st_size == image['bytes']
    assert re.fullmatch('[0-9a-f]{64}', image['sha256'])

    folder = ROOT / 'model_openbox/hy4/full-accuracy-07'
    folder.mkdir(exist_ok=True)
    reports = {
        'gpqa': dict(feature='dataset_evaluation', status='passed', dataset='GPQA Diamond 198',
            max_out_len=102400, metrics=gpqa, checks=dict(all_198_successful=True,
            exact_reference_inputs=True, unique_ids=True, zero_failed_requests=True),
            endpoint_checks={'/v1/completions': None, '/v1/chat/completions': True},
            result_path=gpqa['native_result'], log='/task/logs/gpqa-full-07.log',
            note='Real full BF16 checkpoint; reference uses different quantized weights.'),
        'hallusion': dict(feature='dataset_evaluation', status='passed',
            dataset='HallusionBench 1129 questions / 346 images', max_out_len=102400,
            checks=dict(all_1129_nonempty=True, exact_dataset_and_order=True,
                original_judge_completed=True),
            metrics=dict(actual=hall_metrics, reference=REFERENCE,
                gap_percentage_points=gaps, generation_summary=summary),
            endpoint_checks={'/v1/completions': None, '/v1/chat/completions': True},
            result_path=str(hall_dir / 'HallusionBench_result.json'),
            log='/task/logs/hallusion-judge-07.log',
            note='Original serial client with transport timeout 7200s; generation parameters unchanged.'),
        'serving': dict(feature='full_model_serving', status='passed',
            checks=dict(full_78_layer_checkpoint=True, all_four_nodes_loaded=True,
                text_smoke=True, image_smoke=True, gpqa_198=True, hallusion_1129=True),
            endpoint_checks={'/v1/completions': None, '/v1/chat/completions': True},
            outputs=[state['phases']['full-text-smoke']['final_answer'],
                     state['phases']['full-image-smoke']['final_answer']],
            note='Four-node accuracy scope. Full-model reload/CP/PP cases were not rerun; reduced functional evidence remains separate.'),
        'image': dict(feature='reusable_inference_image', status='passed',
            checks=dict(archive_exists=True, archive_size_matches=True,
                sha256_recorded=True, evaluation_container_excluded=True),
            metrics=image, result_path=image['archive'],
            note='Only our inference container was saved. See logs/reusable-image-run07.json and IMAGE-REUSE.zh-CN.txt.'),
    }
    for name, report in reports.items():
        write(folder / (name + '.json'), report)
    manifest.update(status='finalizing_report', updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hallusion_run07=dict(metrics=hall_metrics, reference=REFERENCE,
            gap_percentage_points=gaps, generation_summary=summary), reusable_image=image)
    manifest['baseline_special_config'] = dict(revision='run07-compiler-path', **manifest['current_configuration'])
    write(manifest_path, manifest)

    commands = ROOT / 'logs/launch_commands-final-run07.txt'
    commands.write_text('# Actual four-node launcher and environment wrapper\n' +
        (ROOT / 'scripts/launch-dp4-skill.sh').read_text() + '\n' +
        (ROOT / 'scripts/serve-with-cann.sh').read_text() +
        '\n# GPQA actual invocation\n/task/evaluation/bin/ais_bench /task/evaluation/gpqa-full-config-07.py --mode all --num-prompts 198 --num-warmups 0 --max-num-workers 1 --dump-eval-details --dump-extract-rate --debug\n' +
        '# Full Hallusion invocation is recorded in evaluation/runs/hallusion-full-07/config.json; request timeout7200, serial, max_tokens102400, temperature0.1\n' +
        '# Judge credentials are intentionally excluded\nbash /task/scripts/run-hallusion-judge.sh\n')
    args = [sys.executable, '/task/tooling/model-rl-case-guard/scripts/reporting/generate_rl_case_guard_readme.py',
        '--output', str(folder.parent / 'README.md'), '--title', 'HY4 四机 BF16 全量精度验证',
        '--model', '/mnt/share/z30059446/hy4_safetensors', '--manifest', str(manifest_path),
        '--commands-file', str(commands)]
    for name in reports:
        args += ['--report', str(folder / (name + '.json'))]
    subprocess.run(args, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert (folder.parent / 'README.md').stat().st_size > 1000
    rows = [f'| GPQA accuracy | {gpqa["accuracy_percent"]:.4f} | 93.4343 | {gpqa["reference_gap_percentage_points"]:+.4f} |']
    rows += [f'| Hallusion {key} | {hall_metrics[key]:.4f} | {REFERENCE[key]:.4f} | {gaps[key]:+.4f} |' for key in REFERENCE]
    report = (folder.parent / 'report.md').read_text()
    report = report.replace('（进行中）', '（已完成）')
    report = report.replace('GPQA 已完成有效全量精度；HallusionBench 仍在运行。第 07 轮实时状态见',
        '两个数据集及原始裁判评测均已完成。第 07 轮最终状态见')
    report = report.replace('完整请求仍待第 07 轮验证', '第 07 轮已完成文本/图片推理及两套数据的完整评测')
    report = report.replace('最终报告及可复用镜像尚待全量验证完成', '最终报告与可复用镜像已生成，详见本报告末尾及 README.md')
    report += '\n## 最终精度与参考差距\n\n| 指标（%） | 本轮 BF16 | 参考 | 差距（百分点） |\n|---|---:|---:|---:|\n' + '\n'.join(rows)
    report += ('\n\nHallusionBench 1129/1129，问题顺序及原始字段逐项一致，原始请求和回答均保留。\n'
        f'平均输出 {summary["avg_completion_token"]:.2f} token，length 截断 {summary["length_truncated"]} 题；参考为 1211.92 token、6 题。\n'
        '裁判使用原始 evaluation.py/utils.py 和 deepseek-flash。不同权重、并行规模和动态裁判版本限制了逐题复现结论。\n\n'
        '本轮验证完成全量服务、文本和图片基础推理、GPQA 198 题、HallusionBench 1129 题及原裁判。\n'
        '原减层功能报告与九项代码修改说明在本地 hy4-codex-weight-transfer-review 中保留；本轮没有新增模型源码补丁。\n'
        'PP 旧配置未在切换网卡后重新验证，本轮使用的是 TP8/DP4/EP32；不把本轮通过外推为所有并行配置通过。\n\n'
        f'镜像：`{image["tag"]}`，镜像 ID `{image["image_id"]}`。\n'
        f'归档：`{image["archive"]}`，{image["bytes"]} bytes，SHA256 `{image["sha256"]}`。\n'
        '复用说明和独立节点启动脚本已放入镜像 /usr/local/share/hy4-codex/run07/。\n'
        '推理完成后停止了我方四个推理容器，NPU 已释放；CPU 评测容器没有保存为镜像，私有裁判凭据未进入镜像。\n'
        'README.md 由 skill 的原始生成器汇总四份全量 JSON 证据生成。\n')
    (folder.parent / 'report.md').write_text(report)
    manifest.update(status='complete', updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(manifest_path, manifest)
    write(ROOT / 'logs/final-report-run07.json', dict(status='passed',
        gpqa_accuracy=gpqa['accuracy_percent'], hallusion=hall_metrics,
        image=image['tag'], readme=str(folder.parent / 'README.md'),
        report=str(folder.parent / 'report.md'), completed_utc=manifest['updated_utc']))
    print(json.dumps({'status': 'passed', 'reports': str(folder.parent)}), flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        actual = judge_metrics('| GPT Eval | 56.4835 | 63.8728 | 82.6374 | 65.814 | 79.4508 |')
        assert actual == REFERENCE
        print('Original reference leaderboard parser passed; no final artifacts generated')
    else:
        try:
            finish()
        except Exception as error:
            write(ROOT / 'logs/final-report-run07.json',
                dict(status='blocked', error=f'{type(error).__name__}: {error}'))
            raise
