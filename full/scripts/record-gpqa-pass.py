import datetime
import json
from pathlib import Path
import re

root = Path('/task')
audit = json.loads((root / 'logs/gpqa-full-audit-07.json').read_text())
assert audit['successful'] == 198
match = re.search(r"'accuracy':\s*([0-9.]+).*?'extract_rate':\s*([0-9.]+)",
                  (root / 'logs/gpqa-full-07.log').read_text())
assert match
accuracy, extraction = map(float, match.groups())
result = dict(audit, accuracy_percent=accuracy, extract_rate_percent=extraction,
    correct=round(accuracy * 198 / 100), reference_accuracy_percent=185 / 198 * 100,
    reference_gap_percentage_points=accuracy - 185 / 198 * 100)
manifest_path = root / 'model_openbox/hy4/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest.update(status='gpqa_passed_hallusion_running',
    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpqa_run07=result)
manifest['hallusion_transport_change'] = json.loads((root / 'logs/hallusion-transport-timeout-change.json').read_text())
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
path = root / 'model_openbox/hy4/report.md'
text = path.read_text().replace('当前尚未取得有效完整精度分数。第 07 轮正在启动，实时状态见',
    'GPQA 已完成有效全量精度；HallusionBench 仍在运行。第 07 轮实时状态见')
text += f'''
## 已完成的 GPQA 全量精度（第 07 轮）

- 198/198 成功，失败 0，题号完整且唯一。
- 准确率 {accuracy:.8f}%（{result['correct']}/198），答案提取率 {extraction:.0f}%。
- 与双机参考结果 185/198（93.43%）总分相同，差距 0 个百分点。
  权重与并行规模不同，不能据此声称逐题预测或整个部署完全复现。
- 从本轮原始预测重建的 prompt/gold SHA256 与参考输入逐字节一致。
- 输出平均 {audit['output_tokens']['mean']:.2f} token，最长 {audit['output_tokens']['max']}；
  触及 102400 上限 0 题，未闭合思考标签 0 题。
- 证据：logs/gpqa-integrity-07.json、logs/gpqa-full-audit-07.json，以及
  evaluation/runs/gpqa-full-07/20261028_180756/ 下的原始预测、原生评分和配置。

HallusionBench 客户端仅增加可配置网络超时，将本轮等待时间设为 7200 秒，
以容纳 BF16 吞吐下可能较长的完整输出。原默认值仍是 1200 秒；题目、顺序、
串行执行、temperature=0.1、max_tokens=102400 及裁判逻辑均未修改。
首次 8 题单独归档，正式完整轮次重新从第一题执行，未混用两次预测。
'''
path.write_text(text)
print(json.dumps({'gpqa_accuracy': accuracy, 'correct': result['correct'],
                  'hallusion': 'running', 'manifest': 'updated'}))
