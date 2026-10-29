import datetime
import json
from pathlib import Path

root = Path('/task')
manifest_path = root / 'model_openbox/hy4/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['status'] = 'run07_initializing'
manifest['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest['current_run'] = '07'
manifest['current_configuration'] = {
    'tp': 8, 'dp': 4, 'pp': 1, 'ep_size': 32, 'mtp_tokens': 3,
    'max_model_len': 147456, 'max_num_seqs': 64, 'max_num_batched_tokens': 8192,
    'kv_cache_bytes': 15032385536, 'gpu_memory_utilization': 0.95,
    'api_server_count': 1, 'engine_ready_timeout_seconds': 3600,
    'communication_iface': 'enp39s0f0', 'communication_link_gbps': 200,
    'compiler_path': '/usr/local/Ascend/cann-9.1.0/bin',
    'async_scheduling': True, 'prefix_caching': True, 'seed': 0}
manifest['run06_result'] = {
    'full_weights_loaded': True, 'text_smoke': 'passed', 'image_smoke': 'failed',
    'root_error': 'TypeError: function object is not subscriptable in slot mapping',
    'cause': 'Our explicit PATH removed the CANN compiler directory; Triton backend detection failed',
    'fix': 'Restore compiler directory in launcher PATH; no model source change',
    'local_validation': 'All four nodes executed slot mapping on NPU and matched expected slots',
    'full_accuracy': 'not_started'}
manifest['judge_credential'] = {'status': 'restored_and_generation_preflight_passed',
    'original_file_on_28': 'deleted', 'evaluation_private_copy': 'retained',
    'evidence': 'logs/judge-generation-preflight-restored.json'}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
report = '''# HY4 四机全量精度验证（进行中）

当前尚未取得有效完整精度分数。第 07 轮正在启动，实时状态见
`logs/evaluation-pipeline.json`，完整证据保存在独立轮次日志及结果目录。

- 权重：官方 BF16 全量，78 层、246 分片；索引、分片头及视觉权重审计通过。
- 当前配置：TP8 / DP4 / PP1 / EP32，MTP3，最大 64 条序列，模型长度 147456，
  批处理 token 上限 8192，固定 KV 14 GiB，单 API，200 Gb 网卡。
- 数据：参考方同一份 GPQA Diamond 198 题及 HallusionBench 1129 题、346 图片，
  输出上限 102400。权重与参考方量化权重不同，不能声称完全复现参考分数。
- GPQA 完整性门槛：198 条成功、非空、唯一题号且无失败记录；否则不接受分数。
  先前部分请求失败时生成的 20.20% 不属于有效全量精度。
- HallusionBench 使用原请求客户端和原 DeepSeek 裁判脚本，裁判调用已重新预检通过。

## 已定位问题及处理

1. 管理网卡仅 1 Gb：32 进程压力测试出现数据校验失败；实际 200 Gb 网卡上
   相同 2048 轮测试通过，服务通信已切换。不能据此断言所有 Gloo 故障同源。
2. 默认四个 API 进程的 600 秒就绪等待在模型初始化期间超时：改为单 API 和
   3600 秒等待，第 06 轮已完成全量加载和文本推理。
3. 第 06 轮图片请求失败：我方启动 PATH 覆盖了 CANN 编译器路径，Triton
   检测不到后端，slot mapping 内核成为普通函数。启动 PATH 补齐后，四台
   NPU 实际内核执行均通过。未为此增加模型源码补丁，完整请求仍待第 07 轮验证。
4. 第 07 轮初次启动前 27 节点 SSH 探测连接重置，脚本退出；保留日志并重试，
   该次尚未加载模型，不计为模型运行故障。

原减层功能验证与九项源码修改的证据仍保留在旧验证报告中；它们不能替代
本轮全量精度、稳定性或多机结论。最终报告及可复用镜像尚待全量验证完成。
'''
(root / 'model_openbox/hy4/report.md').write_text(report)
print(json.dumps({'status': 'updated', 'run': '07'}))
