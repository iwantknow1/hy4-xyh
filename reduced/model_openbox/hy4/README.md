> 交付筛选说明：本文件由原 skill 生成的 README 保留全部 23 项成功证据后整理；被阻塞/未运行项的原始报告不上传，状态边界见仓库根 README 和 docs/PROCESS.md。

# HY4 A5 减层版逐项验证证据

> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。

## Summary

- **Generated:** 2026-10-27T14:12:31+00:00
- **Model / checkpoint:** `HY4 2L / 6L / 2L+MTP`
- **Result:** **23/23** report(s) passed

| Feature | Status | Evidence |
|---|---|---|
| `2L TP1 HCCL 图模式 packed，持续通信器 3 轮` | ✅ **passed** | [`cases/hccl-graph.json`](cases/hccl-graph.json) |
| `2L TP1 HCCL 图模式，重新初始化通信器 1` | ✅ **passed** | [`cases/hccl-reinit-1.json`](cases/hccl-reinit-1.json) |
| `2L TP1 HCCL 图模式，重新初始化通信器 2` | ✅ **passed** | [`cases/hccl-reinit-2.json`](cases/hccl-reinit-2.json) |
| `2L TP1 HCCL 图模式，重新初始化通信器 3` | ✅ **passed** | [`cases/hccl-reinit-3.json`](cases/hccl-reinit-3.json) |
| `2L TP1 HCCL 图模式 unpacked` | ✅ **passed** | [`cases/hccl-unpacked-staged.json`](cases/hccl-unpacked-staged.json) |
| `2L TP2 IPC 图模式 packed` | ✅ **passed** | [`cases/ipc-graph.json`](cases/ipc-graph.json) |
| `2L TP2 checkpoint 图模式，连续 3 轮` | ✅ **passed** | [`cases/checkpoint-graph.json`](cases/checkpoint-graph.json) |
| `2L TP2 sink/expert 权重 A-B-A` | ✅ **passed** | [`cases/variants-graph.json`](cases/variants-graph.json) |
| `6L TP2 HCCL eager，缓存修复后` | ✅ **passed** | [`cases/hccl-6l-eager-cachefix.json`](cases/hccl-6l-eager-cachefix.json) |
| `6L TP2 HCCL 图模式，最终修复代码` | ✅ **passed** | [`cases/hccl-6l-graph-cachefix.json`](cases/hccl-6l-graph-cachefix.json) |
| `6L TP2 checkpoint 图模式，连续 3 轮` | ✅ **passed** | [`cases/checkpoint-6l-graph.json`](cases/checkpoint-6l-graph.json) |
| `6L TP4 IPC 图模式 packed` | ✅ **passed** | [`cases/ipc-6l-tp4-graph.json`](cases/ipc-6l-tp4-graph.json) |
| `feature-baseline` | ✅ **passed** | [`cases/feature-baseline.json`](cases/feature-baseline.json) |
| `feature-sp` | ✅ **passed** | [`cases/feature-sp.json`](cases/feature-sp.json) |
| `feature-pp` | ✅ **passed** | [`cases/feature-pp.json`](cases/feature-pp.json) |
| `feature-dp` | ✅ **passed** | [`cases/feature-dp.json`](cases/feature-dp.json) |
| `DP2 TP2 concurrent requests` | ✅ **passed** | [`cases/feature-dp-concurrent.json`](cases/feature-dp-concurrent.json) |
| `feature-r3` | ✅ **passed** | [`cases/feature-r3.json`](cases/feature-r3.json) |
| `R3 in-process` | ✅ **passed** | [`cases/feature-r3-inproc.json`](cases/feature-r3-inproc.json) |
| `feature-mtp` | ✅ **passed** | [`cases/feature-mtp.json`](cases/feature-mtp.json) |
| `pause_resume` | ✅ **passed** | [`cases/pause-consistency.json`](cases/pause-consistency.json) |
| `profiling` | ✅ **passed** | [`cases/profile-final.json`](cases/profile-final.json) |
| `hardware_efficiency_excel` | ✅ **passed** | [`cases/hardware-efficiency-export.json`](cases/hardware-efficiency-export.json) |

## Baseline 特殊配置

用户答复：待补充；配置版本：``。

以下公共配置应由后续特性继承；实际继承情况以各特性的命令和生效证据为准。

```json
{
  "source": "inherited authorized baseline",
  "tp": 2,
  "ep": true,
  "dp": 1,
  "pp": 1,
  "rl_config": {
    "enabled": true
  },
  "graph": "FULL_DECODE_ONLY",
  "capture_sizes": [
    1,
    2,
    4,
    8
  ],
  "kv_bytes_per_worker": 8589934592,
  "max_model_len": 2048,
  "max_num_seqs": 8,
  "memory_utilization_argument": 0.85,
  "note": "Fixed KV overrides automatic utilization-based KV sizing. Profiling independently prechecked with max_model_len=36864, max_num_seqs=1, capture_sizes=[1]."
}
```

## Commands

The following commands were recorded in `logs/launch_commands.txt`:

```bash
# /codex/scripts/start-hccl-6l-eager.sh
set -e
source /codex/scripts/env.sh
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-6L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --weight-transfer-config '{"backend":"hccl"}' --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --enforce-eager --port 18080


# /codex/scripts/resume-run_hccl_6l_cachefix.sh
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_NPU_SOCKET_PORT_RANGE=62200-62299
export HCCL_HOST_SOCKET_PORT_RANGE=62200-62299
for attempt in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
    sleep 2
done
exec .venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py \
    --server-url http://127.0.0.1:18080 --served-model-name hy4-codex \
    --checkpoint /models/Hy4-6L --device 0 --master-address 127.0.0.1 \
    --master-port 29635 --prompts-json /codex/scripts/prompts.json \
    --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-6l-eager-cachefix.json


# /codex/scripts/start-hccl-6l-graph.sh
set -e
source /codex/scripts/env.sh
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-6L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --weight-transfer-config '{"backend":"hccl"}' --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18080


# /codex/scripts/resume-run_hccl_6l_graph_cachefix.sh
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_NPU_SOCKET_PORT_RANGE=62200-62299
export HCCL_HOST_SOCKET_PORT_RANGE=62200-62299
for attempt in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
    sleep 2
done
exec .venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py \
    --server-url http://127.0.0.1:18080 --served-model-name hy4-codex \
    --checkpoint /models/Hy4-6L --device 0 --master-address 127.0.0.1 \
    --master-port 29636 --prompts-json /codex/scripts/prompts.json \
    --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-6l-graph-cachefix.json


# /codex/scripts/start-ipc-6l-tp4-graph.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-6L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 4 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --weight-transfer-config '{"backend":"npu_ipc"}' --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18080


# /codex/scripts/resume-run_ipc_6l_tp4_graph.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
for attempt in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:18080/v1/models > /dev/null; then break; fi
  sleep 2
done
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/skill-scripts/weight-update/test_npu_ipc_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-6L --devices 0,1,2,3 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --packed-buffer-mb 12288 --timeout 600 --report /codex/logs/resume-exclusive/ipc-6l-tp4-graph.json


# /codex/scripts/start-feature-baseline.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18081


# /codex/scripts/start-feature-sp.sh
set -e
source /codex/scripts/env.sh
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18081


# /codex/scripts/start-feature-pp.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18081


# /codex/scripts/start-feature-tp1-baseline.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_NPU_SOCKET_PORT_RANGE=62500-62599
export HCCL_HOST_SOCKET_PORT_RANGE=62500-62599
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 1 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18082


# /codex/scripts/start-feature-dp.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_NPU_SOCKET_PORT_RANGE=62400-62499
export HCCL_HOST_SOCKET_PORT_RANGE=62400-62499
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 --data-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18082


# /codex/scripts/start-feature-r3.sh
set -e
source /codex/scripts/env.sh
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /models/Hy4-2L \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-return-routed-experts --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18081


# /codex/scripts/run_r3_inproc.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
export HCCL_HOST_SOCKET_PORT_RANGE=62300-62399
exec .venv/bin/python /codex/scripts/r3_inproc.py


# /codex/scripts/start-feature-mtp-baseline.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_NPU_SOCKET_PORT_RANGE=62600-62699
export HCCL_HOST_SOCKET_PORT_RANGE=62600-62699
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /codex/models/Hy4-2L-MTP \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18082


# /codex/scripts/start-feature-mtp.sh
set -e
source /codex/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_NPU_SOCKET_PORT_RANGE=62600-62699
export HCCL_HOST_SOCKET_PORT_RANGE=62600-62699
exec .venv/bin/python -m vllm.entrypoints.cli.main serve /codex/models/Hy4-2L-MTP \
  --served-model-name hy4-codex --trust-remote-code --tensor-parallel-size 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --enable-sleep-mode --enable-expert-parallel --load-format auto --gpu-memory-utilization 0.85 \
  --additional-config '{"rl_config":{"enabled":true}}' \
  --kv-cache-memory-bytes 8589934592 --max-model-len 2048 --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' --port 18082


# /codex/scripts/resume-check_pause_consistency.py
import json
from pathlib import Path
import subprocess
import sys
sys.path[:0] = ['/codex/scripts/skill-scripts/weight-update/common', '/codex/scripts/skill-scripts']
from weight_update_validation import WeightUpdateValidation
report={'feature':'pause_resume', 'checks':{}}
validation=WeightUpdateValidation('http://127.0.0.1:18082','hy4-codex',['What is 2 + 2?','Write one word about the sky.'],16,600,report)
validation.record_baseline()
subprocess.run([sys.executable,'/codex/scripts/skill-scripts/lifecycle/test_pause_resume.py','--server-url','http://127.0.0.1:18082','--model','hy4-codex','--max-tokens','16','--report','/codex/logs/resume-exclusive/pause-domain.json'],check=True)
validation.verify_updated_outputs()
report['status']='passed'
Path('/codex/logs/resume-exclusive/pause-consistency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS pause/resume dual-endpoint exact consistency')


# Profiling exact argv and environment
{
  "command": [
    "/codex/.venv/bin/python",
    "/codex/tooling/skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py",
    "--host",
    "177.8.21.1",
    "--connection-mode",
    "docker",
    "--container",
    "x00951564-hy4-codex-oldhost-features",
    "--key-file",
    "/codex/tools/profile-ssh/id_ed25519",
    "--cann-dir",
    "/usr/local/Ascend",
    "--model",
    "/models/Hy4-2L",
    "--served-model-name",
    "hy4-codex",
    "--tp",
    "2",
    "--dp",
    "1",
    "--enable-expert-parallel",
    "--devices",
    "2,3",
    "--mode",
    "full_decode_only",
    "--tag",
    "hy4-2l-middle",
    "--profile-phase",
    "middle",
    "--random-input-len",
    "32768",
    "--random-output-len",
    "4096",
    "--profile-steps",
    "10",
    "--ignore-frontend",
    "--workload-kind",
    "multimodal",
    "--random-range-ratio",
    "0",
    "--mm-input-token-overhead",
    "95",
    "--image-height",
    "256",
    "--image-width",
    "256",
    "--num-prompts",
    "1",
    "--max-concurrency",
    "1",
    "--max-num-seqs",
    "1",
    "--max-model-len",
    "36864",
    "--gpu-memory-utilization",
    "0.85",
    "--api-server-count",
    "1",
    "--torch-profiler-dir",
    "/codex/profiles/hy4-2l-middle",
    "--analyse-all",
    "--serve-args",
    "--trust-remote-code --load-format auto --enable-sleep-mode --kv-cache-memory-bytes 8589934592 --additional-config '{\"rl_config\":{\"enabled\":true}}' --compilation-config '{\"cudagraph_capture_sizes\":[1]}'",
    "--extra-env",
    "PATH=/codex/.venv/bin:/codex/tools:/usr/local/Ascend/cann-9.1.0/bin:/usr/local/Ascend/cann-9.1.0/tools/ccec_compiler/bin:/usr/local/Ascend/cann-9.1.0/tools/profiler/bin:/usr/local/Ascend/cann-9.1.0/tools/ascend_system_advisor/asys:/usr/local/Ascend/cann-9.1.0/tools/show_kernel_debug_data:/usr/local/Ascend/cann-9.1.0/tools/msobjdump:/root/.cargo/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.12.13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "--extra-env",
    "LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:",
    "--extra-env",
    "VLLM_USE_V2_MODEL_RUNNER=0",
    "--extra-env",
    "HCCL_NPU_SOCKET_PORT_RANGE=62700-62799",
    "--extra-env",
    "HCCL_HOST_SOCKET_PORT_RANGE=62700-62799"
  ],
  "extra_env": {
    "PATH": "/codex/.venv/bin:/codex/tools:/usr/local/Ascend/cann-9.1.0/bin:/usr/local/Ascend/cann-9.1.0/tools/ccec_compiler/bin:/usr/local/Ascend/cann-9.1.0/tools/profiler/bin:/usr/local/Ascend/cann-9.1.0/tools/ascend_system_advisor/asys:/usr/local/Ascend/cann-9.1.0/tools/show_kernel_debug_data:/usr/local/Ascend/cann-9.1.0/tools/msobjdump:/root/.cargo/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.12.13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "HCCL_NPU_SOCKET_PORT_RANGE": "62700-62799",
    "HCCL_HOST_SOCKET_PORT_RANGE": "62700-62799"
  },
  "sampling": "not passed; preserve benchmark/runtime defaults",
  "validation_scope": "functional_smoke",
  "capacity_precheck": "TP2 2L live weights 14.19 GiB/card; retained weight buffers approximately 29 GiB, fixed KV 8 GiB; allow 24 GiB for larger prefill/vision/workspace and 4 GiB runtime, total approximately 65 GiB/card below 91.75 GiB. KV8GiB observed token capacity much larger than requested 36864 context. No concurrent NPU task on 2/3. Stop and preserve report if initialization OOM occurs."
}
```

## Feature evidence

### 1. ✅ 2L TP1 HCCL 图模式 packed，持续通信器 3 轮 — passed

Evidence file: [`cases/hccl-graph.json`](cases/hccl-graph.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_persistent.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 --master-port 29577 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-graph.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 2. ✅ 2L TP1 HCCL 图模式，重新初始化通信器 1 — passed

Evidence file: [`cases/hccl-reinit-1.json`](cases/hccl-reinit-1.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 --master-port 29611 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-reinit-1.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 3. ✅ 2L TP1 HCCL 图模式，重新初始化通信器 2 — passed

Evidence file: [`cases/hccl-reinit-2.json`](cases/hccl-reinit-2.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 --master-port 29612 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-reinit-2.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 4. ✅ 2L TP1 HCCL 图模式，重新初始化通信器 3 — passed

Evidence file: [`cases/hccl-reinit-3.json`](cases/hccl-reinit-3.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 --master-port 29613 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-reinit-3.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 5. ✅ 2L TP1 HCCL 图模式 unpacked — passed

Evidence file: [`cases/hccl-unpacked-staged.json`](cases/hccl-unpacked-staged.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --device 0 --master-address 127.0.0.1 --master-port 29573 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-unpacked-staged.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 6. ✅ 2L TP2 IPC 图模式 packed — passed

Evidence file: [`cases/ipc-graph.json`](cases/ipc-graph.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_consistent` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-2L","format":"safetensors","file_count":4,"total_file_bytes":26629205932,"tensor_count":709,"total_tensor_bytes":26629125812,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"d57c5134b925b670942d630539174d29b5d5f7a0ac4f1fea88f4b877bb63c188"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_npu_ipc_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-2L --devices 0,1 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --packed-buffer-mb 12288 --timeout 600 --report /codex/logs/resume-exclusive/ipc-graph.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 7. ✅ 2L TP2 checkpoint 图模式，连续 3 轮 — passed

Evidence file: [`cases/checkpoint-graph.json`](cases/checkpoint-graph.json)

#### Checks

| Check | Result |
|---|---|
| `round_1_identical` | ✅ pass |
| `round_2_identical` | ✅ pass |
| `round_3_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 8. ✅ 2L TP2 sink/expert 权重 A-B-A — passed

Evidence file: [`cases/variants-graph.json`](cases/variants-graph.json)

#### Checks

| Check | Result |
|---|---|
| `sink-B_restored_identical` | ✅ pass |
| `expert-C_restored_identical` | ✅ pass |

### 9. ✅ 6L TP2 HCCL eager，缓存修复后 — passed

Evidence file: [`cases/hccl-6l-eager-cachefix.json`](cases/hccl-6l-eager-cachefix.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-6L","format":"safetensors","file_count":12,"total_file_bytes":106403824884,"tensor_count":810,"total_tensor_bytes":106403732980,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"93e8d554c12a57ca6c39dc8b372a53ced4de8fc060daae57e33129e82288c970"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-6L --device 0 --master-address 127.0.0.1 --master-port 29635 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-6l-eager-cachefix.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} |
| /v1/completions · Write one word about the sky. | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} |

### 10. ✅ 6L TP2 HCCL 图模式，最终修复代码 — passed

Evidence file: [`cases/hccl-6l-graph-cachefix.json`](cases/hccl-6l-graph-cachefix.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-6L","format":"safetensors","file_count":12,"total_file_bytes":106403824884,"tensor_count":810,"total_tensor_bytes":106403732980,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"93e8d554c12a57ca6c39dc8b372a53ced4de8fc060daae57e33129e82288c970"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_hccl_packed_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-6L --device 0 --master-address 127.0.0.1 --master-port 29636 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --timeout 600 --report /codex/logs/resume-exclusive/hccl-6l-graph-cachefix.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} |
| /v1/completions · Write one word about the sky. | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} |

### 11. ✅ 6L TP2 checkpoint 图模式，连续 3 轮 — passed

Evidence file: [`cases/checkpoint-6l-graph.json`](cases/checkpoint-6l-graph.json)

#### Checks

| Check | Result |
|---|---|
| `round_1_identical` | ✅ pass |
| `round_2_identical` | ✅ pass |
| `round_3_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857],"token_logprobs":[-5.349900245666504,-0.3354434370994568,-3.1330790519714355,-4.652867794036865,-1.1973342895507812,-3.7670233249664307,-4.7741498947143555,-0.05379164591431618,-1.923743724822998,-3.4571495056152344,-3.4921393394470215,-3.2156999111175537,-3.500084400177002,-0.0117262601852417,-4.4454121589660645,-2.2689883708953857]} |
| /v1/completions · prompt-2 | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525],"token_logprobs":[-5.372629165649414,-0.3877597749233246,-0.01635667122900486,-4.413080215454102,-4.234133720397949,-0.17205612361431122,-3.445448875427246,-5.019546031951904,-4.264036655426025,-3.036745548248291,-0.14294523000717163,-0.9140886664390564,-4.179297924041748,-5.162504196166992,-3.692188024520874,-3.4355223178863525]} |
| /v1/chat/completions · prompt-1 | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996],"token_logprobs":[-4.060488224029541,-2.4237818717956543,-1.6350581645965576,-1.4102312326431274,-0.515644371509552,-0.2154325246810913,-2.496633529663086,-3.169853687286377,-4.115867614746094,-3.9432311058044434,-3.0497570037841797,-3.817552328109741,-3.674609899520874,-0.3134908378124237,-4.4170427322387695,-0.5978102684020996]} |
| /v1/chat/completions · prompt-2 | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231],"token_logprobs":[-3.989959955215454,-2.5838241577148438,-2.9352633953094482,-4.458689212799072,-4.792933464050293,-5.1154985427856445,-1.840580940246582,-4.915597915649414,-2.0041909217834473,-4.869478702545166,-3.470384120941162,-4.696456432342529,-4.5472869873046875,-3.5314228534698486,-5.0434417724609375,-0.6057770848274231]} |

### 12. ✅ 6L TP4 IPC 图模式 packed — passed

Evidence file: [`cases/ipc-6l-tp4-graph.json`](cases/ipc-6l-tp4-graph.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_consistent` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/models/Hy4-6L","format":"safetensors","file_count":12,"total_file_bytes":106403824884,"tensor_count":810,"total_tensor_bytes":106403732980,"max_tensor_name":"model.layers.1.mlp.experts.gate_up_proj","max_tensor_bytes":12884901888,"manifest_sha256":"93e8d554c12a57ca6c39dc8b372a53ced4de8fc060daae57e33129e82288c970"}`

<details>
<summary>Verification command</summary>

```text
/codex/.venv/bin/python scripts/skill-scripts/weight-update/test_npu_ipc_cpu_stage.py --server-url http://127.0.0.1:18080 --served-model-name hy4-codex --checkpoint /models/Hy4-6L --devices 0,1,2,3 --prompts-json /codex/scripts/prompts.json --max-tokens 16 --packed-buffer-mb 12288 --timeout 600 --report /codex/logs/resume-exclusive/ipc-6l-tp4-graph.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.360567092895508,-0.3389008045196533,-3.1034607887268066,-4.827348232269287,-1.1965020895004272,-3.7663943767547607,-4.772518157958984,-0.05083649605512619,-1.906234860420227,-3.4789764881134033,-3.45255970954895,-3.138021230697632,-3.4823691844940186,-0.01160348393023014,-4.4107584953308105,-2.178901433944702]} | {"text":"notifycationicMirrorrorsrorsystematicsaticsityyyy恥辱 foldings","tokens":["notify","cation","ic","Mirror","rors","rors","ystem","atics","atics","ity","yyy","","恥","辱"," fold","ings"],"token_logprobs":[-5.360567092895508,-0.3389008045196533,-3.1034607887268066,-4.827348232269287,-1.1965020895004272,-3.7663943767547607,-4.772518157958984,-0.05083649605512619,-1.906234860420227,-3.4789764881134033,-3.45255970954895,-3.138021230697632,-3.4823691844940186,-0.01160348393023014,-4.4107584953308105,-2.178901433944702]} |
| /v1/completions · Write one word about the sky. | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.375425815582275,-0.38397422432899475,-0.015736328437924385,-4.546814918518066,-4.085587978363037,-0.17184263467788696,-3.459044933319092,-4.9710283279418945,-4.235241889953613,-2.9630022048950195,-0.14097866415977478,-0.9365131258964539,-4.218170642852783,-4.9072418212890625,-3.6625869274139404,-3.3739211559295654]} | {"text":" unintentionalitycroftballooning正视版的的确确凿无误 indexedaurusmith","tokens":[" unint","entional","ity","croft","ball","oon","ing","正视","版的","的确","确","凿","无误"," indexed","aurus","mith"],"token_logprobs":[-5.375425815582275,-0.38397422432899475,-0.015736328437924385,-4.546814918518066,-4.085587978363037,-0.17184263467788696,-3.459044933319092,-4.9710283279418945,-4.235241889953613,-2.9630022048950195,-0.14097866415977478,-0.9365131258964539,-4.218170642852783,-4.9072418212890625,-3.6625869274139404,-3.3739211559295654]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.027641773223877,-2.415673017501831,-1.642442226409912,-1.3981924057006836,-0.5214635133743286,-0.22058385610580444,-2.4921875,-3.1606853008270264,-4.1022820472717285,-3.966041326522827,-3.050283670425415,-3.8000874519348145,-3.705132246017456,-0.2910059988498688,-4.410585403442383,-0.5954542756080627]} | {"text":"letsgooseberryhillockscreenwriters strikeoutcomes-orientedtowards honourably","tokens":["lets","go","ose","berry","hill","ocks","creen","writers"," strike","out","comes","-oriented","tow","ards"," honour","ably"],"token_logprobs":[-4.027641773223877,-2.415673017501831,-1.642442226409912,-1.3981924057006836,-0.5214635133743286,-0.22058385610580444,-2.4921875,-3.1606853008270264,-4.1022820472717285,-3.966041326522827,-3.050283670425415,-3.8000874519348145,-3.705132246017456,-0.2910059988498688,-4.410585403442383,-0.5954542756080627]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-4.017858028411865,-2.596029043197632,-2.920264482498169,-4.415066719055176,-4.810184478759766,-5.1250319480896,-1.8196038007736206,-4.872066497802734,-2.3682165145874023,-4.954002857208252,-3.4595391750335693,-4.6801066398620605,-4.496678829193115,-3.4543418884277344,-5.0495781898498535,-0.6087235808372498]} | {"text":"lets go aheadquartersboro sheddershayyyy恷澈的默认值的都比不上","tokens":["lets"," go"," ahead","quarters","boro"," shed","ders","hay","yyy","","恷","澈","的默认","值的","都比","不上"],"token_logprobs":[-4.017858028411865,-2.596029043197632,-2.920264482498169,-4.415066719055176,-4.810184478759766,-5.1250319480896,-1.8196038007736206,-4.872066497802734,-2.3682165145874023,-4.954002857208252,-3.4595391750335693,-4.6801066398620605,-4.496678829193115,-3.4543418884277344,-5.0495781898498535,-0.6087235808372498]} |

### 13. ✅ feature-baseline — passed

Evidence file: [`cases/feature-baseline.json`](cases/feature-baseline.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | <missing> | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | <missing> | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · prompt-1 | <missing> | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | <missing> | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 14. ✅ feature-sp — passed

Evidence file: [`cases/feature-sp.json`](cases/feature-sp.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 15. ✅ feature-pp — passed

Evidence file: [`cases/feature-pp.json`](cases/feature-pp.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.070127487182617,-1.5722265243530273,-4.242175579071045,-3.606135606765747,-4.416794776916504,-4.800335884094238,-4.657891750335693,-4.764471530914307,-5.340261459350586,-0.7771568894386292,-4.442620754241943,-1.6580249071121216,-4.699674606323242,-3.9361395835876465,-4.422239780426025,-3.370418071746826]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} | {"text":"的爱德华的金本位 Giggle的小伙伴虎啸alphabetically built-in玩的儿","tokens":["的爱","德华","的金","本位"," Gig","gle","的小伙伴","虎","啸","alph","abet","ically"," built","-in","玩的","儿"],"token_logprobs":[-6.840651512145996,-3.2745845317840576,-4.229675769805908,-4.238421440124512,-5.353860378265381,-0.1370505392551422,-5.558790683746338,-4.965434551239014,-1.6289533376693726,-4.665212631225586,-4.096260070800781,-0.6598102450370789,-3.970775842666626,-1.4473118782043457,-4.688422679901123,-1.60090970993042]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.3241801261901855,-4.815529823303223,-3.673537254333496,-0.34990033507347107,-4.1290693283081055,-4.255687236785889,-2.5022318363189697,-4.965063571929932,-5.157588005065918,-4.341821193695068,-1.8265459537506104,-2.95933198928833,-3.7309436798095703,-4.65114164352417,-4.864499092102051,-3.900848865509033]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.392412185668945,-4.930185794830322,-3.748257875442505,-0.48519495129585266,-4.172966480255127,-4.352108955383301,-2.5683841705322266,-5.01535177230835,-5.141754150390625,-4.421401023864746,-2.053954839706421,-2.967550039291382,-3.516770839691162,-2.491333484649658,-4.227342128753662,-4.2488884925842285]} |

### 16. ✅ feature-dp — passed

Evidence file: [`cases/feature-dp.json`](cases/feature-dp.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399602890014648,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 17. ✅ DP2 TP2 concurrent requests — passed

Evidence file: [`cases/feature-dp-concurrent.json`](cases/feature-dp-concurrent.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Outputs

- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5783131122589111,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.821530818939209,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5783131122589111,-4.223082065582275,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.821530818939209,-3.6681792736053467,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5783131122589111,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.821530818939209,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5783131122589111,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.821530818939209,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.223082065582275,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6681792736053467,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.223082065582275,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6681792736053467,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}
- {"/v1/completions":{"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.0573296546936035,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]},"/v1/chat/completions":{"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.333907604217529,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]}}

### 18. ✅ feature-r3 — passed

Evidence file: [`cases/feature-r3.json`](cases/feature-r3.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920953750610352,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.338519096374512,-0.7755117416381836,-4.444377422332764,-1.6715059280395508,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.289910316467285,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947864532470703,-4.307109355926514,-3.3007726669311523,-4.597614288330078,-5.000556945800781,-3.9997692108154297,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711971282959,-1.7173662185668945]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806966781616211,-3.6511220932006836,-0.35462284088134766,-4.090996742248535,-4.237446308135986,-2.517526626586914,-4.9553985595703125,-5.160783290863037,-4.334925651550293,-1.8169002532958984,-2.975086212158203,-3.7238917350769043,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552242279052734,-0.4847421646118164,-4.173011302947998,-4.360525131225586,-2.5592174530029297,-5.039780139923096,-5.149765491485596,-4.429548740386963,-2.068330764770508,-2.9887943267822266,-3.5550270080566406,-2.4803695678710938,-4.2264299392700195,-4.247284889221191]} |

### 19. ✅ R3 in-process — passed

Evidence file: [`cases/feature-r3-inproc.json`](cases/feature-r3-inproc.json)

#### Outputs

- {"prompt":"What is 2 + 2?","text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","token_ids":[59131,1771,51145,29059,21484,5909,73808,59513,4235,58578,6669,23293,11904,81044,39297,24159],"token_logprobs":[-5.059730052947998,-1.5920953750610352,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.338519096374512,-0.7755117416381836,-4.444377422332764,-1.6715059280395508,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545],"route_shape":[23,2,8],"route_dtype":"uint8","route_min":0,"route_max":254}
- {"prompt":"Write one word about the sky.","text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","token_ids":[46044,102810,92968,67075,18085,17234,87081,105795,102571,104364,28247,29939,84850,18206,113043,9296],"token_logprobs":[-6.846774578094482,-3.289910316467285,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947864532470703,-4.307109355926514,-3.3007726669311523,-4.597614288330078,-5.000556945800781,-3.9997692108154297,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711971282959,-1.7173662185668945],"route_shape":[22,2,8],"route_dtype":"uint8","route_min":0,"route_max":252}

### 20. ✅ feature-mtp — passed

Evidence file: [`cases/feature-mtp.json`](cases/feature-mtp.json)

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · prompt-2 | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · prompt-1 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · prompt-2 | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 21. ✅ pause_resume — passed

Evidence file: [`cases/pause-consistency.json`](cases/pause-consistency.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · What is 2 + 2? | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} | {"text":"/materialized了的的气itosisesontab极了 excursion镜头的-E类推 scaled-down","tokens":["/material","ized","了的","的气","itos","ises","ontab","极了"," exc","ursion","镜","头的","-E","类推"," scaled","-down"],"token_logprobs":[-5.059730052947998,-1.5920952558517456,-4.227444171905518,-3.6037020683288574,-4.425836563110352,-4.775407314300537,-4.6655755043029785,-4.772834300994873,-5.3385186195373535,-0.7755116820335388,-4.444377422332764,-1.6715060472488403,-4.696792125701904,-3.9438037872314453,-4.445778846740723,-3.375354290008545]} |
| /v1/completions · Write one word about the sky. | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} | {"text":"的爱德华的金沙滩浴盆的改变之城之夜的人身bacilluslld肃的那样子的","tokens":["的爱","德华","的金","沙滩","浴","盆","的改变","之城","之夜","的人身","bac","illus","lld","肃","的那样","子的"],"token_logprobs":[-6.846774578094482,-3.2899105548858643,-4.2017621994018555,-4.2513909339904785,-4.945526123046875,-3.7947866916656494,-4.307109355926514,-3.3007724285125732,-4.597614288330078,-5.000556945800781,-3.9997689723968506,-3.9904255867004395,-4.994872093200684,-3.6239490509033203,-5.269711494445801,-1.7173665761947632]} |
| /v1/chat/completions · What is 2 + 2? | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktot在不断超限itare规矩","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","在不断","超限","itare","规矩"],"token_logprobs":[-5.339831829071045,-4.806967258453369,-3.6511223316192627,-0.3546227514743805,-4.090996742248535,-4.237446308135986,-2.517526388168335,-4.955398082733154,-5.160783290863037,-4.334926128387451,-1.8169002532958984,-2.9750864505767822,-3.723891496658325,-4.648724555969238,-4.8644208908081055,-3.8868088722229004]} |
| /v1/chat/completions · Write one word about the sky. | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} | {"text":"levensburgundy手腕/armitage申请的的同 fightbacktotallyitarian用人之道","tokens":["lev","ens","burg","undy","手腕","/arm","itage","申请的","的同"," fight","back","tot","ally","itarian","用人","之道"],"token_logprobs":[-5.399302959442139,-4.912580966949463,-3.7552244663238525,-0.4847423732280731,-4.173011779785156,-4.360525131225586,-2.559217929840088,-5.039780139923096,-5.149765491485596,-4.429549217224121,-2.068331003189087,-2.9887948036193848,-3.5550272464752197,-2.480370044708252,-4.2264299392700195,-4.247284889221191]} |

### 22. ✅ profiling — passed

Evidence file: [`cases/profile-final.json`](cases/profile-final.json)

#### Checks

| Check | Result |
|---|---|
| `collector_and_analysis_ok` | ✅ pass |
| `two_ranks` | ✅ pass |
| `ten_graph_executions_each_rank` | ✅ pass |
| `actual_input_32768` | ✅ pass |
| `actual_output_4096` | ✅ pass |
| `all_requests_succeeded` | ✅ pass |
| `real_multimodal_input` | ✅ pass |
| `middle_window_configured` | ✅ pass |
- **Note:** Trace uses MODEL_EXECUTE markers rather than ProfilerStep names. Each rank contains exactly ten graph executions and 2380 kernel rows. Worker logs confirm start after delay and automatic stop at ten iterations. Both decoder layers execute in each PP1 graph; no layer extrapolation is used.

### 23. ✅ hardware_efficiency_excel — passed

Evidence file: [`cases/hardware-efficiency-export.json`](cases/hardware-efficiency-export.json)

#### Checks

| Check | Result |
|---|---|
| `all_kernel_formulas_match_json` | ✅ pass |
| `summary_formulas_match_json` | ✅ pass |
| `editable_parameter_recalculation` | ✅ pass |
| `restored_defaults_match` | ✅ pass |
| `saved_workbook_reopened_and_recalculated` | ✅ pass |
- **Note:** 计算算子耗时加权的经验达成率；未计通信、空闲和重叠，不等同端到端吞吐或加速比。

## Interpretation

- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.
- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.
- Text output is runtime evidence only; it does not prove numerical equivalence.



## 镜像最终实机复验

[镜像自包含验证与完整 ID](IMAGE-VALIDATION.md)：6L TP2 HCCL 图模式，在没有源码挂载的最终镜像中完整复验通过。5–7 卡已释放。镜像元数据和归档 SHA256 另行交付，避免镜像自身 ID 的循环依赖。
