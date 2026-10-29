# HY4 四机 BF16 全量精度验证

> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。

## Summary

- **Generated:** 2026-10-29T04:33:54+00:00
- **Model / checkpoint:** `/mnt/share/z30059446/hy4_safetensors`
- **Result:** **4/4** report(s) passed

| Feature | Status | Evidence |
|---|---|---|
| `dataset_evaluation` | ✅ **passed** | [`full-accuracy-07/gpqa.json`](full-accuracy-07/gpqa.json) |
| `dataset_evaluation` | ✅ **passed** | [`full-accuracy-07/hallusion.json`](full-accuracy-07/hallusion.json) |
| `full_model_serving` | ✅ **passed** | [`full-accuracy-07/serving.json`](full-accuracy-07/serving.json) |
| `reusable_inference_image` | ✅ **passed** | [`full-accuracy-07/image.json`](full-accuracy-07/image.json) |

## Baseline 特殊配置

用户答复：待补充；配置版本：`run07-compiler-path`。

以下公共配置应由后续特性继承；实际继承情况以各特性的命令和生效证据为准。

```json
{
  "revision": "run07-compiler-path",
  "tp": 8,
  "dp": 4,
  "pp": 1,
  "ep_size": 32,
  "mtp_tokens": 3,
  "max_model_len": 147456,
  "max_num_seqs": 64,
  "max_num_batched_tokens": 8192,
  "kv_cache_bytes": 15032385536,
  "gpu_memory_utilization": 0.95,
  "api_server_count": 1,
  "engine_ready_timeout_seconds": 3600,
  "communication_iface": "enp39s0f0",
  "communication_link_gbps": 200,
  "compiler_path": "/usr/local/Ascend/cann-9.1.0/bin",
  "async_scheduling": true,
  "prefix_caching": true,
  "seed": 0
}
```

## Commands

The following commands were recorded in `logs/launch_commands.txt`:

```bash
# Actual four-node launcher and environment wrapper
#!/bin/bash
set -euo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/share/z00949185/hy4-work/22.1/aisbench-venv/lib/python3.12/site-packages
exec /task/evaluation/.venv-client/bin/python /task/tooling/MindSpeed-Agent/skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
 --host 197.13.49.1 --key-file /tmp/hy4-ssh/id_ed25519 --container x00951564-hy4-codex-20261028 \
 --worker-hosts 197.13.50.1,197.13.51.1,197.13.52.1 --tp 8 --dp 4 \
 --model /mnt/share/z30059446/hy4_safetensors --served-model-name hy4 \
 --port 18080 --master-port 29740 --health-timeout 3600 --dp-rpc-timeout 300 \
 --wrap-script /task/scripts/serve-with-cann.sh \
 --extra-env PATH=/codex/.venv/bin:/usr/local/Ascend/cann-9.1.0/bin:/usr/local/bin:/usr/bin:/bin \
 --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
 --extra-env VLLM_WORKER_MULTIPROC_METHOD=spawn \
 --extra-env GLOO_SOCKET_IFNAME=enp39s0f0 --extra-env HCCL_SOCKET_IFNAME=enp39s0f0 --extra-env IP_SOCKET_IFNAME=enp39s0f0 \
 --extra-env HCCL_NPU_SOCKET_PORT_RANGE=61000-61999 --extra-env HCCL_HOST_SOCKET_PORT_RANGE=61000-61999 \
 --extra-env HCCL_CONNECT_TIMEOUT=600 --extra-env HCCL_EXEC_TIMEOUT=600 \
 --extra-env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
 --extra-env HCCL_BUFFSIZE=128 --extra-env HCCL_OP_EXPANSION_MODE=AIV \
 --extra-env TOKENIZERS_PARALLELISM=false --extra-env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
 -- --trust-remote-code --distributed-executor-backend mp --enable-expert-parallel \
 --api-server-count 1 \
 --max-model-len 147456 --max-num-seqs 64 --max-num-batched-tokens 8192 \
 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes 15032385536 \
 --enable-prefix-caching --async-scheduling --seed 0 \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_load_config":{}}'

#!/bin/bash
set -euo pipefail
node_ip=${2##*_}
case "$node_ip" in 197.13.49.1|197.13.5[0-2].1) ;; *) echo 'Unexpected node identity in runtime directory' >&2; exit 2;; esac
export VLLM_HOST_IP=$node_ip HCCL_IF_IP=$node_ip
export GLOO_SOCKET_IFNAME=enp39s0f0 HCCL_SOCKET_IFNAME=enp39s0f0 IP_SOCKET_IFNAME=enp39s0f0
exec hy4-entrypoint bash "$1"

# GPQA actual invocation
/task/evaluation/bin/ais_bench /task/evaluation/gpqa-full-config-07.py --mode all --num-prompts 198 --num-warmups 0 --max-num-workers 1 --dump-eval-details --dump-extract-rate --debug
# Full Hallusion invocation is recorded in evaluation/runs/hallusion-full-07/config.json; request timeout7200, serial, max_tokens102400, temperature0.1
# Judge credentials are intentionally excluded
bash /task/scripts/run-hallusion-judge.sh
```

## Feature evidence

### 1. ✅ dataset_evaluation — passed

Evidence file: [`full-accuracy-07/gpqa.json`](full-accuracy-07/gpqa.json)

#### Checks

| Check | Result |
|---|---|
| `all_198_successful` | ✅ pass |
| `exact_reference_inputs` | ✅ pass |
| `unique_ids` | ✅ pass |
| `zero_failed_requests` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ⏭️ not_run |
| `/v1/chat/completions` | ✅ pass |
- **数据集 / 输出长度：** `GPQA Diamond 198` / `102400`

<details>
<summary>评测指标</summary>

```text
{
  "status": "passed",
  "rows": 198,
  "exact_input_sha256": "1705137667c5b9b4e2ad5b3a85a0d94103bad3c303f41aa24609203771a91e2d",
  "raw_prediction_keys": [
    "data_abbr",
    "gold",
    "id",
    "input_tokens",
    "origin_prompt",
    "output_tokens",
    "prediction",
    "success",
    "uuid"
  ],
  "successful": 198,
  "output_tokens": {
    "min": 100,
    "max": 48513,
    "mean": 4311.868686868687,
    "median": 1183.0,
    "sum": 853750
  },
  "output_tokens_at_limit_102400": 0,
  "unclosed_thinking_prediction_ids": [],
  "native_result": "/task/evaluation/runs/gpqa-full-07/20261028_180756/results/hy4-gpu-aligned/GPQA_diamond.json",
  "accuracy_percent": 93.43434343434343,
  "extract_rate_percent": 100.0,
  "correct": 185,
  "reference_accuracy_percent": 93.43434343434343,
  "reference_gap_percentage_points": 0.0
}
```

</details>
- **原始评测结果：** [../../evaluation/runs/gpqa-full-07/20261028_180756/results/hy4-gpu-aligned/GPQA_diamond.json](../../evaluation/runs/gpqa-full-07/20261028_180756/results/hy4-gpu-aligned/GPQA_diamond.json)
- **Log:** `/task/logs/gpqa-full-07.log`
- **Note:** Real full BF16 checkpoint; reference uses different quantized weights.

### 2. ✅ dataset_evaluation — passed

Evidence file: [`full-accuracy-07/hallusion.json`](full-accuracy-07/hallusion.json)

#### Checks

| Check | Result |
|---|---|
| `all_1129_nonempty` | ✅ pass |
| `exact_dataset_and_order` | ✅ pass |
| `original_judge_completed` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ⏭️ not_run |
| `/v1/chat/completions` | ✅ pass |
- **数据集 / 输出长度：** `HallusionBench 1129 questions / 346 images` / `102400`

<details>
<summary>评测指标</summary>

```text
{
  "actual": {
    "qAcc": 54.9451,
    "fAcc": 62.4277,
    "easy_aAcc": 82.6374,
    "hard_aAcc": 65.5814,
    "aAcc": 79.1851
  },
  "reference": {
    "qAcc": 56.4835,
    "fAcc": 63.8728,
    "easy_aAcc": 82.6374,
    "hard_aAcc": 65.814,
    "aAcc": 79.4508
  },
  "gap_percentage_points": {
    "qAcc": -1.5384000000000029,
    "fAcc": -1.4450999999999965,
    "easy_aAcc": 0.0,
    "hard_aAcc": -0.23259999999999081,
    "aAcc": -0.2656999999999954
  },
  "generation_summary": {
    "samples": 1129,
    "token_usage_coverage": 1129,
    "avg_completion_token": 729.5349867139062,
    "length_truncated": 2,
    "note": "Inference only; accuracy requires the matching evaluator"
  }
}
```

</details>
- **原始评测结果：** [../../evaluation/runs/hallusion-full-07/HallusionBench_result.json](../../evaluation/runs/hallusion-full-07/HallusionBench_result.json)
- **Log:** `/task/logs/hallusion-judge-07.log`
- **Note:** Original serial client with transport timeout 7200s; generation parameters unchanged.

### 3. ✅ full_model_serving — passed

Evidence file: [`full-accuracy-07/serving.json`](full-accuracy-07/serving.json)

#### Checks

| Check | Result |
|---|---|
| `full_78_layer_checkpoint` | ✅ pass |
| `all_four_nodes_loaded` | ✅ pass |
| `text_smoke` | ✅ pass |
| `image_smoke` | ✅ pass |
| `gpqa_198` | ✅ pass |
| `hallusion_1129` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ⏭️ not_run |
| `/v1/chat/completions` | ✅ pass |
- **Note:** Four-node accuracy scope. Full-model reload/CP/PP cases were not rerun; reduced functional evidence remains separate.

#### Outputs

- 2
- The dominant color of this image is **red** — a bright, solid red fills the entire frame.

### 4. ✅ reusable_inference_image — passed

Evidence file: [`full-accuracy-07/image.json`](full-accuracy-07/image.json)

#### Checks

| Check | Result |
|---|---|
| `archive_exists` | ✅ pass |
| `archive_size_matches` | ✅ pass |
| `sha256_recorded` | ✅ pass |
| `evaluation_container_excluded` | ✅ pass |
- **原始评测结果：** [../../../mnt/share/x00951564/codex-hy4-20261028-four-node/images/hy4-codex-a5-cann9.1-four-node-run07-20261029.tar.gz](../../../mnt/share/x00951564/codex-hy4-20261028-four-node/images/hy4-codex-a5-cann9.1-four-node-run07-20261029.tar.gz)
- **Note:** Only our inference container was saved. See logs/reusable-image-run07.json and IMAGE-REUSE.zh-CN.txt.

## Interpretation

- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.
- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.
- Text output is runtime evidence only; it does not prove numerical equivalence.

