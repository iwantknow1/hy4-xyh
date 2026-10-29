# Command Recipes

## Fresh start with basic params

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

## Fresh start with extra vllm args

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --tp 4 \
  --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  --extra-env VLLM_USE_V1=1 \
  -- --max-model-len 4096 --gpu-memory-utilization 0.9
```

## Relaunch with same config (e.g. after code change)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch
```

## Relaunch with extra debug env

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch \
  --extra-env VLLM_LOGGING_LEVEL=DEBUG \
  --extra-env VLLM_TRACE_FUNCTION=1
```

## Relaunch and remove a previously set env

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch \
  --unset-env VLLM_TRACE_FUNCTION
```

## Relaunch with a different model

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch \
  --model /data/models/DeepSeek-V3 \
  --served-model-name deepseek-v3
```

## Relaunch and remove a previous vllm arg (value-bearing)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch \
  --unset-args=--max-model-len
```

Note: use `=` syntax to prevent argparse from treating `--max-model-len` as a separate flag. This removes both `--max-model-len` and its value (e.g. `2048`).

## Relaunch and remove a boolean flag

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container --relaunch \
  --unset-args=--enforce-eager
```

Boolean flags like `--enforce-eager` are removed alone (the next token is not consumed).

## Start with a forced port

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4 --port 8000
```

> ⚠️ `--port` 是 serve_start.py 的脚本级参数，必须放在 `--` 之前，**不可**作为 vllm 透传参数放在 `--` 之后。脚本使用该端口做健康检查；若在 `--` 后传入 `--port`，会与脚本分配的端口冲突导致超时。

## Start with extended health timeout (for very large models)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/DeepSeek-V3-685B \
  --tp 8 \
  --health-timeout 1200
```

## Start with auto-selected devices (just specify tp)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4
```

The script probes NPUs, finds 4 free devices, and auto-selects them.

## Start with conda environment (no container)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --conda-env torch_npu \
  --model /data/models/Qwen3-32B \
  --tp 4 --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

When `--conda-env` is set (and `--container` is not), commands run directly on the
host inside the named conda environment via `conda run -n <env>`.  The Ascend
driver libraries are exported automatically; the conda env's own
`workspace-ascend-env.sh` sourcing is **skipped** in favour of the conda-managed
Python and packages.

## Probe NPU availability before deciding

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_probe_npus.py \
  --host 192.168.1.100 --container my-container
```

This probes the **bare-metal host** (not the container) for cross-container NPU visibility. Example output:

```json
{
  "status": "ok",
  "host": "192.168.1.100",
  "total": 8,
  "devices": [0, 1, 2, 3, 4, 5, 6, 7],
  "busy": {"0": [{"pid": 12345, "owner": "root", "name": "python3"}],
           "1": [{"pid": 12345, "owner": "root", "name": "python3"}]},
  "hbm": {"0": 8192, "1": 8192, "2": 0, "3": 0},
  "free": [2, 3, 4, 5, 6, 7],
  "free_count": 6,
  "hbm_busy_threshold_mb": 4096
}
```

## Check status

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_status.py \
  --host 192.168.1.100 --container my-container
```

## Stop gracefully

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --host 192.168.1.100 --container my-container
```

## Force stop

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_stop.py \
  --host 192.168.1.100 --container my-container --force
```

## Start an MoE model (all 8 cards)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3.5-35B-A3B \
  --tp 8 \
  -- --enforce-eager --max-model-len 2048 --trust-remote-code
```

## Start with additional-config (JSON passthrough)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4 \
  -- --enforce-eager --additional-config '{"torchair_graph_config":{"enabled":false}}'
```

JSON double quotes are preserved through the SSH escaping layers.

## Start with chunked prefill

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4 \
  -- --enforce-eager --enable-chunked-prefill
```

## Start with prefix caching

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --model /data/models/Qwen3-32B \
  --tp 4 \
  -- --enforce-eager --enable-prefix-caching
```

## Multi-Node Deployments

Multi-node mode uses vLLM **Data Parallel (DP)** for cross-node scaling.
Head runs API Server + Engine; workers run Engine only (`--headless`).

Worker nodes inherit `--mode` and connection credentials from head.

### Two-node (docker mode)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B \
  --tp 8
```

### Two-node (direct mode — containers with own IP + SSH)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --mode direct --host 10.0.0.5 --password hdpwd \
  --worker-hosts 10.0.0.6 --worker-password wkpwd \
  --model /data/models/Qwen2-72B \
  --tp 8
```

### Two-node (conda mode)

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --mode conda --host 192.168.1.100 --conda-env vllm_env \
  --worker-hosts 192.168.1.101 --worker-conda-env vllm_env \
  --model /data/models/Qwen2-72B \
  --tp 8
```

Auto-behavior: scripts probe NPU on each node. In **direct** mode they prefer the vLLM-visible
die count (`/dev/davinci*`) via the die-aware branch — `dp_local = min(vllm_devices // tp)`,
`dp_size = dp_local × nnodes`, no `ASCEND_RT_VISIBLE_DEVICES` injection (vLLM auto-enumerates its
dies; a single 2-die board handles `--tp 2`). Docker/classic mode falls back to largest contiguous
free block. They compute `dp_local_min` across all nodes and auto-generate DP args:
`--data-parallel-size <dp_local_min × nnodes> --data-parallel-size-local <dp_local_min>
--data-parallel-start-rank <offset> --data-parallel-address <head_ip>
--data-parallel-rpc-port <auto>`.

### Three-node + Pipeline Parallelism

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101,192.168.1.102 \
  --model /data/models/DeepSeek-V3-685B \
  --tp 8 \
  -- --pipeline-parallel-size 3
```

### With explicit DP RPC port

```bash
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container my-container \
  --worker-hosts 192.168.1.101 \
  --model /data/models/Qwen2-72B \
  --tp 4 \
  --master-port 29501
```

### With different worker container/config

```bash
# Worker uses a different container name
python3 skills/inference-model-skills/deployment/vllm-ascend-serving/scripts/serve_start.py \
  --host 192.168.1.100 --container head-ctr \
  --worker-hosts 192.168.1.101 \
  --worker-container worker-ctr \
  --model /data/models/Qwen2-72B \
  --tp 8
```

### Stop multi-node cluster (from head or any worker)

```bash
# docker mode — stop from head
python3 serve_stop.py --host 192.168.1.100 --container my-container

# direct mode — stop from head
python3 serve_stop.py --mode direct --host 10.0.0.5 --password hdpwd

# conda mode — stop from head
python3 serve_stop.py --mode conda --host 192.168.1.100 --conda-env vllm_env
```

### Probe NPUs across all nodes

```bash
# docker mode
python3 serve_probe_npus.py --host 192.168.1.100 --container my-container \
    --worker-hosts 192.168.1.101,192.168.1.102

# direct mode
python3 serve_probe_npus.py --mode direct --host 10.0.0.5 --password hdpwd \
    --worker-hosts 10.0.0.6,10.0.0.7
```

### Check multi-node cluster status

```bash
# docker mode
python3 serve_status.py --host 192.168.1.100 --container my-container

# direct mode
python3 serve_status.py --mode direct --host 10.0.0.5 --password hdpwd
```
