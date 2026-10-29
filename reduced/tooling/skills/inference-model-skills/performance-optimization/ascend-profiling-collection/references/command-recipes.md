# Command Recipes

These are the expected shapes the agent should call. Every invocation is a single shot — there is no "warm collection" mode.

## Minimal text case in eager mode

When the current environment is the target container, collect directly in
local mode:

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --connection-mode local \
  --model /home/vllm/weights/Qwen3-30B-A3B \
  --served-model-name Qwen3-30B-A3B --tp 4 --dp 1 \
  --devices 8,9,10,11 --tag batch-invariance-phase1 \
  --mode enforce_eager --profile-phase middle \
  --random-input-len 5 --random-output-len 20 \
  --num-prompts 8 --max-concurrency 8 --max-num-seqs 8 \
  --extra-env VLLM_BATCH_INVARIANT=1
```

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.100 --container my-container \
  --model /home/weights/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --tp 1 \
  --tag qwen3_8b_eager_text \
  --mode enforce_eager \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 24
```

Direct Docker connection equivalent:

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 173.131.1.2 --container my-vllm-container --password mypwd \
  --model /home/weights/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --tp 1 \
  --tag qwen3_8b_eager_text \
  --mode enforce_eager \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 24
```

## Piecewise graph mode with VL workload

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.101 --container my-container \
  --model /home/weights/Qwen3.5-VL \
  --served-model-name Qwen3.5-VL \
  --tp 4 \
  --tag qwen35_vl_piecewise_vl \
  --mode piecewise_graph \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 24
```

## Long control-plane timeout for many ranks

```bash
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.100 --container my-container \
  --model /home/weights/DeepSeek-V3 \
  --served-model-name DeepSeek-V3 \
  --tp 8 \
  --tag dsv3_tp8_fdo \
  --mode full_decode_only \
  ---random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 24 \
  --profile-control-timeout 1800 \
  --request-timeout 1800
```

Multi-rank torch profiler setup/finalization can take several minutes. Bump `--profile-control-timeout` whenever TP × DP grows.

## Custom output directory

By default profiling data lands in `/home/workspace/` (absolute path). Pass `--torch-profiler-dir` to override:

```bash
# Use a tag-based subdirectory under /home/workspace
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.100 --container my-container \
  --model /home/weights/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --tp 1 \
  --tag qwen3_8b_eager_text \
  --mode enforce_eager \
  --torch-profiler-dir /home/workspace/qwen3_8b_eager_text \
  --random-input-len 128 --random-output-len 64 --num-prompts 24

# Use runtime_dir-relative path (no leading slash)
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --host 192.168.1.100 --container my-container \
  --model /home/weights/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --tp 1 \
  --tag qwen3_8b_eager_text \
  --mode enforce_eager \
  --torch-profiler-dir vllm_profile \
  --random-input-len 128 --random-output-len 64 --num-prompts 24
```

Relative paths are resolved under the vLLM runtime_dir; absolute paths (starting with `/`) are used as-is.

## Re-analyse an already-collected root

Use this when a previous collection ran but `analyse()` was skipped or the agent wants to re-verify outputs without re-collecting.

```bash
# Direct Docker connection
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/run_remote_analyse.py \
  --host 173.131.1.2 --container my-vllm-container --password mypwd \
  --profile-root /home/workspace/.workspace-runtime/serving/<timestamp>_<machine>/vllm_profile \
  --expected-ranks 8
```

Exit 0 means every rank produced `kernel_details.csv` and `trace_view.json` AND the directory count matched `--expected-ranks` (typically `tp * (dp or 1)`). Non-zero means re-collection is needed. Always pass `--expected-ranks` against fresh roots — without it a partial capture where some ranks never produced a directory looks "clean".

## Manually flip the profiler window on a service the agent already started

For debugging the control plane in isolation. The serving skill must have launched a service with a `--profiler-config` argument first.

```bash
# Direct Docker connection
python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/profile_control.py \
  --host 173.131.1.2 --container my-vllm-container --password mypwd \
  --action start_profile --timeout 900

# ... agent sends workload via curl / openai client / etc ...

python3 skills/inference-model-skills/performance-optimization/ascend-profiling-collection/scripts/profile_control.py \
  --host 192.168.1.100 --container my-container --action stop_profile --timeout 900
```

The script reads the service port from `.workspace-local/serving/<alias>.json`.

## What NOT to do

- Do not run `collect_torch_profile_case.py` against a machine the user cannot SSH to.
- When collecting in the current target environment, use `--connection-mode local` instead of SSH.
- Do not pass `--profiler-config` through `--serve-args` of `vllm-ascend-benchmark` to "sneak" profiling into a benchmark run; benchmarks should not own profiler windows.
- Do not call `torch_npu.profiler.profiler.analyse()` from the local Mac. The collection / re-analyse scripts run it inside the container.
- Do not share state with `.workspace-local/service-torch-profiler/` (the old prototype location). The new skill writes only to `.workspace-local/ascend-profiling-collection/runs/`.
