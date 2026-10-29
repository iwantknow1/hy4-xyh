# Acceptance Criteria

## A1. Fresh start — structured input

**Given** a ready managed machine with a valid model path,
**When** `serve_start.py --host <ip> --container <name> --model <path> --tp 4` runs,
**Then**:

- a free port is auto-allocated
- the service starts and `/health` returns 200
- `/v1/models` returns the served model
- stdout is a JSON object with `status=ready`, `base_url`, `pid`, `log_stdout`, `log_stderr`
- `.workspace-local/serving/<alias>.json` is written

## A2. Fresh start — model path not found

**Given** a ready machine,
**When** `--model /nonexistent/path` is passed,
**Then** returns `status=needs_input` with a clear error before launching.

## A3. Relaunch — same config

**Given** a previous successful start recorded in `.workspace-local/serving/<alias>.json`,
**When** `serve_start.py --host <ip> --container <name> --relaunch` runs,
**Then**:

- previous model, tp, devices, env, extra_args are reused
- port is re-allocated (not inherited)
- service starts successfully

## A4. Relaunch — add env delta

**Given** a previous successful start,
**When** `--relaunch --extra-env VLLM_LOGGING_LEVEL=DEBUG`,
**Then** the new env appears in the launched service alongside all previous env vars.

## A5. Relaunch — remove env

**Given** a previous start with `VLLM_LOGGING_LEVEL=DEBUG`,
**When** `--relaunch --unset-env VLLM_LOGGING_LEVEL`,
**Then** that env var is absent from the new launch.

## A6. Relaunch — override model

**Given** a previous start with model A,
**When** `--relaunch --model /path/to/modelB --served-model-name modelB`,
**Then** the service starts with model B.

## A7. Status — service running

**Given** a running service,
**When** `serve_status.py --host <ip> --container <name>` runs,
**Then** returns `status=ready`, `alive=true`, `health=true`, `models_ok=true`.

## A8. Status — service stopped

**Given** a stopped service,
**When** `serve_status.py` runs,
**Then** returns `status=stopped`, `alive=false`, and includes `stderr_tail`.

## A9. Status — no previous service

**Given** no prior start for this machine,
**When** `serve_status.py` runs,
**Then** returns `status=not_found`.

## A10. Stop — graceful

**Given** a running service,
**When** `serve_stop.py --host <ip> --container <name>` runs,
**Then**:

- sends SIGINT first
- waits, then SIGTERM if needed
- does NOT SIGKILL unless `--force`
- updates local state to `stopped`
- returns `status=stopped`

## A11. Stop — force

**Given** a service that ignores SIGINT/SIGTERM,
**When** `serve_stop.py --force`,
**Then** sends SIGKILL and returns `status=stopped`.

## A12. Previous service cleanup

**Given** a running service on the target machine,
**When** a new `serve_start.py` runs for the same machine,
**Then** the old service is stopped before the new one launches.

## A15. Device selection via environment variable

**Given** a machine with 8 NPUs,
**When** `serve_start.py --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 --tp 4` is called,
**Then** only devices 0-3 are visible to vLLM and TP=4 uses those 4 devices.

## A16. Default — all devices visible

**Given** a machine with 8 NPUs,
**When** `serve_start.py --tp 8` is called without `--extra-env ASCEND_RT_VISIBLE_DEVICES`,
**Then** all 8 devices are visible and available to vLLM.

## A17. No device validation at launch

**Given** a machine with any NPU configuration,
**When** `serve_start.py` is called with any `--tp` value,
**Then** the script does NOT validate NPU availability — it trusts the user's configuration and lets vLLM handle device errors at runtime.

## A18. NPU probe standalone

**When** `serve_probe_npus.py --host <ip> --container <name>` runs,
**Then** returns JSON with `devices`, `busy` (with PID details), `hbm` (per-device HBM usage), `free`, `free_count`, and `hbm_busy_threshold_mb`. Probing is done on the bare-metal host for cross-container visibility.

## A19. Escaping safety

**Given** model paths, env values, or args containing spaces, quotes, or shell metacharacters,
**When** any serve script runs,
**Then** the values are correctly escaped and the remote command executes without shell injection or breakage.

## A21. Launch directory isolation

**When** `serve_start.py` launches vLLM,
**Then** the process working directory is the runtime dir (e.g. `/home/workspace/.workspace-runtime/serving/<ts>/`), NOT `/home//workspace`, preventing Python from resolving `import vllm` to the source tree.

## A22. Unset boolean flag

**Given** a previous launch with `--enforce-eager --max-model-len 2048`,
**When** `--relaunch --unset-args=--enforce-eager`,
**Then** `--enforce-eager` is removed but `--max-model-len 2048` is preserved.

## A23. Unset value-bearing flag

**Given** a previous launch with `--enforce-eager --max-model-len 2048`,
**When** `--relaunch --unset-args=--max-model-len`,
**Then** both `--max-model-len` and `2048` are removed but `--enforce-eager` is preserved.

## A24. JSON arg passthrough

**Given** `-- --additional-config '{"torchair_graph_config":{"enabled":false}}'` in extra args,
**When** `serve_start.py` launches,
**Then** the JSON double quotes survive the SSH + bash escaping layers and `vllm serve` receives valid JSON.

## A26. MoE model support

**Given** a Mixture-of-Experts model (e.g. Qwen3.5-35B-A3B),
**When** `serve_start.py` launches with `--tp 2 --dp 2 --trust-remote-code`,
**Then** the model loads and `/v1/completions` returns valid responses.

## A27. Chunked prefill

**When** `serve_start.py` launches with `--enable-chunked-prefill`,
**Then** the service starts successfully.

## A28. Prefix caching

**When** `serve_start.py` launches with `--enable-prefix-caching`,
**Then** the service starts successfully.

## Multi-Node Acceptance

### M1. Two-node fresh start

**Given** two reachable machines each with 8 NPUs and a shared model path,
**When** `serve_start.py --host <head> --container <c> --worker-hosts <worker> --model <path> --tp 8` runs,
**Then**:

- Both head and worker vLLM processes start successfully
- Cluster state is persisted with `cluster_type: "multi-node"`
- `/health` returns 200 on head
- `serve_status.py` shows `cluster_type: "multi-node"` with all nodes alive

### M2. Worker SSH failure during preflight

**Given** a head and an unreachable worker,
**When** `serve_start.py --worker-hosts <unreachable>` runs,
**Then** returns `status=needs_input` with per-node failure details before any vLLM launch.

### M3. Worker NPU device shortage

**Given** a worker with 2 free NPUs and `--tp 4`,
**When** `serve_start.py --worker-hosts <worker>` runs,
**Then** returns `status=needs_input` before any vLLM launch.

### M4. Stop cluster from head

**Given** a running multi-node cluster,
**When** `serve_stop.py --host <head>` runs,
**Then** stops vLLM on head and all workers, returns `status=stopped`.

### M5. Stop cluster from worker

**Given** a running multi-node cluster,
**When** `serve_stop.py --host <worker>` runs,
**Then** finds cluster via `cluster_head` reference, stops all nodes.

### M6. Single-node backward compatibility

**Given** a single-node launch without `--worker-hosts`,
**When** `serve_start.py --host <ip> --container <c> --model <path> --tp 4` runs,
**Then** behaves identically to previous single-node version — no multi-node fields in output.

### M7. Multi-node status aggregation

**Given** a running multi-node cluster,
**When** `serve_status.py --host <head>` runs,
**Then** returns aggregated status with `head`, `workers` list, `all_nodes_healthy`, `nnodes`.

### M8. Multi-node NPU probe

**Given** two reachable machines,
**When** `serve_probe_npus.py --host <head> --worker-hosts <worker>` runs,
**Then** returns per-node NPU info with `cluster_type: "multi"` and summary totals.
