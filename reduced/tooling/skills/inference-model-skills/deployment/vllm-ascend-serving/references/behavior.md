# Behavior Reference

## Remote execution

This skill owns service lifecycle semantics; its scripts handle all remote
execution for managed workspace sessions.

## Escaping safety

The core value of this skill is that all SSH escaping is handled inside `serve_start.py`. The agent passes structured arguments via CLI flags; the script internally builds a bash script with proper `shlex.quote` on every dynamic value, then wraps the entire script for SSH transport. The agent should never construct raw `ssh ... "export ... && vllm serve ..."` commands for serving.

## Launch lifecycle

1. **resolve-target** — resolve the remote host via direct SSH connection
2. **stop-existing** — if a previous service is recorded for that target, send SIGINT+SIGTERM
3. **probe-npus** — check NPU device availability via `npu-smi info`; validate or auto-select devices
4. **validate** — check model path exists remotely via `test -d` / `test -f`
5. **allocate-port** — allocate a free port locally for the service
6. **launch** — build and execute the launch script via SSH
7. **persist-starting-state** — after PID capture, write serving state with `status=starting`
8. **probe-health** — poll `GET /health` (HTTP 200)
9. **probe-models** — poll `GET /v1/models` (non-empty `data` array)
10. **persist-final-state** — update serving state to `ready` or `started`
11. **output** — print JSON to stdout

## NPU device probing

Before launching, the script SSHes to the **bare-metal host** (port 22, via `host_endpoint()`) and runs `npu-smi info`. Host-level probing can see processes from **all** containers, bypassing PID namespace isolation. It determines:

- Total available NPU devices
- Which devices have running processes (PID-visible from the host)
- Which devices have high HBM usage (above 4096 MB), indicating occupancy even when PIDs are not visible from another container
- Which devices are free (no PID and HBM below threshold)

Device selection: use `--extra-env ASCEND_RT_VISIBLE_DEVICES=<ids>` to specify which NPU
devices are visible to vLLM.  When not set, all devices on the node are available.

> **Die-aware device selection (multi-node)**: on Ascend a single physical board often
> holds multiple logical dies (e.g. davinci14 + davinci15), and vLLM counts those dies as its
> `--tensor-parallel-size` devices — not the physical board. `probe_npus` therefore counts
> `/dev/davinci*` in direct mode (returning `vllm_devices`/`vllm_device_count`).
> When every node exposes `>= tp` vLLM devices, `serve_start` uses the **die-aware** branch:
> `dp_local = min(vllm_devices // tp)`, `dp_size = dp_local × nnodes`, and it does **NOT** inject
> `ASCEND_RT_VISIBLE_DEVICES` (vLLM auto-enumerates its dies; injecting `npu-smi` board ids such
> as `7` gives the wrong id space and breaks `--tp` on single-board nodes).

## Relaunch merge rules

When `--relaunch` is used:

- Previous launch parameters are loaded from `.workspace-local/serving/<alias>.json`.
- Any CLI argument provided this time **overrides** the previous value
- `--extra-env KEY=VALUE` is **merged** into the previous env map (new keys added, existing keys overwritten)
- `--unset-env KEY` **removes** a key from the inherited env map
- `--unset-args PREFIX` removes args starting with that prefix from the inherited extra args. Use `=` syntax (`--unset-args=--enforce-eager`) to avoid argparse treating the prefix as a separate flag. Boolean flags (where the next token starts with `-` or is absent) are removed alone; value-bearing flags (where the next token does not start with `-`) remove both the flag and its value.
- Extra vllm args after `--` are **appended** to inherited extra args
- Runtime-only fields (port, pid, log paths) are always recalculated

## Ascend environment

The launch script prepends the Ascend driver library paths to `LD_LIBRARY_PATH`. Device visibility is controlled via `ASCEND_RT_VISIBLE_DEVICES`.

## Extra args escaping

Extra vllm args after `--` are passed to `vllm serve` as individual tokens. Each token is independently `shlex.quote`-wrapped for bash safety. This means JSON values like `--additional-config '{"key":"value"}'` are correctly preserved through the SSH + bash layers — double quotes inside JSON are not consumed.

The args are stored in local state as a flat list of strings. On relaunch, the inherited list is used as-is without re-splitting.

## Process detachment

Services are launched with `nohup ... </dev/null &` followed by `disown` to fully detach from the SSH session. The PID is captured and written to `<runtime_dir>/pid`.

## Remote runtime directory

Each launch instance gets its own directory under:

```text
<workdir>/.workspace-runtime/serving/<timestamp>_<machine>/
```

Where `<machine>` identifies the node the instance runs on (its IP/hostname,
e.g. `80.5.27.39`). On multi-node clusters the head and each worker use their
own machine identifier, so a given timestamp maps to one directory per node
(e.g. `20260808_120000_80.5.27.39/` and `20260808_120000_80.5.27.38/`).

This directory contains:

- `stdout.log` — vllm server stdout
- `stderr.log` — vllm server stderr
- `pid` — process ID file

The `<workdir>` is typically `/home/workspace`.

The vLLM process is launched **from** this runtime directory, not from `/home/workspace`, to prevent Python from resolving the `vllm` package to the source tree instead of the installed package.

## Stop sequence

1. `SIGINT` (graceful shutdown)
2. Wait 5 seconds
3. `SIGTERM` if still alive
4. Wait 5 seconds
5. `SIGKILL` only if `--force` is given

## Status probes

- **alive**: `kill -0 <pid>` succeeds
- **health**: `GET /health` returns HTTP 200
- **models_ok**: `GET /v1/models` returns a non-empty `data` array

Combined status:

- `ready` = alive + health + models_ok
- `alive_healthy` = alive + health but models not confirmed
- `alive` = process exists but health endpoint not responding
- `stopped` = process does not exist

## Multi-Node Data Parallel Lifecycle

When `--worker-hosts` is specified, the script enters multi-node mode. vLLM uses
**Data Parallel (DP)** mode with `--data-parallel-size` / `--data-parallel-size-local` /
`--data-parallel-start-rank` / `--data-parallel-address` / `--data-parallel-rpc-port`
for cross-node scaling. Head runs API Server + Engine, workers run Engine only
(`--headless`). No external Ray cluster is required.

### Launch sequence

1. **resolve-worker-targets** — parse `--worker-hosts`, resolve each worker as an `ExecutionTarget` inheriting head config (including `--mode` and connection credentials)
2. **preflight-all** — verify SSH connectivity and model path on ALL nodes before any work begins
3. **stop-existing-cluster** — check persisted cluster state and stop any previous processes
4. **probe-npus-all** — probe NPU devices on each node independently, select largest contiguous free block, trim to uniform `dp_local_min` across nodes
5. **allocate-ports** — allocate service port (head only) and DP RPC port for cross-node engine communication
6. **launch-head** — start vLLM on head with `--data-parallel-size <total> --data-parallel-size-local <N> --data-parallel-start-rank 0 --data-parallel-address <ip> --data-parallel-rpc-port <port>`
7. **wait-dp-rpc** — poll head DP RPC port until listening (timeout --dp-rpc-timeout, default 1800s), ensuring workers can connect
8. **launch-workers** — start vLLM on each worker with matching DP args + `--data-parallel-start-rank <offset>` + `--headless`. Any worker failure triggers rollback of launched nodes
9. **wait-ready** — poll `/health` and `/v1/models` on head
10. **persist-state** — save cluster state and per-machine serving states (including `connection_mode`)

### Multi-node stop sequence

1. Load cluster state to find all nodes
2. Send SIGINT → SIGTERM → SIGKILL (if `--force`) to vLLM on each node
3. Clean up orphan NPU workers on each node
4. Mark cluster state as `stopped`

### Worker configuration inheritance

Worker nodes inherit the following from head unless explicitly overridden
via `--worker-*` CLI arguments:

- SSH port, user, key file, password
- Connection mode (`--mode` / `connection_mode`)
- Container name (docker mode) or conda environment name (conda mode)
All workers use the same connection topology; mixed modes are not supported.

### Cluster state

Cluster state is persisted at `.workspace-local/serving/cluster/<head_ip>.json` separately
from per-machine serving state. This ensures single-node state files are never broken by
multi-node additions. Each per-machine state gains `cluster_type` and `cluster_head` fields
to enable stop/status from any node.
