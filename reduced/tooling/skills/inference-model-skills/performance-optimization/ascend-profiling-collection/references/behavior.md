# Behavior Reference

## Execution transport

This skill owns profiler collection semantics; its scripts handle all remote
execution for managed workspace sessions. Direct Docker connections (`--host`,
`--container`, `--password` / `--key-file`) are supported — when used the
SSH commands are automatically
wrapped with ``docker exec``.

When the agent already runs in the target container or Conda environment, use
`--connection-mode local`. Commands and profiler-control HTTP
requests then execute directly in the current environment; no SSH connection,
credential, or port-forwarding tunnel is used. Local mode is single-node only.

## Why profiler control lives here, not in the serving skill

`/start_profile` and `/stop_profile` only exist because vLLM has a built-in torch profiler. Moving the control client into `vllm-ascend-serving` would force serving to grow profiling-specific knobs (multi-rank long timeout, multi-api-server quirks). The serving skill stays simple by treating `--profiler-config` as an opaque blob it forwards to `vllm serve`. Anything that flips, waits on, or interprets the profiler window belongs here.

## Mode → vLLM flag mapping

| `--mode` | Forwarded to `vllm serve` |
| --- | --- |
| `enforce_eager` | `--enforce-eager` |
| `full_decode_only` | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` |
| `piecewise_graph` | `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` |

`CUDAGraphMode` enum values come from `vllm/vllm/config/compilation.py`. Add a new mode here only after verifying the underlying enum value still exists in the active `vllm/` submodule.

## Control-plane timeout

`/start_profile` and `/stop_profile` setup/finalization touches every rank. With TP=8 it is normal to see 60+ seconds of latency on each call. The default `--profile-control-timeout 600` covers most cases; bump to 1200–1800 for large-scale runs. `--request-timeout` is independent and applies only to the chat-completions requests during the profile window.

## Request wave shape

The default wave is `--num-prompt 24` at `--max-concurrency 24`, followed by exactly one `--random-output-len 24` tail request. The follow-up exists so the resulting trace contains a clean steady-state step that is not contaminated by the wave's tail decay. The analysis skill relies on this shape for its tail-step detection.

## Workload transport

Chat requests are sent from the local machine through an ephemeral `ssh -L` tunnel. This is so multimodal (VL) payloads — base64 image data URLs — can be assembled locally without round-tripping through SSH heredocs. For text-only workloads the tunnel is still used (consistency); request bodies stay small so there is no transport cost.

With `--connection-mode local`, requests instead use the service's loopback URL
directly and record `transport: local` in the manifest; no SSH tunnel is opened.

The `sips` image resizer is macOS-only. The collection skill assumes the agent runs on a Mac workstation. If the workstation is ever Linux, swap `_parse_sips_dimensions` / `_build_image_data_url` for a PIL implementation.

## Multi-api-server interactions

`--api-server-count` is exposed because vLLM's multi-api-server mode has historically had control-plane routing quirks: a `/start_profile` POST may be answered by an API server that is not the one driving the worker processes. When investigating a "trace empty" symptom, set `--api-server-count 1` to eliminate this variable before blaming the profiler.

## Workload success gate

`benchmark_results` and `followup_result` are not just diagnostic data — the
orchestrator computes a `workload_status` over them and the top-level
`status` is `failed` whenever:

- the follow-up tail request did not return 2xx (`workload_status.status ==
  "followup_failed"`), or
- the benchmark wave's success rate fell below
  `--benchmark-success-threshold` (default 0.8) — `bench_success_rate <
  bench_threshold` ⇒ `workload_status.status == "benchmark_below_threshold"`.

A profile window that ran with no real model traffic produces a trace that
*looks* fine to `analyse()` (kernel_details.csv lands, only it describes
nothing useful). The gate prevents that root from leaking into downstream
analysis. Lower the threshold only when you explicitly expect flakiness in
the wave.

## Output verification

After `analyse()`, every `*_ascend_pt` directory is expected to contain:

- `ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- `ASCEND_PROFILER_OUTPUT/trace_view.json`

Historical captures show that `analyse()` can "succeed" but produce no `kernel_details.csv` because the profile window was too short or `FRAMEWORK/torch.op_range` never made it to disk. The skill turns that into `analysis_status == missing_kernel_details` and fails the run. Re-collection is the only fix; offline `analyse()` cannot recover the missing device data.

In addition, the orchestrator passes `--expected-ranks = tp * (dp or 1)` to
`run_remote_analyse.py`. When the actual `*_ascend_pt` count differs from the
expected number, `analysis_status` becomes `rank_count_mismatch`. Without
this gate a partial capture (e.g. only rank 0 dumped) can look "clean"
because every directory that *did* land was complete, masking a topology
failure.

## Post-stop flush window

The orchestrator sleeps `POST_STOP_FLUSH_SECONDS` (5s) between `/stop_profile` returning and `serve_stop.py` being called. `/stop_profile` should already block until profiler threads quiesce, but historical traces show flush latency on some CANN versions. The window is short enough not to bother humans and long enough to cover known races.

## Local state layout

```text
.workspace-local/ascend-profiling-collection/runs/
  <YYYYmmdd_HHMMSS>_<tag>/
    manifest.json
```

One directory per invocation. Nothing else is ever written here. The remote profiling root (`<runtime_dir>/<torch_profiler_dir>`) lives on the container and is referenced from the manifest via `remote_profile_root`.

## Manifest schema versioning

`schema_version` starts at 1. Bump only when a field is renamed or removed. Adding new fields (e.g. richer profiler config knobs) does not require a bump; the analysis skill should treat unknown fields as advisory.
