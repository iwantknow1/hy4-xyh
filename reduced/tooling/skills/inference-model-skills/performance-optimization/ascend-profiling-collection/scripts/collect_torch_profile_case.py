#!/usr/bin/env python3
from __future__ import annotations

"""Collect one torch-profiler case on a workspace-managed remote NPU container.

This is the single agent-facing entry point for the
``ascend-profiling-collection`` skill. It chains together:

    1. start a service via vllm-ascend-serving with ``--profiler-config``
       (delay_iterations/max_iterations configured for auto-triggered profiling)
    2. POST /start_profile to activate torch profiler
    3. run ``vllm bench`` workload on the remote container
    4. POST /stop_profile and wait briefly for profiler files to flush
    5. stop the service via vllm-ascend-serving
    6. analyse every ``*_ascend_pt`` directory and verify outputs
       (run_remote_analyse.py)
    7. write a manifest the analysis skill can consume
"""

import argparse
import contextlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    add_ssh_connection_args,
    ascend_env_preamble,
    call_serve_start,
    call_serve_stop,
    emit_progress,
    now_utc,
    open_local_tunnel,
    print_json,
    resolve_execution_target,
    resolve_worker_target,
    ssh_exec_target,
    unique_collection_run_dir,
)
from profile_control import post_remote_action
from run_remote_analyse import analyse_profile_root


DEFAULT_TORCH_PROFILER_DIRNAME = "/home/vllm/vllm-profile"
PROFILE_CONTROL_TIMEOUT_SECONDS = 600
POST_STOP_FLUSH_SECONDS = 5


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_json(manifest)


# ---------------------------------------------------------------------------
# Serving args assembly
# ---------------------------------------------------------------------------

def _build_serve_args(args: argparse.Namespace, profiler_config: dict[str, Any]) -> list[str]:
    serve_args: list[str] = [
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--tp", str(args.tp),
    ]
    # ---- target selection (prepended to serve_args) ----
    target_prefix: list[str] = ["--host", args.host]
    if args.ssh_port != 22:
        target_prefix.extend(["--ssh-port", str(args.ssh_port)])
    if args.user != "root":
        target_prefix.extend(["--user", args.user])
    if args.password:
        target_prefix.extend(["--password", args.password])
    if args.key_file:
        target_prefix.extend(["--key-file", args.key_file])
    if args.container:
        target_prefix.extend(["--container", args.container])
    connection_mode = getattr(args, "connection_mode", "docker")
    if connection_mode != "docker":
        target_prefix.extend(["--mode", connection_mode])
    if getattr(args, "conda_env", None):
        target_prefix.extend(["--conda-env", args.conda_env])
    cann_dir = getattr(args, "cann_dir", "/usr/local/Ascend")
    if cann_dir != "/usr/local/Ascend":
        target_prefix.extend(["--cann-dir", cann_dir])
    serve_args[:0] = target_prefix
    # ---- serving params ----
    if args.dp is not None and args.dp > 1:
        serve_args.extend(["--dp", str(args.dp)])
    if getattr(args, "devices", None):
        serve_args.extend(["--extra-env", f"ASCEND_RT_VISIBLE_DEVICES={args.devices}"])
    serve_args.extend([
        "--extra-env",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ])
    extra_env = getattr(args, "extra_env", None)
    if extra_env:
        for item in extra_env:
            serve_args.extend(["--extra-env", item])

    # ---- multi-node ----
    worker_hosts = getattr(args, "worker_hosts", None)
    if worker_hosts:
        serve_args.extend(["--worker-hosts", worker_hosts])
    master_port = getattr(args, "master_port", None)
    if master_port is not None:
        serve_args.extend(["--master-port", str(master_port)])
    if getattr(args, "worker_key_file", None):
        serve_args.extend(["--worker-key-file", args.worker_key_file])
    if getattr(args, "worker_password", None):
        serve_args.extend(["--worker-password", args.worker_password])
    if getattr(args, "worker_container", None):
        serve_args.extend(["--worker-container", args.worker_container])
    if getattr(args, "worker_conda_env", None):
        serve_args.extend(["--worker-conda-env", args.worker_conda_env])
    worker_extra_env = getattr(args, "worker_extra_env", None)
    if worker_extra_env:
        for item in worker_extra_env:
            serve_args.extend(["--worker-extra-env", item])

    serve_args.append("--")
    serve_args.extend([
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
    ])
    if args.api_server_count is not None:
        if worker_hosts:
            # --api-server-count is incompatible with multi-node --headless
            # workers (each worker runs engine-only with --headless, which
            # cannot host more than one API server).
            emit_progress(
                "constraint",
                "--api-server-count is incompatible with multi-node --headless "
                "workers; ignoring it (multi-node must use a single API server)",
            )
        else:
            serve_args.extend(["--api-server-count", str(args.api_server_count)])

    serve_args.extend([
        "--profiler-config", json.dumps(profiler_config, separators=(",", ":")),
    ])

    if args.enable_expert_parallel:
        serve_args.append("--enable-expert-parallel")

    # ---- user-provided extra serve args (appended after built-in ones) ----
    serve_args_extra = getattr(args, "serve_args", "") or ""
    if serve_args_extra.strip():
        try:
            extra_tokens = shlex.split(serve_args_extra)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse --serve-args: {exc}. "
                "Tip: use shlex-style quoting, e.g. "
                "--serve-args \"--additional-config '{\\\"key\\\":true}'\""
            )
        managed_options = {"--max-model-len", "--max-num-seqs", "--gpu-memory-utilization", "--api-server-count", "--profiler-config"}
        for token in extra_tokens:
            option = token.split("=", 1)[0].replace("_", "-")
            if option in managed_options:
                raise ValueError(f"{option} is managed by the collector; use its dedicated flag instead of --serve-args")
        serve_args.extend(extra_tokens)

    if args.mode not in {"enforce_eager", "full_decode_only", "piecewise_graph"}:
        raise ValueError(f"unknown --mode: {args.mode}")
    # Merge the graph mode into the baseline JSON rather than overwriting its
    # capture sizes or other confirmed settings with a second config argument.
    compilation = {}
    merged_args = []
    cursor = 0
    seen_compilation = False
    while cursor < len(serve_args):
        token = serve_args[cursor]
        option = token.split("=", 1)[0].replace("_", "-")
        if option == "--compilation-config":
            if seen_compilation:
                raise ValueError("Provide one combined --compilation-config JSON")
            seen_compilation = True
            if "=" in token:
                value = token.split("=", 1)[1]
            else:
                cursor += 1
                if cursor >= len(serve_args):
                    raise ValueError("Missing --compilation-config value")
                value = serve_args[cursor]
            compilation = json.loads(value)
            if not isinstance(compilation, dict):
                raise ValueError("--compilation-config must be a JSON object")
        else:
            merged_args.append(token)
        cursor += 1
    if args.mode == "enforce_eager":
        if compilation.get("cudagraph_mode") not in (None, "NONE", 0):
            raise ValueError("Baseline cudagraph_mode conflicts with --mode enforce_eager")
        if "--enforce-eager" not in merged_args:
            merged_args.append("--enforce-eager")
    else:
        mode = "FULL_DECODE_ONLY" if args.mode == "full_decode_only" else "PIECEWISE"
        if compilation.get("cudagraph_mode", mode) != mode or "--enforce-eager" in merged_args:
            raise ValueError("Baseline graph configuration conflicts with selected --mode")
        compilation["cudagraph_mode"] = mode
    if compilation:
        merged_args.extend(["--compilation-config", json.dumps(compilation, separators=(",", ":"))])
    return merged_args


def _append_sampling_args(command: list[str], args: argparse.Namespace) -> None:
    """Append explicitly provided vllm bench serve sampling parameters."""
    for option, attribute in (
        ("--top-p", "top_p"),
        ("--top-k", "top_k"),
        ("--min-p", "min_p"),
        ("--temperature", "temperature"),
        ("--frequency-penalty", "frequency_penalty"),
        ("--presence-penalty", "presence_penalty"),
        ("--repetition-penalty", "repetition_penalty"),
    ):
        value = getattr(args, attribute)
        if value is not None:
            command.extend([option, str(value)])


def _build_benchmark_command(args: argparse.Namespace, base_url: str) -> list[str]:
    """Build fixed-length text or image/text traffic for the same collector."""
    multimodal = args.workload_kind == "multimodal"
    input_length = args.random_input_len
    if multimodal:
        if args.mm_input_token_overhead is None or args.mm_input_token_overhead < 0:
            raise ValueError("Multimodal collection requires measured --mm-input-token-overhead (image + chat template tokens)")
        input_length -= args.mm_input_token_overhead
        if input_length <= 0:
            raise ValueError("Multimodal overhead must be smaller than target input length")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai-chat" if multimodal else "vllm",
        "--tokenizer", args.model, "--model", args.served_model_name,
        "--dataset-name", "random-mm" if multimodal else "random",
        "--random-input-len", str(input_length),
        "--random-output-len", str(args.random_output_len), "--ignore-eos",
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", str(args.max_concurrency or args.num_prompts),
        "--metric-percentiles", args.metric_percentiles, "--base-url", base_url,
    ]
    if args.random_range_ratio is not None:
        command.extend(["--random-range-ratio", str(args.random_range_ratio)])
    if multimodal:
        command.extend([
            "--endpoint", "/v1/chat/completions",
            "--random-mm-base-items-per-request", "1",
            "--random-mm-num-mm-items-range-ratio", "0",
            "--random-mm-limit-mm-per-prompt", '{"image":1,"video":0}',
            "--random-mm-bucket-config", f"{{({args.image_height},{args.image_width},1):1.0}}",
        ])
    _append_sampling_args(command, args)
    return command


def _build_profiler_config(args: argparse.Namespace, output_dir: str) -> dict[str, Any]:
    if args.profile_steps <= 0 or args.random_output_len <= 0 or args.random_input_len <= 0:
        raise ValueError("Profile steps and input/output lengths must be positive")
    if args.max_model_len < args.random_input_len + args.random_output_len:
        raise ValueError("--max-model-len must cover input plus output lengths (and any template overhead)")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("Image dimensions must be positive")
    if args.num_prompts <= 0 or (args.max_concurrency is not None and args.max_concurrency <= 0):
        raise ValueError("Prompt count and concurrency must be positive")
    if args.random_range_ratio is not None and not 0 <= args.random_range_ratio <= 1:
        raise ValueError("--random-range-ratio must be between 0 and 1")
    delay = (0 if args.profile_phase == "head" else args.random_output_len // 2
             if args.profile_phase == "middle" else max(args.random_output_len - 100, 0))
    if delay + args.profile_steps > args.random_output_len:
        raise ValueError("Requested profiling window extends beyond generated output")
    return {"profiler": "torch", "delay_iterations": delay,
            "max_iterations": args.profile_steps, "torch_profiler_dir": output_dir,
            "torch_profiler_with_stack": False, "ignore_frontend": args.ignore_frontend}


def _stop_profile_and_flush(target, port: int, *, conda_env: str | None = None) -> dict[str, Any]:
    """Finalize every profiler rank before the serving process is stopped."""
    result = post_remote_action(
        target,
        port,
        "stop_profile",
        PROFILE_CONTROL_TIMEOUT_SECONDS,
        conda_env=conda_env,
    )
    time.sleep(POST_STOP_FLUSH_SECONDS)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    # ---- direct SSH connection ----
    add_ssh_connection_args(p, include_cann=True)
    p.add_argument("--model", required=True,
                   help="absolute remote path to model weights")
    p.add_argument("--served-model-name", required=True,
                   help="name vLLM exposes via /v1/models")
    p.add_argument("--tp", type=int, required=True, help="tensor-parallel size")
    p.add_argument("--tag", required=True,
                   help="stable identifier for this collection run; used in run dir name")
    p.add_argument(
        "--mode",
        required=True,
        choices=("enforce_eager", "full_decode_only", "piecewise_graph"),
        help="graph mode for the service",
    )

    p.add_argument("--dp", type=int, default=None,
                   help="data-parallel size (forwarded to serving as --dp)")
    p.add_argument("--devices", default=None,
                   help="comma-separated NPU device IDs (e.g. 8,9,10,11); "
                        "forwarded to serving as --extra-env "
                        "ASCEND_RT_VISIBLE_DEVICES=... (serve_start no longer "
                        "accepts --devices)")
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument(
        "--api-server-count", type=int, default=1,
        help="override vLLM --api-server-count",
    )

    p.add_argument(
        "--analyse-all",
        action="store_true",
        help=(
            "post-process and verify every *_ascend_pt directory on each node; "
            "by default only the first directory per node is analysed"
        ),
    )
    p.add_argument(
        "--profile-phase",
        choices=("head", "middle", "tail"),
        default="middle",
        help="when to capture profiling window: "
             "head → delay_iterations=0 (start of workload); "
             "middle → delay_iterations=random_output_len//2 (current default); "
             "tail → delay_iterations=random_output_len-100 (near end of workload)",
    )
    p.add_argument("--random-input-len", type=int, default=1024,
                   help="--random-input-len for vllm bench serve")
    p.add_argument("--random-output-len", type=int, default=1024,
                   help="--random-output-len for vllm bench serve (controls tail start when --profile-phase=tail)")
    p.add_argument("--profile-steps", type=int, default=20, help="Number of worker iterations to capture")
    p.add_argument("--ignore-frontend", action="store_true", help="Capture worker iterations without whole-request frontend tracing")
    p.add_argument("--workload-kind", choices=("text", "multimodal"), default="text")
    p.add_argument("--random-range-ratio", type=float, default=None, help="Use 0 for fixed input/output lengths")
    p.add_argument("--mm-input-token-overhead", type=int, help="Measured image + chat-template tokens; deducted from target input length")
    p.add_argument("--image-height", type=int, default=256)
    p.add_argument("--image-width", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=24,
                   help="--num-prompts for vllm bench serve")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="--max-concurrency for vllm bench serve (defaults to num-prompts)")
    p.add_argument("--metric-percentiles", type=str, default="50,90,99",
                   help="--metric-percentiles for vllm bench serve")
    sampling_group = p.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)
    sampling_group.add_argument("--frequency-penalty", type=float, default=None)
    sampling_group.add_argument("--presence-penalty", type=float, default=None)
    sampling_group.add_argument("--repetition-penalty", type=float, default=None)

    p.add_argument("--torch-profiler-dir", default=DEFAULT_TORCH_PROFILER_DIRNAME,
                   help="remote path where vLLM writes profiling traces; "
                        "relative paths are resolved under runtime_dir, "
                        "absolute paths are used as-is")
    p.add_argument("--extra-env", action="append", default=None,
                   help="extra KEY=VALUE environment variables for the service (repeatable)")
    p.add_argument(
        "--serve-args",
        default="",
        help="additional vLLM serve arguments appended after the built-in ones "
             "(e.g. --additional-config '{\"enable_cpu_binding\":true}' "
             "or --compilation_config '{\"cudagraph_capture_sizes\":[4,8,16]}'). "
             "Use shlex-style quoting for values with spaces.",
    )

    # ---- multi-node (optional) ----
    p.add_argument("--worker-hosts", help="comma-separated worker IPs for multi-node deployment")
    p.add_argument("--master-port", type=int, help="DP RPC port (default: auto-allocate)")
    p.add_argument("--worker-key-file", help="SSH key for worker nodes (default: same as head)")
    p.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    p.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    p.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    p.add_argument("--worker-extra-env", action="append", default=None,
                   help="extra env vars for worker nodes (repeatable)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _normalize_target_args(args: argparse.Namespace) -> None:
    """Fill the loopback target for local mode and validate remote mode."""
    if args.connection_mode == "local":
        args.host = args.host or "127.0.0.1"
    elif not args.host:
        raise ValueError("--host is required unless --connection-mode local is used")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        _normalize_target_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.connection_mode == "local" and args.worker_hosts:
        raise SystemExit("--connection-mode local supports single-node collection only")

    session_target = None

    run_dir = unique_collection_run_dir(
        tag=args.tag,
        host=args.host,
    )

    # Construct torch_profiler_dir as <base>/<tag> for per-run isolation
    base_dir = args.torch_profiler_dir
    torch_profiler_dir = (
        f"{base_dir}/{args.tag}" if base_dir.startswith("/")
        else f"./{base_dir}/{args.tag}"
    )

    profiler_config = _build_profiler_config(args, torch_profiler_dir)
    # Validate workload before allocating an NPU service.
    _build_benchmark_command(args, "http://127.0.0.1:8000")

    serve_args = _build_serve_args(args, profiler_config)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": now_utc(),
        "tag": args.tag,
        "host": args.host,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "tp": args.tp,
        "dp": args.dp,
        "mode": args.mode,
        "enable_expert_parallel": bool(args.enable_expert_parallel),
        "api_server_count": args.api_server_count,
        "torch_profiler_with_stack": False,
        "torch_profiler_dir": torch_profiler_dir,
        "profile_phase": args.profile_phase,
        "delay_iterations": profiler_config["delay_iterations"],
        "max_iterations": profiler_config["max_iterations"],
        "ignore_frontend": profiler_config["ignore_frontend"],
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "workload_kind": args.workload_kind,
        "random_range_ratio": args.random_range_ratio,
        "mm_input_token_overhead": args.mm_input_token_overhead,
        "image_height": args.image_height if args.workload_kind == "multimodal" else None,
        "image_width": args.image_width if args.workload_kind == "multimodal" else None,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency or args.num_prompts,
        "metric_percentiles": args.metric_percentiles,
        "expected_ranks": args.tp * (args.dp if args.dp else 1),
        "run_dir": str(run_dir),
        "connection_mode": getattr(args, "connection_mode", "docker"),
        "devices": args.devices,
        "serve_args": serve_args,
        "analyse_all_ranks": bool(getattr(args, "analyse_all", False)),
    }
    worker_hosts = getattr(args, "worker_hosts", None)
    if worker_hosts:
        nodes = 1 + worker_hosts.count(",") + 1
        manifest["worker_hosts"] = worker_hosts
        manifest["nnodes"] = nodes

    service_result: dict[str, Any] | None = None
    stop_result: dict[str, Any] | None = None
    try:
        emit_progress("serve_start", f"starting service on {args.host}")
        service_result = call_serve_start(serve_args)
        manifest["service_result"] = service_result
        if service_result.get("status") != "ready":
            raise RuntimeError(f"service did not become ready: {service_result}")

        # Multi-node serve_start reports the port as ``service_port`` while the
        # single-node path uses ``port``; accept either so the workload can be
        # driven against the head API server in both topologies.
        port = int(service_result.get("port") or service_result.get("service_port"))
        profile_root = torch_profiler_dir

        session_target = resolve_execution_target(
            host=args.host, port=args.ssh_port, user=args.user,
            password=args.password, key_file=args.key_file,
            container=args.container,
            conda_env=getattr(args, "conda_env", None),
            connection_mode=getattr(args, "connection_mode", "docker"),
        )
        ep = session_target.endpoint

        # Extract auth for tunnel and SSH calls
        _password, _key_file = None, None
        record = getattr(session_target, "record", {}) or {}
        if record.get("_direct"):
            _password = record.get("_password")
            _key_file = record.get("_key_file")

        request_base_url = f"http://127.0.0.1:{port}"

        tunnel_context = (
            contextlib.nullcontext({
                "local_port": port,
                "base_url": request_base_url,
                "transport": "local",
            })
            if args.connection_mode == "local"
            else open_local_tunnel(ep, port, password=_password, key_file=_key_file)
        )
        with tunnel_context as tunnel:
            manifest["request_tunnel"] = tunnel

            emit_progress("profile_control", "POST /start_profile")
            manifest["start_profile"] = post_remote_action(
                session_target, port, "start_profile", PROFILE_CONTROL_TIMEOUT_SECONDS,
                conda_env=getattr(session_target, "conda_env", None),
            )

            try:
                emit_progress(
                    "workload",
                    f"vllm bench: --num-prompts {args.num_prompts} "
                    f"--max-concurrency {args.max_concurrency or args.num_prompts} "
                    f"--random-input-len {args.random_input_len} "
                    f"--random-output-len {args.random_output_len}",
                )
                bench_cmd = _build_benchmark_command(args, request_base_url)
                # Wrap with conda run when using a conda environment
                conda_env = getattr(args, "conda_env", None)
                if conda_env and not args.container:
                    bench_cmd = [
                        "conda", "run", "-n", conda_env,
                        "--no-capture-output",
                    ] + bench_cmd
                # Build the full remote script with CANN env preamble
                bench_cmd_str = " ".join(shlex.quote(str(s)) for s in bench_cmd)
                cann_dir = getattr(args, "cann_dir", "/usr/local/Ascend")
                # Forward --extra-env vars to benchmark so version pins (e.g.
                # VLLM_VERSION) take effect for both the service AND the workload.
                bench_env_prefix = ""
                extra_env = getattr(args, "extra_env", None)
                if extra_env:
                    bench_env_prefix = "".join(
                        f"export {shlex.quote(e)}; " for e in extra_env
                    )
                remote_bench_script = (
                    ascend_env_preamble(cann_dir) + bench_env_prefix + bench_cmd_str
                )
                emit_progress("workload", f"running: {bench_cmd_str}")
                bench_result = ssh_exec_target(session_target, remote_bench_script, check=False)
                (run_dir / "benchmark.stdout.log").write_text(bench_result.stdout or "", encoding="utf-8")
                (run_dir / "benchmark.stderr.log").write_text(bench_result.stderr or "", encoding="utf-8")
                manifest["benchmark_result"] = {
                    "returncode": bench_result.returncode,
                    "stdout_log": str(run_dir / "benchmark.stdout.log"),
                    "stderr_log": str(run_dir / "benchmark.stderr.log"),
                    "stdout": (bench_result.stdout or "")[-4000:],
                    "stderr": (bench_result.stderr or "")[-4000:],
                }

                if bench_result.returncode != 0:
                    manifest["workload_status"] = {
                        "status": "benchmark_failed",
                        "bench_returncode": bench_result.returncode,
                    }
                else:
                    manifest["workload_status"] = {"status": "ok"}
            finally:
                emit_progress("profile_control", "POST /stop_profile")
                manifest["stop_profile"] = _stop_profile_and_flush(
                    session_target,
                    port,
                    conda_env=getattr(session_target, "conda_env", None),
                )

        emit_progress("serve_stop", "stopping service")
        stop_result = call_serve_stop(
            host=args.host,
            ssh_port=args.ssh_port,
            user=args.user,
            password=args.password,
            key_file=args.key_file,
            container=args.container,
            conda_env=getattr(args, "conda_env", None),
            connection_mode=getattr(args, "connection_mode", "docker"),
            worker_key_file=getattr(args, "worker_key_file", None),
            worker_password=getattr(args, "worker_password", None),
            worker_container=getattr(args, "worker_container", None),
            worker_conda_env=getattr(args, "worker_conda_env", None),
            model=args.model,
        )
        manifest["stop_result"] = stop_result

        # Skip analysis when benchmark failed — traces may be empty/incomplete.
        if bench_result.returncode != 0:
            manifest["remote_profile_root"] = profile_root
            manifest["remote_profile_dirs"] = []
            manifest["rank_count"] = 0
            manifest["analysis_status"] = "skipped_benchmark_failed"
            manifest["completed_at"] = now_utc()
        else:
            cann_dir = getattr(args, "cann_dir", "/usr/local/Ascend")
            cluster_nodes = (service_result or {}).get("nodes", [])
            if len(cluster_nodes) > 1:
                # Multi-node: analyse EVERY node's trace dir (head + workers).
                # Each node runs its own vLLM engine, so each writes its own
                # *_ascend_pt dirs under the (shared) torch_profiler_dir.
                head_dp_local = 1
                for node in cluster_nodes:
                    if node.get("role") == "head" and node.get("dp_size_local"):
                        head_dp_local = int(node["dp_size_local"])
                        break
                per_node_expected = head_dp_local * (manifest["tp"] or 1)
                manifest["expected_ranks"] = per_node_expected
                node_bundles: list[dict[str, Any]] = []
                for node in cluster_nodes:
                    role = node.get("role", "unknown")
                    host = node.get("host", "")
                    if role == "head" or host == args.host:
                        tgt = session_target
                    else:
                        tgt = resolve_worker_target(
                            host,
                            head_record=session_target.record,
                            key_file=getattr(args, "worker_key_file", None) or None,
                            password=getattr(args, "worker_password", None) or None,
                            container=getattr(args, "worker_container", None) or None,
                            conda_env=getattr(args, "worker_conda_env", None) or None,
                        )
                    emit_progress(
                        "analyse",
                        f"[{role} {host}] analysing {profile_root} (expected_ranks={per_node_expected})",
                    )
                    bundle = analyse_profile_root(
                        tgt, profile_root, expected_ranks=per_node_expected,
                        conda_env=getattr(tgt, "conda_env", None) if tgt else None,
                        cann_dir=cann_dir,
                        analyse_all_ranks=getattr(args, "analyse_all", False),
                    )
                    node_bundles.append({"role": role, "host": host, **bundle})

                manifest["remote_profile_root"] = profile_root
                manifest["remote_profile_dirs"] = [
                    d for b in node_bundles for d in b.get("dirs", [])
                ]
                manifest["rank_count"] = sum(b.get("rank_count", 0) for b in node_bundles)
                # Worst-of across nodes: missing_kernel_details > rank_count_mismatch
                # > partial > no_profile_dirs > ok.
                worst = "ok"
                for b in node_bundles:
                    s = b.get("analysis_status", "ok")
                    if s == "missing_kernel_details":
                        worst = "missing_kernel_details"
                    elif s == "rank_count_mismatch" and worst != "missing_kernel_details":
                        worst = "rank_count_mismatch"
                    elif s == "partial" and worst not in ("missing_kernel_details", "rank_count_mismatch"):
                        worst = "partial"
                    elif s == "no_profile_dirs" and worst == "ok":
                        worst = "no_profile_dirs"
                manifest["analysis_status"] = worst
                manifest["node_analysis"] = [
                    {"role": b["role"], "host": b["host"],
                     "rank_count": b.get("rank_count"),
                     "analysis_status": b.get("analysis_status")}
                    for b in node_bundles
                ]
                manifest["completed_at"] = now_utc()
            else:
                # Single node: analyse the head's trace dir only.
                expected_ranks = manifest["expected_ranks"]
                emit_progress(
                    "analyse",
                    f"analysing {profile_root} (expected_ranks={expected_ranks})",
                )
                analyse_bundle = analyse_profile_root(
                    session_target, profile_root, expected_ranks=expected_ranks,
                    conda_env=getattr(session_target, "conda_env", None) if session_target else None,
                    cann_dir=cann_dir,
                    analyse_all_ranks=getattr(args, "analyse_all", False),
                )
                manifest["remote_profile_root"] = profile_root
                manifest["remote_profile_dirs"] = analyse_bundle["dirs"]
                manifest["rank_count"] = analyse_bundle.get("rank_count")
                manifest["analysis_status"] = analyse_bundle["analysis_status"]
                manifest["completed_at"] = now_utc()

        analysis_worst = manifest["analysis_status"]
        workload_worst = manifest["workload_status"]["status"]

        # 输出 profile 文件存放位置提示
        remote_dirs = manifest.get("remote_profile_dirs", [])
        remote_dir_paths = [d["path"] for d in remote_dirs]
        emit_progress(
            "summary",
            f"profiling 数据存放位置: 远端 {profile_root}/ "
            f"(共 {len(remote_dir_paths)} 个 *_ascend_pt 目录), "
            f"本地 manifest: {run_dir / 'manifest.json'}",
        )
        for dir_info in remote_dirs:
            status_icon = "\u2705" if dir_info.get("analysis_status") == "ok" else ("\u23ed\ufe0f" if dir_info.get("analysis_status") == "skipped" else "\u274c")
            emit_progress(
                "summary",
                f"  {status_icon} {dir_info['path']}",
            )

        if analysis_worst == "ok" and workload_worst == "ok":
            manifest["status"] = "ok"
            _write_manifest(run_dir, manifest)
            return 0

        reasons: list[str] = []
        if analysis_worst != "ok":
            reasons.append(f"analysis_status={analysis_worst}")
        if workload_worst != "ok":
            reasons.append(f"workload_status={workload_worst}")
        manifest["status"] = "failed"
        manifest["error"] = (
            "profiling collection produced an unusable trace ("
            + "; ".join(reasons)
            + "); re-collect required"
        )
        _write_manifest(run_dir, manifest)
        return 1

    except (RuntimeError, OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc) or "interrupted"
        manifest["failed_at"] = now_utc()
        if isinstance(exc, KeyboardInterrupt):
            emit_progress("cleanup", "interrupted; stopping launched service to avoid leftover processes")
            manifest["interrupted"] = True
        if stop_result is None:
            try:
                stop_result = call_serve_stop(
                    host=args.host,
                    ssh_port=args.ssh_port,
                    user=args.user,
                    password=args.password,
                    key_file=args.key_file,
                    container=args.container,
                    conda_env=getattr(args, "conda_env", None),
                    connection_mode=getattr(args, "connection_mode", "docker"),
                    worker_key_file=getattr(args, "worker_key_file", None),
                    worker_password=getattr(args, "worker_password", None),
                    worker_container=getattr(args, "worker_container", None),
                    worker_conda_env=getattr(args, "worker_conda_env", None),
                    model=args.model,
                )
                manifest["stop_result"] = stop_result
                # If graceful stop returned but the process refused to die,
                # retry with SIGKILL so we don't leak a zombie vLLM process.
                if stop_result.get("status") not in ("stopped", "not_found"):
                    stop_result = call_serve_stop(
                        host=args.host,
                        ssh_port=args.ssh_port,
                        user=args.user,
                        password=args.password,
                        key_file=args.key_file,
                        container=args.container,
                        conda_env=getattr(args, "conda_env", None),
                        connection_mode=getattr(args, "connection_mode", "docker"),
                        worker_key_file=getattr(args, "worker_key_file", None),
                        worker_password=getattr(args, "worker_password", None),
                        worker_container=getattr(args, "worker_container", None),
                        worker_conda_env=getattr(args, "worker_conda_env", None),
                        model=args.model,
                        force=True,
                    )
                    manifest["stop_result"] = stop_result
            except (RuntimeError, OSError):
                try:
                    stop_result = call_serve_stop(
                        host=args.host,
                        ssh_port=args.ssh_port,
                        user=args.user,
                        password=args.password,
                        key_file=args.key_file,
                        container=args.container,
                        conda_env=getattr(args, "conda_env", None),
                        connection_mode=getattr(args, "connection_mode", "docker"),
                        worker_key_file=getattr(args, "worker_key_file", None),
                        worker_password=getattr(args, "worker_password", None),
                        worker_container=getattr(args, "worker_container", None),
                        worker_conda_env=getattr(args, "worker_conda_env", None),
                        model=args.model,
                        force=True,
                    )
                    manifest["stop_result"] = stop_result
                except (RuntimeError, OSError) as stop_exc:
                    manifest["stop_error"] = str(stop_exc)

        _write_manifest(run_dir, manifest)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
