#!/usr/bin/env python3
from __future__ import annotations

"""Stop a vllm-ascend service on a workspace-managed remote container.

Usage:
    python3 serve_stop.py --host <ip> --container <name>
    python3 serve_stop.py --host <ip> --container <name> --force

Progress on stderr, final JSON on stdout.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    _parse_npu_smi,
    check_alive,
    emit_progress,
    load_cluster_state,
    load_cluster_state_for_worker,
    load_serving_state,
    now_utc,
    print_json,
    resolve_execution_target,
    resolve_worker_target,
    save_cluster_state,
    save_serving_state,
    ssh_exec_target,
    write_serving_history,
)


GRACE_PERIOD_SECONDS = 5


def _cleanup_npu_workers(target, state: dict[str, Any]) -> int:
    """Kill orphan NPU worker processes on devices recorded in serving state.

    For non-container (conda / bare-metal) services, child worker processes
    may survive after the main vLLM process is killed.  This function uses
    ``npu-smi info`` to locate any remaining processes on the specific NPU
    chips that were assigned to this service and sends them SIGKILL.

    Returns the number of processes killed.
    """
    devices_csv = state.get("devices", "")
    if not devices_csv:
        return 0

    device_ids: set[int] = set()
    for part in devices_csv.split(","):
        try:
            device_ids.add(int(part.strip()))
        except ValueError:
            continue
    if not device_ids:
        return 0

    # Each physical NPU exposes 2 logical devices → NPU chip = device // 2
    npu_chips: set[int] = {d // 2 for d in device_ids}

    result = ssh_exec_target(target, "npu-smi info 2>/dev/null", check=False, use_host=True)
    if result.returncode != 0:
        return 0

    parsed = _parse_npu_smi(result.stdout)
    busy = parsed.get("busy", {})

    killed = 0
    for npu_str in sorted(busy.keys()):
        try:
            npu_id = int(npu_str)
        except ValueError:
            continue
        if npu_id not in npu_chips:
            continue
        for proc in busy[npu_str]:
            pid_val = proc.get("pid")
            if not pid_val:
                continue
            emit_progress(
                "stop",
                f"cleaning up orphan NPU worker on NPU {npu_id}: pid={pid_val}",
            )
            ssh_exec_target(
                target,
                f"kill -9 {pid_val} 2>/dev/null || true",
                check=False,
                use_host=True,
            )
            killed += 1

    return killed


def _cleanup_orphan_vllm(target, model: str | None = None) -> int:
    """Scan the remote container for leftover ``vllm serve`` processes and SIGKILL them.

    Used as a fallback when serving state is missing or stale (e.g. the
    collector was hard-killed before persisting state), so orphaned leftovers
    from an interrupted launch are still reclaimed.  When *model* is given,
    only processes whose command line contains that model path are matched, to
    avoid killing unrelated services on the same host.  Returns the number of
    top-level processes killed (child ``VLLM::`` processes are reaped too).
    """
    ps = ssh_exec_target(target, "ps -ef", check=False)
    if ps.returncode != 0:
        return 0

    serve_pids: list[int] = []
    for line in ps.stdout.splitlines():
        low = line.lower()
        if "vllm" not in low:
            continue
        if "[vllm]" in low or "<defunct>" in low or "vllm::" in low:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        cmdline = parts[2]
        if "vllm serve" not in cmdline:
            continue
        if model and model not in cmdline:
            continue
        serve_pids.append(pid)

    if not serve_pids:
        return 0

    emit_progress(
        "stop",
        f"orphan cleanup: killing leftover vllm serve pid(s) {serve_pids}",
    )
    killed = 0
    for pid in serve_pids:
        ssh_exec_target(target, f"kill -9 {pid} 2>/dev/null || true", check=False)
        killed += 1
    ssh_exec_target(
        target,
        "for p in $(ps -ef | awk '/[V]LLM::/{print $2}'); do kill -9 $p 2>/dev/null || true; done",
        check=False,
    )
    time.sleep(1)
    return killed


def _stop_multi_node_cluster(
    cluster_state: dict[str, Any],
    head_target,
    head_alias: str,
    model: str | None = None,
    force: bool = False,
    worker_key_file: str | None = None,
    worker_password: str | None = None,
    worker_container: str | None = None,
    worker_conda_env: str | None = None,
) -> int:
    """Stop all nodes in a multi-node cluster.

    *head_target* is the ExecutionTarget for whichever node we connected to
    (head or a worker).  All other worker targets are resolved from the
    cluster state using stored credentials, overridden by the explicit
    ``worker_*`` args (needed when workers use different passwords/keys).
    """
    nodes = cluster_state.get("nodes", [])
    if not nodes:
        print_json({"status": "not_found", "message": "cluster state has no nodes"})
        return 0

    emit_progress("stop", f"stopping multi-node cluster (head={head_alias}, {len(nodes)} nodes)")

    all_stopped = True
    results: list[dict[str, Any]] = []

    for node_info in nodes:
        host = node_info.get("host", "")
        pid = node_info.get("pid")
        role = node_info.get("role", "unknown")

        if not pid:
            tgt = head_target if host == head_alias else resolve_worker_target(
                host,
                head_record=dict(head_target.record),
                key_file=worker_key_file,
                password=worker_password,
                container=worker_container,
                conda_env=worker_conda_env,
            )
            killed = _cleanup_orphan_vllm(tgt, model=model)
            emit_progress("stop", f"[{role} {host}] no pid recorded; orphan sweep killed {killed}")
            results.append({"host": host, "role": role, "status": "swept" if killed else "skipped",
                            "reason": "no pid", "orphan_killed": killed})
            continue

        # Use the provided target for the directly-connected node,
        # otherwise create a target from stored cluster credentials.
        if host == head_alias:
            tgt = head_target
        else:
            head_record = dict(head_target.record)
            # Prefer persisted connection_mode from cluster state so workers
            # are reachable even if the stop invocation uses a different --mode.
            if cluster_state.get("connection_mode"):
                head_record["_connection_mode"] = cluster_state["connection_mode"]
            tgt = resolve_worker_target(
                host,
                head_record=head_record,
                key_file=worker_key_file if worker_key_file else head_record.get("_key_file"),
                password=worker_password if worker_password else head_record.get("_password"),
                container=worker_container if worker_container else head_record.get("container_name"),
                conda_env=worker_conda_env if worker_conda_env else head_record.get("conda_env"),
            )

        alive = check_alive(tgt, int(pid))
        if not alive:
            emit_progress("stop", f"[{role} {host}] pid={pid} already gone")
            # Clean up orphan NPU workers
            _cleanup_npu_workers(tgt, node_info)
            results.append({"host": host, "role": role, "pid": pid, "status": "already_stopped"})
            continue

        # SIGINT
        ssh_exec_target(tgt, f"kill -2 {pid} 2>/dev/null || true", check=False)
        time.sleep(GRACE_PERIOD_SECONDS)

        if check_alive(tgt, int(pid)):
            ssh_exec_target(tgt, f"kill -15 {pid} 2>/dev/null || true", check=False)
            time.sleep(GRACE_PERIOD_SECONDS)

        if check_alive(tgt, int(pid)):
            if force:
                emit_progress("stop", f"[{role} {host}] still alive, sending SIGKILL")
                ssh_exec_target(tgt, f"kill -9 {pid} 2>/dev/null || true", check=False)
                time.sleep(1)
            else:
                all_stopped = False
                results.append({"host": host, "role": role, "pid": pid, "status": "alive", "error": "did not exit after SIGINT+SIGTERM; rerun with --force"})
                continue

        stopped = not check_alive(tgt, int(pid))
        _cleanup_npu_workers(tgt, node_info)
        results.append({"host": host, "role": role, "pid": pid, "status": "stopped" if stopped else "alive"})
        if not stopped:
            all_stopped = False

    # Mark cluster state as stopped
    cluster_state["status"] = "stopped"
    cluster_state["stopped_at"] = now_utc()
    save_cluster_state(head_alias, cluster_state)

    output: dict[str, Any] = {
        "status": "stopped" if all_stopped else "partial",
        "cluster_type": "multi-node",
        "machine": head_alias,
        "nodes": results,
    }
    write_serving_history("serve_stop", sys.argv[1:], output, alias=head_alias)
    print_json(output)
    return 0 if all_stopped else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--host", required=True, help="direct SSH host IP")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    p.add_argument("--user", default="root", help="SSH user")
    p.add_argument("--password", help="SSH password")
    p.add_argument("--key-file", help="SSH private key file")
    p.add_argument("--container", help="Docker container name")
    p.add_argument("--conda-env", help="conda environment name on the remote host")
    p.add_argument("--mode", "--connection-mode", dest="connection_mode",
                   choices=["docker", "conda", "direct"], default="docker",
                   help="连接模式: docker (SSH 到宿主机 + docker exec 进容器), "
                   "conda (SSH 到宿主机 + 激活 conda 环境), "
                   "direct (SSH 直连容器/虚拟机 IP) — 默认: docker")
    p.add_argument("--worker-key-file", help="SSH key for worker nodes (default: same as head)")
    p.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    p.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    p.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    p.add_argument("--model", default=None,
                   help="model path; when state is missing, only kill leftover vllm serve processes whose command line matches this model")
    p.add_argument("--force", action="store_true", help="use SIGKILL if graceful stop fails")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        target = resolve_execution_target(
            host=args.host,
            port=args.ssh_port,
            user=args.user,
            password=args.password,
            key_file=args.key_file,
            container=args.container,
            conda_env=getattr(args, "conda_env", None),
            connection_mode=getattr(args, "connection_mode", "docker"),
        )
        alias = target.alias

        # ---- detect multi-node cluster ----
        cluster_state = load_cluster_state(alias)
        if cluster_state is None:
            # Maybe we're targeting a worker node directly
            cluster_state = load_cluster_state_for_worker(alias)

        if cluster_state and cluster_state.get("cluster_type") == "multi-node":
            return _stop_multi_node_cluster(
                cluster_state, target, alias,
                model=args.model,
                force=getattr(args, "force", False),
                worker_key_file=getattr(args, "worker_key_file", None),
                worker_password=getattr(args, "worker_password", None),
                worker_container=getattr(args, "worker_container", None),
                worker_conda_env=getattr(args, "worker_conda_env", None),
            )

        state = load_serving_state(
            alias,
            state_repo_root=target.state_repo_root,
        )
        if state is None:
            killed = _cleanup_orphan_vllm(target, model=args.model)
            if killed:
                state = {"model": args.model, "machine": alias, "status": "stopped",
                         "stopped_at": now_utc(), "mode": target.mode,
                         "connection_mode": target.connection_mode,
                         "cann_dir": getattr(args, "cann_dir", "/usr/local/Ascend")}
                save_serving_state(alias, state, state_repo_root=target.state_repo_root)
                print_json({
                    "status": "stopped",
                    "machine": alias,
                    "mode": target.mode,
                    "message": "no serving state recorded; swept leftover vllm processes",
                    "orphan_killed": killed,
                })
            else:
                print_json({
                    "status": "not_found",
                    "machine": alias,
                    "mode": target.mode,
                    "message": "no serving state recorded for this machine",
                })
            return 0

        pid = state.get("pid")
        if not pid:
            killed = _cleanup_orphan_vllm(target, model=args.model)
            state["status"] = "stopped"
            state["stopped_at"] = now_utc()
            save_serving_state(alias, state, state_repo_root=target.state_repo_root)
            print_json({
                "status": "stopped",
                "machine": alias,
                "mode": target.mode,
                "message": "serving state has no pid; swept leftover vllm processes",
                "orphan_killed": killed,
            })
            return 0

        alive = check_alive(target, pid)
        if not alive:
            emit_progress("stop", f"pid={pid} is already gone")
            # Clean up orphan NPU workers for non-container services
            _cleanup_npu_workers(target, state)
            state["status"] = "stopped"
            state["stopped_at"] = now_utc()
            save_serving_state(
                alias,
                state,
                state_repo_root=target.state_repo_root,
            )
            print_json({
                "status": "stopped",
                "machine": alias,
                "mode": target.mode,
                "pid": pid,
                "message": "process was already stopped",
            })
            return 0

        # SIGINT first (graceful)
        emit_progress("stop", f"sending SIGINT to pid={pid}")
        ssh_exec_target(target, f"kill -2 {pid} 2>/dev/null || true", check=False)
        time.sleep(GRACE_PERIOD_SECONDS)

        if check_alive(target, pid):
            emit_progress("stop", f"still alive, sending SIGTERM to pid={pid}")
            ssh_exec_target(target, f"kill -15 {pid} 2>/dev/null || true", check=False)
            time.sleep(GRACE_PERIOD_SECONDS)

        if check_alive(target, pid):
            if args.force:
                emit_progress("stop", f"still alive, sending SIGKILL to pid={pid}")
                ssh_exec_target(target, f"kill -9 {pid} 2>/dev/null || true", check=False)
                time.sleep(1)
            else:
                print_json({
                    "status": "failed",
                    "machine": alias,
                    "mode": target.mode,
                    "pid": pid,
                    "error": f"process {pid} did not exit after SIGINT+SIGTERM; rerun with --force to SIGKILL",
                })
                return 1

        stopped = not check_alive(target, pid)
        # Clean up orphan NPU workers for non-container services
        npu_cleaned = _cleanup_npu_workers(target, state)
        state["status"] = "stopped" if stopped else "alive"
        state["stopped_at"] = now_utc()
        save_serving_state(
            alias,
            state,
            state_repo_root=target.state_repo_root,
        )

        output: dict[str, Any] = {
            "status": "stopped" if stopped else "failed",
            "machine": alias,
            "mode": target.mode,
            "pid": pid,
            "stopped": stopped,
        }
        if npu_cleaned > 0:
            output["npu_workers_cleaned"] = npu_cleaned
        if not stopped:
            output["error"] = "process refused to exit"

        write_serving_history("serve_stop", sys.argv[1:], output, alias=alias)
        print_json(output)
        return 0 if stopped else 1

    except Exception as exc:
        fail_output = {
            "status": "failed",
            "error": str(exc),
            "host": getattr(args, "host", None),
        }
        write_serving_history("serve_stop", sys.argv[1:], fail_output,
                              alias=getattr(args, "host", None) or "")
        print_json(fail_output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
