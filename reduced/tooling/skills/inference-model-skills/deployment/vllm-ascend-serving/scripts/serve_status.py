#!/usr/bin/env python3
from __future__ import annotations

"""Check the status of a running vllm-ascend service.

Usage:
    python3 serve_status.py --host <ip> --container <name>

Progress on stderr, final JSON on stdout.
"""

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    check_alive,
    check_health,
    check_models,
    emit_progress,
    load_serving_state,
    load_cluster_state,
    load_cluster_state_for_worker,
    now_utc,
    print_json,
    resolve_execution_target,
    resolve_worker_target,
    save_serving_state,
    ssh_exec_target,
    write_serving_history,
)


def _show_multi_node_status(
    cluster_state,
    head_target,
    *,
    worker_key_file: str | None = None,
    worker_password: str | None = None,
    worker_container: str | None = None,
    worker_conda_env: str | None = None,
):
    """Display aggregated status for a multi-node cluster.

    Each node's liveness must be checked ON that node — a pid is only valid in
    its own PID namespace (especially across ``docker exec``).  Head checks
    also run on the head node (so the query may come from a worker node).
    """
    nodes = cluster_state.get("nodes", [])
    connected_alias = head_target.alias
    head_info: dict[str, Any] = {}
    worker_list: list[dict[str, Any]] = []

    for node in nodes:
        host = node.get("host", "")
        pid = node.get("pid")
        port = node.get("port", 0)
        role = node.get("role", "unknown")

        if host == connected_alias:
            tgt = head_target
        else:
            head_record = dict(head_target.record)
            # Prefer persisted connection_mode from cluster state so workers
            # are reachable even if this invocation uses a different --mode.
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

        alive = check_alive(tgt, pid) if pid else False

        if role == "head":
            health = check_health(tgt, port) if alive and port else False
            models = check_models(tgt, port) if health else None
            head_info = {
                "host": host, "role": "head", "pid": pid, "port": port,
                "alive": alive, "health": health, "models_ok": models is not None,
                "base_url": node.get("base_url"),
                "devices": node.get("devices"),
                "local_world_size": node.get("local_world_size"),
            }
        else:
            # Workers run headless (no HTTP endpoint); only liveness is checked.
            worker_list.append({
                "host": host, "role": "worker", "pid": pid,
                "alive": alive,
                "devices": node.get("devices"),
                "local_world_size": node.get("local_world_size"),
            })

    head_ok = head_info.get("alive") and head_info.get("health") and head_info.get("models_ok")
    workers_ok = all(w.get("alive") for w in worker_list)
    all_healthy = head_ok and workers_ok

    output: dict[str, Any] = {
        "status": "ready" if all_healthy else "degraded",
        "cluster_type": "multi-node",
        "machine": cluster_state.get("head_addr", ""),
        "nnodes": cluster_state.get("nnodes"),
        "head": head_info,
        "workers": worker_list,
        "total_nodes": len(nodes),
        "all_nodes_healthy": all_healthy,
        "model": cluster_state.get("model"),
        "served_model_name": cluster_state.get("served_model_name"),
        "started_at": cluster_state.get("started_at"),
    }
    write_serving_history("serve_status", sys.argv[1:], output, alias=cluster_state.get("head_addr", ""))
    print_json(output)
    return 0 if all_healthy else 1


def _show_multi_node_status_from_cluster(cluster_state):
    """Display cluster status when querying a worker node directly."""
    nodes = cluster_state.get("nodes", [])
    head_info = next((n for n in nodes if n.get("role") == "head"), {})
    output: dict[str, Any] = {
        "status": "ok",
        "cluster_type": "multi-node",
        "machine": cluster_state.get("head_addr", ""),
        "message": "this host is a worker in a multi-node cluster; showing cluster-level status",
        "nnodes": cluster_state.get("nnodes"),
        "head": head_info,
        "workers": [n for n in nodes if n.get("role") == "worker"],
        "model": cluster_state.get("model"),
        "started_at": cluster_state.get("started_at"),
    }
    print_json(output)
    return 0


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

        state = load_serving_state(
            alias,
            state_repo_root=target.state_repo_root,
        )
        if state is None:
            # check if this host belongs to a multi-node cluster as a worker
            cluster_state = load_cluster_state_for_worker(alias)
            if cluster_state:
                return _show_multi_node_status_from_cluster(cluster_state)
            print_json({
                "status": "not_found",
                "machine": alias,
                "mode": target.mode,
                "message": "no serving state recorded for this machine",
            })
            return 0

        # ---- multi-node cluster status ----
        if state.get("cluster_type") == "multi-node":
            cluster_state = load_cluster_state(state.get("cluster_head", alias))
            if cluster_state:
                return _show_multi_node_status(
                    cluster_state, target,
                    worker_key_file=getattr(args, "worker_key_file", None),
                    worker_password=getattr(args, "worker_password", None),
                    worker_container=getattr(args, "worker_container", None),
                    worker_conda_env=getattr(args, "worker_conda_env", None),
                )
            # fall through to per-machine status if cluster state is missing

        pid = state.get("pid")
        port = state.get("port")
        if not pid or not port:
            print_json({
                "status": "not_found",
                "machine": alias,
                "mode": target.mode,
                "message": "serving state is missing pid or port",
                "state": state,
            })
            return 0

        emit_progress("probe", f"checking pid={pid} port={port}")

        alive = check_alive(target, pid)
        health = check_health(target, port) if alive else False
        models = check_models(target, port) if health else None

        if alive and health and models is not None:
            status = "ready"
        elif alive and health:
            status = "alive_healthy"
        elif alive:
            status = "alive"
        else:
            status = "stopped"

        state["status"] = status
        state["status_checked_at"] = now_utc()
        if status == "stopped":
            if state.get("stopped_at") is None:
                state["stopped_at"] = state["status_checked_at"]
        save_serving_state(
            alias,
            state,
            state_repo_root=target.state_repo_root,
        )


        output: dict[str, Any] = {
            "status": status,
            "machine": alias,
            "mode": target.mode,
            "alive": alive,
            "health": health,
            "models_ok": models is not None,
            "pid": pid,
            "port": port,
            "base_url": state.get("base_url"),
            "served_model_name": state.get("served_model_name"),
            "model": state.get("model"),
            "log_stdout": state.get("log_stdout"),
            "log_stderr": state.get("log_stderr"),
            "runtime_dir": state.get("runtime_dir"),
            "started_at": state.get("started_at"),
        }

        if not alive:
            stderr_path = state.get("log_stderr")
            if stderr_path:
                r = ssh_exec_target(
                    target,
                    f"tail -20 {shlex.quote(stderr_path)} 2>/dev/null || echo '(no log)'",
                    check=False,
                )
                output["stderr_tail"] = r.stdout.strip()

        write_serving_history("serve_status", sys.argv[1:], output, alias=alias)
        print_json(output)
        return 0

    except Exception as exc:
        fail_output = {
            "status": "failed",
            "error": str(exc),
            "host": getattr(args, "host", None),
        }
        write_serving_history("serve_status", sys.argv[1:], fail_output,
                              alias=getattr(args, "host", None) or "")
        print_json(fail_output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
