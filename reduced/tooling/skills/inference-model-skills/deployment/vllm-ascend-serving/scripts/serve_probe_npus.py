#!/usr/bin/env python3
from __future__ import annotations

"""Probe NPU device availability on a remote host.

Runs ``npu-smi info`` on the host (optionally inside a container).

Usage:
    # Direct SSH:
    python3 serve_probe_npus.py --host xxx --password xxx
    python3 serve_probe_npus.py --host xxx --password xxx --container my-container

Returns JSON with total devices, which are busy (with PIDs), and which are free.
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    SshEndpoint,
    _ensure_ssh_key,
    emit_progress,
    print_json,
    probe_npus,
    write_serving_history,
    parse_worker_hosts,
    resolve_worker_target,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    # Direct mode
    p.add_argument("--host", required=True, help="direct SSH host IP")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    p.add_argument("--user", default="root", help="SSH user (default: root)")
    p.add_argument("--password", help="SSH password")
    p.add_argument("--key-file", help="SSH private key file path")
    p.add_argument("--container", help="Docker container name to exec into")
    p.add_argument("--mode", "--connection-mode", dest="connection_mode",
                   choices=["docker", "conda", "direct"], default="docker",
                   help="连接模式: docker (SSH 到宿主机 + docker exec 进容器), "
                   "conda (SSH 到宿主机 + 激活 conda 环境), "
                   "direct (SSH 直连容器/虚拟机 IP) — 默认: docker")
    p.add_argument("--worker-hosts", help="comma-separated worker node IPs for multi-node NPU probing")
    p.add_argument("--worker-key-file", help="SSH key for worker nodes (default: same as head)")
    p.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    p.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    p.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        connection_mode = getattr(args, "connection_mode", "docker")
        ep = SshEndpoint(host=args.host, port=args.ssh_port, user=args.user)
        # In direct mode, skip key generation — the target may not support
        # key injection. Use password or explicit key file directly.
        if connection_mode == "direct":
            key_file = args.key_file
        else:
            key_file = _ensure_ssh_key(ep, password=args.password, key_file=args.key_file)
        # In direct mode, npu-smi runs natively (no docker exec wrapping).
        probe_container = args.container if connection_mode != "direct" else None
        emit_progress("probe-npus", f"probing NPU devices on {args.host}:{args.ssh_port}")
        npu_info = probe_npus(ep, password=args.password, key_file=key_file, container=probe_container)
        output: dict[str, Any] = {
            "status": "ok",
            "host": args.host,
            "mode": connection_mode,
            **npu_info,
        }

        # ---- multi-node: probe workers too ----
        if getattr(args, "worker_hosts", None):
            worker_hosts = parse_worker_hosts(args.worker_hosts)
            head_record = {
                "_key_file": key_file,
                "_password": args.password,
                "container_name": args.container if connection_mode != "direct" else None,
                "_connection_mode": connection_mode,
                "host": {"port": args.ssh_port, "user": args.user},
            }
            worker_results: dict[str, Any] = {}
            for wh in worker_hosts:
                wt = resolve_worker_target(
                    wh, head_record=head_record,
                    key_file=getattr(args, "worker_key_file", None) or None,
                    password=getattr(args, "worker_password", None) or None,
                    container=getattr(args, "worker_container", None) or None,
                    conda_env=getattr(args, "worker_conda_env", None) or None,
                )
                w_ep = SshEndpoint(host=wh, port=wt.record["host"]["port"],
                                   user=wt.record["host"]["user"])
                w_mode = wt.record.get("_connection_mode", "docker")
                if w_mode == "direct":
                    w_key = wt.record.get("_key_file")
                else:
                    w_key = _ensure_ssh_key(w_ep, password=wt.record.get("_password"),
                                           key_file=wt.record.get("_key_file"))
                emit_progress("probe-npus", f"probing NPU devices on worker {wh}")
                w_container = wt.record.get("container_name") if w_mode != "direct" else None
                w_npu = probe_npus(w_ep, password=wt.record.get("_password"),
                                  key_file=w_key, container=w_container)
                worker_results[wh] = {"role": "worker", **w_npu}

            all_nodes = {args.host: {"role": "head", **npu_info}, **worker_results}
            total_all = sum(n.get("total", 0) for n in all_nodes.values())
            total_free = sum(n.get("free_count", 0) for n in all_nodes.values())
            output = {
                "status": "ok",
                "cluster_type": "multi",
                "nodes": all_nodes,
                "summary": {
                    "total_devices": total_all,
                    "total_free": total_free,
                    "nodes_probed": len(all_nodes),
                },
            }

        write_serving_history("serve_probe", sys.argv[1:], output,
                              alias=args.host or "")
        print_json(output)
        return 0

    except Exception as exc:
        fail_output = {
            "status": "failed",
            "error": str(exc),
            "host": getattr(args, "host", None),
        }
        write_serving_history("serve_probe", sys.argv[1:], fail_output,
                              alias=getattr(args, "host", None) or "")
        print_json(fail_output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
