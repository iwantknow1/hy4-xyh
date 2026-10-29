#!/usr/bin/env python3
"""Shared utilities for vllm-ascend-serving scripts."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]

# TODO: Move it to a shared common directory once one is established
#       These are currently scattered across each skill and should be
#       consolidated into a single shared directory.

from workspace_validate import parse_device_csv
from workspace_output import emit_progress as _emit_progress, print_json, now_utc

SERVING_STATE_DIR = ROOT / ".workspace-local" / "serving"
CLUSTER_STATE_DIR = SERVING_STATE_DIR / "cluster"
PROGRESS_SENTINEL = "__WORKSPACE_SERVING_PROGRESS__="

emit_progress = partial(_emit_progress, sentinel=PROGRESS_SENTINEL)


# ---------------------------------------------------------------------------
# SSH endpoint
# ---------------------------------------------------------------------------

@dataclass
class SshEndpoint:
    """SSH connection target: host + port + user."""
    host: str
    port: int
    user: str = "root"

    def destination(self) -> str:
        return f"{self.user}@{self.host}"

    def known_hosts_key(self) -> str:
        return self.host if self.port == 22 else f"[{self.host}]:{self.port}"

    def to_dict(self, *, plane: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "destination": self.destination(),
            "known_hosts_key": self.known_hosts_key(),
        }
        if plane:
            payload["plane"] = plane
        return payload


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionTarget:
    mode: str
    alias: str
    endpoint: SshEndpoint
    host_endpoint: SshEndpoint
    runtime_base: str
    record: dict[str, Any]
    state_repo_root: Path
    conda_env: str | None = None
    container_name: str | None = None
    connection_mode: str = "docker"  # "docker" | "conda" | "direct"


# Cache dir for auto-generated SSH keys per host
_SSH_KEY_DIR = ROOT / ".workspace-local" / "ssh-keys"


def _ssh_askpass_script(password: str) -> Path:
    """Write a temporary askpass script that prints *password* when invoked.

    Uses an atomic write (mkstemp + os.replace) so concurrent password-auth
    processes do not corrupt the shared askpass file.

    On Windows, OpenSSH can only spawn the askpass helper via CreateProcessW,
    which cannot execute a bare ``.py`` file and mangles non-ASCII paths. We
    therefore write an ``echo``-based ``.cmd`` helper into the (ASCII-only)
    system temp directory so password auth works from PowerShell.
    """
    if os.name != "nt":
        _SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        target = _SSH_KEY_DIR / ".askpass.py"
        content = (
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "sys.stdout.write(os.environ.get('_WORKSPACE_SSH_PASSWORD', ''))\n"
            "sys.stdout.flush()\n"
        )
        _atomic_write_text(target, content)
        return target

    # Windows: an echo-based .cmd in the ASCII system temp dir. The workspace
    # root may contain non-ASCII characters, which breaks OpenSSH's askpass
    # spawn (CreateProcessW receives an ANSI path). Avoid launching `py -3`
    # here too, since quoting/relative-path resolution can also fail.
    #
    # The password is NOT echoed through ``%_WORKSPACE_SSH_PASSWORD%``: cmd
    # re-parses the expanded value, so an ``&`` (or ``|``/``<``/``>``) inside
    # the password is treated as a command separator and truncates it (seen as
    # "'A@vv' is not recognized as an internal or external command" for the
    # password ``2wp1_$4&A@vv``). Instead the password is written verbatim to a
    # sibling .pwd file that ``type`` streams out byte-for-byte — ``type`` does
    # no cmd metacharacter interpretation, so any password is safe.
    #
    # The .pwd/.cmd pair is unique per invocation so concurrent password-auth
    # sessions with different passwords (e.g. multi-node head vs worker) cannot
    # clobber each other's shared askpass file; both are removed at exit.
    uniq = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    pwd_file = Path(tempfile.gettempdir()) / f"mindspeed-askpass-{uniq}.pwd"
    _atomic_write_text(pwd_file, password + "\r\n")
    target = Path(tempfile.gettempdir()) / f"mindspeed-askpass-{uniq}.cmd"
    content = (
        "@echo off\r\n"
        f"type \"{pwd_file}\"\r\n"
    )
    _atomic_write_text(target, content)
    atexit.register(_cleanup_askpass, pwd_file, target)
    return target


def _cleanup_askpass(*paths: Path) -> None:
    """Best-effort removal of the per-invocation askpass helper files."""
    for p in paths:
        with contextlib.suppress(OSError):
            p.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write *content* to *path* (mkstemp + os.replace)."""
    handle, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name != "nt":
            os.chmod(tmp, 0o700)
        try:
            os.replace(tmp, path)
        except OSError as exc:
            # Windows: a concurrent process (e.g. OpenSSH spawning the askpass
            # helper via CreateProcessW) may still hold the destination open,
            # which makes os.replace() raise WinError 5 (access denied). This
            # fallback is only safe when the destination's content is equivalent
            # to what we are writing — true for the fixed-template POSIX
            # .askpass.py helper. The Windows .pwd/.cmd helpers use
            # per-invocation unique names, so they never reach this branch.
            if path.exists() and getattr(exc, "winerror", None) == 5:
                pass  # destination already present with equivalent content
            else:
                raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _ssh_env_for_password(password: str) -> dict[str, str]:
    """Return env dict with SSH_ASKPASS wired to supply *password*."""
    askpass = _ssh_askpass_script(password)
    return {
        "SSH_ASKPASS": str(askpass),
        "SSH_ASKPASS_REQUIRE": "force",
        "DISPLAY": "dummy",
        "_WORKSPACE_SSH_PASSWORD": password,
    }


def _ssh_base_cmd(
    endpoint: SshEndpoint,
    *,
    password: str | None = None,
    key_file: str | None = None,
) -> list[str]:
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
        "-p", str(endpoint.port),
    ]
    if key_file:
        cmd.extend(["-i", key_file])
        cmd.extend(["-o", "BatchMode=yes"])
    elif password:
        cmd.extend(["-o", "PreferredAuthentications=password"])
        cmd.extend(["-o", "PubkeyAuthentication=no"])
    else:
        cmd.extend(["-o", "BatchMode=yes"])
    cmd.append(endpoint.destination())
    return cmd


def ssh_exec(
    endpoint: SshEndpoint,
    script: str,
    *,
    check: bool = True,
    password: str | None = None,
    key_file: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [*_ssh_base_cmd(endpoint, password=password, key_file=key_file), "bash", "-c", shlex.quote(script)]
    env = dict(os.environ)
    if password:
        env.update(_ssh_env_for_password(password))
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace",
        env=env,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"remote command failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr[:2000]}"
        )
    return result


def _container_wrap(script: str, container_name: str | None) -> str:
    """If container_name is given, wrap the script with 'docker exec'."""
    if not container_name:
        return script
    return f"docker exec -i {shlex.quote(container_name)} bash -c {shlex.quote(script)}"


def _target_auth(target: ExecutionTarget) -> tuple[str | None, str | None]:
    """Extract (password, key_file) from an ExecutionTarget's record."""
    record = target.record
    return record.get("_password"), record.get("_key_file")


def check_alive(target: ExecutionTarget, pid: int) -> bool:
    """Return True if *pid* is alive on the remote container.

    pid<=0 is never alive: ``kill -0 0`` succeeds on Linux (PID 0 = current
    process group), which would falsely mark a failed-launch worker (pid=0)
    as alive and later signal the whole process group.

    A zombie (defunct, state Z) process is still present in the process
    table, so ``kill -0`` reports it as alive even though it can never run or
    be signalled.  Such a process is treated as dead here so stop logic does
    not loop on SIGINT/SIGTERM/SIGKILL against an unkillable zombie.
    """
    if not pid or pid <= 0:
        return False
    r = ssh_exec_target(
        target,
        f"stat=$(ps -o stat= -p {pid} 2>/dev/null); "
        f"if [ -z \"$stat\" ] || [ \"$stat\" = \"Z\" ]; then echo dead; "
        f"else echo alive; fi",
        check=False,
    )
    return r.stdout.strip() == "alive"


def ssh_exec_target(
    target: ExecutionTarget,
    script: str,
    *,
    check: bool = True,
    use_host: bool = False,
) -> subprocess.CompletedProcess[str]:
    """ssh_exec that automatically uses the target's auth and container."""
    ep = target.host_endpoint if use_host else target.endpoint
    password, key_file = _target_auth(target)
    wrapped = _container_wrap(script, target.container_name)
    return ssh_exec(ep, wrapped, check=check, password=password, key_file=key_file)


def check_health(target: ExecutionTarget, port: int) -> bool:
    """Check whether the service at *port* responds to /health."""
    script = (
        f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 3"
        f" http://127.0.0.1:{port}/health 2>/dev/null || echo 000"
    )
    r = ssh_exec_target(target, script, check=False)
    return r.stdout.strip() == "200"


def check_models(target: ExecutionTarget, port: int) -> dict[str, Any] | None:
    """Fetch /v1/models from the service at *port*. Returns the parsed JSON or None."""
    script = (
        f"curl -s --connect-timeout 3 http://127.0.0.1:{port}/v1/models"
        f" 2>/dev/null || true"
    )
    r = ssh_exec_target(target, script, check=False)
    text = r.stdout.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if data.get("data"):
            return data
    except json.JSONDecodeError:
        pass
    return None


def ascend_env_preamble(cann_dir: str = "/usr/local/Ascend") -> str:
    """Shell preamble that sources CANN + ATB environment scripts."""
    cann_setup = f"{cann_dir.rstrip('/')}/cann/set_env.sh"
    atb_setup = f"{cann_dir.rstrip('/')}/nnal/atb/set_env.sh"
    return (
        "set -e; " +
        'for _pybin in /usr/local/python3.12.13/bin /usr/local/python3/bin; do   if [ -d "$_pybin" ] && ! printf "%s" ":$PATH:" | grep -q ":$_pybin:"; then     export PATH="$_pybin:$PATH";   fi; done; ' +
        f"if [ -f {shlex.quote(cann_setup)} ]; then"
        f"  set +u; source {shlex.quote(cann_setup)}; set -u;"
        f" else"
        f"  export LD_LIBRARY_PATH="
        f'"{cann_dir.rstrip("/")}/driver/lib64/driver'
        f':{cann_dir.rstrip("/")}/driver/lib64'
        f'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}";'
        f" fi; "
        f"if [ -f {shlex.quote(atb_setup)} ]; then"
        f"  set +u; source {shlex.quote(atb_setup)}; set -u;"
        f" fi; "
    )


def add_ssh_connection_args(parser: argparse.ArgumentParser, *, include_cann: bool = False) -> None:
    """Add standard SSH connection arguments to an argparse parser."""
    parser.add_argument("--host", required=True, help="direct SSH host IP")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--user", default="root", help="SSH user (default: root)")
    parser.add_argument("--password", help="SSH password (if not using key)")
    parser.add_argument("--key-file", help="SSH private key file path")
    parser.add_argument("--container", help="Docker container name to exec into")
    parser.add_argument("--conda-env", help="conda environment name on the remote host "
                       "(mutually exclusive with --container)")
    parser.add_argument("--connection-mode", dest="connection_mode",
                       choices=["docker", "conda", "direct"], default="docker",
                       help="连接模式: docker (SSH 到宿主机 + docker exec 进容器), "
                       "conda (SSH 到宿主机 + 激活 conda 环境), "
                       "direct (SSH 直连容器/虚拟机 IP, 不做 docker exec 包装) — 默认: docker")
    if include_cann:
        parser.add_argument("--cann-dir", default="/usr/local/Ascend",
                           help="CANN installation root directory (default: /usr/local/Ascend)")


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _ensure_ssh_key(endpoint: SshEndpoint, *, password: str | None, key_file: str | None) -> str | None:
    """Ensure we have working SSH key auth. If password is given, auto-generate
    a key pair and copy the public key to the remote host. Returns key_file path
    or None (if password-only auth should be used)."""
    if key_file:
        return key_file
    if not password:
        return None
    # Generate a host-specific key pair
    _SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
    host_key_name = f"id_ed25519_{endpoint.host.replace('.', '_')}_{endpoint.port}"
    private_key = _SSH_KEY_DIR / host_key_name
    if private_key.exists():
        return str(private_key)
    # Generate key
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(private_key), "-N", "", "-q"],
        check=True, capture_output=True,
    )
    pubkey_path = Path(str(private_key) + ".pub")
    pubkey = pubkey_path.read_text().strip()
    # Copy to remote using password
    setup = (
        f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"echo {shlex.quote(pubkey)} >> ~/.ssh/authorized_keys && "
        f"chmod 600 ~/.ssh/authorized_keys"
    )
    result = ssh_exec(endpoint, setup, check=False, password=password)
    if result.returncode != 0:
        # Clean up failed key
        private_key.unlink(missing_ok=True)
        pubkey_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to copy SSH key to {endpoint.destination()}: {result.stderr[:500]}")
    return str(private_key)


def resolve_direct_target(
    host: str,
    *,
    port: int = 22,
    user: str = "root",
    password: str | None = None,
    key_file: str | None = None,
    container: str | None = None,
    conda_env: str | None = None,
    connection_mode: str = "docker",
) -> ExecutionTarget:
    """Build an ExecutionTarget from direct connection params.

    Three connection modes (``connection_mode``):

    ``"docker"`` (default)
        SSH to the host, then ``docker exec -i <container>`` to run commands
        inside the named container.  If *container* is None, commands run
        directly on the host (bare metal).

    ``"conda"``
        SSH to the host, then ``conda activate <env>`` before running commands.
        *conda_env* is required.  No docker involved.

    ``"direct"``
        SSH directly to the target IP (a container or VM running its own SSH
        server).  No ``docker exec`` wrapping.  Password auth is used directly
        without auto-generating a key pair (the target may not support key
        injection).  *container* and *conda_env* must NOT be set.
    """
    _VALID_MODES = ("docker", "conda", "direct")
    if connection_mode not in _VALID_MODES:
        raise RuntimeError(
            f"invalid connection_mode {connection_mode!r}; expected one of {_VALID_MODES}"
        )

    if connection_mode == "direct":
        if container:
            raise RuntimeError(
                "--mode direct is incompatible with --container. "
                "In direct mode, SSH connects to the container's own IP."
            )
        if conda_env:
            raise RuntimeError(
                "--mode direct is incompatible with --conda-env. "
                "In direct mode, the target runs commands natively."
            )

    host_ep = SshEndpoint(host=host, port=port, user=user)
    container_ep = SshEndpoint(host=host, port=port, user=user)

    # In direct mode, skip key generation — the target may not support key injection.
    if connection_mode == "direct":
        resolved_key = key_file
    else:
        resolved_key = _ensure_ssh_key(host_ep, password=password, key_file=key_file)

    # Determine NPU count by probing.
    # In direct mode, npu-smi runs natively in the target (no docker exec).
    # In docker mode with a container, npu-smi is wrapped with docker exec.
    npu_info: dict[str, Any] = {"devices": [], "total": 0}
    try:
        probe_cmd = _container_wrap("npu-smi info", container if connection_mode != "direct" else None)
        result = ssh_exec(host_ep, probe_cmd, check=False, password=password, key_file=resolved_key)
        if result.returncode == 0:
            npu_info = _parse_npu_smi(result.stdout)
    except Exception:
        pass

    if conda_env and container:
        emit_progress(
            "resolve-target",
            f"both --conda-env ({conda_env}) and --container ({container}) specified; "
            f"--conda-env is ignored when --container is set",
        )

    # In docker mode, conda_env is ignored when container is also set.
    effective_conda = conda_env
    if container and connection_mode == "docker":
        effective_conda = None

    record = {
        "host": {"ip": host, "port": port, "user": user},
        "container": {"ssh_port": port, "name": container or "direct"},
        "container_name": container if connection_mode == "docker" else None,
        "conda_env": effective_conda,
        "_direct": True,
        "_password": password,
        "_key_file": resolved_key,
        "_connection_mode": connection_mode,
        "npu_info": npu_info,
    }

    has_container = bool(container and connection_mode == "docker")
    is_container_target = bool(has_container or connection_mode == "direct")

    return ExecutionTarget(
        mode="direct",
        alias=host,
        endpoint=container_ep,
        host_endpoint=host_ep,
        runtime_base="/home/vllm" if is_container_target else "/home/vllm",
        record=record,
        state_repo_root=ROOT,
        conda_env=effective_conda,
        container_name=container if connection_mode == "docker" else None,
        connection_mode=connection_mode,
    )


def resolve_execution_target(
    *,
    host: str,
    port: int = 22,
    user: str = "root",
    password: str | None = None,
    key_file: str | None = None,
    container: str | None = None,
    conda_env: str | None = None,
    connection_mode: str = "docker",
) -> ExecutionTarget:
    """Resolve execution target via direct SSH connection.

    Raises RuntimeError if --host is not provided.
    """
    if not host:
        raise RuntimeError("--host is required")
    return resolve_direct_target(
        host=host, port=port, user=user,
        password=password, key_file=key_file, container=container,
        conda_env=conda_env, connection_mode=connection_mode,
    )


# ---------------------------------------------------------------------------
# Local serving state
# ---------------------------------------------------------------------------

def load_serving_state(
    machine_alias: str,
    *,
    state_repo_root: Path = ROOT,
) -> dict[str, Any] | None:
    path = SERVING_STATE_DIR / f"{machine_alias}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_serving_state(
    machine_alias: str,
    data: dict[str, Any],
    *,
    state_repo_root: Path = ROOT,
) -> Path:
    path = SERVING_STATE_DIR / f"{machine_alias}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
    return path


# ---------------------------------------------------------------------------
# NPU probe
# ---------------------------------------------------------------------------

_HBM_BUSY_THRESHOLD_MB = 6144  # 6 GiB: A5 (Ascend950DT) idle HBM ~4.8 GiB must not be misread as busy


def _parse_npu_smi(output: str) -> dict[str, Any]:
    """Parse ``npu-smi info`` output into a structured dict.

    Handles both the legacy A3-style layout (``| 7     Ascend910 |``, no
    separator between NPU ID and Name) and the separator-delimited A5-style
    layout (``| 0      | Ascend950DT |``).  Auto-detects the machine type and
    per-NPU die count so callers can enforce the correct TP*DP ceiling
    (A3: 8 NPU x 2 dies = 16; A5: 8 x 1 = 8).

    Returns dict with keys: devices, total, busy, free, free_count, hbm,
    hbm_busy_threshold_mb, npu_name, machine_type, per_device_dies.
    """
    dev_ids: set[int] = set()
    hbm: dict[int, dict[str, int]] = {}
    current_npu: int | None = None
    npu_names: list[str] = []
    lines = output.splitlines()
    # NPU header row matches both layouts:
    #   A3: | 7     Ascend910           | OK  | ...   (ID then Name, no ``|``)
    #   A5: | 0      | Ascend950DT      | OK  | ...   (``|`` between ID and Name)
    # The optional ``\|`` between (\d+) and the Name covers A5.  A3 chip rows
    # whose second field is a numeric Phy-ID are never matched because the
    # Name group requires a letter first.
    hdr_re = re.compile(r"^\|\s*(\d+)\s+(?:\|\s*)?([A-Za-z][\w.-]*)\s+\|")
    for line in lines:
        hdr = hdr_re.match(line)
        if hdr:
            current_npu = int(hdr.group(1))
            name = hdr.group(2)
            if name not in npu_names:
                npu_names.append(name)
            dev_ids.add(current_npu)
            continue
        if current_npu is None:
            continue
        # HBM row: A3 carries a PCI ``0000:`` Bus-Id; A5 carries an empty first
        # column plus two adjacent ``x / y`` pairs (Hugepages + HBM).
        has_pci = "0000:" in line
        has_hbm_pair = bool(re.search(r"\d+\s*/\s*\d+\s+\d+\s*/\s*\d+", line))
        if has_pci or has_hbm_pair:
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if len(pairs) >= 2:
                hbm[current_npu] = {
                    "used_mb": int(pairs[-1][0]),
                    "total_mb": int(pairs[-1][1]),
                }
            current_npu = None

    proc_busy: dict[int, list[dict[str, Any]]] = {}
    in_proc = False
    for line in lines:
        if "Process name" in line or "Process memory" in line:
            in_proc = True
            continue
        if in_proc and "No running processes" in line:
            continue
        if in_proc and line.startswith("|"):
            m = re.match(r"\|\s*(\d+)\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)", line)
            if m:
                dev = int(m.group(1))
                proc_busy.setdefault(dev, []).append({
                    "pid": int(m.group(2)),
                    "owner": m.group(3),
                    "name": m.group(4),
                })

    busy: dict[int, list[dict[str, Any]]] = {}
    for dev in sorted(dev_ids):
        reasons: list[dict[str, Any]] = []
        if dev in proc_busy:
            reasons.extend(proc_busy[dev])
        h = hbm.get(dev)
        if h and h["used_mb"] >= _HBM_BUSY_THRESHOLD_MB and dev not in proc_busy:
            reasons.append({
                "pid": None,
                "name": "unknown (HBM occupied, likely another container)",
                "hbm_used_mb": h["used_mb"],
                "detection": "hbm_threshold",
            })
        if reasons:
            busy[dev] = reasons

    free = sorted(d for d in dev_ids if d not in busy)
    npu_name = npu_names[0] if npu_names else None
    machine_type = NPU_NAME_TO_MACHINE.get(npu_name or "") if npu_name else None
    hw = MACHINE_HARDWARE.get(machine_type or "") or {}
    return {
        "devices": sorted(dev_ids),
        "total": len(dev_ids),
        "busy": {str(k): v for k, v in sorted(busy.items())},
        "hbm": {str(k): v for k, v in sorted(hbm.items())},
        "free": free,
        "free_count": len(free),
        "hbm_busy_threshold_mb": _HBM_BUSY_THRESHOLD_MB,
        "npu_name": npu_name,
        "machine_type": machine_type,
        "per_device_dies": hw.get("per_device_dies"),
        "comm_iface": hw.get("comm_iface") or "",
    }


def probe_npus(
    host_ep: SshEndpoint,
    *,
    password: str | None = None,
    key_file: str | None = None,
    container: str | None = None,
) -> dict[str, Any]:
    """Probe NPU device availability via SSH.

    When *container* is set, runs npu-smi inside the container.  Otherwise runs
    on the bare-metal host (recommended for cross-container visibility).

    ``npu-smi`` depends on the CANN toolkit runtime libraries, so we source the
    toolkit and ATB environment scripts (if present) before invoking it.
    """
    probe_lines = [
        "[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && ",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; ",
        "[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && ",
        "source /usr/local/Ascend/nnal/atb/set_env.sh; ",
    ]
    # vLLM on Ascend counts *logical dies* as tensor-parallel devices (a single
    # physical NPU board often contains 2 dies, e.g. davinci14 + davinci15).
    # npu-smi groups those dies under one board id, which is NOT the id vLLM
    # expects in ASCEND_RT_VISIBLE_DEVICES.  We additionally count
    # /dev/davinci* nodes (inside the container when *container* is set — the
    # device nodes are mapped into it) so callers can reason about TP/DP in
    # vLLM device terms and avoid setting a bogus device id.
    probe_lines.append(
        "npu_devs=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l); "
        "echo VLLM_NPU_DEVICE_COUNT=$npu_devs; "
    )
    probe_lines.append("npu-smi info")
    probe_script = "".join(probe_lines)
    script = _container_wrap(probe_script, container)
    result = ssh_exec(host_ep, script, check=False, password=password, key_file=key_file)
    if result.returncode != 0:
        raise RuntimeError(
            f"npu-smi failed (rc={result.returncode}): "
            f"{result.stderr[:500]}"
        )

    # Extract the vLLM-visible logical device count echoed above.
    vllm_dev_count = 0
    for line in result.stdout.splitlines():
        if line.startswith("VLLM_NPU_DEVICE_COUNT="):
            try:
                vllm_dev_count = int(line.split("=", 1)[1])
            except ValueError:
                vllm_dev_count = 0

    parsed = _parse_npu_smi(result.stdout)
    parsed["npu_smi_ok"] = True
    # vLLM-visible logical (die) devices, remapped to 0..n-1 to match the index
    # space torch.npu uses inside the container.
    if vllm_dev_count:
        parsed["vllm_devices"] = list(range(vllm_dev_count))
        parsed["vllm_device_count"] = vllm_dev_count
    return parsed


MACHINE_HARDWARE: dict[str, dict[str, Any]] = {
    # npus_per_node x per_device_dies bounds the per-node TP*DP ceiling.
    # A3 (Ascend910): 8 NPU x 2 chip/die each = 16, TP*DP <= 16.
    # A5 (Ascend950DT): 8 NPU x 1 die each = 8, TP*DP <= 8.
    # comm_iface: mandatory cross-node NIC when a machine type has a special
    # interconnect.  A3 auto-detects via local_ip; A5 must use data0.3001.
    "A3": {"npus_per_node": 8, "per_device_dies": 2, "tp_dp_max": 16, "comm_iface": ""},
    "A5": {"npus_per_node": 8, "per_device_dies": 1, "tp_dp_max": 8, "comm_iface": "data0.3001"},
}

# npu-smi NPU model name -> machine type (drives auto TP*DP ceiling selection).
NPU_NAME_TO_MACHINE: dict[str, str] = {
    "Ascend910": "A3",
    "Ascend950DT": "A5",
}

NPU_DEVICE_RANGE_BY_MACHINE_TYPE: dict[str, tuple[int, int]] = {
    "A3": (0, 15),
    "A5": (0, 7),
}


def select_devices(
    npu_info: dict[str, Any],
    *,
    requested_devices: str | None,
    tp: int | None,
    dp: int | None = None,
    machine_type: str | None = None,
) -> tuple[str | None, str | None]:
    """Validate or auto-select NPU devices.

    Returns (devices_csv, error_message).
    On success error_message is None. On failure devices_csv is None.
    """
    free: list[int] = npu_info.get("free", [])
    busy: dict[str, list] = npu_info.get("busy", {})

    visible_base = set(npu_info.get("devices", []))
    if machine_type:
        device_range = NPU_DEVICE_RANGE_BY_MACHINE_TYPE.get(machine_type.upper())
        if device_range:
            max_device = device_range[1]
            visible = set(range(max_device + 1))
        else:
            visible = visible_base
    else:
        # Container may expose 2 logical devices per physical NPU.
        # Extend the visible range to cover the full logical device space.
        max_physical = max(visible_base) if visible_base else 7
        visible = set(range(max_physical * 2 + 2))
        emit_progress(
            "select-devices",
            f"no machine_type given — assuming up to {max_physical} physical NPUs "
            f"with 2 logical devices each "
            f"(visible range 0–{max(visible)}; detected {sorted(visible_base)})",
        )

    if requested_devices is not None:
        requested = parse_device_csv(requested_devices) or []
        missing = [d for d in requested if d not in visible]
        if missing:
            return None, (
                f"requested devices {missing} are not valid for {machine_type or 'unknown'}; "
                f"valid range={sorted(visible)}"
            )
        conflicts = [d for d in requested if str(d) in busy]
        if conflicts:
            details = {
                str(d): busy[str(d)] for d in conflicts if str(d) in busy
            }
            return None, (
                f"requested devices {conflicts} are busy: {json.dumps(details)}; "
                f"free devices: {free}"
            )
        return ",".join(str(d) for d in requested), None

    if tp is None:
        return None, None

    # Sentinel: tp=0 means "select all free devices" (used in multi-node DP mode
    # where we want to maximize dp_size_local = devices / per_node_tp).
    if tp == 0:
        if not free:
            return None, "no free NPU devices available"
        return ",".join(str(d) for d in free), None

    # Enforce per-node TP*DP ceiling from the machine hardware map
    # (A3: 8 NPU x 2 dies = 16, A5: 8 NPU x 1 = 8).
    if machine_type:
        hw = MACHINE_HARDWARE.get(machine_type.upper())
        if hw and tp is not None:
            tp_dp = tp * (dp or 1)
            if tp_dp > hw["tp_dp_max"]:
                return None, (
                    f"tp*dp={tp_dp} exceeds {machine_type.upper()} machine ceiling "
                    f"{hw['npus_per_node']} NPU x {hw['per_device_dies']} die = "
                    f"{hw['tp_dp_max']} per node"
                )
    need = tp * (dp or 1)
    if len(free) < need:
        return None, (
            f"need {need} free NPUs (tp={tp}, dp={dp or 1}) but only {len(free)} available; "
            f"free={free}, busy={list(busy.keys())}"
        )
    selected = free[:need]
    return ",".join(str(d) for d in selected), None


def largest_contiguous_free(free_devices: list[int]) -> list[int]:
    """Return the largest contiguous block of free NPU device IDs.

    vLLM's DP mode requires ``ASCEND_RT_VISIBLE_DEVICES`` to be contiguous
    (e.g. ``0,1,2,3``, not ``0,1,5,6``).  This helper extracts the longest
    uninterrupted run from the sorted free-device list.
    """
    if not free_devices:
        return []
    free_devices = sorted(free_devices)
    best_start = 0
    best_len = 1
    cur_start = 0
    for i in range(1, len(free_devices)):
        if free_devices[i] == free_devices[i - 1] + 1:
            if i - cur_start + 1 > best_len:
                best_len = i - cur_start + 1
                best_start = cur_start
        else:
            cur_start = i
    if len(free_devices) - cur_start > best_len:
        best_start = cur_start
        best_len = len(free_devices) - cur_start
    return free_devices[best_start:best_start + best_len]


# ---------------------------------------------------------------------------
# Invocation history archive (one record per CLI invocation)
# ---------------------------------------------------------------------------

SERVING_HISTORY_DIR = ROOT / ".workspace-local" / "serving" / "history"


_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_token(value: str, *, fallback: str = "item", max_len: int = 63) -> str:
    """Sanitize a string into a filesystem-safe token with collision-resistant hash suffix."""
    token = _SAFE_TOKEN_RE.sub("-", value.strip()).strip(".-_")
    if not token:
        return fallback
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    keep = max(1, max_len - len(digest) - 1)
    return f"{token[:keep].rstrip('.-_')}-{digest}"


def write_serving_history(
    action: str,
    cli_args: list[str],
    output: dict[str, Any],
    *,
    alias: str = "",
) -> Path:
    """Persist a record of every script invocation to serving history.

    Mirrors ``write_local_result`` in the benchmark skill: one timestamped
    JSON file per call. Call from each top-level script's ``main()`` on both
    success and failure paths.
    """
    ts = now_utc().replace(":", "-")
    tag = safe_token(alias or "direct", fallback="direct")
    filename = f"{ts}_{tag}_{os.getpid()}.json"
    target_dir = SERVING_HISTORY_DIR / action
    target_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "action": action,
        "machine": alias or None,
        "timestamp_utc": now_utc(),
        "cli_args": cli_args,
        "output": output,
    }
    path = target_dir / filename
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Multi-node cluster types and helpers
# ---------------------------------------------------------------------------

@dataclass
class MultiNodeDPConfig:
    """Resolved multi-node cluster configuration (DP-based).

    Uses vLLM's ``--data-parallel-size`` / ``--data-parallel-size-local`` /
    ``--data-parallel-start-rank`` for cross-node scaling, instead of
    torch.distributed ``--nnodes`` / ``--node-rank``.
    """
    head_addr: str                   # head node IP for --data-parallel-address
    dp_rpc_port: int                 # --data-parallel-rpc-port
    dp_size: int                     # total DP instances across all nodes
    dp_size_local: int               # DP instances per node
    nnodes: int                      # total number of nodes (head + workers)


def parse_worker_hosts(raw: str) -> list[str]:
    """Parse comma-separated worker host IPs.

    Returns a deduplicated list, preserving order.
    Raises ``ValueError`` on empty or malformed input.
    """
    if not raw or not raw.strip():
        raise ValueError("--worker-hosts must be a non-empty comma-separated list of IPs")
    hosts: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        host = token.strip()
        if not host:
            raise ValueError("--worker-hosts contains an empty entry")
        if host in seen:
            raise ValueError(f"--worker-hosts contains a duplicate host: {host!r}")
        seen.add(host)
        hosts.append(host)
    return hosts


def resolve_worker_target(
    worker_host: str,
    *,
    head_record: dict[str, Any],
    ssh_port: int | None = None,
    user: str | None = None,
    key_file: str | None = None,
    password: str | None = None,
    container: str | None = None,
    conda_env: str | None = None,
    connection_mode: str | None = None,
) -> ExecutionTarget:
    """Resolve a single worker node target, inheriting config from head.

    Worker inherits container, key_file, user, ssh_port, and
    connection_mode from head unless explicitly overridden by the
    corresponding ``worker_*`` arguments.
    """
    head_host_info = head_record.get("host", {})
    resolved_port = ssh_port if ssh_port is not None else head_host_info.get("port", 22)
    resolved_user = user if user is not None else head_host_info.get("user", "root")
    resolved_key = key_file if key_file is not None else head_record.get("_key_file")
    resolved_password = password if password is not None else head_record.get("_password")
    resolved_container = container if container is not None else head_record.get("container_name")
    resolved_conda = conda_env if conda_env is not None else head_record.get("conda_env")
    resolved_mode = connection_mode if connection_mode is not None else head_record.get("_connection_mode", "docker")

    return resolve_direct_target(
        host=worker_host,
        port=resolved_port,
        user=resolved_user,
        password=resolved_password,
        key_file=resolved_key,
        container=resolved_container,
        conda_env=resolved_conda,
        connection_mode=resolved_mode,
    )


def resolve_multi_node_dp_config(
    *,
    head_addr: str,
    dp_size_local: int,
    nnodes: int,
    dp_rpc_port: int,
) -> MultiNodeDPConfig:
    """Build a ``MultiNodeDPConfig`` from pre-computed DP sizing.

    ``dp_size_local`` is the number of DP (TP) groups per node (same across
    all nodes for symmetry).  Total DP size = dp_size_local * nnodes.
    """
    return MultiNodeDPConfig(
        head_addr=head_addr,
        dp_rpc_port=dp_rpc_port,
        dp_size=dp_size_local * nnodes,
        dp_size_local=dp_size_local,
        nnodes=nnodes,
    )


def build_multi_node_dp_args(
    config: MultiNodeDPConfig,
    node_index: int,  # 0 = head, 1+ = worker
) -> list[str]:
    """Return DP-based vLLM CLI args for a specific node in a multi-node cluster.

    ``node_index``:
      0 → head node (API server + engine)
      1..N → worker nodes (engine only, ``--headless``)
    """
    if node_index < 0 or node_index >= config.nnodes:
        raise ValueError(
            f"node_index {node_index} out of range [0, {config.nnodes})"
        )
    args = [
        "--data-parallel-size", str(config.dp_size),
        "--data-parallel-size-local", str(config.dp_size_local),
        "--data-parallel-address", config.head_addr,
        "--data-parallel-rpc-port", str(config.dp_rpc_port),
    ]
    # Workers (node_index > 0) are engine-only and must declare their DP start
    # rank.  The head node must NOT pass --data-parallel-start-rank: passing an
    # explicit rank 0 makes vLLM 0.28 treat the cluster as external/hybrid DP
    # LB mode, which then forbids --headless on worker engines
    # ("Remote engine N must not use --headless in external or hybrid dp lb
    # mode").  Leaving it unset on the head yields the default coordinated
    # mode where remote engines are headless as expected.
    if node_index > 0:
        dp_start_rank = config.dp_size_local * node_index
        args.append("--data-parallel-start-rank")
        args.append(str(dp_start_rank))
        # Workers only run the engine — no API server
        args.append("--headless")
    return args


# ---------------------------------------------------------------------------
# Multi-node (cluster) state persistence
# ---------------------------------------------------------------------------

def load_cluster_state(
    head_alias: str,
    *,
    state_repo_root: Path = ROOT,
) -> dict[str, Any] | None:
    """Load a persisted multi-node cluster state.

    Cluster states are stored separately from per-machine serving states so
    that the single-node format is never broken.
    """
    path = CLUSTER_STATE_DIR / f"{head_alias}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cluster_state(
    head_alias: str,
    data: dict[str, Any],
    *,
    state_repo_root: Path = ROOT,
) -> Path:
    """Persist a multi-node cluster state atomically."""
    path = CLUSTER_STATE_DIR / f"{head_alias}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
    return path


def load_cluster_state_for_worker(
    worker_alias: str,
    *,
    state_repo_root: Path = ROOT,
) -> dict[str, Any] | None:
    """Find a cluster state that includes *worker_alias* as a worker node.

    Scans all persisted cluster states and returns the first one whose
    ``workers`` list contains *worker_alias*.  Returns ``None`` if the host
    is not part of any known multi-node cluster.
    """
    if not CLUSTER_STATE_DIR.exists():
        return None
    for entry in CLUSTER_STATE_DIR.iterdir():
        if not entry.suffix == ".json":
            continue
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        workers = data.get("workers", [])
        if isinstance(workers, list):
            for w in workers:
                if isinstance(w, dict) and w.get("host") == worker_alias:
                    return data
    return None
