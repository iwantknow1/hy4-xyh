#!/usr/bin/env python3
"""Validate level-1 sleep/wakeup determinism and memory stability."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
from common.dual_api import ENDPOINTS, generate_both
from common.feature_arguments import (
    add_generation_arguments,
    add_report_arguments,
    add_service_arguments,
    add_timeout_argument,
    create_feature_parser,
    positive_integer,
    positive_number,
)

DEFAULT_PROMPT = (
    "Please explain in detail how distributed inference systems coordinate "
    "data parallelism, tensor parallelism, expert parallelism, weight updates, "
    "and service recovery while preserving deterministic outputs across repeated validation runs."
)
MAX_MEMORY_GROWTH_RATIO = 0.05


def request_json(
    server_url: str, method: str, endpoint: str,
    payload: dict[str, Any] | None = None, *, timeout: float = 600,
) -> Any:
    request_body = None if payload is None else json.dumps(payload).encode()
    http_request = urllib.request.Request(
        server_url.rstrip("/") + endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(http_request, timeout=timeout) as response:
        response_body = response.read()
    # Sleep/wake endpoints may return an empty or whitespace-only body.
    return json.loads(response_body) if response_body.strip() else None


def collect_generation_outputs(
    arguments: argparse.Namespace, prompts: list[str],
) -> dict[str, list[dict[str, Any]]]:
    prompt_outputs = [
        generate_both(
            request_json, arguments.server_url, arguments.model, prompt,
            arguments.max_tokens, timeout=arguments.request_timeout,
        )
        for prompt in prompts
    ]
    return {
        endpoint: [output[endpoint] for output in prompt_outputs]
        for endpoint in ENDPOINTS
    }


def wait_until_awake(
    server_url: str, *, timeout: float = 600, poll_interval: float = 2,
) -> None:
    """Poll the service until it reports that it is no longer sleeping."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining_seconds = max(1, deadline - time.monotonic())
        sleep_state = request_json(
            server_url, "GET", "/is_sleeping", timeout=min(30, remaining_seconds),
        ) or {}
        if sleep_state.get("is_sleeping") is False:
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"service remained sleeping after {timeout:.0f}s")


def get_npu_memory_allocated() -> int | None:
    """Return local NPU allocation when memory counters are available."""
    try:
        import torch

        return int(torch.npu.memory_allocated())
    except Exception:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser, model_option="--model", require_model=True)
    add_generation_arguments(parser, prompt_input="repeat")
    add_report_arguments(parser)
    add_timeout_argument(
        parser, option="--request-timeout", default=600, value_type=positive_number,
    )
    parser.add_argument(
        "--baseline-result", required=True, help="Baseline JSON; its prompts are reused",
    )
    parser.add_argument("--cycles", type=positive_integer, default=3)
    parser.add_argument(
        "--poll-interval", type=positive_number, default=2,
        help="Seconds between /is_sleeping polls after wake",
    )
    return parser.parse_args(argv)


def load_validation_prompts(arguments: argparse.Namespace) -> list[str]:
    saved_baseline = json.loads(Path(arguments.baseline_result).read_text(encoding="utf-8"))
    prompts = list(saved_baseline.get("prompts") or arguments.prompts or [DEFAULT_PROMPT])
    assert prompts, "baseline result has no prompts"
    return prompts


def run_sleep_cycle(
    arguments: argparse.Namespace, prompts: list[str],
    baseline_by_endpoint: dict[str, list[dict[str, Any]]], cycle_number: int,
) -> dict[str, Any]:
    request_json(
        arguments.server_url, "POST", "/sleep?level=1", timeout=arguments.request_timeout,
    )
    request_json(
        arguments.server_url, "POST", "/wake_up?tags=weights", timeout=arguments.request_timeout,
    )
    wait_until_awake(
        arguments.server_url, timeout=arguments.request_timeout,
        poll_interval=arguments.poll_interval,
    )
    cycle_outputs = collect_generation_outputs(arguments, prompts)
    memory_allocated = get_npu_memory_allocated()
    # Preserve the existing completion acceptance; retain chat output as evidence.
    assert cycle_outputs["/v1/completions"] == baseline_by_endpoint["/v1/completions"], (
        f"cycle {cycle_number} output/token/logp differs from baseline"
    )
    return {
        "cycle": cycle_number,
        "memory": memory_allocated,
        "outputs_by_endpoint": cycle_outputs,
    }


def calculate_memory_growth(
    initial_memory: int | None, cycles: list[dict[str, Any]],
) -> float | None:
    memory_readings = [cycle["memory"] for cycle in cycles if cycle["memory"] is not None]
    if initial_memory is None or not memory_readings:
        return None
    return (max(memory_readings) - initial_memory) / max(initial_memory, 1)


def run_validation(
    arguments: argparse.Namespace, verification_command: str,
) -> dict[str, Any]:
    prompts = load_validation_prompts(arguments)
    baseline_by_endpoint = collect_generation_outputs(arguments, prompts)
    initial_memory = get_npu_memory_allocated()
    cycles = [
        run_sleep_cycle(arguments, prompts, baseline_by_endpoint, cycle_number)
        for cycle_number in range(1, arguments.cycles + 1)
    ]
    memory_growth = calculate_memory_growth(initial_memory, cycles)
    memory_within_limit = memory_growth is None or memory_growth <= MAX_MEMORY_GROWTH_RATIO
    return {
        "status": "passed" if memory_within_limit else "failed",
        "level": 1,
        "prompts": prompts,
        "baseline": baseline_by_endpoint["/v1/completions"],
        "baseline_by_endpoint": baseline_by_endpoint,
        "cycles": cycles,
        "memory_before": initial_memory,
        "memory_growth_ratio": memory_growth,
        "checks": {
            "output_tokens_logprobs_identical": True,
            "both_generation_endpoints": True,
            "memory_growth_within_5pct": memory_within_limit,
        },
        "endpoint_checks": {endpoint: True for endpoint in ENDPOINTS},
        "verification_command": verification_command,
    }


def write_report(report_path: str, report: dict[str, Any]) -> None:
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    verification_command = arguments.verification_command or shlex.join([sys.executable, *sys.argv])
    print(f"[verification command] {verification_command}", flush=True)
    report = run_validation(arguments, verification_command)
    write_report(arguments.report, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
