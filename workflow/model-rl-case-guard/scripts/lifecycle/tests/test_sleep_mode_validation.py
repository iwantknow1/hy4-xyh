"""Stateful sleep regression checks without NPU hardware or external services."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_sleep_mode as sleep_test
from common.dual_api import ENDPOINTS, generate_both


class InferenceService:
    """Model the executor's two sleeping tags and both OpenAI response schemas."""

    def __init__(self, fault: str | None = None):
        self.fault = fault
        self.tags: set[str] = set()
        self.calls: list[str] = []
        self.payloads: list[tuple[str, dict]] = []
        self.wakes = 0
        self.sleep_attempted = False

    def request(self, base, method, endpoint, payload=None, *, timeout=600):
        self.calls.append(endpoint)
        if endpoint == "/is_sleeping":
            if self.fault == "state_unavailable" and self.sleep_attempted:
                raise ConnectionError("state unavailable")
            return {"is_sleeping": bool(self.tags)}
        if endpoint == "/health":
            return None
        if endpoint.startswith("/sleep?"):
            self.sleep_attempted = True
            if self.fault != "sleep_noop":
                self.tags = {"weights", "kv_cache"}
            if self.fault == "sleep_timeout":
                raise TimeoutError("sleep request timeout")
            return None
        if endpoint == "/wake_up?tags=weights":
            self.tags.discard("weights")
            if self.fault == "early_wake":
                self.tags.clear()
            return None
        if endpoint == "/wake_up?tags=kv_cache":
            if self.fault != "wake_stuck":
                self.tags.discard("kv_cache")
            self.wakes += 1
            return None
        if endpoint not in ENDPOINTS:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        if self.tags:
            raise RuntimeError("generation attempted before all tags were awake")
        self.payloads.append((endpoint, payload))
        text, tokens, logprobs = "answer", ["answer"], [-0.25]
        if self.wakes and self.fault and self.fault.startswith("chat_") and endpoint == ENDPOINTS[1]:
            if self.fault == "chat_text":
                text = "different"
            elif self.fault == "chat_tokens":
                tokens = ["different"]
            elif self.fault == "chat_logprobs":
                logprobs = [-0.5]
            elif self.fault == "chat_missing_logprobs":
                logprobs = [None]
        if endpoint == ENDPOINTS[0]:
            return {"choices": [{"text": text, "logprobs": {"tokens": tokens, "token_logprobs": logprobs}}]}
        return {"choices": [{"message": {"content": text}, "logprobs": {"content": [
            {"token": token, "logprob": probability} for token, probability in zip(tokens, logprobs)
        ]}}]}


class SleepValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(json.dumps({"prompts": ["first", "second"]}), encoding="utf-8")
        self.args = SimpleNamespace(
            server_url="http://fixture.invalid", model="policy", max_tokens=32,
            request_timeout=0.02, poll_interval=0.001, cycles=1, prompts=None,
            baseline_result=str(self.baseline), report=str(self.root / "sleep.json"),
            verification_command="fixture sleep command", memory_probe_command=None, memory_devices=[],
        )

    def validate(self, service=None):
        service = service or InferenceService()
        with patch.object(sleep_test, "request_json", side_effect=service.request):
            return sleep_test.run_validation(self.args, "fixture")

    def probe(self, readings):
        self.args.memory_probe_command = "python probe.py --host head --worker-hosts worker"
        self.args.memory_devices = ["head/0", "worker/1"]
        results = []
        for head_used, worker_used in readings:
            data = {"status": "ok", "nodes": {
                "head": {"hbm": {"0": {"used_mb": head_used, "total_mb": 2000}}},
                "worker": {"hbm": {"1": {"used_mb": worker_used, "total_mb": 2000}}},
            }}
            results.append(SimpleNamespace(returncode=0, stdout=json.dumps(data, indent=2), stderr="progress\n"))
        return patch.object(sleep_test.subprocess, "run", side_effect=results)

    def test_complete_two_stage_wake_and_all_prompts(self):
        self.args.cycles = 2
        service = InferenceService()
        report = self.validate(service)
        self.assertEqual(report["status"], "passed_http_only")
        self.assertEqual(service.wakes, 2)
        for cycle in report["cycles"]:
            self.assertTrue(cycle["sleep_state_after_sleep"])
            self.assertTrue(cycle["sleep_state_after_weights"])
            self.assertTrue(cycle["checks"]["fully_awake"])
            self.assertEqual(cycle["outputs_identical_by_endpoint"], {endpoint: [True, True] for endpoint in ENDPOINTS})
        for endpoint, payload in service.payloads:
            self.assertEqual(payload["logprobs"], True if endpoint == ENDPOINTS[1] else 1)
            if endpoint == ENDPOINTS[1]:
                self.assertEqual(payload["top_logprobs"], 1)

    def test_missing_memory_is_not_claimed_as_passed(self):
        report = self.validate()
        self.assertEqual(report["status"], "passed_http_only")
        self.assertEqual(report["memory_validation"]["status"], "not_run")
        self.assertIsNone(report["checks"]["memory_growth_within_5pct"])
        self.assertIsNone(report["checks"]["memory_released"])

    def test_chat_only_changes_fail_for_each_field(self):
        for field in ("text", "tokens", "logprobs"):
            with self.subTest(field=field):
                report = self.validate(InferenceService("chat_" + field))
                self.assertEqual(report["status"], "failed")
                self.assertIs(report["endpoint_checks"][ENDPOINTS[1]], False)
                self.assertIs(report["endpoint_checks"][ENDPOINTS[0]], True)
                self.assertEqual(report["failed_phase"], "compare_outputs")
                self.assertEqual(len(report["baseline_outputs_by_endpoint"][ENDPOINTS[1]]), 2)

    def test_missing_logprobs_keep_partial_completed_outputs(self):
        report = self.validate(InferenceService("chat_missing_logprobs"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_phase"], "generation_after_wake")
        self.assertEqual(len(report["feature_outputs_by_endpoint"][ENDPOINTS[0]]), 1)
        self.assertEqual(report["feature_outputs_by_endpoint"][ENDPOINTS[1]], [])
        self.assertIsNone(report["endpoint_checks"][ENDPOINTS[1]])

    def test_noop_sleep_cannot_pass(self):
        service = InferenceService("sleep_noop")
        report = self.validate(service)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["cycles"][0]["checks"]["sleep_state_confirmed"])
        self.assertNotIn("/wake_up?tags=weights", service.calls)

    def test_partial_wake_must_keep_kv_cache_asleep(self):
        report = self.validate(InferenceService("early_wake"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_phase"], "wake_weights")
        self.assertFalse(report["cycles"][0]["checks"]["partial_wake_still_sleeping"])

    def test_sleep_timeout_preserves_sleeping_service_state(self):
        report = self.validate(InferenceService("sleep_timeout"))
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["server_left_sleeping"])
        self.assertEqual(report["failed_phase"], "sleep")
        self.assertIn("recovery", report)

    def test_state_query_failure_is_reported_as_unknown(self):
        report = self.validate(InferenceService("state_unavailable"))
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["server_state_unknown"])
        self.assertIn("state_query_error", report)

    def test_wake_timeout_still_writes_failure_report(self):
        service = InferenceService("wake_stuck")
        with patch.object(sleep_test, "parse_args", return_value=self.args), \
                patch.object(sleep_test, "request_json", side_effect=service.request), \
                contextlib.redirect_stdout(io.StringIO()):
            code = sleep_test.main([])
        report = json.loads(Path(self.args.report).read_text(encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertTrue(report["server_left_sleeping"])
        self.assertEqual(report["failed_phase"], "wake_kv_cache")
        self.assertIn("TimeoutError", report["error"])

    def test_bad_baseline_writes_failure_report_before_any_sleep(self):
        self.baseline.write_text('{"prompts":"invalid"}', encoding="utf-8")
        with patch.object(sleep_test, "parse_args", return_value=self.args), \
                patch.object(sleep_test, "request_json") as request, \
                contextlib.redirect_stdout(io.StringIO()):
            code = sleep_test.main([])
        report = json.loads(Path(self.args.report).read_text(encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertEqual(report["failed_phase"], "load_prompts")
        request.assert_not_called()

    def test_remote_hbm_release_and_growth_per_device(self):
        self.args.cycles = 2
        with self.probe([(100, 1000), (20, 200), (101, 1001), (21, 201), (102, 1002)]):
            report = self.validate()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["checks"]["memory_released"])
        self.assertAlmostEqual(report["memory_growth_ratio"], 0.02)
        self.assertEqual(report["memory_before"]["head/0"]["used_bytes"], 100 * 1024**2)

    def test_single_device_growth_cannot_be_hidden_by_total(self):
        with self.probe([(100, 1000), (20, 200), (106, 1000)]):
            report = self.validate()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["memory_growth_within_5pct"])
        self.assertAlmostEqual(report["memory_growth_by_device"]["head/0"], 0.06)
        self.assertEqual(report["failed_phase"], "memory_acceptance")

    def test_memory_not_released_is_failure_with_awake_service(self):
        with self.probe([(100, 1000), (100, 1000), (100, 1000)]):
            report = self.validate()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["memory_released"])
        self.assertFalse(report["server_left_sleeping"])

    def test_probe_missing_device_blocks_before_sleep(self):
        with self.probe([(100, 1000)]):
            self.args.memory_devices.append("worker/99")
            service = InferenceService()
            report = self.validate(service)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["memory_validation"]["status"], "failed")
        self.assertFalse(service.sleep_attempted)

    def test_shared_dual_api_helper_keeps_both_response_schemas(self):
        service = InferenceService()
        output = generate_both(service.request, "http://fixture.invalid", "policy", "prompt", 32, timeout=1)
        self.assertEqual(set(output), set(ENDPOINTS))
        self.assertEqual(output[ENDPOINTS[0]], output[ENDPOINTS[1]])

    def test_real_http_cli_failure_under_python_optimization(self):
        service = InferenceService("chat_logprobs")

        class Handler(BaseHTTPRequestHandler):
            def handle_request(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length)) if length else None
                result = service.request("", self.command, self.path, payload)
                body = json.dumps(result).encode() if result is not None else b" \n"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            do_GET = handle_request
            do_POST = handle_request

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([
                sys.executable, "-O", str(Path(sleep_test.__file__)),
                "--server-url", f"http://127.0.0.1:{server.server_port}", "--model", "policy",
                "--baseline-result", str(self.baseline), "--cycles", "1", "--report", self.args.report,
            ], capture_output=True, timeout=10)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        report = json.loads(Path(self.args.report).read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 2, result.stderr.decode())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["endpoint_checks"][ENDPOINTS[1]])


if __name__ == "__main__":
    unittest.main()
