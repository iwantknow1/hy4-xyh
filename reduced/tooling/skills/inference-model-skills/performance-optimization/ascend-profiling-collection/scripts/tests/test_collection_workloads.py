"""Exercise collector configuration without an NPU or remote connection."""
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "collect_torch_profile_case.py"
SPEC = importlib.util.spec_from_file_location("collection_workloads", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CollectionWorkloadTests(unittest.TestCase):
    def arguments(self, *extra):
        return MODULE.build_parser().parse_args([
            "--host", "example.test", "--model", "/weights/model", "--served-model-name", "model",
            "--tp", "2", "--mode", "full_decode_only", "--tag", "offline-test",
            "--random-input-len", "2048", "--random-output-len", "6144",
            "--max-model-len", "8192", "--profile-steps", "10", "--ignore-frontend",
            "--random-range-ratio", "0", *extra,
        ])

    def test_text_middle_ten_steps_and_fixed_lengths(self):
        args = self.arguments()
        config = MODULE._build_profiler_config(args, "/profiles/test")
        self.assertEqual(config["delay_iterations"], 3072)
        self.assertEqual(config["max_iterations"], 10)
        self.assertTrue(config["ignore_frontend"])
        command = MODULE._build_benchmark_command(args, "http://localhost:8000")
        self.assertEqual(command[command.index("--dataset-name") + 1], "random")
        self.assertEqual(command[command.index("--random-range-ratio") + 1], "0.0")
        self.assertIn("--ignore-eos", command)

    def test_multimodal_sends_image_and_deducts_measured_overhead(self):
        args = self.arguments("--workload-kind", "multimodal", "--random-input-len", "32768",
                              "--random-output-len", "4096", "--max-model-len", "36864",
                              "--mm-input-token-overhead", "512")
        command = MODULE._build_benchmark_command(args, "http://localhost:8000")
        self.assertEqual(command[command.index("--dataset-name") + 1], "random-mm")
        self.assertEqual(command[command.index("--backend") + 1], "openai-chat")
        self.assertEqual(command[command.index("--endpoint") + 1], "/v1/chat/completions")
        self.assertEqual(command[command.index("--random-input-len") + 1], "32256")
        self.assertIn("--random-mm-bucket-config", command)
        self.assertEqual(MODULE._build_profiler_config(args, "/profiles/test")["delay_iterations"], 2048)

    def test_multimodal_overhead_and_context_validated_before_launch(self):
        with self.assertRaisesRegex(ValueError, "overhead"):
            MODULE._build_benchmark_command(self.arguments("--workload-kind", "multimodal"), "http://localhost")
        with self.assertRaisesRegex(ValueError, "max-model-len"):
            MODULE._build_profiler_config(self.arguments("--max-model-len", "4096"), "/profiles/test")
        with self.assertRaisesRegex(ValueError, "window"):
            MODULE._build_profiler_config(self.arguments("--random-output-len", "10"), "/profiles/test")

    def test_baseline_graph_settings_are_preserved(self):
        args = self.arguments("--serve-args", "--compilation-config '{\"cudagraph_capture_sizes\":[1,2]}'")
        command = MODULE._build_serve_args(args, {})
        self.assertEqual(command.count("--compilation-config"), 1)
        config = json.loads(command[command.index("--compilation-config") + 1])
        self.assertEqual(config, {"cudagraph_capture_sizes": [1, 2], "cudagraph_mode": "FULL_DECODE_ONLY"})

    def test_conflicting_baseline_graph_mode_is_rejected(self):
        args = self.arguments("--serve-args", "--compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}'")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            MODULE._build_serve_args(args, {})

    def test_extra_arguments_cannot_override_ten_step_profiler(self):
        args = self.arguments('--serve-args=--profiler-config={"max_iterations":100}')
        with self.assertRaisesRegex(ValueError, "managed"):
            MODULE._build_serve_args(args, {})


if __name__ == "__main__":
    unittest.main()
