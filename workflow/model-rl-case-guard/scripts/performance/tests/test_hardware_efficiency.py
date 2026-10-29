"""Offline checks of weighting, empirical ceilings, and source filtering."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "estimate_hardware_efficiency.py"
SPEC = importlib.util.spec_from_file_location("hardware_efficiency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HardwareEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "kernel_details.csv"

    def write(self, rows):
        with self.source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Name", "Task Type", "Duration(us)", "Device_id", "mte2_ratio", "mac_ratio", "dur_ratio"])
            writer.writerows(rows)

    def analyze(self, **kwargs):
        return MODULE.analyze_kernels(self.source, device="0", **kwargs)

    def test_gmm_empirical_ceiling_and_all_layer_aggregation(self):
        self.write([["GMM1", "AI_CORE", 70, 0, .709, .224, .99],
                    ["GMM1", "AI_CORE", 70, 0, .709, .224, .99],
                    ["Other", "AI_VECTOR", 860, 0, .4, .2, .99]])
        result = self.analyze(rules=[{"operator": "GMM1", "metric": "mte2_ratio", "ceiling": .85, "source": "operator team"}])
        self.assertAlmostEqual(result["kernels"][0]["attainment"], .709 / .85)
        gmm = next(group for group in result["operator_summary"] if group["operator"] == "GMM1")
        self.assertEqual(gmm["calls"], 2)
        self.assertAlmostEqual(gmm["dur_ratio"], .14)
        self.assertAlmostEqual(gmm["contribution"], .14 * .709 / .85)
        self.assertAlmostEqual(result["weighted_attainment"], .14 * .709 / .85 + .86 * .4 / .8)

    def test_short_invocations_not_reclassified_after_grouping(self):
        self.write([["tiny", "AI_CORE", 6, 0, .1, .1, 0],
                    ["tiny", "AI_CORE", 6, 0, .1, .1, 0],
                    ["boundary", "AI_CORE", 10, 0, .4, .1, 0]])
        result = self.analyze()
        self.assertAlmostEqual(result["short_contribution"], 12 / 22)
        self.assertAlmostEqual(result["long_contribution"], 10 / 22 * .5)

    def test_priority_strict_boundaries_and_raw_max_ratio(self):
        self.write([["at20", "AI_CORE", 20, 0, .59, .1, 0],
                    ["at06", "AI_CORE", 21, 0, .6, .1, 0],
                    ["priority", "AI_CORE", 21, 0, .59, .1, 0]])
        self.assertEqual([kernel["priority"] for kernel in self.analyze()["kernels"]], [False, False, True])

    def test_filter_communication_other_devices_and_runtime(self):
        self.write([["compute", "AI_CORE", 30, 0, .8, 0, 0],
                    ["hcom_allReduce", "AI_CORE", 900, 0, "N/A", "N/A", 0],
                    ["runtime", "HOST", 100, 0, "", "", 0],
                    ["othercard", "AI_CORE", 500, 1, "", "", 0]])
        result = self.analyze()
        self.assertEqual(result["compute_duration_us"], 30)
        self.assertEqual(len(result["excluded"]), 3)
        self.assertAlmostEqual(result["weighted_attainment"], 1)

    def test_vector_core_included_in_duration_weights_and_operator_summary(self):
        for task_type in ("AI_VECTOR_CORE", "ai_vector_core", " AI VECTOR CORE "):
            with self.subTest(task_type=task_type):
                self.write([["matrix", "AI_CORE", 40, 0, .8, .2, 0],
                            ["vector", task_type, 60, 0, .4, .1, 0]])
                result = self.analyze()
                self.assertEqual(result["compute_duration_us"], 100)
                self.assertEqual(result["excluded"], [])
                vector = next(kernel for kernel in result["kernels"] if kernel["operator"] == "vector")
                self.assertEqual(vector["task_type"], "AI_VECTOR_CORE")
                self.assertAlmostEqual(vector["dur_ratio"], .6)
                self.assertAlmostEqual(vector["contribution"], .3)
                self.assertTrue(vector["priority"])
                group = next(group for group in result["operator_summary"] if group["operator"] == "vector")
                self.assertEqual(group["calls"], 1)
                self.assertAlmostEqual(group["dur_ratio"], .6)
                self.assertAlmostEqual(group["contribution"], .3)
                self.assertAlmostEqual(result["weighted_attainment"], .7)

    def test_vector_core_only_uses_short_rule_and_retains_other_exclusions(self):
        self.write([["short", "AI_VECTOR_CORE", 6, 0, .1, .1, 0],
                    ["boundary", "AI_VECTOR_CORE", 10, 0, .4, .1, 0],
                    ["hcom_allReduce", "AI_VECTOR_CORE", 900, 0, "N/A", "N/A", 0],
                    ["memcpy", "AI_VECTOR_CORE", 500, 0, "", "", 0],
                    ["othercard", "AI_VECTOR_CORE", 100, 1, "", "", 0]])
        result = self.analyze()
        self.assertEqual(result["compute_duration_us"], 16)
        self.assertAlmostEqual(result["short_contribution"], 6 / 16)
        self.assertAlmostEqual(result["long_contribution"], 10 / 16 * .5)
        self.assertAlmostEqual(result["weighted_attainment"], .6875)
        self.assertEqual([row["reason"] for row in result["excluded"]],
                         ["communication_or_memory_copy", "communication_or_memory_copy", "other_device"])

    def test_percent_normalization_and_no_clamping(self):
        self.write([["compute", "AI_CORE", 30, 0, "90%", "22.4", 0]])
        result = self.analyze(ratio_unit="percent")
        self.assertAlmostEqual(result["weighted_attainment"], 1.125)

    def test_invalid_ratios_and_missing_selected_metric_fail(self):
        for value in ("NaN", "inf", "1.1", "-0.1"):
            with self.subTest(value=value):
                self.write([["compute", "AI_CORE", 30, 0, value, "", 0]])
                with self.assertRaises(ValueError):
                    self.analyze()
        self.write([["compute", "AI_CORE", 30, 0, "", .2, 0]])
        with self.assertRaisesRegex(ValueError, "Missing selected metric"):
            self.analyze(rules=[{"operator": "compute", "metric": "mte2_ratio", "ceiling": .85, "source": "team"}])

    def test_ambiguous_rules_fail(self):
        self.write([["GMM1", "AI_CORE", 30, 0, .7, .2, 0]])
        rule = {"operator": "GMM*", "metric": "mte2_ratio", "ceiling": .85, "source": "team"}
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.analyze(rules=[rule, rule])

    def test_zero_duration_contributes_zero_but_all_zero_fails(self):
        self.write([["zero", "AI_CORE", 0, 0, .2, .1, 0], ["compute", "AI_CORE", 30, 0, .8, .1, 0]])
        result = self.analyze()
        self.assertEqual(result["kernels"][0]["contribution"], 0)
        self.assertAlmostEqual(result["weighted_attainment"], 1)
        self.write([["zero", "AI_CORE", 0, 0, .2, .1, 0]])
        with self.assertRaisesRegex(ValueError, "duration is zero"):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
