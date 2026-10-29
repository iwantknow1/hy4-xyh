"""Keep new artifact evidence readable in the final task report."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "reporting" / "generate_rl_case_guard_readme.py"
SPEC = importlib.util.spec_from_file_location("performance_readme", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PerformanceReadmeTests(unittest.TestCase):
    def test_analysis_scope_links_and_evaluation_choice(self):
        root = Path.cwd().resolve()
        report = {"feature": "hardware_efficiency", "status": "passed", "device": "0",
                  "validation_scope": "functional_smoke", "weighted_attainment": .8,
                  "priority_calls": 2, "workbook_path": str(root / "performance result.xlsx")}
        evaluation = {"feature": "dataset_evaluation", "status": "not_run", "dataset": "math500",
                      "max_out_len": None, "reason": "User declined evaluation"}
        result = MODULE.build_readme([(root / "performance.json", report), (root / "evaluation.json", evaluation)],
                                     "", title="Offline evidence", model="fixture", output_dir=root)
        self.assertIn("80.0000%", result)
        self.assertIn("functional_smoke", result)
        self.assertIn("[performance result.xlsx](performance%20result.xlsx)", result)
        self.assertIn("User declined evaluation", result)
        self.assertIn("math500", result)


if __name__ == "__main__":
    unittest.main()
