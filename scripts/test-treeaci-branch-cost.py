#!/usr/bin/env python3
"""Check dimension adjustment against known coefficients."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("branch_cost", Path(__file__).with_name("run-treeaci-branch-cost.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BranchCostTests(unittest.TestCase):
    def test_case_matrix_is_complete_and_unique(self):
        for full, count in ((False, 120), (True, 432)):
            cases = list(runner.cases(full))
            self.assertEqual(len(cases), count)
            self.assertEqual(len(set(cases)), count)
            self.assertEqual({c[0] for c in cases}, {2, 3, 4})

    def test_dimension_fit_recovers_degree_residual(self):
        records = []
        for degree in (2, 3, 4):
            for size in (32, 128, 512, 2048):
                time = 11 + 0.25 * size + {2: 0, 3: 7, 4: 19}[degree]
                records.append({"config": {"scalar":"f64", "d":2, "mode":"cold", "batch":32},
                    "nodes":[{"node":"tree:0", "degree":degree, "local_elements":size,
                              "guard_ns":time*16, "guard_misses":16,
                              "guard_batches":{"points":16, "calls":1},
                              "frame_batches":{"points":0, "calls":0}}]})
        fits = runner.fit_nodes(records)
        self.assertEqual(len(fits), 1)
        for field, expected in (("intercept_ns", 11), ("beta_ns_per_element", 0.25),
                                ("degree3_residual_ns", 7), ("degree4_residual_ns", 19)):
            self.assertAlmostEqual(fits[0][field], expected, places=10)
        self.assertLess(fits[0]["rms_residual_ns"], 1e-10)

    def test_incomplete_degrees_are_not_fitted(self):
        self.assertEqual(runner.fit_nodes([]), [])
        self.assertEqual(runner.fit_nodes([{"error":"failed"}]), [])

    def test_paired_statistics_keep_regression_direction(self):
        self.assertEqual(runner.interval([1.25]*5), [1.25, 1.25])
        self.assertEqual(runner.relative_mad([10,10,10,11,9]), 0)

    def experiment(self, candidate_times, bound=None, failed=False):
        # Exercise the complete runner, including preserving every failed pair,
        # without depending on host scheduling or building numerical binaries.
        calls = {"baseline": 0, "candidate": 0}

        def observe(binary, case, repeats, cpu, commit):
            side = binary.name
            iteration = calls[side]
            calls[side] += 1
            if failed and side == "candidate" and iteration == 0:
                return {"error": "deliberate execution failure"}
            elapsed = candidate_times[iteration] if side == "candidate" else 100
            return {"times_ns": [elapsed], "nodes": [], "config": {},
                    "host": {"load_before": [0, 0, 0], "load_after": [0, 0, 0],
                             "frequency_before": None, "frequency_after": None}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = ["runner", "--baseline", "baseline", "--candidate", "candidate",
                    "--baseline-commit", "a", "--candidate-commit", "b",
                    "--pairs", str(len(candidate_times)), "--repeats", "1",
                    "--cpu", "2", "--output", str(output)]
            if bound is not None:
                argv.extend(["--max-regression", str(bound)])
            with (patch("sys.argv", argv), patch.object(runner, "observe", observe),
                  patch.object(runner, "digest", return_value="test digest"),
                  patch.object(runner.os, "sched_getaffinity", return_value={2}),
                  patch.object(runner.os, "sched_setaffinity"),
                  patch.object(runner.subprocess, "check_output", return_value="test host"),
                  patch.object(runner, "cases", return_value=[(2, 0, 2, "f64", "cold", 8)]),
                  patch("builtins.print")):
                status = runner.main()
            report = json.loads(output.read_text())
        self.assertEqual(calls, dict.fromkeys(("baseline", "candidate"), len(candidate_times)))
        self.assertEqual(len(report["runs"]), len(candidate_times))
        return status, report

    def test_regression_gate_is_explicit_and_directional(self):
        for times, bound, expected, status in (
                ([130]*3, None, "DESCRIPTIVE", 0),
                ([110]*3, 0.2, "PASS", 0),
                ([130]*3, 0.2, "FAIL", 1)):
            with self.subTest(expected=expected):
                actual, report = self.experiment(times, bound)
                self.assertEqual((actual, report["verdict"]), (status, expected))

    def test_noisy_run_cannot_pass_even_with_a_loose_gate(self):
        status, report = self.experiment([100, 200, 400], bound=10)
        self.assertEqual((status, report["verdict"]), (1, "INCONCLUSIVE"))
        self.assertIn("dispersion", [f["reason"] for f in report["validity_failures"]])

    def test_failed_pair_is_preserved_and_invalidates_the_whole_run(self):
        status, report = self.experiment([100]*3, failed=True)
        self.assertEqual((status, report["verdict"]), (1, "INCONCLUSIVE"))
        self.assertIn("error", report["runs"][0]["candidate"])
        self.assertEqual(report["summary"], [])


if __name__ == "__main__":
    unittest.main()
