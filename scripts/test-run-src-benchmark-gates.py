#!/usr/bin/env python3
"""Tests for run-src-benchmark-gates.py."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run-src-benchmark-gates.py")
SPEC = importlib.util.spec_from_file_location("src_benchmark_gates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SrcBenchmarkGateTests(unittest.TestCase):
    def test_parse_records_and_statistics(self):
        records = MODULE.parse_records(
            "record=build git_commit=abc profile=release\n"
            "record=case name=x per_run_seconds=0.25 relative_error=1e-12\n"
        )
        self.assertEqual(records["build"][0]["git_commit"], "abc")
        self.assertEqual(records["case"][0]["name"], "x")
        self.assertEqual(MODULE.relative_mad([1.0, 1.0, 1.0]), 0.0)
        self.assertEqual(
            MODULE.bootstrap_median_ci([0.9, 1.0, 1.1], 7, 100),
            MODULE.bootstrap_median_ci([0.9, 1.0, 1.1], 7, 100),
        )

    def test_suite_shapes_are_fixed(self):
        quick = MODULE.cases("quick")
        self.assertEqual(len(quick), 4)
        self.assertTrue(all(case.max_rank == 32 for case in quick))
        self.assertTrue(all(case.rank_increment == 3 for case in quick))
        full = MODULE.cases("full")
        self.assertEqual(len(full), 30)
        self.assertNotIn(15, {case.n_sites for case in full if case.mode == "tree"})

    def run_gate(
        self,
        baseline_error: str,
        candidate_error: str,
        baseline_features: str = "fake",
        candidate_features: str = "fake",
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.write_fake(
                root / "baseline", "base", 1.0, baseline_error, baseline_features
            )
            candidate = self.write_fake(
                root / "candidate", "cand", 0.99, candidate_error, candidate_features
            )
            report = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--baseline",
                    str(baseline),
                    "--baseline-commit",
                    "base",
                    "--candidate",
                    str(candidate),
                    "--candidate-commit",
                    "cand",
                    "--expected-backend",
                    "fake",
                    "--expected-features",
                    "fake",
                    "--pairs",
                    "1",
                    "--reps",
                    "1",
                    "--max-dispersion-percent",
                    "100",
                    "--output",
                    str(report),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            return completed.returncode, json.loads(report.read_text())

    @staticmethod
    def write_fake(
        path: Path,
        commit: str,
        seconds: float,
        relative_error: str,
        features: str = "fake",
    ) -> Path:
        path.write_text(
            "#!/bin/sh\n"
            f"echo 'record=build git_commit={commit} profile=release backend=fake features={features}'\n"
            f"echo 'record=case name=fake reps=1 elapsed_seconds={seconds} per_run_seconds={seconds} nodes=1 edges=0 requested_max_rank=32 effective_max_bond=1 src_seed=1234 relative_error={relative_error}'\n"
        )
        path.chmod(0o700)
        return path

    def test_paired_synthetic_gate_passes(self):
        code, report = self.run_gate("0", "0")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")

    def test_candidate_correctness_failure_is_fail(self):
        code, report = self.run_gate("0", "1")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_baseline_identity_failure_is_inconclusive(self):
        code, report = self.run_gate("0", "0", baseline_features="wrong")
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "INCONCLUSIVE")

    def test_baseline_correctness_failure_is_fail(self):
        code, report = self.run_gate("1", "0")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_candidate_feature_mismatch_is_fail(self):
        code, report = self.run_gate("0", "0", candidate_features="wrong")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
