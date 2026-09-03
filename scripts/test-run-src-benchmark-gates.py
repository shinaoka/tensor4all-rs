#!/usr/bin/env python3
"""Tests for run-src-benchmark-gates.py."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
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
        pairs: int = 5,
        required_improvement: str = "0",
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
                    str(pairs),
                    "--required-improvement-percent",
                    required_improvement,
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
                env=os.environ
                | {
                    "T4A_BENCH_CENTER": "stale-center",
                    "T4A_PROFILE_CONTRACT": "stale-profile",
                },
            )
            report_data = json.loads(report.read_text()) if report.exists() else None
            return completed.returncode, report_data

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
            "n=$1; bond=$2; reps=$3; mode=$4; increment=$5; final_svd=$6; max_rank=$7\n"
            "if [ \"$mode\" = mpo-mps ]; then seed=7; center=S0; name=mpo-mps/$T4A_BENCH_ALGORITHM; else seed=13; center=N0000; name=tree-mpo-mpo/$T4A_BENCH_ALGORITHM; fi\n"
            f"echo 'record=build git_commit={commit} profile=release backend=fake features={features}'\n"
            "echo \"record=config n_sites=$n physical_dim=2 mpo_bond=$bond mps_bond=$bond requested_max_rank=$max_rank reps=$reps mode=$mode algorithm=$T4A_BENCH_ALGORITHM rank_increment=$increment final_svd=$final_svd src_seed=1234 adaptive_rtol=1e-4 adaptive_atol=0 adaptive_min_rank=2\"\n"
            "echo \"record=preflight mode=$mode network_seed=$seed total_input_bytes=8 largest_input_tensor_bytes=8 dense_output_bytes=8 max_degree=2 max_input_bytes=1024 max_dense_bytes=1024\"\n"
            f"echo \"record=case name=$name reps=$reps elapsed_seconds={seconds} per_run_seconds={seconds} nodes=1 edges=0 requested_max_rank=$max_rank effective_max_bond=1 src_seed=1234 center=$center relative_error={relative_error}\"\n"
        )
        path.chmod(0o700)
        return path

    def test_paired_synthetic_gate_passes(self):
        code, report = self.run_gate("0", "0")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["protocol"]["pairs"], 5)

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

    def test_non_finite_error_is_fail(self):
        code, report = self.run_gate("0", "nan")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_paired_quick_gate_rejects_fewer_than_five_pairs(self):
        code, report = self.run_gate("0", "0", pairs=1)
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_non_finite_improvement_is_rejected(self):
        code, report = self.run_gate("0", "0", required_improvement="nan")
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_timeout_kills_the_benchmark_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid = root / "child.pid"
            binary = root / "hang"
            binary.write_text(
                "#!/bin/sh\n"
                f"sleep 30 & echo $! > {child_pid}\n"
                "wait\n"
            )
            binary.chmod(0o700)
            with self.assertRaises(subprocess.TimeoutExpired):
                MODULE.run_once(
                    binary,
                    None,
                    MODULE.cases("quick")[0],
                    1,
                    0.1,
                    1 << 30,
                    1 << 20,
                    1.0e-8,
                    "fake",
                    "fake",
                )
            pid = int(child_pid.read_text())
            time.sleep(0.1)
            self.assertNotEqual(
                subprocess.run(
                    ["ps", "-p", str(pid)], capture_output=True, check=False
                ).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()
