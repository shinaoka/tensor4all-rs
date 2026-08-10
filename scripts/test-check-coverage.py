#!/usr/bin/env python3
"""Tests for the coverage threshold checker (tensor4all/tensor4all-rs#566 Task 12).

Proves: default threshold pass/fail, exact per-file override, missing-file
behavior (report files without a threshold entry use the default), and that
top-level ``_comment_*`` rationale keys are ignored for enforcement.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check-coverage.py")
PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[1]


def make_report(percentages: dict[str, float]) -> dict:
    return {
        "data": [
            {
                "files": [
                    {
                        "filename": f"{REPO_ROOT}/{path}",
                        "summary": {"lines": {"percent": percent}},
                    }
                    for path, percent in percentages.items()
                ]
            }
        ]
    }


def run_check(thresholds: dict, percentages: dict[str, float]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        thresholds_path = root / "thresholds.json"
        thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
        report_path = root / "coverage.json"
        report_path.write_text(json.dumps(make_report(percentages)), encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(SCRIPT), "--thresholds", str(thresholds_path), str(report_path)],
            capture_output=True,
            text=True,
        )


class CoverageCheckTests(unittest.TestCase):
    def test_default_threshold_fail(self) -> None:
        result = run_check({"default": 75, "files": {}}, {"crates/x.rs": 60.0})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("crates/x.rs: 60.0% < 75%", result.stdout)

    def test_default_threshold_pass(self) -> None:
        result = run_check({"default": 75, "files": {}}, {"crates/x.rs": 90.0})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("All files meet their coverage thresholds.", result.stdout)

    def test_per_file_override(self) -> None:
        # A low-coverage file passes when its per-file threshold is set exactly.
        result = run_check(
            {"default": 75, "files": {"crates/x.rs": 60}},
            {"crates/x.rs": 61.0, "crates/y.rs": 90.0},
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        # And an override below the default is enforced for its own file.
        result = run_check(
            {"default": 75, "files": {"crates/x.rs": 70}},
            {"crates/x.rs": 69.0},
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("crates/x.rs: 69.0% < 70%", result.stdout)

    def test_missing_file_uses_default(self) -> None:
        # A report file with no per-file entry falls back to the default.
        result = run_check(
            {"default": 75, "files": {"crates/other.rs": 0}},
            {"crates/unknown.rs": 74.9},
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("crates/unknown.rs: 74.9% < 75%", result.stdout)

    def test_comment_rationale_keys_are_ignored_for_enforcement(self) -> None:
        # Top-level _comment_* keys document rationale but never change numbers.
        thresholds = {
            "default": 75,
            "_comment_tooling": "tooling files are exercised by subprocess tests",
            "_comment_capi": "capi boundary files covered by FFI integration tests",
            "files": {"crates/x.rs": 60},
        }
        result = run_check(thresholds, {"crates/x.rs": 61.0})
        self.assertEqual(result.returncode, 0, result.stdout)
        # And a failing file is still reported with the unchanged default.
        result = run_check(thresholds, {"crates/x.rs": 61.0, "crates/y.rs": 10.0})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("crates/y.rs: 10.0% < 75%", result.stdout)


if __name__ == "__main__":
    unittest.main()
