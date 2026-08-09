#!/usr/bin/env python3
"""Fixture-based self-tests for the library panic audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit-library-panics.py"


def run_audit(files: dict[str, str], baseline: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        for name, source in files.items():
            path = fixture / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        baseline_path = fixture / "baseline.json"
        baseline_path.write_text(json.dumps(baseline or []), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(fixture),
                "--baseline",
                str(baseline_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )


def assert_output(result: subprocess.CompletedProcess[str], *lines: str) -> None:
    output = result.stdout + result.stderr
    for line in lines:
        assert line in output, f"missing {line!r} in output:\n{output}"


def test_production_hit_fails_with_normalized_kind() -> None:
    result = run_audit({"crates/demo/src/lib.rs": 'pub fn bad() { panic!("boom"); }\n'})
    assert result.returncode == 1
    assert_output(result, "crates/demo/src/lib.rs:1:panic")


def test_all_raw_panic_kinds_fail() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """
fn bad() {
    unreachable!();
    let _ = None::<bool>.unwrap();
    let _ = None::<bool>.expect(\"missing\");
}
"""
        }
    )
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:3:unreachable",
        "crates/demo/src/lib.rs:4:unwrap",
        "crates/demo/src/lib.rs:5:expect",
    )


def test_test_modules_and_files_are_excluded() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """
#[cfg(test)]
mod inline_tests {
    fn bad() { panic!(\"inline\"); }
}
#[cfg(test)]
mod multiline_tests
{
    fn bad() { panic!(\"multiline\"); }
}
#[cfg(test)]
fn test_function() {
    panic!(\"function\");
}
#[cfg(test)]
mod test_support;
""",
            "crates/demo/src/test_support.rs": 'pub fn bad() { unreachable!("external"); }\n',
            "crates/demo/src/tests.rs": 'pub fn bad() { panic!("file"); }\n',
            "crates/demo/src/tests/file.rs": 'pub fn bad() { panic!("directory"); }\n',
            "crates/demo/src/production.rs": 'pub fn ok() { let _ = 1; }\n',
        }
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_output(result, "Audit passed: 0 unbaselined findings, 0 stale baseline entries")


def test_test_support_is_not_excluded_without_cfg_test() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": "mod test_support;\n",
            "crates/demo/src/test_support.rs": 'pub fn bad() { panic!("production"); }\n',
        }
    )
    assert result.returncode == 1
    assert_output(result, "crates/demo/src/test_support.rs:1:panic")


def test_comments_rustdoc_strings_and_macros_are_ignored() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": r'''// panic!("line comment")
/// unreachable!("rustdoc")
const TEXT: &str = r###"panic!("nested") .unwrap() .expect("x")"###;
const CHAR: char = '!';
macro_rules! harmless {
    () => { assert_eq!(1, 1) };
}
pub fn ok() {
    harmless!();
    let _ = "debug_assert!(false)";
}
'''
        }
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_output(result, "Audit passed: 0 unbaselined findings, 0 stale baseline entries")


def test_public_assertions_are_reported_but_private_helpers_are_not() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """
pub fn public_path(value: bool) {
    assert!(value);
    debug_assert!(value);
    assert_eq!(value, value);
    debug_assert_eq!(value, value);
}
fn private_helper(value: bool) {
    assert!(value);
    debug_assert!(value);
}
pub(crate) fn crate_helper(value: bool) {
    assert!(value);
}
"""
        }
    )
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:3:assert",
        "crates/demo/src/lib.rs:4:debug_assert",
    )
    output = result.stdout + result.stderr
    assert "crates/demo/src/lib.rs:8:assert" not in output
    assert "crates/demo/src/lib.rs:9:debug_assert" not in output
    assert "crates/demo/src/lib.rs:12:assert" not in output
    assert "crates/demo/src/lib.rs:5:assert" not in output
    assert "crates/demo/src/lib.rs:6:debug_assert" not in output


def test_matching_baseline_passes_with_normalized_entry() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit(
        {"crates/demo/src/lib.rs": "pub fn public_path() {\n    assert!(true);\n}\n"},
        [entry],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_output(result, f"Baseline matched: {entry}")
    assert_output(result, "Audit passed: 0 unbaselined findings, 0 stale baseline entries")


def test_new_finding_fails_against_existing_baseline() -> None:
    baseline = "crates/demo/src/lib.rs:3:assert"
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """
pub fn public_path() {
    assert!(true);
    debug_assert!(true);
}
"""
        },
        [baseline],
    )
    assert result.returncode == 1
    assert_output(result, f"Baseline matched: {baseline}", "crates/demo/src/lib.rs:4:debug_assert")
    assert_output(result, "Audit failed: 1 unbaselined finding, 0 stale baseline entries")


def test_stale_baseline_fails() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit({"crates/demo/src/lib.rs": "pub fn public_path() {}\n"}, [entry])
    assert result.returncode == 1
    assert_output(result, f"Stale baseline: {entry}")
    assert_output(result, "Audit failed: 0 unbaselined findings, 1 stale baseline entry")


def test_raw_panic_style_entries_cannot_be_baselined() -> None:
    entry = "crates/demo/src/lib.rs:1:panic"
    result = run_audit({"crates/demo/src/lib.rs": 'pub fn bad() { panic!("boom"); }\n'}, [entry])
    assert result.returncode == 1
    assert_output(result, f"Baseline contains forbidden raw panic-style entry: {entry}")


def main() -> int:
    tests = [
        test_production_hit_fails_with_normalized_kind,
        test_all_raw_panic_kinds_fail,
        test_test_modules_and_files_are_excluded,
        test_test_support_is_not_excluded_without_cfg_test,
        test_comments_rustdoc_strings_and_macros_are_ignored,
        test_public_assertions_are_reported_but_private_helpers_are_not,
        test_matching_baseline_passes_with_normalized_entry,
        test_new_finding_fails_against_existing_baseline,
        test_stale_baseline_fails,
        test_raw_panic_style_entries_cannot_be_baselined,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} scanner tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
