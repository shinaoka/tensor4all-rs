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


def test_masking_handles_bytes_nested_comments_and_scope_corruption() -> None:
    source = r'''const BYTE: u8 = b'}';
const BYTE_STRING: &[u8] = b"panic!(\\\") .unwrap()";
const RAW_BYTE_STRING: &[u8] = br###"unreachable!() } .expect(\"x\")"###;
const RAW_STRING: &str = r###"panic!() } .unwrap()"###;
/* outer comment { panic!("x") /* nested } .unwrap() */ still masked */
pub fn production() {
    assert!(true);
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_output(result, "crates/demo/src/lib.rs:7:assert")
    output = result.stdout + result.stderr
    assert output.count("crates/demo/src/lib.rs:") == 1, output


def test_multiline_calls_and_known_ufcs_are_reported_without_user_function_false_matches() -> None:
    source = '''fn user_unwrap(value: Option<bool>) -> bool { value.unwrap_or(false) }
fn unwrap(value: bool) -> bool { value }
fn bad(value: Option<bool>, result: Result<bool, ()>) {
    panic
        !
        ("boom");
    unreachable
        ! { "boom" };
    let _ = value
        . /* comment between tokens */
        unwrap
        ();
    let _ = result\n        .expect\n        ("missing");
    let _ = Option
        ::
        unwrap(value);
    let _ = std::option::Option::<bool>
        ::
        unwrap(value);
    let _ = Result::<bool, ()>
        ::
        expect(result, "missing");
    let _ = user::std::option::Option::<bool>::unwrap(value);
    let _ = user::unwrap(value);
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:4:panic",
        "crates/demo/src/lib.rs:7:unreachable",
        "crates/demo/src/lib.rs:10:unwrap",
        "crates/demo/src/lib.rs:14:expect",
        "crates/demo/src/lib.rs:16:unwrap",
        "crates/demo/src/lib.rs:19:unwrap",
        "crates/demo/src/lib.rs:22:expect",
    )
    output = result.stdout + result.stderr
    assert "user_unwrap" not in output
    assert "user::std::option" not in output
    assert "crates/demo/src/lib.rs:25:unwrap" not in output
    assert "crates/demo/src/lib.rs:26:unwrap" not in output


def test_public_trait_defaults_impls_and_signature_modifiers_are_public_paths() -> None:
    source = '''pub trait PublicTrait {
    fn default_method(&self) {
        assert!(true);
    }
}
trait PrivateTrait {
    fn private_default(&self) {
        assert!(true);
    }
}
struct PublicType;
impl PublicTrait for PublicType {
    fn impl_method(&self) {
        debug_assert!(true);
    }
}
impl PrivateTrait for PublicType {
    fn private_impl(&self) {
        assert!(true);
    }
}
impl PublicType {
    fn private_inherent(&self) {
        assert!(true);
    }
    pub
    async fn explicit_async(&self) {
        assert!(true);
    }
    pub
    unsafe fn explicit_unsafe(&self) {
        assert!(true);
    }
    pub const fn explicit_const(&self) {
        assert!(true);
    }
    pub extern "C"
    fn explicit_extern(&self) {
        assert!(true);
    }
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:3:assert",
        "crates/demo/src/lib.rs:14:debug_assert",
        "crates/demo/src/lib.rs:28:assert",
        "crates/demo/src/lib.rs:32:assert",
        "crates/demo/src/lib.rs:35:assert",
        "crates/demo/src/lib.rs:39:assert",
    )
    output = result.stdout + result.stderr
    assert "crates/demo/src/lib.rs:8:assert" not in output
    assert "crates/demo/src/lib.rs:19:assert" not in output
    assert "crates/demo/src/lib.rs:24:assert" not in output


def test_generic_public_trait_impl_and_qualified_ufcs_are_detected() -> None:
    source = '''pub trait PublicTrait<T> {
    fn default_method(&self) {
        assert!(true);
    }
}
trait PrivateTrait<T> {
    fn private_method(&self) {
        assert!(true);
    }
}
struct PublicType;
impl<T> PublicTrait<T> for PublicType {
    fn impl_method(&self) {
        debug_assert!(true);
    }
}
impl<T> PrivateTrait<T> for PublicType {
    fn private_impl(&self) {
        assert!(true);
    }
}
fn ufcs(value: Option<bool>, result: Result<bool, ()>) {
    let _ = <Option<bool>>::unwrap(value);
    let _ = <std::result::Result<bool, ()>>::expect(result, "missing");
    let _ = <user::Option<bool>>::unwrap(value);
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:3:assert",
        "crates/demo/src/lib.rs:14:debug_assert",
        "crates/demo/src/lib.rs:23:unwrap",
        "crates/demo/src/lib.rs:24:expect",
    )
    output = result.stdout + result.stderr
    assert "crates/demo/src/lib.rs:8:assert" not in output
    assert "crates/demo/src/lib.rs:18:assert" not in output
    assert "crates/demo/src/lib.rs:25:unwrap" not in output


def test_cfg_test_attributes_do_not_leak_to_following_items() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": '''#[cfg(test)] # [path = "fixtures.rs"] mod external_tests; pub fn production_one() { assert!(true); }
#[cfg(test)] # [cfg_attr(test, allow(dead_code))] mod inline_tests { fn bad() { panic!("test"); } } pub fn production_two() { debug_assert!(true); }
''',
            "crates/demo/src/fixtures.rs": 'pub fn bad() { panic!("external test"); }\n',
        }
    )
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:1:assert",
        "crates/demo/src/lib.rs:2:debug_assert",
    )
    output = result.stdout + result.stderr
    assert "fixtures.rs" not in output
    assert "panic" not in output


def test_cfg_test_function_does_not_hide_following_production_module() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": '''#[cfg(test)] fn test_only() { panic!("test"); }
mod production;
pub fn production_path() { assert!(true); }
''',
            "crates/demo/src/production.rs": 'pub fn bad() { panic!("production"); }\n',
        }
    )
    assert result.returncode == 1
    assert_output(
        result,
        "crates/demo/src/lib.rs:3:assert",
        "crates/demo/src/production.rs:1:panic",
    )
    output = result.stdout + result.stderr
    assert "crates/demo/src/lib.rs:1:panic" not in output


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


def test_duplicate_baseline_entries_fail_configuration() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit(
        {"crates/demo/src/lib.rs": "pub fn public_path() {\n    assert!(true);\n}\n"},
        [entry, entry],
    )
    assert result.returncode == 2
    assert_output(result, "baseline contains duplicate entries")


def test_malformed_and_non_normalized_baselines_fail_configuration() -> None:
    source = {"crates/demo/src/lib.rs": "pub fn public_path() { assert!(true); }\n"}
    for baseline, message in [
        (["crates/demo/src/lib.rs:1"], "invalid baseline entry"),
        (["crates/demo/src/lib.rs:1:assert:extra"], "invalid baseline entry"),
        (["crates/demo/src\\\\lib.rs:1:assert"], "baseline path is not normalized"),
        (["/absolute/lib.rs:1:assert"], "baseline path is not normalized"),
        (["crates/../demo/src/lib.rs:1:assert"], "baseline path is not normalized"),
        (["demo/src/lib.rs:1:assert"], "baseline path is not normalized"),
        (["crates/demo/src/lib.rs:0:assert"], "invalid baseline entry"),
    ]:
        result = run_audit(source, baseline)
        assert result.returncode == 2, (baseline, result.stdout, result.stderr)
        assert_output(result, message)


def test_report_claims_include_exact_sorted_findings_matches_and_stale_entries() -> None:
    baseline = "crates/demo/src/lib.rs:2:assert"
    result = run_audit(
        {
            "crates/demo/src/lib.rs": "pub fn public_path() {\n    assert!(true);\n    debug_assert!(true);\n}\n",
            "crates/demo/src/other.rs": "fn bad() { panic!(\"boom\"); }\n",
        },
        [baseline],
    )
    assert result.returncode == 1
    assert_output(
        result,
        f"Baseline matched: {baseline}",
        "crates/demo/src/lib.rs:3:debug_assert",
        "crates/demo/src/other.rs:1:panic",
        "Audit failed: 2 unbaselined findings, 0 stale baseline entries",
    )


def main() -> int:
    tests = [
        test_production_hit_fails_with_normalized_kind,
        test_all_raw_panic_kinds_fail,
        test_test_modules_and_files_are_excluded,
        test_test_support_is_not_excluded_without_cfg_test,
        test_comments_rustdoc_strings_and_macros_are_ignored,
        test_masking_handles_bytes_nested_comments_and_scope_corruption,
        test_multiline_calls_and_known_ufcs_are_reported_without_user_function_false_matches,
        test_public_trait_defaults_impls_and_signature_modifiers_are_public_paths,
        test_generic_public_trait_impl_and_qualified_ufcs_are_detected,
        test_cfg_test_attributes_do_not_leak_to_following_items,
        test_cfg_test_function_does_not_hide_following_production_module,
        test_public_assertions_are_reported_but_private_helpers_are_not,
        test_matching_baseline_passes_with_normalized_entry,
        test_new_finding_fails_against_existing_baseline,
        test_stale_baseline_fails,
        test_raw_panic_style_entries_cannot_be_baselined,
        test_duplicate_baseline_entries_fail_configuration,
        test_malformed_and_non_normalized_baselines_fail_configuration,
        test_report_claims_include_exact_sorted_findings_matches_and_stale_entries,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} scanner tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
