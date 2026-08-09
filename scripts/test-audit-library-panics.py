#!/usr/bin/env python3
"""Fixture-based self-tests for the library panic audit."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit-library-panics.py"
TOOL_ENV = "T4A_PANIC_AUDIT_BIN"
TOOL_PACKAGE = "library-panic-audit"


def _build_tool_once() -> None:
    configured = os.environ.get(TOOL_ENV)
    if configured and Path(configured).is_file():
        return
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--message-format=json-render-diagnostics",
            "-p",
            TOOL_PACKAGE,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    executable = None
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == TOOL_PACKAGE
            and "bin" in target.get("kind", [])
            and message.get("executable")
        ):
            executable = Path(message["executable"])
            if not executable.is_absolute():
                executable = (ROOT / executable).resolve()
            break
    assert executable is not None, result.stdout
    assert executable.is_file(), executable
    os.environ[TOOL_ENV] = str(executable)


def _write_fixture(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def run_audit_at_root(
    fixture: Path,
    baseline: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
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
        timeout=5,
    )


def run_audit(
    files: dict[str, str], baseline: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(fixture, files)
        return run_audit_at_root(fixture, baseline)


def assert_exact_output(
    result: subprocess.CompletedProcess[str],
    stdout: list[str],
    stderr: list[str] | None = None,
) -> None:
    expected_stderr = [] if stderr is None else stderr
    actual_stdout = result.stdout.splitlines()
    actual_stderr = result.stderr.splitlines()
    assert actual_stdout == stdout, f"stdout differs:\nexpected {stdout!r}\nactual {actual_stdout!r}"
    assert actual_stderr == expected_stderr, f"stderr differs:\nexpected {expected_stderr!r}\nactual {actual_stderr!r}"


def test_wrapper_discovers_json_compiler_artifact_path() -> None:
    wrapper_spec = importlib.util.spec_from_file_location("panic_wrapper", SCRIPT)
    assert wrapper_spec and wrapper_spec.loader
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    with tempfile.TemporaryDirectory() as directory:
        fake_bin = Path(directory) / "cargo"
        artifact = Path(directory) / "custom-target" / "release" / TOOL_PACKAGE
        artifact.parent.mkdir(parents=True)
        fake_bin.write_text(
            "#!/bin/sh\n"
            f"touch {artifact}\n"
            "printf '%s\\n' "
            + repr(
                json.dumps(
                    {
                        "reason": "compiler-artifact",
                        "target": {"name": TOOL_PACKAGE, "kind": ["bin"]},
                        "executable": str(artifact),
                    }
                )
            )
            + "\n",
            encoding="utf-8",
        )
        fake_bin.chmod(0o755)
        old_path = os.environ.get("PATH")
        old_override = os.environ.pop(TOOL_ENV, None)
        os.environ["PATH"] = f"{directory}{os.pathsep}{old_path or ''}"
        try:
            discovered = wrapper._tool_path(ROOT)
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
            if old_override is not None:
                os.environ[TOOL_ENV] = old_override
        assert discovered == artifact.resolve()


def test_wrong_and_empty_roots_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        results = [run_audit_at_root(fixture)]
        (fixture / "crates").mkdir()
        results.append(run_audit_at_root(fixture))
        results.append(
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(fixture / "missing"),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        )
    for result in results:
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr.startswith("library panic audit configuration error:")


def test_parse_failure_fails_closed() -> None:
    result = run_audit({"crates/demo/src/lib.rs": "pub fn broken( {\n"})
    assert result.returncode == 2
    assert result.stdout == ""
    assert "cannot parse Rust source" in result.stderr


def test_production_hit_fails_with_normalized_kind() -> None:
    result = run_audit({"crates/demo/src/lib.rs": 'pub fn bad() { panic!("boom"); }\n'})
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/lib.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:unreachable",
            "crates/demo/src/lib.rs:4:unwrap",
            "crates/demo/src/lib.rs:5:expect",
        ],
        ["Audit failed: 3 unbaselined findings, 0 stale baseline entries."],
    )


def test_structurally_test_only_modules_and_inline_items_are_excluded() -> None:
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
#[cfg(test)]
#[path = "tests.rs"]
mod named_tests;
#[path = "tests/file.rs"]
#[cfg(test)]
mod directory_tests;
""",
            "crates/demo/src/test_support.rs": 'pub fn bad() { unreachable!("external"); }\n',
            "crates/demo/src/tests.rs": 'pub fn bad() { panic!("file"); }\n',
            "crates/demo/src/tests/file.rs": 'pub fn bad() { panic!("directory"); }\n',
            "crates/demo/src/production.rs": 'pub fn ok() { let _ = 1; }\n',
        }
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_exact_output(
        result,
        ["Audit passed: 0 unbaselined findings, 0 stale baseline entries"],
    )


def test_test_support_is_not_excluded_without_cfg_test() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": "mod test_support;\n",
            "crates/demo/src/test_support.rs": 'pub fn bad() { panic!("production"); }\n',
        }
    )
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/test_support.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        ["Audit passed: 0 unbaselined findings, 0 stale baseline entries"],
    )


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
    assert_exact_output(
        result,
        ["crates/demo/src/lib.rs:7:assert"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_macro_arguments_and_production_definitions_are_scanned_without_expansion() -> None:
    source = r'''macro_rules! literal {
    () => { panic!("definition"); };
}
macro_rules! passthrough {
    ($panic:ident) => { $panic!(); };
}
fn calls(value: Option<bool>) {
    wrapper!(panic!("statement argument"));
    dbg!(value.unwrap());
    wrapper!(Option::expect(value, "associated argument"));
}
wrapper! {
    fn generated() {
        unreachable!("item body");
    }
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:2:panic",
            "crates/demo/src/lib.rs:8:panic",
            "crates/demo/src/lib.rs:9:unwrap",
            "crates/demo/src/lib.rs:10:expect",
            "crates/demo/src/lib.rs:14:unreachable",
        ],
        ["Audit failed: 5 unbaselined findings, 0 stale baseline entries."],
    )


def test_macro_definitions_scan_generic_associated_calls_with_metavariables() -> None:
    source = r'''macro_rules! generic {
    ($value:expr) => { Option::<bool>::unwrap($value); };
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/lib.rs:2:unwrap"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_macro_rules_scan_only_transcribers_and_recurse_repetitions() -> None:
    source = r'''macro_rules! mixed {
    (panic!() Option::<bool>::unwrap($value:expr) $($ty:ty),+; $panic:ident) => {
        $($(
            Option::<($($ty,)+)>::unwrap($value);
            Option::<bool>::unwrap($value);
            panic!("literal transcriber");
            $panic!();
        )+)+
    };
}
fn expression(value: Option<bool>) {
    wrapper!(panic!("expression argument"));
    let _ = dbg!(value.unwrap());
}
wrapper! {
    fn item() {
        unreachable!("item argument");
    }
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "lib.rs"
        source_path.write_text(
            source.split("fn expression", 1)[0] + "fn main() {}\n", encoding="utf-8"
        )
        compiled = subprocess.run(
            ["rustc", "--crate-type", "lib", str(source_path), "-o", str(Path(directory) / "lib.rlib")],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert compiled.returncode == 0, compiled.stderr
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:4:unwrap",
            "crates/demo/src/lib.rs:5:unwrap",
            "crates/demo/src/lib.rs:6:panic",
            "crates/demo/src/lib.rs:12:panic",
            "crates/demo/src/lib.rs:13:unwrap",
            "crates/demo/src/lib.rs:17:unreachable",
        ],
        ["Audit failed: 6 unbaselined findings, 0 stale baseline entries."],
    )


def test_imported_panic_macro_aliases_are_detected() -> None:
    source = '''use std::panic as fail;\n\nfn production() {\n    fail!("aliased panic");\n}\n'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/lib.rs:4:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:4:panic",
            "crates/demo/src/lib.rs:7:unreachable",
            "crates/demo/src/lib.rs:10:unwrap",
            "crates/demo/src/lib.rs:14:expect",
            "crates/demo/src/lib.rs:16:unwrap",
            "crates/demo/src/lib.rs:19:unwrap",
            "crates/demo/src/lib.rs:22:expect",
        ],
        ["Audit failed: 7 unbaselined findings, 0 stale baseline entries."],
    )


def test_ufcs_reports_both_option_result_methods_and_rejects_collisions() -> None:
    source = """fn ufcs(option: Option<bool>, result: Result<bool, ()>) {
    let _ = Option::unwrap(option);
    let _ = Option::expect(option, "missing");
    let _ = Result::unwrap(result);
    let _ = Result::expect(result, "missing");
    let _ = std::option::Option::<bool>::expect(option, "missing");
    let _ = std::result::Result::<bool, ()>::unwrap(result);
    let _ = <Option<bool>>::expect(option, "missing");
    let _ = <std::result::Result<bool, ()>>::unwrap(result);
    let _ = ::core::option::Option::<bool>::unwrap(option);
    let _ = ::std::result::Result::<bool, ()>::expect(result, "missing");
    let _ = user::Option::expect(option, "not standard");
    let _ = Optionish::unwrap(option);
    let _ = Result2::expect(result, "not standard");
    let _ = user::std::option::Option::<bool>::unwrap(option);
}
"""
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    expected = [
        "crates/demo/src/lib.rs:2:unwrap",
        "crates/demo/src/lib.rs:3:expect",
        "crates/demo/src/lib.rs:4:unwrap",
        "crates/demo/src/lib.rs:5:expect",
        "crates/demo/src/lib.rs:6:expect",
        "crates/demo/src/lib.rs:7:unwrap",
        "crates/demo/src/lib.rs:8:expect",
        "crates/demo/src/lib.rs:9:unwrap",
        "crates/demo/src/lib.rs:10:unwrap",
        "crates/demo/src/lib.rs:11:expect",
    ]
    assert_exact_output(
        result,
        expected,
        ["Audit failed: 10 unbaselined findings, 0 stale baseline entries."],
    )


def test_public_trait_defaults_and_external_impls_are_public_conservatively() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": "mod api; mod implementations;\n",
            "crates/demo/src/api.rs": """pub trait ExternalTrait {
    fn default_method(&self) {
        assert!(true);
    }
}
trait SameName {
    fn private_default(&self) {
        assert!(true);
    }
}
""",
            "crates/demo/src/implementations.rs": """struct PublicType;
impl crate::api::ExternalTrait for PublicType {
    fn impl_method(&self) {
        debug_assert!(true);
    }
}
impl SameName for PublicType {
    fn private_trait_impl(&self) {
        assert!(true);
    }
}
impl PublicType {
    fn private_inherent(&self) {
        assert!(true);
    }
}
""",
            "crates/consumer/src/lib.rs": """struct ConsumerType;
impl demo::api::ExternalTrait for ConsumerType {
    fn cross_crate_impl(&self) {
        assert!(true);
    }
}
""",
        }
    )
    assert result.returncode == 1
    expected = [
        "crates/consumer/src/lib.rs:4:assert",
        "crates/demo/src/api.rs:3:assert",
        "crates/demo/src/implementations.rs:4:debug_assert",
        "crates/demo/src/implementations.rs:9:assert",
    ]
    assert_exact_output(
        result,
        expected,
        ["Audit failed: 4 unbaselined findings, 0 stale baseline entries."],
    )


def test_const_generic_where_body_and_attribute_visibility_are_structural() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """trait Bound<T> {}
#[some(pub)]
fn private_attribute(value: bool) {
    assert!(value);
}
#[some(pub)]
pub fn public_attribute(value: bool) {
    assert!(value);
}
pub fn const_generic<T: Bound<{ 1 }>>(value: bool)
where
    T: Sized,
{
    assert!(value);
}
"""
        }
    )
    assert result.returncode == 1
    expected = [
        "crates/demo/src/lib.rs:8:assert",
        "crates/demo/src/lib.rs:14:assert",
    ]
    assert_exact_output(
        result,
        expected,
        ["Audit failed: 2 unbaselined findings, 0 stale baseline entries."],
    )


def test_cfg_test_external_modules_are_order_independent_and_transitive() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """#[path = r"root_tests.rs"]
#[cfg(test)]
mod root_tests;
#[cfg(test)]
mod inline_tests {
    #[path = r"custom/inline_helper.rs"]
    mod helper;
}
""",
            "crates/demo/src/root_tests.rs": """mod nested_tests;
pub fn test_only() { panic!("root"); }
""",
            "crates/demo/src/root_tests/nested_tests.rs": 'pub fn test_only() { unreachable!("nested"); }\n',
            "crates/demo/src/inline_tests/custom/inline_helper.rs": 'pub fn test_only() { panic!("inline helper"); }\n',
        }
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_exact_output(
        result,
        ["Audit passed: 0 unbaselined findings, 0 stale baseline entries"],
    )


def test_inline_path_parent_changes_external_child_base() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": '''#[path = "alt"]
mod production {
    mod child;
}
''',
            "crates/demo/src/alt/child.rs": 'pub fn bad() { panic!("alt child"); }\n',
        }
    )
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/alt/child.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_graph_referenced_paths_follow_logical_symlinked_crate_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(
            fixture,
            {
                "crates/demo/real_root.rs": "mod child;\n",
                "crates/demo/src/child.rs": 'pub fn bad() { panic!("logical child"); }\n',
            },
        )
        logical_root = fixture / "crates/demo/src/lib.rs"
        try:
            logical_root.symlink_to(fixture / "crates/demo/real_root.rs")
        except OSError as error:
            raise AssertionError(f"symlink fixture unavailable: {error}") from error
        result = run_audit_at_root(fixture)
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/child.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_graph_referenced_production_paths_outside_src_are_scanned() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": '#[path = r"../shared.rs"] mod shared;\n',
            "crates/demo/shared.rs": 'pub fn bad() { panic!("outside src"); }\n',
        }
    )
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/shared.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )

    escaping = run_audit(
        {
            "crates/demo/src/lib.rs": '#[path = r"../../../../outside.rs"] mod outside;\n',
            "outside.rs": 'pub fn bad() { panic!("escape"); }\n',
        }
    )
    assert escaping.returncode == 2
    assert escaping.stdout == ""
    assert escaping.stderr.startswith("library panic audit configuration error:")


def test_production_reference_overrides_test_only_alias() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """#[cfg(test)]
#[path = "shared.rs"]
mod test_shared;
#[path = "shared.rs"]
mod production_shared;
""",
            "crates/demo/src/shared.rs": 'pub fn production() { panic!("shared"); }\n',
        }
    )
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/shared.rs:1:panic",
        ],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_cfg_test_macro_rules_range_is_excluded() -> None:
    result = run_audit(
        {
            "crates/demo/src/lib.rs": """#[cfg(test)]
macro_rules! test_helper {
    () => { panic!("test macro"); };
}
"""
        }
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_exact_output(
        result,
        ["Audit passed: 0 unbaselined findings, 0 stale baseline entries"],
    )


def test_source_names_and_checkout_ancestors_do_not_exclude_production_files() -> None:
    outer = Path(tempfile.mkdtemp())
    try:
        fixture = outer / "tests" / "checkout"
        _write_fixture(
            fixture,
            {
                "crates/demo/src/tests.rs": 'pub fn bad() { panic!("tests"); }\n',
                "crates/demo/src/test_utils.rs": 'pub fn bad() { unreachable!("utils"); }\n',
                "crates/demo/src/_tests.rs": 'pub fn bad() { panic!("suffix"); }\n',
                "crates/demo/src/tests/unreferenced.rs": 'pub fn bad() { panic!("directory"); }\n',
            },
        )
        result = run_audit_at_root(fixture)
    finally:
        shutil.rmtree(outer)
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/_tests.rs:1:panic",
            "crates/demo/src/test_utils.rs:1:unreachable",
            "crates/demo/src/tests.rs:1:panic",
            "crates/demo/src/tests/unreferenced.rs:1:panic",
        ],
        ["Audit failed: 4 unbaselined findings, 0 stale baseline entries."],
    )


def test_outside_root_symlink_is_rejected_safely() -> None:
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
        fixture = Path(directory)
        outside_source = Path(outside) / "outside.rs"
        outside_source.write_text('pub fn bad() { panic!("outside"); }\n', encoding="utf-8")
        link = fixture / "crates/demo/src/link.rs"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(outside_source)
        except OSError as error:
            raise AssertionError(f"symlink fixture unavailable: {error}") from error
        result = run_audit_at_root(fixture)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("library panic audit configuration error:")


def test_in_root_symlink_is_canonicalized_and_reported_once() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        real = fixture / "crates/demo/src/real.rs"
        alias = fixture / "crates/demo/src/alias.rs"
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text('pub fn bad() { panic!("inside"); }\n', encoding="utf-8")
        try:
            alias.symlink_to(real)
        except OSError as error:
            raise AssertionError(f"symlink fixture unavailable: {error}") from error
        result = run_audit_at_root(fixture)
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/src/real.rs:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_logical_rs_symlink_discovers_extensionless_in_root_target() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(
            fixture,
            {
                "crates/demo/src/lib.rs": "pub fn root() {}\n",
                "crates/demo/shared_target": 'pub fn bad() { panic!("target"); }\n',
            },
        )
        link = fixture / "crates/demo/src/bin/tool.rs"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(fixture / "crates/demo/shared_target")
        except OSError as error:
            raise AssertionError(f"symlink fixture unavailable: {error}") from error
        result = run_audit_at_root(fixture)
    assert result.returncode == 1
    assert_exact_output(
        result,
        ["crates/demo/shared_target:1:panic"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_large_fixture_stays_within_subprocess_timeout() -> None:
    source = "\n".join(
        f"pub fn function_{index}() {{ assert!(true); }}" for index in range(2_500)
    )
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 2_500
    assert lines[0] == "crates/demo/src/lib.rs:1:assert"
    assert lines[-1] == "crates/demo/src/lib.rs:2500:assert"
    assert result.stderr.splitlines() == [
        "Audit failed: 2500 unbaselined findings, 0 stale baseline entries."
    ]


def test_method_heavy_fixture_stays_within_subprocess_timeout() -> None:
    source = "\n".join(
        f"fn function_{index}(value: Option<bool>) {{\n    let _ = value\n        .unwrap();\n}}"
        for index in range(6_000)
    )
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 6_000
    assert lines[0] == "crates/demo/src/lib.rs:3:unwrap"
    assert lines[-1] == "crates/demo/src/lib.rs:23999:unwrap"
    assert result.stderr.splitlines() == [
        "Audit failed: 6000 unbaselined findings, 0 stale baseline entries."
    ]


def test_ufcs_aliases_and_local_type_shadowing() -> None:
    source = r'''use std::option::Option as Maybe;
use core::result::Result as R;
use user::Option as LocalOption;
type Alias<T> = Maybe<T>;
mod local {
    struct Option<T>(T);
    fn shadow(value: Option<bool>) {
        let _ = Option::unwrap(value);
    }
}
fn calls(value: Maybe<bool>, result: R<bool, ()>) {
    let _ = Maybe::unwrap(value);
    let _ = R::r#expect(result, "missing");
    let _ = Alias::<bool>::unwrap(value);
    let _ = <Maybe<bool>>::unwrap(value);
    let _ = <Alias<bool>>::r#expect(value, "missing");
    let _ = LocalOption::unwrap(value);
}
fn local_shadow(value: Maybe<bool>) {
    type Maybe = LocalOption<bool>;
    let _ = Maybe::unwrap(value);
}
#[cfg(test)]
type Maybe = LocalOption<bool>;
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:12:unwrap",
            "crates/demo/src/lib.rs:13:expect",
            "crates/demo/src/lib.rs:14:unwrap",
            "crates/demo/src/lib.rs:15:unwrap",
            "crates/demo/src/lib.rs:16:expect",
        ],
        ["Audit failed: 5 unbaselined findings, 0 stale baseline entries."],
    )


def test_module_declared_aliases_resolve_through_crate_self_super_and_groups() -> None:
    source = r'''mod aliases {
    pub use std::option::Option as Maybe;
    pub type ResultAlias<T, E> = core::result::Result<T, E>;
    mod nested {
        use super::{Maybe as InnerMaybe, ResultAlias as InnerResult};
        fn calls(option: InnerMaybe<bool>, result: InnerResult<bool, ()>) {
            let _ = InnerMaybe::unwrap(option);
            let _ = InnerResult::r#expect(result, "missing");
        }
    }
}
use crate::aliases::{Maybe as ChainedMaybe, ResultAlias as ChainedResult};
fn production(option: ChainedMaybe<bool>, result: ChainedResult<bool, ()>) {
    let _ = ChainedMaybe::unwrap(option);
    let _ = ChainedResult::expect(result, "missing");
}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:7:unwrap",
            "crates/demo/src/lib.rs:8:expect",
            "crates/demo/src/lib.rs:14:unwrap",
            "crates/demo/src/lib.rs:15:expect",
        ],
        ["Audit failed: 4 unbaselined findings, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:assert",
            "crates/demo/src/lib.rs:14:debug_assert",
            "crates/demo/src/lib.rs:19:assert",
            "crates/demo/src/lib.rs:28:assert",
            "crates/demo/src/lib.rs:32:assert",
            "crates/demo/src/lib.rs:35:assert",
            "crates/demo/src/lib.rs:39:assert",
        ],
        ["Audit failed: 7 unbaselined findings, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:assert",
            "crates/demo/src/lib.rs:14:debug_assert",
            "crates/demo/src/lib.rs:19:assert",
            "crates/demo/src/lib.rs:23:unwrap",
            "crates/demo/src/lib.rs:24:expect",
        ],
        ["Audit failed: 5 unbaselined findings, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:1:assert",
            "crates/demo/src/lib.rs:2:debug_assert",
        ],
        ["Audit failed: 2 unbaselined findings, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:assert",
            "crates/demo/src/production.rs:1:panic",
        ],
        ["Audit failed: 2 unbaselined findings, 0 stale baseline entries."],
    )


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
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:assert",
            "crates/demo/src/lib.rs:4:debug_assert",
        ],
        ["Audit failed: 2 unbaselined findings, 0 stale baseline entries."],
    )


def test_matching_baseline_passes_with_normalized_entry() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit(
        {"crates/demo/src/lib.rs": "pub fn public_path() {\n    assert!(true);\n}\n"},
        [entry],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_exact_output(
        result,
        [
            f"Baseline matched: {entry}",
            "Audit passed: 0 unbaselined findings, 0 stale baseline entries",
        ],
    )


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
    assert_exact_output(
        result,
        [f"Baseline matched: {baseline}", "crates/demo/src/lib.rs:4:debug_assert"],
        ["Audit failed: 1 unbaselined finding, 0 stale baseline entries."],
    )


def test_stale_baseline_fails() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit({"crates/demo/src/lib.rs": "pub fn public_path() {}\n"}, [entry])
    assert result.returncode == 1
    assert_exact_output(
        result,
        [f"Stale baseline: {entry}"],
        ["Audit failed: 0 unbaselined findings, 1 stale baseline entry."],
    )

def test_raw_panic_style_entries_cannot_be_baselined() -> None:
    entry = "crates/demo/src/lib.rs:1:panic"
    result = run_audit({"crates/demo/src/lib.rs": 'pub fn bad() { panic!("boom"); }\n'}, [entry])
    assert result.returncode == 1
    assert_exact_output(
        result,
        [f"Baseline matched: {entry}"],
        [
            f"Baseline contains forbidden raw panic-style entry: {entry}",
            "Audit failed: 0 unbaselined findings, 0 stale baseline entries.",
        ],
    )

def test_duplicate_baseline_entries_fail_configuration() -> None:
    entry = "crates/demo/src/lib.rs:2:assert"
    result = run_audit(
        {"crates/demo/src/lib.rs": "pub fn public_path() {\n    assert!(true);\n}\n"},
        [entry, entry],
    )
    assert result.returncode == 2
    assert_exact_output(
        result,
        [],
        [
            "library panic audit configuration error: panic baseline contains duplicate entries"
        ],
    )


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
        (["crates/demo/src/lib.rs:01:assert"], "invalid baseline entry"),
    ]:
        result = run_audit(source, baseline)
        assert result.returncode == 2, (baseline, result.stdout, result.stderr)
        assert_exact_output(
            result,
            [],
            [f"library panic audit configuration error: {message}: {baseline[0]!r}"],
        )


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
    assert_exact_output(
        result,
        [
            f"Baseline matched: {baseline}",
            "crates/demo/src/lib.rs:3:debug_assert",
            "crates/demo/src/other.rs:1:panic",
        ],
        ["Audit failed: 2 unbaselined findings, 0 stale baseline entries."],
    )


def test_round3_ast_edges_are_not_lexically_approximated() -> None:
    source = '''
fn raw_ident(value: Option<bool>) {
    let _ = value.r#unwrap();
    let _ = Option::r#expect(value, "missing");
    let _ = Option::<fn() -> bool>::unwrap(value);
}
pub fn const_generic<T: Bound<{ 1 < 2 }>>() {
    assert!(true);
}
trait Bound<const N: usize> {}
'''
    result = run_audit({"crates/demo/src/lib.rs": source})
    assert result.returncode == 1
    assert_exact_output(
        result,
        [
            "crates/demo/src/lib.rs:3:unwrap",
            "crates/demo/src/lib.rs:4:expect",
            "crates/demo/src/lib.rs:5:unwrap",
            "crates/demo/src/lib.rs:8:assert",
        ],
        ["Audit failed: 4 unbaselined findings, 0 stale baseline entries."],
    )


def main() -> int:
    _build_tool_once()
    tests = [
        test_round3_ast_edges_are_not_lexically_approximated,
        test_wrapper_discovers_json_compiler_artifact_path,
        test_wrong_and_empty_roots_fail_closed,
        test_parse_failure_fails_closed,
        test_production_hit_fails_with_normalized_kind,
        test_all_raw_panic_kinds_fail,
        test_structurally_test_only_modules_and_inline_items_are_excluded,
        test_test_support_is_not_excluded_without_cfg_test,
        test_comments_rustdoc_strings_and_macros_are_ignored,
        test_masking_handles_bytes_nested_comments_and_scope_corruption,
        test_macro_arguments_and_production_definitions_are_scanned_without_expansion,
        test_macro_definitions_scan_generic_associated_calls_with_metavariables,
        test_macro_rules_scan_only_transcribers_and_recurse_repetitions,
        test_imported_panic_macro_aliases_are_detected,
        test_multiline_calls_and_known_ufcs_are_reported_without_user_function_false_matches,
        test_ufcs_reports_both_option_result_methods_and_rejects_collisions,
        test_public_trait_defaults_and_external_impls_are_public_conservatively,
        test_const_generic_where_body_and_attribute_visibility_are_structural,
        test_cfg_test_external_modules_are_order_independent_and_transitive,
        test_inline_path_parent_changes_external_child_base,
        test_graph_referenced_paths_follow_logical_symlinked_crate_root,
        test_graph_referenced_production_paths_outside_src_are_scanned,
        test_production_reference_overrides_test_only_alias,
        test_cfg_test_macro_rules_range_is_excluded,
        test_source_names_and_checkout_ancestors_do_not_exclude_production_files,
        test_outside_root_symlink_is_rejected_safely,
        test_in_root_symlink_is_canonicalized_and_reported_once,
        test_logical_rs_symlink_discovers_extensionless_in_root_target,
        test_large_fixture_stays_within_subprocess_timeout,
        test_method_heavy_fixture_stays_within_subprocess_timeout,
        test_ufcs_aliases_and_local_type_shadowing,
        test_module_declared_aliases_resolve_through_crate_self_super_and_groups,
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
