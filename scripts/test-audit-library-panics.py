#!/usr/bin/env python3
"""Focused tests for the compiler-backed library panic audit."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit-library-panics.py"
TOOL_ENV = "T4A_PANIC_AUDIT_BIN"
TOOL_PACKAGE = "library-panic-audit"


FIXTURE_MANIFEST = """\
[workspace]
members = ["crates/demo"]
resolver = "2"
"""

FIXTURE_PACKAGE = """\
[package]
name = "demo"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[[bin]]
name = "demo-bin"
path = "src/shared.rs"

[features]
default = []
production = []
"""

FIXTURE_LIB = r'''#![allow(dead_code)]

macro_rules! passthrough {
    ($value:expr) => { $value };
}
macro_rules! dormant_panic {
    () => { panic!("dormant macro"); };
}
macro_rules! safe_local_macro {
    () => { () };
}
mod safe_local_panic {
    macro_rules! panic {
        () => { () };
    }
    pub(crate) use panic;
    pub(crate) fn call() {
        panic!();
    }
}

use std::option::Option as Maybe;
use std::panic as fail;

struct Custom;
impl Custom {
    fn new() -> Self { Self }
    fn unwrap(self) {}
    fn expect(self, _: &str) {}
}

pub trait PublicTrait {
    fn default_assertion(&self) {
        assert!(true);
    }

    fn trait_impl_assertion(&self);
}

impl PublicTrait for Custom {
    fn trait_impl_assertion(&self) {
        assert!(true);
    }
}

impl Custom {
    pub fn inherent_assertion(&self) {
        assert!(true);
    }

    fn private_inherent_assertion(&self) {
        assert!(true);
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
pub fn forced_warnings(option: Option<bool>) {
    panic!("allow cannot hide this");
    let _ = option.unwrap();
}

pub fn direct(option: Option<bool>) {
    let _ = option.unwrap();
}

pub fn raw_unreachable() {
    unreachable!("production unreachable");
}

pub fn raw_expect(option: Option<bool>) {
    let _ = option.expect("production expect");
}

pub fn aliases(option: Maybe<bool>) {
    let _ = Maybe::unwrap(option);
    fail!("aliased panic");
}

pub fn local_macro(option: Option<bool>) {
    passthrough!(option.unwrap());
    passthrough!(panic!("invoked macro argument"));
}

pub fn custom_methods() {
    Custom::new().unwrap();
    Custom::new().expect("safe custom method");
    safe_local_macro!();
    safe_local_panic::call();
}

pub fn public_assertions() {
    assert!(true);
    debug_assert!(true);
    assert_eq!(1, 1);
    debug_assert_eq!(1, 1);
}

fn private_assertions() {
    assert!(true);
    debug_assert!(true);
}

pub(crate) fn crate_assertions() {
    assert!(true);
}

pub fn cfg_statement_assertions(value: bool) {
    #[cfg(test)]
    assert!(true);
    #[cfg(test)]
    {
        assert!(true);
    }
    #[cfg(not(test))]
    {
        assert!(true);
    }
    let _ = {
        #[cfg(test)]
        assert!(true);
        #[cfg(not(test))]
        {
            assert!(true);
        }
    };
    match value {
        #[cfg(test)]
        true => assert!(true),
        #[cfg(not(test))]
        false => assert!(true),
        _ => {}
    }
}

#[cfg(target_arch = "definitely-not-this")]
pub fn target_arch_assertion() {
    assert!(true);
}

#[cfg_attr(not(test), cfg_attr(not(test), cfg(test)))]
pub fn nested_cfg_attr_test_only() {
    assert!(true);
}

#[path = "shared.rs"]
mod path_shared;
#[cfg_attr(all(feature = "production", not(test)), path = "feature_selected.rs")]
mod selected;
mod nested;
#[path = "file_level.rs"]
mod file_level;

#[cfg(test)]
pub fn test_only_function() {
    panic!("test function");
    let _ = None::<bool>.unwrap();
}

#[cfg(test)]
mod tests {
    pub fn test_only_module() {
        panic!("test module");
    }
}

#[cfg(feature = "production")]
pub fn feature_production() {
    panic!("all-features production");
}

#[path = "macro_fixture.rs"]
mod macro_fixture;
'''

FIXTURE_SHARED = r'''#[path = "shared/nested.rs"]
mod nested;

pub fn shared_root_assertion() {
    assert!(true);
}

fn main() {
    std::panic!("production binary");
}
'''

FIXTURE_SELECTED = 'pub fn default_selected_assertion() { assert!(true); }\n'
FIXTURE_FEATURE_SELECTED = 'pub fn feature_selected_assertion() { assert!(true); }\n'
FIXTURE_BIN_NESTED = 'pub fn bin_root_assertion() { assert!(true); }\n'
FIXTURE_LIB_NESTED = 'pub fn lib_module_assertion() { assert!(true); }\n'
FIXTURE_MACRO = r'''macro_rules! matcher_only {
    (assert!($($argument:tt)*)) => { () };
}
macro_rules! transcribed_assertion {
    ($assert:ident) => {
        $assert!();
        assert!(true);
    };
}
macro_rules! passthrough_assertion {
    ($value:expr) => { $value };
}
pub fn macro_assertions() {
    passthrough_assertion!(assert!(true));
}
macro_rules! nested_outer {
    () => {
        macro_rules! nested_inner {
            (assert!($value:expr)) => {};
            (_unused:expr) => { assert!(true); };
        }
    };
}
pub fn nested_macro_use() { assert!(true); }
'''
FIXTURE_FILE_LEVEL_TEST = '''#![cfg(test)]
pub fn file_level_test_only() {
    assert!(true);
}
'''


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
        capture_output=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    wrapper_spec = importlib.util.spec_from_file_location("panic_wrapper", SCRIPT)
    assert wrapper_spec and wrapper_spec.loader
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    executable = wrapper._artifact_executable(result.stdout, ROOT)
    assert executable.is_file(), executable
    os.environ[TOOL_ENV] = str(executable)


def _write_fixture(root: Path) -> None:
    files = {
        "Cargo.toml": FIXTURE_MANIFEST,
        "crates/demo/Cargo.toml": FIXTURE_PACKAGE,
        "crates/demo/src/lib.rs": FIXTURE_LIB,
        "crates/demo/src/shared.rs": FIXTURE_SHARED,
        "crates/demo/src/selected.rs": FIXTURE_SELECTED,
        "crates/demo/src/feature_selected.rs": FIXTURE_FEATURE_SELECTED,
        "crates/demo/src/nested.rs": FIXTURE_BIN_NESTED,
        "crates/demo/src/shared/nested.rs": FIXTURE_LIB_NESTED,
        "crates/demo/src/macro_fixture.rs": FIXTURE_MACRO,
        "crates/demo/src/file_level.rs": FIXTURE_FILE_LEVEL_TEST,
        "crates/demo/tests/ignored.rs": 'pub fn ignored() { panic!("integration test"); }\n',
        "crates/demo/examples/ignored.rs": 'fn main() { panic!("example"); }\n',
        "crates/demo/benches/ignored.rs": 'fn main() { panic!("bench"); }\n',
    }
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _line(source: str, needle: str, occurrence: int = 1) -> int:
    seen = 0
    for number, value in enumerate(source.splitlines(), 1):
        if needle in value:
            seen += 1
            if seen == occurrence:
                return number
    raise AssertionError(f"missing {needle!r}")


def _finding_sort_key(entry: str) -> tuple[str, int, str]:
    path, line, kind = entry.rsplit(":", 2)
    return path, int(line), kind


def _run_audit(root: Path, baseline: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    baseline_path = root / "baseline.json"
    baseline_path.write_text(json.dumps(baseline or []), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--baseline",
            str(baseline_path),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )


def test_dep_info_selects_each_production_source_once() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(fixture)
        baseline = [
            *[
                f"crates/demo/src/lib.rs:{line}:assert"
                for line in (34, 42, 48, 92, 116, 123, 130, 137)
            ],
            "crates/demo/src/lib.rs:93:debug_assert",
            f"crates/demo/src/nested.rs:{_line(FIXTURE_BIN_NESTED, 'assert!(true);')}:assert",
            f"crates/demo/src/shared.rs:{_line(FIXTURE_SHARED, 'assert!(true);')}:assert",
            f"crates/demo/src/shared/nested.rs:{_line(FIXTURE_LIB_NESTED, 'assert!(true);')}:assert",
            f"crates/demo/src/selected.rs:{_line(FIXTURE_SELECTED, 'assert!(true);')}:assert",
            f"crates/demo/src/feature_selected.rs:{_line(FIXTURE_FEATURE_SELECTED, 'assert!(true);')}:assert",
        ]
        result = _run_audit(fixture, baseline)

    assert result.returncode == 1, result.stderr
    assert result.stderr.splitlines() == [
        "Audit failed: 15 unbaselined findings, 0 stale baseline entries."
    ]
    findings = result.stdout.splitlines()
    assert findings[: len(baseline)] == [
        f"Baseline matched: {entry}"
        for entry in sorted(baseline, key=_finding_sort_key)
    ]
    assert findings[len(baseline) :] == [
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    panic!(\"allow cannot hide this\");')}:panic",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    let _ = option.unwrap();', 1)}:unwrap",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    let _ = option.unwrap();', 2)}:unwrap",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    unreachable!(\"production unreachable\");')}:unreachable",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    let _ = option.expect(\"production expect\");')}:expect",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    let _ = Maybe::unwrap(option);')}:unwrap",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    fail!(\"aliased panic\");')}:panic",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    passthrough!(option.unwrap());')}:unwrap",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    passthrough!(panic!(\"invoked macro argument\"));')}:panic",
        f"crates/demo/src/lib.rs:{_line(FIXTURE_LIB, '    panic!(\"all-features production\");')}:panic",
        f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '        assert!(true);')}:assert",
        f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '    passthrough_assertion!(assert!(true));')}:assert",
        f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '            (_unused:expr) => { assert!(true); };')}:assert",
        f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, 'pub fn nested_macro_use() { assert!(true); }')}:assert",
        f"crates/demo/src/shared.rs:{_line(FIXTURE_SHARED, '    std::panic!(\"production binary\");')}:panic",
    ]
    assert {finding.rsplit(':', 1)[-1] for finding in findings[len(baseline) :]} >= {
        "panic",
        "unreachable",
        "unwrap",
        "expect",
    }
    assert all("ignored" not in finding for finding in findings)
    assert all("test_only" not in finding for finding in findings)
    assert all("file_level" not in finding for finding in findings)


def test_macro_assertion_arguments_and_transcribers_are_reported() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(fixture)
        result = _run_audit(fixture)
    assert result.returncode == 1, result.stderr
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '        assert!(true);')}:assert" in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '    passthrough_assertion!(assert!(true));')}:assert" in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '    (assert!($($argument:tt)*)) => { () };')}:assert" not in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '        $assert!();')}:assert" not in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '            (assert!($value:expr)) => {};')}:assert" not in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, '            (_unused:expr) => { assert!(true); };')}:assert" in result.stdout
    assert f"crates/demo/src/macro_fixture.rs:{_line(FIXTURE_MACRO, 'pub fn nested_macro_use() { assert!(true); }')}:assert" in result.stdout


def test_unknown_cfg_assertion_is_included_conservatively() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        _write_fixture(fixture)
        entry = "crates/demo/src/lib.rs:137:assert"
        result = _run_audit(fixture, [entry])
    assert result.returncode == 1
    assert f"Baseline matched: {entry}" in result.stdout


def test_missing_build_finished_fails_closed() -> None:
    wrapper_spec = importlib.util.spec_from_file_location("panic_wrapper", SCRIPT)
    assert wrapper_spec and wrapper_spec.loader
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    artifact = json.dumps(
        {
            "reason": "compiler-artifact",
            "package_id": "path+file:///fixture#library-panic-audit@0.1.0",
            "target": {"name": TOOL_PACKAGE, "kind": ["bin"]},
            "executable": "/tmp/library-panic-audit",
        }
    ).encode() + b"\n"
    try:
        wrapper._artifact_executable(artifact, Path("/tmp"))
    except RuntimeError as error:
        assert "build-finished" in str(error)
    else:
        raise AssertionError("missing build-finished was accepted")


def test_wrapper_discovers_custom_target_artifact() -> None:
    wrapper_spec = importlib.util.spec_from_file_location("panic_wrapper", SCRIPT)
    assert wrapper_spec and wrapper_spec.loader
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        fake_cargo = directory_path / "cargo"
        artifact = directory_path / "custom-target" / "release" / TOOL_PACKAGE
        artifact.parent.mkdir(parents=True)
        fake_cargo.write_text(
            "#!/bin/sh\n"
            f"touch {artifact}\n"
            "printf '%s\\n' "
            + repr(
                json.dumps(
                    {
                        "reason": "compiler-artifact",
                        "package_id": "path+file:///fixture#library-panic-audit@0.1.0",
                        "target": {"name": TOOL_PACKAGE, "kind": ["bin"]},
                        "executable": str(artifact),
                    }
                )
            )
            + "; printf '%s\\n' "
            + repr(json.dumps({"reason": "build-finished", "success": True}))
            + "\n",
            encoding="utf-8",
        )
        fake_cargo.chmod(0o755)
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
        for malformed in [b"\xff", b"null\n", b'{"reason":"unknown"}\n']:
            try:
                wrapper._artifact_executable(malformed, directory_path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("malformed Cargo JSON was accepted")


def test_missing_manifest_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = _run_audit(Path(directory))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("library panic audit configuration error:")


def test_timeout_nesting_leaves_margin_for_ci() -> None:
    wrapper_spec = importlib.util.spec_from_file_location("panic_wrapper", SCRIPT)
    assert wrapper_spec and wrapper_spec.loader
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    assert wrapper.BUILD_TIMEOUT_SECONDS == 600
    assert wrapper.AUDIT_TIMEOUT_SECONDS == 1500
    assert wrapper.BUILD_TIMEOUT_SECONDS + wrapper.AUDIT_TIMEOUT_SECONDS < 40 * 60


def test_wrapper_rejects_missing_override() -> None:
    old_override = os.environ.get(TOOL_ENV)
    os.environ[TOOL_ENV] = "/does/not/exist"
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(ROOT)],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    finally:
        if old_override is None:
            os.environ.pop(TOOL_ENV, None)
        else:
            os.environ[TOOL_ENV] = old_override
    assert result.returncode == 2
    assert result.stdout == ""
    assert "does not point to a file" in result.stderr


def main() -> int:
    _build_tool_once()
    tests = [
        test_dep_info_selects_each_production_source_once,
        test_unknown_cfg_assertion_is_included_conservatively,
        test_macro_assertion_arguments_and_transcribers_are_reported,
        test_timeout_nesting_leaves_margin_for_ci,
        test_missing_build_finished_fails_closed,
        test_wrapper_discovers_custom_target_artifact,
        test_missing_manifest_fails_closed,
        test_wrapper_rejects_missing_override,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} scanner tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
