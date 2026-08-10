#!/usr/bin/env python3
"""Tests for the incremental public-error-doc audit.

Ported from tenferro-rs and extended with repository-specific changed-mode
cases required by tensor4all/tensor4all-rs#566 Task 9: a pre-existing
undocumented API is ignored in changed mode; an added undocumented ``Result``
API fails; a concrete ``# Errors`` section passes; deleted files do not
trigger a whole-repo fallback; a missing base commit fails loudly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check-public-error-docs.py")
SPEC = importlib.util.spec_from_file_location("check_public_error_docs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PYTHON = sys.executable


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_repo(prefix: str) -> Path:
    directory = tempfile.mkdtemp(prefix=prefix)
    root = Path(directory) / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "public-error-docs-test")
    _git(root, "config", "user.email", "public-error-docs-test@example.invalid")
    return root


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, str(SCRIPT), "--root-dir", str(root), *extra],
        capture_output=True,
        text=True,
    )


class PublicErrorDocsUnitTests(unittest.TestCase):
    def audit(self, source: str, filename: str = "sample.rs"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_text(source, encoding="utf-8")
            return MODULE.audit_file(path)

    def test_public_result_requires_errors_section(self) -> None:
        findings = self.audit(
            "/// Compute a value.\npub fn compute() -> Result<(), MyError> { Ok(()) }\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].reason, "missing # Errors")

    def test_errors_section_must_name_concrete_failure(self) -> None:
        findings = self.audit(
            "/// Compute a value.\n///\n/// # Errors\n///\n"
            "/// Returns an error when computation fails.\n"
            "pub fn compute() -> Result<(), MyError> { Ok(()) }\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("concrete", findings[0].reason)

    def test_trait_method_and_variant_are_accepted(self) -> None:
        findings = self.audit(
            "pub trait Compute {\n"
            "    /// Compute a value.\n"
            "    ///\n"
            "    /// # Errors\n"
            "    ///\n"
            "    /// Returns `ValidationError::ShapeMismatch` for incompatible input.\n"
            "    fn compute(&self) -> Result<(), Error>;\n"
            "}\n"
        )
        self.assertEqual(findings, [])

    def test_doc_attributes_are_treated_as_generated_rustdoc(self) -> None:
        findings = self.audit(
            '#[doc = "Register an extension."]\n'
            '#[doc = "\\n# Errors\\n\\nReturns `Error::InvalidArgument` for an invalid family id."]\n'
            "pub fn register() -> Result<(), Error> { Ok(()) }\n"
        )
        self.assertEqual(findings, [])

    def test_non_result_function_is_not_a_finding(self) -> None:
        findings = self.audit("/// Return a value.\npub fn value() -> usize { 1 }\n")
        self.assertEqual(findings, [])

    def test_multiline_trait_does_not_leak_scope_into_private_fns(self) -> None:
        source = (
            "pub trait Bounds: \n"
            "    Send \n"
            "{ \n"
            "    /// Documented.\n"
            "    ///\n"
            "    /// # Errors\n"
            "    ///\n"
            "    /// Returns `Error::InvalidArgument` for bad input.\n"
            "    fn documented(&self) -> Result<(), Error>;\n"
            "}\n"
            "fn private_after_trait() -> Result<(), Error> { Ok(()) }\n"
        )
        # The private function after the multiline trait must not be treated as
        # a public trait method.
        self.assertEqual(self.audit(source), [])

    def test_multiline_trait_leaks_scope_into_trait_body(self) -> None:
        source = (
            "pub trait Bounds: \n"
            "    Send \n"
            "{ \n"
            "    /// Undocumented public trait method.\n"
            "    fn missing(&self) -> Result<(), Error>;\n"
            "}\n"
        )
        findings = self.audit(source)
        self.assertEqual(len(findings), 1)
        self.assertIn("missing # Errors", findings[0].reason)

    def test_brace_in_trait_comment_does_not_disable_auditing(self) -> None:
        source = (
            "pub trait T: \n"
            "    Send // { not an opening brace \n"
            "{ \n"
            "    /// Undocumented public trait method.\n"
            "    fn missing(&self) -> Result<(), Error>;\n"
            "}\n"
        )
        findings = self.audit(source)
        self.assertEqual(len(findings), 1)
        self.assertIn("missing # Errors", findings[0].reason)

    def test_comment_semicolon_does_not_truncate_signature(self) -> None:
        source = (
            "pub fn legacy(\n"
            "    input: usize, // compatibility; \n"
            ") -> Result<(), Error> { Ok(()) }\n"
        )
        # The commented semicolon must not end the signature; the Result return
        # type must still be audited.
        findings = self.audit(source)
        self.assertEqual(len(findings), 1)
        self.assertIn("missing # Errors", findings[0].reason)


class PublicErrorDocsChangedModeTests(unittest.TestCase):
    def _write(self, root: Path, name: str, body: str) -> None:
        (root / name).write_text(body, encoding="utf-8")

    def test_pre_existing_undocumented_api_is_ignored_in_changed_mode(self) -> None:
        root = _git_repo("error-docs-preexisting-")
        # Pre-existing undocumented Result API at the base commit.
        self._write(
            root,
            "sample.rs",
            "pub fn legacy() -> Result<(), Error> { Ok(()) }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "base")
        base = _git(root, "rev-parse", "HEAD")

        # Change an unrelated line: the pre-existing API must not be flagged.
        self._write(
            root,
            "sample.rs",
            "// changed\npub fn legacy() -> Result<(), Error> { Ok(()) }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "doc change")
        result = _run(root, "--changed-from", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("public-error-docs-ok", result.stdout)

    def test_added_undocumented_result_api_fails(self) -> None:
        root = _git_repo("error-docs-added-")
        self._write(root, "sample.rs", "pub fn existing() -> usize { 1 }\n")
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "base")
        base = _git(root, "rev-parse", "HEAD")

        # Adding a new public Result API without # Errors must fail.
        self._write(
            root,
            "sample.rs",
            "pub fn existing() -> usize { 1 }\n"
            "pub fn new_api() -> Result<(), Error> { Ok(()) }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "add api")
        result = _run(root, "--changed-from", base)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("missing # Errors", result.stderr)
        self.assertIn("new_api", result.stderr)

    def test_added_result_api_with_concrete_errors_passes(self) -> None:
        root = _git_repo("error-docs-concrete-")
        self._write(root, "sample.rs", "pub fn existing() -> usize { 1 }\n")
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "base")
        base = _git(root, "rev-parse", "HEAD")

        self._write(
            root,
            "sample.rs",
            "pub fn existing() -> usize { 1 }\n"
            "/// Compute.\n"
            "///\n"
            "/// # Errors\n"
            "///\n"
            "/// Returns `ValidationError::ShapeMismatch` for unequal ranks.\n"
            "pub fn new_api() -> Result<(), Error> { Ok(()) }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "add documented api")
        result = _run(root, "--changed-from", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("public-error-docs-ok", result.stdout)

    def test_multiline_signature_change_to_result_is_audited(self) -> None:
        root = _git_repo("error-docs-multisig-")
        self._write(
            root,
            "sample.rs",
            "pub fn legacy(\n"
            "    input: usize,\n"
            ") -> usize\n"
            "{ input }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "base")
        base = _git(root, "rev-parse", "HEAD")

        # Only the return-type line changes; the declaration line is untouched.
        self._write(
            root,
            "sample.rs",
            "pub fn legacy(\n"
            "    input: usize,\n"
            ") -> Result<(), Error>\n"
            "{ Ok(()) }\n",
        )
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "make fallible")
        result = _run(root, "--changed-from", base)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("missing # Errors", result.stderr)
        self.assertIn("legacy", result.stderr)

    def test_deleted_files_do_not_trigger_whole_repo_fallback(self) -> None:
        root = _git_repo("error-docs-delete-")
        self._write(
            root,
            "gone.rs",
            "/// Old.\npub fn gone() -> Result<(), Error> { Ok(()) }\n",
        )
        self._write(root, "kept.rs", "/// Old.\npub fn kept() -> u64 { 0 }\n")
        _git(root, "add", "gone.rs", "kept.rs")
        _git(root, "commit", "-m", "base")
        base = _git(root, "rev-parse", "HEAD")
        (root / "gone.rs").unlink()
        _git(root, "add", "gone.rs")
        _git(root, "commit", "-m", "delete")
        result = _run(root, "--changed-from", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("public-error-docs-ok", result.stdout)

    def test_missing_base_commit_fails_loudly(self) -> None:
        root = _git_repo("error-docs-missingbase-")
        self._write(root, "sample.rs", "pub fn x() -> u8 { 0 }\n")
        _git(root, "add", "sample.rs")
        _git(root, "commit", "-m", "base")
        result = _run(root, "--changed-from", "0000000000000000000000000000000000000000")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
