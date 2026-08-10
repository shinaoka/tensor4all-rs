#!/usr/bin/env python3
"""Tests for the crate-boundary checker (tensor4all/tensor4all-rs#566 Task 9).

Fixture manifests must cover: the tensorbackend normal tenferro route passes;
a new feature crate with a normal tenferro dependency fails; a renamed
``package = "tenferro-*"`` dependency fails; a dev-only tenferro dependency is
allowed; an acyclic dev graph passes; the tcicore -> tensorci (dev) ->
tcicore (normal) cycle fails with the full path; a stale exception tuple
fails.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check-crate-boundaries.py")
PYTHON = sys.executable

TENSORBACKEND_MANIFEST = """\
[package]
name = "tensor4all-tensorbackend"
version = "0.1.0"
edition = "2021"

[dependencies]
tenferro-einsum.workspace = true
tenferro-tensor.workspace = true
"""

PLAIN_MANIFEST = """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }
"""

WORKSPACE_MANIFEST = """\
[workspace]
members = ["crates/*"]
resolver = "2"
"""


def write_fixture(files: dict[str, str]) -> Path:
    directory = tempfile.mkdtemp(prefix="crate-boundaries-")
    root = Path(directory)
    (root / "Cargo.toml").write_text(WORKSPACE_MANIFEST, encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def run_check(
    root: Path, exceptions: str | None = ""
) -> subprocess.CompletedProcess[str]:
    """Run the checker with an isolated exception map (test fixture).

    ``exceptions=None`` keeps the built-in map; ``""`` (default) uses no
    exceptions so fixture crates never trip the removed-crate stale check.
    """
    import os

    env = dict(os.environ)
    if exceptions is None:
        env.pop("T4A_TEST_TENFERRO_EXCEPTIONS", None)
    else:
        env["T4A_TEST_TENFERRO_EXCEPTIONS"] = exceptions
    return subprocess.run(
        [PYTHON, str(SCRIPT), "--root-dir", str(root)],
        capture_output=True,
        text=True,
        env=env,
    )


class CrateBoundaryTests(unittest.TestCase):
    def test_tensorbackend_normal_tenferro_route_passes(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("crate-boundary-ok", result.stdout)

    def test_new_feature_crate_normal_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "tenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-demo", result.stderr)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_renamed_package_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + 'sneaky = { workspace = true, package = "tenferro-tensor" }\n',
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)

    def test_dev_only_tenferro_dependency_is_allowed(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "\n[dev-dependencies]\ntenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_acyclic_dev_graph_passes(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
                "crates/tensor4all-demo2/Cargo.toml": """\
[package]
name = "tensor4all-demo2"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-demo = { path = "../tensor4all-demo" }

[dev-dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tcicore_tensorci_dev_cycle_fails_with_full_path(self) -> None:
        # The exact cycle the checker exists to reject: tcicore dev-depends on
        # tensorci while tensorci normal-depends on tcicore.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-tcicore/Cargo.toml": """\
[package]
name = "tensor4all-tcicore"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }

[dev-dependencies]
tensor4all-tensorci = { path = "../tensor4all-tensorci" }
""",
                "crates/tensor4all-tensorci/Cargo.toml": """\
[package]
name = "tensor4all-tensorci"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tcicore = { path = "../tensor4all-tcicore" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-tcicore -> tensor4all-tensorci(dev)", result.stderr)
        self.assertIn("tensor4all-tensorci -> tensor4all-tcicore(normal)", result.stderr)

    def test_dev_cycle_survives_coexisting_pure_normal_cycle(self) -> None:
        # A pure-normal cycle must not suppress a dev-containing cycle with
        # the same node set.
        root = write_fixture(
            {
                "crates/tensor4all-a/Cargo.toml": """\
[package]
name = "tensor4all-a"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-b = { path = "../tensor4all-b" }

[dev-dependencies]
tensor4all-b = { path = "../tensor4all-b" }
""",
                "crates/tensor4all-b/Cargo.toml": """\
[package]
name = "tensor4all-b"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-a = { path = "../tensor4all-a" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-a -> tensor4all-b(dev)", result.stderr)

    def test_subtable_renamed_tenferro_dependency_fails(self) -> None:
        # ``[dependencies.name]`` sub-tables must not bypass the tenferro rule.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies.sneaky]
package = "tenferro-tensor"
workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_target_specific_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[target.'cfg(unix)'.dependencies]
tenferro-tensor.workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_quoted_subtable_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies."tenferro-tensor"]
workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_single_quoted_package_rename_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "sneaky = { workspace = true, package = 'tenferro-tensor' }\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_removed_exception_crate_is_stale(self) -> None:
        # An exception tuple whose crate no longer exists must be flagged.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
            }
        )
        result = run_check(root, exceptions="tensor4all-ghost:tenferro-tensor")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("no crate of that name exists", result.stderr)
        self.assertIn("tensor4all-ghost", result.stderr)

    def test_stale_exception_tuple_fails(self) -> None:
        # A crate whose sanctioned tuple lists a dependency it no longer has.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "tenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(
            root, exceptions="tensor4all-demo:tenferro-tensor,tenferro-einsum"
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("stale tenferro exception", result.stderr)
        self.assertIn("tenferro-einsum", result.stderr)


if __name__ == "__main__":
    unittest.main()
