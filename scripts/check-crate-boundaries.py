#!/usr/bin/env python3
"""Check crate-boundary and dev-dependency-cycle rules mechanically.

Rules enforced here:

1. Normal ``tenferro-*`` dependencies are allowed only in
   ``tensor4all-tensorbackend`` (the single sanctioned tenferro route),
   plus exact temporary exception tuples for the current core, simplett,
   and treetci dependencies. A new tuple, an unsanctioned crate,
   a dependency beyond a crate's tuple, or a stale tuple all fail.
2. Dev-dependency cycles are rejected with the full cycle path, e.g.
   a dev-depends on ``tensorci`` while ``tensorci`` depends on ``a``.

Dev-dependencies on tenferro-* are permitted (test fixtures). The Cargo.toml
parser is stdlib-only (tomllib requires Python 3.11, while CI's runner python3
is 3.10) and understands the bare ``name = value`` forms plus
``name = { package = "..." }`` renames used in this workspace.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENSORBACKEND = "tensor4all-tensorbackend"

# Approved exceptions: normal tenferro deps permitted outside the
# tensorbackend route. These encode the sanctioned tenferro-backed abstractions
# per the architecture rules (use tensorbackend and established tenferro-backed
# abstractions rather than local dense/linalg implementations):
# - tensor4all-core: eager autodiff integration (tenferro-ad EagerTensor
#   reverse-mode graph, tenferro-einsum AD einsum, tenferro-linalg eager
#   full-piv-LU) and the DType/tensor interop used by the native eager path;
#   the tensorbackend crate cannot host the AD layer.
# - tensor4all-simplett: the tensor abstraction layer (tenferro-tensor
#   Tensor/TypedTensor/TensorScalar) and einsum subscripts used by MPO/TT
#   construction; routing every tensor op through tensorbackend would create a
#   circular dependency.
# - tensor4all-treetci: tenferro-linalg LinalgBackend + tenferro-tensor for
#   materializing TreeTN site tensors during cross-interpolation assembly.
# Tuples are exact: adding a dependency to the tuple, or keeping the tuple when
# the crate drops the dependency, both fail.
TENFERRO_EXCEPTIONS: dict[str, frozenset[str]] = {
    "tensor4all-core": frozenset(
        {"tenferro-ad", "tenferro-einsum", "tenferro-linalg", "tenferro-tensor"}
    ),
    "tensor4all-simplett": frozenset({"tenferro-einsum", "tenferro-tensor"}),
    "tensor4all-treetci": frozenset({"tenferro-linalg", "tenferro-tensor"}),
}


def _exceptions_from_env() -> dict[str, frozenset[str]]:
    """Test seam: allow the self-tests to override the exception map.

    Format: ``crate:dep1,dep2;crate2:dep3``. Never set in CI.
    """
    import os

    raw = os.environ.get("T4A_TEST_TENFERRO_EXCEPTIONS")
    if raw is None:
        # Production runs use the built-in map.
        return TENFERRO_EXCEPTIONS
    result: dict[str, frozenset[str]] = {}
    for entry in raw.split(";"):
        if not entry.strip():
            continue
        crate, _, deps = entry.partition(":")
        result[crate.strip()] = frozenset(deps.split(","))
    return result


EXCEPTIONS = _exceptions_from_env()


def crate_manifests(root: Path) -> list[Path]:
    return sorted((root / "crates").glob("*/Cargo.toml"))


def package_name(manifest: Path) -> str:
    text = manifest.read_text(encoding="utf-8")
    match = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"no package name in {manifest}")
    return match.group(1)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_dependencies(manifest: Path) -> tuple[set[str], set[str]]:
    """Return (normal, dev) dependency package names (renames resolved).

    Handles bare ``name = value`` lines, the ``name.workspace`` shorthand,
    ``name = { package = "..." }`` renames, ``[dependencies.name]`` and
    target-specific ``[target.'cfg(...)'.dependencies]`` sub-tables (with an
    optional ``package`` attribute), and quoted dependency keys.
    """
    text = manifest.read_text(encoding="utf-8")
    normal: set[str] = set()
    dev: set[str] = set()
    section: str | None = None
    pending: str | None = None

    def commit() -> None:
        nonlocal pending
        if pending is not None:
            if section == "dependencies":
                normal.add(pending)
            elif section == "dev-dependencies":
                dev.add(pending)
            pending = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            commit()
            name = stripped[1:-1].strip()
            if name == "dependencies":
                section = "dependencies"
            elif name.startswith("dependencies."):
                # ``[dependencies.name]`` sub-table.
                section = "dependencies"
                pending = _unquote(name[len("dependencies."):])
            elif name.endswith(".dependencies"):
                # Target-specific ``[target.'cfg'.dependencies]`` table:
                # its dependency lines are bare entries, no single pending key.
                section = "dependencies"
            elif name == "dev-dependencies":
                section = "dev-dependencies"
            elif name.startswith("dev-dependencies."):
                section = "dev-dependencies"
                pending = _unquote(name[len("dev-dependencies."):])
            elif name.endswith(".dev-dependencies"):
                section = "dev-dependencies"
            else:
                section = "other"
            continue
        if section not in ("dependencies", "dev-dependencies"):
            continue
        match = re.match(r"^(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_\-]+))\s*[=.]", stripped)
        if match is None:
            continue
        key = _unquote(match.group(1) or match.group(2) or match.group(3))
        if key in {"version", "path", "package", "features", "default-features",
                   "optional", "workspace", "registry", "git", "branch", "tag", "rev"}:
            # Attribute lines inside a dep sub-table, not a dependency itself.
            if key == "package" and pending is not None:
                package_match = re.search(r"package\s*=\s*[\"']([^\"']+)[\"']", stripped)
                if package_match is not None:
                    if section == "dependencies":
                        normal.add(package_match.group(1))
                    else:
                        dev.add(package_match.group(1))
                    pending = None
            continue
        package_match = re.search(r"package\s*=\s*[\"']([^\"']+)[\"']", stripped)
        real = package_match.group(1) if package_match else key
        if section == "dependencies":
            normal.add(real)
        else:
            dev.add(real)
    commit()
    return normal, dev


def check_tenferro_boundaries(manifests: list[Path]) -> list[str]:
    violations: list[str] = []
    discovered = {package_name(manifest) for manifest in manifests}
    for crate in sorted(set(EXCEPTIONS) - discovered):
        violations.append(
            f"{crate}: stale tenferro exception tuple {sorted(EXCEPTIONS[crate])}; "
            "no crate of that name exists under crates/ (remove the exception)"
        )
    for manifest in manifests:
        name = package_name(manifest)
        if name == TENSORBACKEND:
            continue
        normal, _ = parse_dependencies(manifest)
        actual = {dep for dep in normal if dep.startswith("tenferro-")}
        expected = EXCEPTIONS.get(name, frozenset())
        if actual == expected:
            continue
        if not expected:
            violations.append(
                f"{name}: normal tenferro dependencies {sorted(actual)} are "
                f"not allowed; only {TENSORBACKEND} may name tenferro-* (no "
                "exception tuple exists)"
            )
        elif not actual:
            violations.append(
                f"{name}: stale tenferro exception tuple {sorted(expected)}; "
                "the crate has no normal tenferro dependencies (remove the exception)"
            )
        elif actual < expected:
            missing = sorted(expected - actual)
            violations.append(
                f"{name}: stale tenferro exception tuple lists {missing} which "
                f"the crate no longer depends on; shrink the exception to {sorted(actual)}"
            )
        else:
            extra = sorted(actual - expected)
            violations.append(
                f"{name}: normal tenferro dependencies {extra} exceed the "
                f"sanctioned exception tuple {sorted(expected)}"
            )
    return violations


def find_dev_dependency_cycles(manifests: list[Path]) -> list[str]:
    """Return formatted cycles that contain at least one dev-dependency edge."""
    names = {package_name(manifest) for manifest in manifests}
    adjacency: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for manifest in manifests:
        name = package_name(manifest)
        normal, dev = parse_dependencies(manifest)
        for dep in sorted(dep for dep in normal if dep in names):
            adjacency[name].append((dep, "normal"))
        for dep in sorted(dep for dep in dev if dep in names):
            adjacency[name].append((dep, "dev"))

    reported: set[tuple[str, ...]] = set()
    cycles: list[str] = []

    def dfs(node: str, path: list[str], kinds: list[str], on_path: set[str]) -> None:
        for dep, kind in adjacency[node]:
            if dep in on_path:
                start = path.index(dep)
                cycle = path[start:] + [dep]
                kinds_cycle = kinds[start:] + [kind]
                # Pure-normal cycles cannot exist in a Cargo workspace and are
                # not our concern; only cycles containing a dev edge are
                # rejected, so a dev-containing cycle is never suppressed by
                # a same-node-set pure-normal cycle.
                if "dev" not in kinds_cycle:
                    continue
                key = tuple(sorted(cycle))
                if key in reported:
                    continue
                reported.add(key)
                segments = [
                    f"{cycle[i]} -> {cycle[i + 1]}({kinds_cycle[i]})"
                    for i in range(len(cycle) - 1)
                ]
                cycles.append("  ".join(segments))
                continue
            if dep in path:
                continue
            dfs(dep, path + [dep], kinds + [kind], on_path | {dep})

    for name in sorted(names):
        dfs(name, [name], [], {name})
    return sorted(cycles)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root_dir.resolve()
    manifests = crate_manifests(root)
    violations = [
        *check_tenferro_boundaries(manifests),
        *find_dev_dependency_cycles(manifests),
    ]
    if violations:
        print("crate-boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("crate-boundary-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
