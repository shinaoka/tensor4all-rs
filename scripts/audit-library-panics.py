#!/usr/bin/env python3
"""Run the Rust AST-based library panic audit.

This compatibility wrapper keeps the historical ``--root``/``--baseline``
interface and exact tool output while keeping parsing and source reachability
in the focused ``library-panic-audit`` Rust workspace tool.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TOOL_ENV = "T4A_PANIC_AUDIT_BIN"
TOOL_PACKAGE = "library-panic-audit"
TIMEOUT_SECONDS = 120


def _tool_path(root: Path) -> Path:
    configured = os.environ.get(TOOL_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"{TOOL_ENV} does not point to a file: {configured}")

    # Always ask Cargo to build. Cargo performs the freshness check; reusing a
    # discovered target binary would let a standalone audit run stale code.
    command = ["cargo", "build", "--release", "-p", TOOL_PACKAGE]
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"failed to build {TOOL_PACKAGE}: {detail}")
    path = root / "target" / "release" / TOOL_PACKAGE
    if not path.is_file():
        raise RuntimeError(f"cargo build did not produce {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    baseline = (args.baseline or root / "scripts" / "library-panics-baseline.json").resolve()
    try:
        tool = _tool_path(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [str(tool), "--root", str(root), "--baseline", str(baseline)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"library panic audit configuration error: {error}", file=sys.stderr)
        return 2
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
