#!/usr/bin/env python3
"""Run the compiler-backed production library panic audit.

This wrapper keeps the historical ``--root``/``--baseline`` interface and
``T4A_PANIC_AUDIT_BIN`` override. The Rust tool asks Cargo/Clippy to compile
workspace ``crates/`` library and binary targets, parses only the four exact
panic-path diagnostics, and uses ``syn`` only for the reviewed public
``assert!``/``debug_assert!`` baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TOOL_ENV = "T4A_PANIC_AUDIT_BIN"
TOOL_PACKAGE = "library-panic-audit"
TIMEOUT_SECONDS = 120


def _artifact_executable(output: str, root: Path) -> Path | None:
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        executable = message.get("executable")
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == TOOL_PACKAGE
            and "bin" in target.get("kind", [])
            and executable
        ):
            path = Path(executable)
            return path if path.is_absolute() else (root / path).resolve()
    return None


def _tool_path(root: Path) -> Path:
    configured = os.environ.get(TOOL_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"{TOOL_ENV} does not point to a file: {configured}")

    # Cargo is the source of truth for freshness and target layout. The JSON
    # artifact message handles CARGO_TARGET_DIR, build.target-dir,
    # CARGO_BUILD_TARGET, and target-specific executable suffixes without
    # guessing a fixed target path.
    command = [
        "cargo",
        "build",
        "--release",
        "--message-format=json-render-diagnostics",
        "-p",
        TOOL_PACKAGE,
    ]
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
    path = _artifact_executable(result.stdout, root)
    if path is None or not path.is_file():
        raise RuntimeError(
            f"cargo build produced no executable artifact for {TOOL_PACKAGE}"
        )
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
