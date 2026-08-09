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
# Keep the wrapper timeout above the Rust tool's 10-minute Clippy timeout;
# CI allows another five minutes for process startup and teardown.
BUILD_TIMEOUT_SECONDS = 600
AUDIT_TIMEOUT_SECONDS = 1200
KNOWN_CARGO_REASONS = {
    "build-finished",
    "build-script-executed",
    "compiler-artifact",
    "compiler-message",
}


def _decode_cargo_output(output: bytes) -> str:
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"Cargo emitted invalid UTF-8: {error}") from error


def _target_record(message: dict[str, object]) -> tuple[str, list[str]]:
    package_id = message.get("package_id")
    if not isinstance(package_id, str):
        raise RuntimeError("Cargo artifact record omitted string package_id")
    target = message.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("Cargo artifact record omitted target object")
    name = target.get("name")
    kinds = target.get("kind")
    if not isinstance(name, str) or not isinstance(kinds, list) or not kinds:
        raise RuntimeError("Cargo artifact record contained an invalid target")
    if not all(isinstance(kind, str) for kind in kinds):
        raise RuntimeError("Cargo artifact target kind was not a string")
    return name, kinds


def _artifact_executable(output: bytes, root: Path) -> Path:
    text = _decode_cargo_output(output)
    executable: Path | None = None
    build_finished = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Cargo emitted malformed JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(message, dict):
            raise RuntimeError(f"Cargo emitted a non-object record on line {line_number}")
        reason = message.get("reason")
        if not isinstance(reason, str):
            raise RuntimeError(f"Cargo record omitted string reason on line {line_number}")
        if reason not in KNOWN_CARGO_REASONS:
            raise RuntimeError(f"Cargo emitted unknown JSON reason: {reason!r}")
        if reason == "compiler-artifact":
            name, kinds = _target_record(message)
            if name == TOOL_PACKAGE and "bin" in kinds:
                value = message.get("executable")
                if not isinstance(value, str) or not value:
                    raise RuntimeError("library-panic-audit artifact omitted executable")
                path = Path(value)
                executable = path if path.is_absolute() else (root / path).resolve()
        elif reason == "compiler-message":
            _target_record(message)
            if not isinstance(message.get("message"), dict):
                raise RuntimeError("Cargo compiler-message record omitted message object")
        elif reason == "build-finished":
            success = message.get("success")
            if not isinstance(success, bool):
                raise RuntimeError("Cargo build-finished record omitted boolean success")
            if not success:
                raise RuntimeError("Cargo reported an unsuccessful build")
            build_finished = True
    if not build_finished:
        raise RuntimeError("Cargo output omitted a successful build-finished record")
    if executable is None or not executable.is_file():
        raise RuntimeError(f"cargo build produced no executable artifact for {TOOL_PACKAGE}")
    return executable


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
        capture_output=True,
        check=False,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = _decode_cargo_output(result.stderr or result.stdout).strip()
        raise RuntimeError(f"failed to build {TOOL_PACKAGE}: {detail}")
    return _artifact_executable(result.stdout, root)


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
            timeout=AUDIT_TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"library panic audit configuration error: {error}", file=sys.stderr)
        return 2
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
