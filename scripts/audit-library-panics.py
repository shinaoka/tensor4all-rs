#!/usr/bin/env python3
"""Audit production Rust sources for library panic paths.

The audit is deliberately lexical rather than a Rust parser: it masks comments,
rustdoc, strings, and character literals before matching the small set of
panic-style calls.  ``assert!`` and ``debug_assert!`` are reported only inside
lexically public functions; raw panic-style calls are reported everywhere in
production source.  Exact, normalized baseline entries may suppress reviewed
public assertions, but never raw panic-style calls, and every baseline entry
must still exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RAW_KINDS = frozenset({"panic", "unreachable", "unwrap", "expect"})
ENTRY_KINDS = "panic|unreachable|unwrap|expect|assert|debug_assert"
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")
EXTERNAL_MOD_RE = re.compile(r"\bmod\s+([A-Za-z_]\w*)\s*;")
PATH_ATTR_RE = re.compile(r"#\s*\[\s*path\s*=\s*\"([^\"]+)\"\s*\]")
PUBLIC_FN_RE = re.compile(
    r"\bpub(?P<qual>\s*\([^)]*\))?"
    r"(?:\s+(?:async|unsafe|const|extern)\b)*\s+fn\b"
)
RAW_CALL_RE = re.compile(r"\b(panic|unreachable)\s*!\s*(?=[({[])\s*")
METHOD_RE = re.compile(r"\.\s*(unwrap|expect)\s*(?=\()")
ASSERT_RE = re.compile(r"\b(assert|debug_assert)\s*!\s*(?=[({[])\s*")
ENTRY_RE = re.compile(rf"^(.+):([1-9][0-9]*):({ENTRY_KINDS})$")


@dataclass(frozen=True, order=True)
class Finding:
    """A normalized audit finding."""

    path: str
    line: int
    kind: str

    @property
    def entry(self) -> str:
        return f"{self.path}:{self.line}:{self.kind}"


def _mask(chars: list[str], start: int, end: int) -> None:
    """Replace source text with spaces while preserving line positions."""
    for index in range(start, end):
        if chars[index] not in "\r\n":
            chars[index] = " "


def _raw_string_end(source: str, start: int) -> int | None:
    """Return the end of a Rust raw string beginning at ``start``."""
    if source.startswith("br", start):
        hash_start = start + 2
    elif source.startswith("r", start):
        hash_start = start + 1
    else:
        return None

    hash_end = hash_start
    while hash_end < len(source) and source[hash_end] == "#":
        hash_end += 1
    if hash_end >= len(source) or source[hash_end] != '"':
        return None

    closing = '"' + ("#" * (hash_end - hash_start))
    end = source.find(closing, hash_end + 1)
    return len(source) if end < 0 else end + len(closing)


def _char_literal_end(source: str, start: int) -> int | None:
    """Return the end of a character literal, not a lifetime label."""
    if source[start] != "'" or start > 0 and (source[start - 1].isalnum() or source[start - 1] in "_'"):
        return None
    index = start + 1
    if index >= len(source) or source[index] in "\r\n":
        return None
    if source[index] == "\\":
        index += 2
    else:
        index += 1
    if index < len(source) and source[index] == "'":
        return index + 1
    return None


def sanitize_rust(source: str) -> str:
    """Mask comments and literals without changing line/column offsets."""
    chars = list(source)
    index = 0
    block_depth = 0

    while index < len(source):
        if block_depth:
            if source.startswith("/*", index):
                _mask(chars, index, index + 2)
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                _mask(chars, index, index + 2)
                block_depth -= 1
                index += 2
            else:
                _mask(chars, index, index + 1)
                index += 1
            continue

        if source.startswith("//", index):
            line_end = source.find("\n", index)
            end = len(source) if line_end < 0 else line_end
            _mask(chars, index, end)
            index = end
            continue

        raw_end = _raw_string_end(source, index)
        if raw_end is not None:
            _mask(chars, index, raw_end)
            index = raw_end
            continue

        if source.startswith("/*", index):
            _mask(chars, index, index + 2)
            block_depth = 1
            index += 2
            continue

        if source[index] == '"':
            end = index + 1
            escaped = False
            while end < len(source):
                char = source[end]
                if char == "\n" and not escaped:
                    break
                if char == '"' and not escaped:
                    end += 1
                    break
                if char == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
                end += 1
            _mask(chars, index, end)
            index = end
            continue

        char_end = _char_literal_end(source, index) if source[index] == "'" else None
        if char_end is not None:
            _mask(chars, index, char_end)
            index = char_end
            continue

        index += 1

    return "".join(chars)


def brace_delta(line: str) -> int:
    """Return brace balance after comments and literals have been masked."""
    return line.count("{") - line.count("}")


def _is_test_file(path: Path) -> bool:
    """Recognize conventional test files without hiding production helpers."""
    return (
        "tests" in path.parts
        or path.name in {"tests.rs", "test_utils.rs"}
        or path.name.endswith("_tests.rs")
    )


def _module_candidates(parent: Path, module: str) -> Iterable[Path]:
    if parent.name in {"lib.rs", "main.rs", "mod.rs"}:
        base = parent.parent
    else:
        base = parent.parent / parent.stem
    yield base / f"{module}.rs"
    yield base / module / "mod.rs"


def _cfg_test_external_files(paths: Iterable[Path]) -> set[Path]:
    """Resolve external modules guarded by an exact ``cfg(test)`` attribute."""
    path_set = set(paths)
    excluded: set[Path] = set()

    for parent in sorted(path_set):
        source = parent.read_text(encoding="utf-8")
        clean = sanitize_rust(source)
        pending_cfg = False
        pending_path: str | None = None

        for raw_line, clean_line in zip(source.splitlines(), clean.splitlines()):
            cfg_match = CFG_TEST_RE.search(clean_line)
            if cfg_match:
                pending_cfg = True
                clean_line = clean_line[: cfg_match.start()] + " " * (cfg_match.end() - cfg_match.start()) + clean_line[cfg_match.end() :]
                path_match = PATH_ATTR_RE.search(raw_line)
                if path_match:
                    pending_path = path_match.group(1)

            stripped = clean_line.strip()
            if not pending_cfg:
                continue
            if not stripped or stripped.startswith("#"):
                path_match = PATH_ATTR_RE.search(raw_line)
                if path_match:
                    pending_path = path_match.group(1)
                continue

            module_match = EXTERNAL_MOD_RE.search(clean_line)
            if module_match:
                module = module_match.group(1)
                if pending_path is not None:
                    candidate = (parent.parent / pending_path).resolve()
                    if candidate in path_set:
                        excluded.add(candidate)
                else:
                    for candidate in _module_candidates(parent, module):
                        if candidate in path_set:
                            excluded.add(candidate)
                pending_cfg = False
                pending_path = None
                continue

            # A different item consumes the attribute; do not guess that it is
            # test-only. This keeps exclusions structural and conservative.
            pending_cfg = False
            pending_path = None

    return excluded


def _audit_file_with_offsets(path: Path, root: Path, cfg_test_files: set[Path]) -> list[Finding]:
    """Offset-aware implementation used by ``audit_file``."""
    # Kept as a separate implementation detail so line/column preservation is
    # explicit and easy to test; callers use the public ``audit_file`` wrapper.
    if _is_test_file(path) or path in cfg_test_files:
        return []
    source = path.read_text(encoding="utf-8")
    clean = sanitize_rust(source)
    relative = path.relative_to(root).as_posix()
    findings: list[Finding] = []
    depth = 0
    public_body_depth: int | None = None
    pending_public_fn = False
    pending_cfg = False
    pending_test_item = False
    test_body_depth: int | None = None
    offset = 0

    for raw_line, clean_line in zip(source.splitlines(keepends=True), clean.splitlines(keepends=True)):
        line_text = clean_line.rstrip("\r\n")
        line_delta = brace_delta(line_text)
        line_number = source.count("\n", 0, offset) + 1

        if test_body_depth is not None:
            depth += line_delta
            if depth < test_body_depth:
                test_body_depth = None
            offset += len(raw_line)
            continue

        cfg_match = CFG_TEST_RE.search(line_text)
        if cfg_match:
            pending_cfg = True
            line_text = line_text[: cfg_match.start()] + " " * (cfg_match.end() - cfg_match.start()) + line_text[cfg_match.end() :]
            line_delta = brace_delta(line_text)

        stripped = line_text.strip()
        if pending_cfg or pending_test_item:
            if pending_cfg and (not stripped or stripped.startswith("#")):
                depth += line_delta
                offset += len(raw_line)
                continue
            if EXTERNAL_MOD_RE.search(line_text):
                pending_cfg = False
                pending_test_item = False
                depth += line_delta
                offset += len(raw_line)
                continue
            starts_test_item = pending_test_item or re.search(
                r"\b(?:mod|fn|const|static|struct|enum|trait|impl)\b", line_text
            )
            if starts_test_item and "{" in line_text:
                old_depth = depth
                depth += line_delta
                pending_cfg = False
                pending_test_item = False
                if depth > old_depth:
                    test_body_depth = depth
                offset += len(raw_line)
                continue
            if starts_test_item and ";" not in line_text:
                pending_cfg = False
                pending_test_item = True
                depth += line_delta
                offset += len(raw_line)
                continue
            pending_cfg = False
            pending_test_item = False

        if public_body_depth is not None and depth < public_body_depth:
            public_body_depth = None
        if pending_public_fn:
            if ";" in line_text and "{" not in line_text:
                pending_public_fn = False
            elif "{" in line_text:
                public_body_depth = depth + 1
                pending_public_fn = False

        public_match = PUBLIC_FN_RE.search(line_text)
        if public_match and public_match.group("qual") is None:
            if "{" in line_text[public_match.end() :]:
                public_body_depth = depth + 1
                pending_public_fn = False
            elif ";" not in line_text[public_match.end() :]:
                pending_public_fn = True

        line_offset = offset
        for match in RAW_CALL_RE.finditer(line_text):
            findings.append(Finding(relative, line_number, match.group(1)))
        for match in METHOD_RE.finditer(line_text):
            findings.append(Finding(relative, line_number, match.group(1)))
        if public_body_depth is not None:
            for match in ASSERT_RE.finditer(line_text):
                findings.append(Finding(relative, line_number, match.group(1)))

        depth += line_delta
        offset = line_offset + len(raw_line)

    return sorted(findings)


def audit_file(path: Path, root: Path, cfg_test_files: set[Path]) -> list[Finding]:
    """Scan one production file for normalized panic-style findings."""
    return _audit_file_with_offsets(path, root, cfg_test_files)


def discover_source_files(root: Path) -> list[Path]:
    source_root = root / "crates"
    return sorted(source_root.glob("*/src/**/*.rs"))


def scan_tree(root: Path) -> list[Finding]:
    paths = discover_source_files(root)
    cfg_test_files = _cfg_test_external_files(paths)
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_audit_file_with_offsets(path, root, cfg_test_files))
    return sorted(findings)


def _parse_entry(value: object) -> Finding:
    if not isinstance(value, str):
        raise ValueError("baseline entries must be strings")
    match = ENTRY_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid baseline entry: {value!r}")
    path, line, kind = match.groups()
    normalized = Path(path).as_posix()
    if (
        path != normalized
        or "\\" in path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not path.startswith("crates/")
    ):
        raise ValueError(f"baseline path is not normalized: {value!r}")
    return Finding(path, int(line), kind)


def load_baseline(path: Path) -> set[Finding]:
    if not path.exists():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("panic baseline must be a JSON array of path:line:kind strings")
    entries = [_parse_entry(item) for item in value]
    if len(set(entries)) != len(entries):
        raise ValueError("panic baseline contains duplicate entries")
    return set(entries)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="repository root (defaults to this script's repository)")
    parser.add_argument("--baseline", type=Path, help="exact JSON baseline path")
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    baseline_path = (args.baseline or root / "scripts" / "library-panics-baseline.json").resolve()

    try:
        findings = set(scan_tree(root))
        baseline = load_baseline(baseline_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"library panic audit configuration error: {error}", file=sys.stderr)
        return 2

    forbidden_baseline = sorted(entry for entry in baseline if entry.kind in RAW_KINDS)
    matched = sorted(findings & baseline)
    unbaselined = sorted(findings - baseline)
    stale = sorted(baseline - findings)

    for entry in matched:
        print(f"Baseline matched: {entry.entry}")
    for entry in unbaselined:
        print(entry.entry)
    for entry in stale:
        print(f"Stale baseline: {entry.entry}")
    for entry in forbidden_baseline:
        print(f"Baseline contains forbidden raw panic-style entry: {entry.entry}", file=sys.stderr)

    if unbaselined or stale or forbidden_baseline:
        print(
            "Audit failed: "
            f"{len(unbaselined)} {_plural(len(unbaselined), 'unbaselined finding')}, "
            f"{len(stale)} {_plural(len(stale), 'stale baseline entry', 'stale baseline entries')}.",
            file=sys.stderr,
        )
        return 1

    print(
        "Audit passed: "
        f"{len(unbaselined)} unbaselined findings, "
        f"{len(stale)} stale baseline entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
