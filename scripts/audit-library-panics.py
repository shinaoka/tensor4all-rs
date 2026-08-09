#!/usr/bin/env python3
"""Audit production Rust sources for library panic paths.

The audit is lexical rather than a Rust parser.  Comments and literals are
masked in one source-sized buffer, so every finding keeps its original source
offset and line mapping while calls may span lines or comments.  Exact,
normalized baseline entries may suppress reviewed public assertions, but never
raw panic-style calls, and every baseline entry must still exist.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RAW_KINDS = frozenset({"panic", "unreachable", "unwrap", "expect"})
ENTRY_KINDS = "panic|unreachable|unwrap|expect|assert|debug_assert"
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")
PATH_ATTR_RE = re.compile(r"#\s*\[\s*path\s*=\s*\"([^\"]+)\"\s*\]")
RAW_CALL_RE = re.compile(r"\b(panic|unreachable)\s*!\s*(?=[({\[])\s*")
METHOD_RE = re.compile(r"\.\s*(unwrap|expect)\s*(?=\()")
ASSERT_RE = re.compile(r"\b(assert|debug_assert)\s*!\s*(?=[({\[])\s*")
ENTRY_RE = re.compile(rf"^(.+):([1-9][0-9]*):({ENTRY_KINDS})$")
IDENT_RE = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True, order=True)
class Finding:
    """A normalized audit finding."""

    path: str
    line: int
    kind: str

    @property
    def entry(self) -> str:
        return f"{self.path}:{self.line}:{self.kind}"


@dataclass(frozen=True)
class Token:
    """A token in the comment/literal-masked source."""

    value: str
    start: int
    end: int


@dataclass(frozen=True)
class CfgTestItem:
    """A source range selected by an exact ``#[cfg(test)]`` item."""

    start: int
    end: int


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
    """Return the end of a character or byte-character literal, not a lifetime."""
    if source[start] != "'":
        return None
    previous = source[start - 1] if start else ""
    if previous.isalnum() or previous in "_'":
        # ``b'x'`` is the one literal form whose quote is preceded by an
        # identifier character.  A longer identifier followed by a quote is a
        # lifetime/invalid source, not a byte character.
        if previous != "b" or start < 2 or source[start - 2].isalnum() or source[start - 2] in "_'":
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
    """Mask comments and literals without changing source offsets or lines."""
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


def tokenize(clean: str) -> list[Token]:
    """Tokenize identifiers and punctuation in a masked source buffer."""
    tokens: list[Token] = []
    index = 0
    while index < len(clean):
        match = IDENT_RE.match(clean, index)
        if match:
            tokens.append(Token(match.group(), match.start(), match.end()))
            index = match.end()
            continue
        if clean.startswith("::", index):
            tokens.append(Token("::", index, index + 2))
            index += 2
            continue
        if clean[index] in "{}()[]<>;:,.!?#+-*=/&|%^~@":
            tokens.append(Token(clean[index], index, index + 1))
        index += 1
    return tokens


def _brace_pairs(tokens: list[Token]) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, token in enumerate(tokens):
        if token.value == "{":
            stack.append(index)
        elif token.value == "}" and stack:
            opening = stack.pop()
            pairs[opening] = index
            pairs[index] = opening
    return pairs


def _line_map(source: str) -> list[int]:
    return [0] + [index + 1 for index, char in enumerate(source) if char == "\n"]


def _line_at(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def _skip_attribute(clean: str, start: int) -> int | None:
    """Skip one ``#[...]`` attribute and return its exclusive end."""
    index = start
    while index < len(clean) and clean[index].isspace():
        index += 1
    if index >= len(clean) or clean[index] != "#":
        return None
    index += 1
    while index < len(clean) and clean[index].isspace():
        index += 1
    if index >= len(clean) or clean[index] != "[":
        return None
    depth = 1
    for position in range(index + 1, len(clean)):
        if clean[position] == "[":
            depth += 1
        elif clean[position] == "]":
            depth -= 1
            if depth == 0:
                return position + 1
    return len(clean)


def _after_cfg_attributes(clean: str, source: str, cfg_end: int) -> tuple[int, str | None]:
    """Skip following attributes and return item offset plus optional path."""
    index = cfg_end
    path: str | None = None
    while True:
        attribute_end = _skip_attribute(clean, index)
        if attribute_end is None:
            break
        attribute = source[index:attribute_end]
        path_match = PATH_ATTR_RE.search(attribute)
        if path_match:
            path = path_match.group(1)
        index = attribute_end
    return index, path


def _next_token_index(tokens: list[Token], offset: int) -> int:
    return bisect.bisect_left([token.start for token in tokens], offset)


def _item_body(tokens: list[Token], keyword_index: int, pairs: dict[int, int]) -> tuple[int, int | None] | None:
    """Find an item body or semicolon after an item keyword."""
    paren = 0
    bracket = 0
    for index in range(keyword_index + 1, len(tokens)):
        value = tokens[index].value
        if value == "(" or value == "[":
            if value == "(":
                paren += 1
            else:
                bracket += 1
            continue
        if value == ")" and paren:
            paren -= 1
            continue
        if value == "]" and bracket:
            bracket -= 1
            continue
        if paren or bracket:
            continue
        if value == "{":
            closing = pairs.get(index)
            if closing is not None:
                return index, closing
            return None
        if value == ";":
            return index, None
    return None


def _cfg_test_items(source: str, clean: str, tokens: list[Token], pairs: dict[int, int]) -> list[CfgTestItem]:
    """Return exact source ranges belonging to ``#[cfg(test)]`` items."""
    ranges: list[CfgTestItem] = []
    for cfg_match in CFG_TEST_RE.finditer(clean):
        item_offset, _ = _after_cfg_attributes(clean, source, cfg_match.end())
        item_index = _next_token_index(tokens, item_offset)
        # Visibility/modifier tokens can precede the actual item keyword.
        keyword_index = None
        for index in range(item_index, len(tokens)):
            if tokens[index].value in {"mod", "fn", "const", "static", "struct", "enum", "trait", "impl"}:
                keyword_index = index
                break
            if tokens[index].value in {";", "{"}:
                break
        if keyword_index is None:
            continue
        body = _item_body(tokens, keyword_index, pairs)
        if body is None:
            continue
        opening, closing = body
        end = tokens[opening].end if closing is None else tokens[closing].end
        ranges.append(CfgTestItem(cfg_match.start(), end))
    return ranges


def _cfg_test_external_files(paths: Iterable[Path]) -> set[Path]:
    """Resolve external modules guarded by an exact ``#[cfg(test)]`` item."""
    path_set = set(paths)
    excluded: set[Path] = set()
    for parent in sorted(path_set):
        source = parent.read_text(encoding="utf-8")
        clean = sanitize_rust(source)
        tokens = tokenize(clean)
        pairs = _brace_pairs(tokens)
        for item in _cfg_test_items(source, clean, tokens, pairs):
            cfg_match = CFG_TEST_RE.search(clean, item.start, item.end)
            if cfg_match is None:
                continue
            item_offset, path_attr = _after_cfg_attributes(clean, source, cfg_match.end())
            item_start = _next_token_index(tokens, item_offset)
            item_end = _next_token_index(tokens, item.end)
            keyword_index = next(
                (
                    index
                    for index in range(item_start, item_end)
                    if tokens[index].value == "mod"
                ),
                None,
            )
            if keyword_index is None or keyword_index + 1 >= item_end:
                continue
            module = tokens[keyword_index + 1].value
            if path_attr is not None:
                candidate = (parent.parent / path_attr).resolve()
                if candidate in path_set:
                    excluded.add(candidate)
            else:
                for candidate in _module_candidates(parent, module):
                    if candidate in path_set:
                        excluded.add(candidate)
    return excluded


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


def _declaration_start(tokens: list[Token], index: int, brace_depth: list[int]) -> int:
    depth = brace_depth[index]
    start = index
    while start > 0:
        previous = tokens[start - 1].value
        if brace_depth[start - 1] != depth or previous in {";", "{", "}"}:
            break
        start -= 1
    return start


def _is_public_visibility(tokens: list[Token], start: int, end: int) -> bool:
    for index in range(start, end):
        if tokens[index].value != "pub":
            continue
        if index + 1 >= end or tokens[index + 1].value != "(":
            return True
    return False


def _trait_name_before_for(tokens: list[Token], impl_index: int, for_index: int) -> str | None:
    """Return the implemented trait's final name, skipping generic arguments."""
    angle_depth = 0
    for index in range(for_index - 1, impl_index, -1):
        value = tokens[index].value
        if value == ">":
            angle_depth += 1
        elif value == "<" and angle_depth:
            angle_depth -= 1
        elif angle_depth == 0 and IDENT_RE.fullmatch(value):
            return value
    return None


def _public_body_ranges(tokens: list[Token], pairs: dict[int, int]) -> list[tuple[int, int]]:
    """Find public function bodies, including public trait method bodies."""
    if not tokens:
        return []
    brace_depth: list[int] = []
    depth = 0
    for token in tokens:
        brace_depth.append(depth)
        if token.value == "{":
            depth += 1
        elif token.value == "}" and depth:
            depth -= 1

    public_traits: set[str] = set()
    trait_blocks: dict[int, int] = {}
    for index, token in enumerate(tokens):
        if token.value != "trait":
            continue
        declaration_start = _declaration_start(tokens, index, brace_depth)
        if not _is_public_visibility(tokens, declaration_start, index):
            continue
        body = _item_body(tokens, index, pairs)
        if body is None or body[1] is None:
            continue
        name_index = index + 1
        if name_index < len(tokens) and IDENT_RE.fullmatch(tokens[name_index].value):
            public_traits.add(tokens[name_index].value)
        trait_blocks[body[0]] = body[1]  # type: ignore[index]

    public_impl_blocks: set[int] = set()
    for index, token in enumerate(tokens):
        if token.value != "impl":
            continue
        body = _item_body(tokens, index, pairs)
        if body is None or body[1] is None:
            continue
        opening, _ = body
        for for_index in range(index + 1, opening):
            if tokens[for_index].value == "for":
                trait_name = _trait_name_before_for(tokens, index, for_index)
                if trait_name in public_traits:
                    public_impl_blocks.add(opening)
                break

    ranges: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token.value != "fn":
            continue
        body = _item_body(tokens, index, pairs)
        if body is None or body[1] is None:
            continue
        opening, closing = body
        declaration_start = _declaration_start(tokens, index, brace_depth)
        public = _is_public_visibility(tokens, declaration_start, index)
        enclosing_openings = [
            opening_index
            for opening_index, closing_index in pairs.items()
            if opening_index < index and isinstance(closing_index, int) and closing_index > index and tokens[opening_index].value == "{"
        ]
        if enclosing_openings:
            enclosing = max(enclosing_openings)
            if enclosing in trait_blocks or enclosing in public_impl_blocks:
                public = True
        if public:
            ranges.append((tokens[opening].start, tokens[closing].end))
    return ranges


def _angle_end(tokens: list[Token], start: int) -> int | None:
    if start >= len(tokens) or tokens[start].value != "<":
        return None
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].value == "<":
            depth += 1
        elif tokens[index].value == ">":
            depth -= 1
            if depth == 0:
                return index
    return None


def _is_known_ufcs_type(tokens: list[Token], index: int, kind: str) -> bool:
    """Accept only Option/Result paths from the standard type namespaces."""
    if tokens[index].value != kind:
        return False
    expected_module = "option" if kind == "Option" else "result"
    previous = index - 1
    if previous < 0 or tokens[previous].value != "::":
        return True
    # A qualified path is accepted only for std/core's canonical module.
    path_start = previous - 3
    if path_start >= 0 and tokens[previous - 1].value == expected_module and tokens[previous - 2].value == "::":
        namespace = tokens[path_start].value
        if namespace not in {"std", "core"}:
            return False
        if path_start >= 2 and tokens[path_start - 1].value == "::":
            return not IDENT_RE.fullmatch(tokens[path_start - 2].value)
        return True
    return False


def _known_ufcs_kind(path: list[str]) -> str | None:
    if path == ["Option"] or path in (["std", "option", "Option"], ["core", "option", "Option"]):
        return "unwrap"
    if path == ["Result"] or path in (["std", "result", "Result"], ["core", "result", "Result"]):
        return "expect"
    return None


def _qualified_ufcs_match(tokens: list[Token], start: int) -> tuple[int, str] | None:
    """Match ``<Option<T>>::unwrap``/``<Result<T>>::expect`` syntax."""
    if tokens[start].value != "<" or (start and tokens[start - 1].value == "::"):
        return None
    index = start + 1
    if index < len(tokens) and tokens[index].value == "::":
        index += 1
    path: list[str] = []
    while index < len(tokens):
        value = tokens[index].value
        if not IDENT_RE.fullmatch(value):
            break
        path.append(value)
        index += 1
        if index >= len(tokens) or tokens[index].value != "::":
            break
        index += 1
    kind = _known_ufcs_kind(path)
    if kind is None or not path:
        return None
    if index < len(tokens) and tokens[index].value == "::":
        index += 1
        if index >= len(tokens) or tokens[index].value != "<":
            return None
        angle_end = _angle_end(tokens, index)
        if angle_end is None:
            return None
        index = angle_end + 1
    elif index < len(tokens) and tokens[index].value == "<":
        angle_end = _angle_end(tokens, index)
        if angle_end is None:
            return None
        index = angle_end + 1
    if index >= len(tokens) or tokens[index].value != ">":
        return None
    index += 1
    if index + 1 >= len(tokens) or tokens[index].value != "::":
        return None
    index += 1
    if index + 1 >= len(tokens) or tokens[index].value != kind or tokens[index + 1].value != "(":
        return None
    return tokens[start].start, kind


def _ufcs_matches(tokens: list[Token]) -> Iterable[tuple[int, str]]:
    for index, token in enumerate(tokens):
        if token.value == "<":
            match = _qualified_ufcs_match(tokens, index)
            if match is not None:
                yield match
            continue
        if token.value not in {"Option", "Result"}:
            continue
        kind = "unwrap" if token.value == "Option" else "expect"
        if not _is_known_ufcs_type(tokens, index, token.value):
            continue
        next_index = index + 1
        if next_index >= len(tokens) or tokens[next_index].value != "::":
            continue
        next_index += 1
        if next_index < len(tokens) and tokens[next_index].value == "<":
            angle_end = _angle_end(tokens, next_index)
            if angle_end is None:
                continue
            next_index = angle_end + 1
            if next_index >= len(tokens) or tokens[next_index].value != "::":
                continue
            next_index += 1
        if next_index + 1 < len(tokens) and tokens[next_index].value == kind and tokens[next_index + 1].value == "(":
            yield token.start, kind


def _audit_file_with_offsets(path: Path, root: Path, cfg_test_files: set[Path]) -> list[Finding]:
    """Scan a complete masked source buffer and map matches back to source lines."""
    if _is_test_file(path) or path in cfg_test_files:
        return []
    source = path.read_text(encoding="utf-8")
    clean = sanitize_rust(source)
    tokens = tokenize(clean)
    pairs = _brace_pairs(tokens)
    cfg_ranges = _cfg_test_items(source, clean, tokens, pairs)
    public_ranges = _public_body_ranges(tokens, pairs)
    line_starts = _line_map(source)
    relative = path.relative_to(root).as_posix()

    def excluded(offset: int) -> bool:
        return any(item.start <= offset < item.end for item in cfg_ranges)

    findings: list[Finding] = []
    for match in RAW_CALL_RE.finditer(clean):
        if not excluded(match.start()):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
    for match in METHOD_RE.finditer(clean):
        if not excluded(match.start()):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
    for offset, kind in _ufcs_matches(tokens):
        if not excluded(offset):
            findings.append(Finding(relative, _line_at(line_starts, offset), kind))
    for match in ASSERT_RE.finditer(clean):
        if not excluded(match.start()) and any(start <= match.start() < end for start, end in public_ranges):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
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
