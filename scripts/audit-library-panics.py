#!/usr/bin/env python3
"""Audit production Rust sources for library panic paths.

The audit is lexical rather than a Rust parser. Comments and literals are
masked in source-sized buffers, and the small amount of item structure needed
for reachability and public-path classification comes from masked tokens.
External modules are excluded only when they are reachable exclusively through
``#[cfg(test)]`` module declarations; file names and directory names never
change audit coverage.

Trait implementation methods are conservatively treated as public paths. A
trait can be declared in another file or crate, and resolving that visibility
lexically is less safe than reporting an assertion in every trait impl method.
Trait default methods are reported when their containing trait is public.
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
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ITEM_KEYWORDS = frozenset(
    {"mod", "fn", "const", "static", "struct", "enum", "trait", "impl", "macro_rules"}
)
UFCS_METHODS = frozenset({"unwrap", "expect"})


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
class Attribute:
    """One source attribute and its masked representation."""

    start: int
    end: int
    source: str
    clean: str


@dataclass(frozen=True)
class CfgTestItem:
    """A source range selected by an exact ``#[cfg(test)]`` item."""

    start: int
    end: int


@dataclass(frozen=True)
class ModuleDecl:
    """An external module declaration and its test-only condition."""

    name: str
    cfg_test: bool
    path_attr: str | None


@dataclass(frozen=True)
class IntervalIndex:
    """Merged source intervals queried in logarithmic time."""

    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @classmethod
    def from_ranges(cls, ranges: Iterable[tuple[int, int]]) -> "IntervalIndex":
        ordered = sorted((start, end) for start, end in ranges if start < end)
        merged: list[list[int]] = []
        for start, end in ordered:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return cls(
            tuple(start for start, _ in merged),
            tuple(end for _, end in merged),
        )

    def contains(self, offset: int) -> bool:
        index = bisect.bisect_right(self.starts, offset) - 1
        return index >= 0 and offset < self.ends[index]


@dataclass(frozen=True)
class ParsedSource:
    """Token and source metadata reused by all scans for one file."""

    source: str
    clean: str
    tokens: tuple[Token, ...]
    token_starts: tuple[int, ...]
    pairs: dict[int, int]
    enclosing_braces: tuple[int | None, ...]
    brace_depth: tuple[int, ...]
    attributes: tuple[Attribute, ...]
    attribute_starts: tuple[int, ...]
    cfg_ranges: tuple[tuple[int, int], ...]


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
        # identifier character. A longer identifier followed by a quote is a
        # lifetime/invalid source, not a byte character.
        if previous != "b" or (
            start >= 2
            and (source[start - 2].isalnum() or source[start - 2] in "_'")
        ):
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
        if clean[index] in "{}()[]<>;:,.!?#+-*=/&|%^~@$'":
            tokens.append(Token(clean[index], index, index + 1))
        index += 1
    return tokens


def _brace_structure(tokens: list[Token]) -> tuple[dict[int, int], list[int | None], list[int]]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    enclosing: list[int | None] = []
    depths: list[int] = []
    for index, token in enumerate(tokens):
        enclosing.append(stack[-1] if stack else None)
        depths.append(len(stack))
        if token.value == "{":
            stack.append(index)
        elif token.value == "}" and stack:
            opening = stack.pop()
            pairs[opening] = index
            pairs[index] = opening
    return pairs, enclosing, depths


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


def _find_attributes(source: str, clean: str) -> list[Attribute]:
    attributes: list[Attribute] = []
    search = 0
    while search < len(clean):
        start = clean.find("#", search)
        if start < 0:
            break
        end = _skip_attribute(clean, start)
        if end is None:
            search = start + 1
            continue
        attributes.append(Attribute(start, end, source[start:end], clean[start:end]))
        search = end
    return attributes


def _is_cfg_test_attribute(attribute: Attribute) -> bool:
    return CFG_TEST_RE.fullmatch(attribute.clean.strip()) is not None


def _attribute_block_bounds(
    attributes: list[Attribute], clean: str
) -> tuple[list[int], list[int]]:
    """Precompute contiguous attribute-block bounds in one sweep."""
    first = list(range(len(attributes)))
    last = list(range(len(attributes)))
    for index in range(1, len(attributes)):
        if clean[attributes[index - 1].end : attributes[index].start].strip() == "":
            first[index] = first[index - 1]
    for index in range(len(attributes) - 2, -1, -1):
        if clean[attributes[index].end : attributes[index + 1].start].strip() == "":
            last[index] = last[index + 1]
    return first, last


def _attribute_index(attributes: list[Attribute]) -> IntervalIndex:
    return IntervalIndex.from_ranges((attribute.start, attribute.end) for attribute in attributes)


def _next_token_index(token_starts: tuple[int, ...] | list[int], offset: int) -> int:
    return bisect.bisect_left(token_starts, offset)


def _item_body(
    tokens: list[Token], keyword_index: int, pairs: dict[int, int]
) -> tuple[int, int | None] | None:
    """Find an item body or semicolon after an item keyword.

    Angle tokens are tracked alongside parentheses and brackets. This matters
    for const-generic expressions such as ``Bound<{ 1 }>``, whose braces are
    not the function or impl body. The same tracking keeps ``where`` clauses
    from ending a range at a generic bound.
    """
    paren = 0
    bracket = 0
    angle = 0
    for index in range(keyword_index + 1, len(tokens)):
        value = tokens[index].value
        if value == "(":
            paren += 1
            continue
        if value == ")" and paren:
            paren -= 1
            continue
        if value == "[":
            bracket += 1
            continue
        if value == "]" and bracket:
            bracket -= 1
            continue
        if value == "<" and not paren and not bracket:
            angle += 1
            continue
        if value == ">" and angle and not paren and not bracket:
            angle -= 1
            continue
        if paren or bracket or angle:
            continue
        if value == "{":
            closing = pairs.get(index)
            if closing is not None:
                return index, closing
            return None
        if value == ";":
            return index, None
    return None


def _declaration_start(tokens: list[Token], index: int, brace_depth: list[int]) -> int:
    depth = brace_depth[index]
    start = index
    while start > 0:
        previous = tokens[start - 1].value
        if brace_depth[start - 1] != depth or previous in {";", "{", "}"}:
            break
        start -= 1
    return start


def _attributes_between(
    attributes: list[Attribute], starts: tuple[int, ...] | list[int], start: int, end: int
) -> list[Attribute]:
    left = bisect.bisect_left(starts, start)
    right = bisect.bisect_left(starts, end)
    return [attribute for attribute in attributes[left:right] if attribute.end <= end]


def _is_public_visibility(
    tokens: list[Token], start: int, end: int, attributes: IntervalIndex
) -> bool:
    for index in range(start, end):
        token = tokens[index]
        if attributes.contains(token.start):
            continue
        if token.value == "pub":
            return index + 1 >= end or tokens[index + 1].value != "("
    return False


def _find_item_keyword(tokens: list[Token], start: int) -> int | None:
    for index in range(start, len(tokens)):
        value = tokens[index].value
        if value in ITEM_KEYWORDS:
            return index
        if value in {";", "{", "}"}:
            return None
    return None


def _cfg_test_items(
    clean: str,
    tokens: list[Token],
    pairs: dict[int, int],
    token_starts: tuple[int, ...],
    attributes: list[Attribute],
) -> list[CfgTestItem]:
    """Return exact source ranges belonging to ``#[cfg(test)]`` items."""
    ranges: list[CfgTestItem] = []
    block_first, block_last = _attribute_block_bounds(attributes, clean)
    for attribute_index, attribute in enumerate(attributes):
        if not _is_cfg_test_attribute(attribute):
            continue
        first, last = block_first[attribute_index], block_last[attribute_index]
        item_index = _find_item_keyword(
            tokens, _next_token_index(token_starts, attributes[last].end)
        )
        if item_index is None:
            continue
        body = _item_body(tokens, item_index, pairs)
        if body is None:
            continue
        opening, closing = body
        end = tokens[opening].end if closing is None else tokens[closing].end
        ranges.append(CfgTestItem(attributes[first].start, end))
    return ranges


def _module_declarations(
    tokens: list[Token],
    pairs: dict[int, int],
    brace_depth: list[int],
    attributes: list[Attribute],
    attribute_starts: tuple[int, ...] | list[int],
    attribute_index: IntervalIndex,
    cfg_ranges: IntervalIndex,
) -> list[ModuleDecl]:
    declarations: list[ModuleDecl] = []
    for index, token in enumerate(tokens):
        if token.value != "mod" or attribute_index.contains(token.start):
            continue
        if index + 1 >= len(tokens) or not IDENT_RE.fullmatch(tokens[index + 1].value):
            continue
        body = _item_body(tokens, index, pairs)
        if body is None or body[1] is not None:
            continue
        start = _declaration_start(tokens, index, brace_depth)
        attrs = _attributes_between(
            attributes, attribute_starts, tokens[start].start, token.start
        )
        path_attr = next(
            (
                match.group(1)
                for attr in attrs
                if (match := PATH_ATTR_RE.search(attr.source))
            ),
            None,
        )
        declarations.append(
            ModuleDecl(
                tokens[index + 1].value,
                cfg_ranges.contains(token.start)
                or any(_is_cfg_test_attribute(attr) for attr in attrs),
                path_attr,
            )
        )
    return declarations


def _parse_source(source: str) -> ParsedSource:
    clean = sanitize_rust(source)
    tokens = tokenize(clean)
    pairs, enclosing, brace_depth = _brace_structure(tokens)
    token_starts = tuple(token.start for token in tokens)
    attributes = _find_attributes(source, clean)
    cfg_items = _cfg_test_items(
        clean, tokens, pairs, token_starts, attributes
    )
    return ParsedSource(
        source=source,
        clean=clean,
        tokens=tuple(tokens),
        token_starts=token_starts,
        pairs=pairs,
        enclosing_braces=tuple(enclosing),
        brace_depth=tuple(brace_depth),
        attributes=tuple(attributes),
        attribute_starts=tuple(attribute.start for attribute in attributes),
        cfg_ranges=tuple((item.start, item.end) for item in cfg_items),
    )


def _module_candidates(parent: Path, module: str, path_attr: str | None) -> Iterable[Path]:
    if path_attr is not None:
        path = Path(path_attr)
        yield path if path.is_absolute() else parent.parent / path
        return
    if parent.name in {"lib.rs", "main.rs", "mod.rs"}:
        base = parent.parent
    else:
        base = parent.parent / parent.stem
    yield base / f"{module}.rs"
    yield base / module / "mod.rs"


def _resolve_inside(root: Path, path: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _cfg_test_external_files(
    root: Path, paths: Iterable[Path], parsed: dict[Path, ParsedSource]
) -> set[Path]:
    """Return external modules with no production-reachable path.

    A production edge always wins over a test-only edge to the same canonical
    file. Edges inside a test-only external module remain unreachable from a
    production root, so exclusion is transitive without relying on filenames.
    """
    path_set = set(paths)
    production_edges: dict[Path, set[Path]] = {path: set() for path in path_set}
    incoming_production: set[Path] = set()
    incoming_test: set[Path] = set()

    for parent in sorted(path_set):
        metadata = parsed[parent]
        attributes = list(metadata.attributes)
        attribute_index = _attribute_index(attributes)
        cfg_ranges = IntervalIndex.from_ranges(metadata.cfg_ranges)
        declarations = _module_declarations(
            list(metadata.tokens),
            metadata.pairs,
            list(metadata.brace_depth),
            attributes,
            metadata.attribute_starts,
            attribute_index,
            cfg_ranges,
        )
        for declaration in declarations:
            target = next(
                (
                    resolved
                    for candidate in _module_candidates(
                        parent, declaration.name, declaration.path_attr
                    )
                    if (resolved := _resolve_inside(root, candidate)) in path_set
                ),
                None,
            )
            if target is None:
                continue
            if declaration.cfg_test:
                incoming_test.add(target)
            else:
                production_edges[parent].add(target)
                incoming_production.add(target)

    roots = {
        path
        for path in path_set
        if path not in incoming_production and path not in incoming_test
    }
    # Explicit crate entry points remain roots even in malformed/cyclic fixture
    # graphs. Bin targets are also independent production roots.
    for path in path_set:
        relative = path.relative_to(root)
        if (
            len(relative.parts) >= 4
            and relative.parts[0] == "crates"
            and relative.parts[2] == "src"
        ):
            if relative.parts[3] in {"lib.rs", "main.rs"} or (
                relative.parts[3] == "bin" and len(relative.parts) == 5
            ):
                roots.add(path)

    reachable: set[Path] = set()
    stack = sorted(roots, reverse=True)
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(sorted(production_edges[current] - reachable, reverse=True))

    return path_set - reachable


def _is_trait_impl(tokens: list[Token], impl_index: int, opening: int) -> bool:
    angle = paren = bracket = 0
    for index in range(impl_index + 1, opening):
        value = tokens[index].value
        if not angle and not paren and not bracket and value == "where":
            break
        if not angle and not paren and not bracket and value == "for":
            return True
        if value == "<" and not paren and not bracket:
            angle += 1
        elif value == ">" and angle and not paren and not bracket:
            angle -= 1
        elif value == "(":
            paren += 1
        elif value == ")" and paren:
            paren -= 1
        elif value == "[":
            bracket += 1
        elif value == "]" and bracket:
            bracket -= 1
    return False


def _public_body_ranges(metadata: ParsedSource) -> IntervalIndex:
    """Find public function bodies, including conservative trait impl paths."""
    tokens = list(metadata.tokens)
    pairs = metadata.pairs
    attributes = _attribute_index(list(metadata.attributes))
    brace_depth = list(metadata.brace_depth)
    public_trait_blocks: set[int] = set()
    trait_impl_blocks: set[int] = set()

    for index, token in enumerate(tokens):
        if token.value == "trait" and not attributes.contains(token.start):
            body = _item_body(tokens, index, pairs)
            if body is None or body[1] is None:
                continue
            start = _declaration_start(tokens, index, brace_depth)
            if _is_public_visibility(tokens, start, index, attributes):
                public_trait_blocks.add(body[0])
        elif token.value == "impl" and not attributes.contains(token.start):
            body = _item_body(tokens, index, pairs)
            if body is not None and body[1] is not None and _is_trait_impl(tokens, index, body[0]):
                trait_impl_blocks.add(body[0])

    ranges: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token.value != "fn" or attributes.contains(token.start):
            continue
        body = _item_body(tokens, index, pairs)
        if body is None or body[1] is None:
            continue
        start = _declaration_start(tokens, index, brace_depth)
        public = _is_public_visibility(tokens, start, index, attributes)
        enclosing = metadata.enclosing_braces[index]
        if enclosing in public_trait_blocks or enclosing in trait_impl_blocks:
            # Every trait impl method is public by policy. This intentionally
            # avoids a cross-file trait-name table and errs toward detection.
            public = True
        if public:
            opening, closing = body
            ranges.append((tokens[opening].start, tokens[closing].end))
    return IntervalIndex.from_ranges(ranges)


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


def _known_type_path(tokens: list[Token], start: int) -> tuple[str, int] | None:
    """Match only canonical Option/Result paths at identifier boundaries."""
    if start >= len(tokens):
        return None
    value = tokens[start].value
    if value in {"Option", "Result"}:
        if start and tokens[start - 1].value == "::":
            return None
        return value, start + 1

    leading_path = value == "::"
    if leading_path:
        if start and (
            IDENT_RE.fullmatch(tokens[start - 1].value)
            or tokens[start - 1].value in {")", "]", "}"}
        ):
            return None
        start += 1
        if start >= len(tokens):
            return None
        value = tokens[start].value
    if value not in {"std", "core"}:
        return None
    if not leading_path and start and tokens[start - 1].value == "::":
        return None
    if start + 4 >= len(tokens):
        return None
    if tokens[start + 1].value != "::" or tokens[start + 3].value != "::":
        return None
    module = tokens[start + 2].value
    type_name = tokens[start + 4].value
    if module not in {"option", "result"} or type_name not in {"Option", "Result"}:
        return None
    if (module == "option") != (type_name == "Option"):
        return None
    return type_name, start + 5


def _skip_optional_generic(tokens: list[Token], index: int) -> int | None:
    if index < len(tokens) and tokens[index].value == "::":
        if index + 1 >= len(tokens) or tokens[index + 1].value != "<":
            return index
        index += 1
    if index < len(tokens) and tokens[index].value == "<":
        end = _angle_end(tokens, index)
        return None if end is None else end + 1
    return index


def _method_after_type(tokens: list[Token], index: int) -> tuple[int, str] | None:
    if index >= len(tokens) or tokens[index].value != "::":
        return None
    method_index = index + 1
    if method_index >= len(tokens) or tokens[method_index].value not in UFCS_METHODS:
        return None
    after_method = method_index + 1
    if after_method < len(tokens) and tokens[after_method].value == "<":
        end = _angle_end(tokens, after_method)
        if end is None:
            return None
        after_method = end + 1
    if after_method >= len(tokens) or tokens[after_method].value != "(":
        return None
    return method_index, tokens[method_index].value


def _qualified_ufcs_match(tokens: list[Token], start: int) -> tuple[int, str] | None:
    """Match ``<Option<T>>::unwrap`` and equivalent Result syntax."""
    if tokens[start].value != "<":
        return None
    known = _known_type_path(tokens, start + 1)
    if known is None:
        return None
    _, index = known
    index = _skip_optional_generic(tokens, index)
    if index is None or index >= len(tokens) or tokens[index].value != ">":
        return None
    method = _method_after_type(tokens, index + 1)
    if method is None:
        return None
    _, kind = method
    return tokens[start].start, kind


def _ufcs_matches(tokens: list[Token]) -> Iterable[tuple[int, str]]:
    for index, token in enumerate(tokens):
        if token.value == "<":
            match = _qualified_ufcs_match(tokens, index)
            if match is not None:
                yield match
            continue
        known = _known_type_path(tokens, index)
        if known is None:
            continue
        _, after_type = known
        after_type = _skip_optional_generic(tokens, after_type)
        if after_type is None:
            continue
        method = _method_after_type(tokens, after_type)
        if method is not None:
            yield token.start, method[1]


def _audit_file_with_offsets(
    path: Path,
    root: Path,
    test_only_files: set[Path],
    metadata: ParsedSource,
) -> list[Finding]:
    """Scan one complete masked source buffer and map matches to source lines."""
    if path in test_only_files:
        return []
    tokens = list(metadata.tokens)
    cfg_ranges = IntervalIndex.from_ranges(metadata.cfg_ranges)
    public_ranges = _public_body_ranges(metadata)
    line_starts = _line_map(metadata.source)
    relative = path.relative_to(root).as_posix()

    def excluded(offset: int) -> bool:
        return cfg_ranges.contains(offset)

    findings: list[Finding] = []
    for match in RAW_CALL_RE.finditer(metadata.clean):
        if not excluded(match.start()):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
    for match in METHOD_RE.finditer(metadata.clean):
        if not excluded(match.start()):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
    for offset, kind in _ufcs_matches(tokens):
        if not excluded(offset):
            findings.append(Finding(relative, _line_at(line_starts, offset), kind))
    for match in ASSERT_RE.finditer(metadata.clean):
        if not excluded(match.start()) and public_ranges.contains(match.start()):
            findings.append(Finding(relative, _line_at(line_starts, match.start()), match.group(1)))
    return sorted(findings)


def audit_file(path: Path, root: Path, test_only_files: set[Path]) -> list[Finding]:
    """Scan one production file for normalized panic-style findings."""
    source = path.read_text(encoding="utf-8")
    return _audit_file_with_offsets(path, root, test_only_files, _parse_source(source))


def discover_source_files(root: Path) -> list[Path]:
    """Discover canonical in-root crate sources without trusting symlink names."""
    root = root.resolve()
    source_root = root / "crates"
    candidates = sorted(source_root.glob("*/src/**/*.rs"), key=lambda path: path.as_posix())
    discovered: set[Path] = set()
    for candidate in candidates:
        resolved = _resolve_inside(root, candidate)
        if resolved is None or not resolved.is_file():
            continue
        relative = resolved.relative_to(root)
        if len(relative.parts) < 3 or relative.parts[0] != "crates" or relative.parts[2] != "src":
            continue
        discovered.add(resolved)
    return sorted(discovered, key=lambda path: path.relative_to(root).as_posix())


def scan_tree(root: Path) -> list[Finding]:
    root = root.resolve()
    paths = discover_source_files(root)
    parsed: dict[Path, ParsedSource] = {}
    for path in paths:
        parsed[path] = _parse_source(path.read_text(encoding="utf-8"))
    test_only_files = _cfg_test_external_files(root, paths, parsed)
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_audit_file_with_offsets(path, root, test_only_files, parsed[path]))
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
    parser.add_argument(
        "--root", type=Path, help="repository root (defaults to this script's repository)"
    )
    parser.add_argument("--baseline", type=Path, help="exact JSON baseline path")
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    baseline_path = (
        args.baseline or root / "scripts" / "library-panics-baseline.json"
    ).resolve()

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
