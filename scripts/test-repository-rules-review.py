#!/usr/bin/env python3
"""Self-contained tests for scripts/repository-rules-review.py.

Run with `python3 scripts/test-repository-rules-review.py`; no pytest needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "repository-rules-review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repository_rules_review", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_diff(path: str, added: list[str], *, start: int = 1) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "index abc..def 100644",
            f"--- a/{path}",
            f"+++ b/{path}",
            f"@@ -{start},1 +{start},{len(added) + 1} @@",
            " unchanged",
            *(f"+{line}" for line in added),
        ]
    )


# Secret-shaped fixtures are assembled at runtime so this file contains no
# contiguous secret-shaped literal of its own. Otherwise the guard under test
# blocks the LLM pass on every PR that touches its own tests, and the only way
# to review such a PR is a maintainer waiver. Short names keep the interpolated
# span below the 12-character threshold the quoted-credential pattern uses.
PAT = "ghp" + "_" + "abcdefghijklmnopqrstuvwxyz0123"
PW = "correct " + "horse " + "battery " + "staple"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
SK = "sk-" + "0123456789abcdef0123456789abcdef"
VALUE = "abcdefghij" + "klmnopqrst"
BEARER = "Authorization: Bearer " + VALUE


# --- diff parsing ------------------------------------------------------------


def test_added_line_numbers_tracks_context_offsets() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/foo.rs b/foo.rs",
            "index abc..def 100644",
            "--- a/foo.rs",
            "+++ b/foo.rs",
            "@@ -1,3 +1,4 @@",
            " unchanged",
            "+added",
            " context",
        ]
    )
    lines = mod.added_line_numbers(mod.added_lines_with_text(diff))
    assert lines["foo.rs"] == {2}


def test_added_lines_with_text_matches_line_numbers() -> None:
    mod = load_module()
    diff = make_diff("crates/tensor4all-core/src/lib.rs", ["let a = 1;", "let b = 2;"])
    entries = mod.added_lines_with_text(diff)
    assert entries["crates/tensor4all-core/src/lib.rs"] == [(2, "let a = 1;"), (3, "let b = 2;")]
    numbers = mod.added_line_numbers(entries)
    assert numbers["crates/tensor4all-core/src/lib.rs"] == {2, 3}


def test_added_lines_with_text_ignores_deleted_file() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/gone.rs b/gone.rs",
            "--- a/gone.rs",
            "+++ /dev/null",
            "@@ -1,1 +0,0 @@",
            "-removed",
        ]
    )
    assert mod.added_lines_with_text(diff) == {}


# --- model defaults ----------------------------------------------------------


def test_default_deepseek_model_uses_current_v4_name() -> None:
    mod = load_module()
    assert mod.DEFAULT_MODEL == "deepseek-v4-pro"
    assert mod.DEFAULT_API_URL == "https://api.deepseek.com/chat/completions"


# --- finding filtering -------------------------------------------------------


def test_filter_findings_drops_unchanged_files() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="other.rs",
        line=1,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {1}}) == []


def test_filter_findings_keeps_added_line() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=2,
        summary="test",
        detail="detail",
    )
    assert len(mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {2}})) == 1


def test_filter_findings_drops_line_finding_without_added_lines() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="deleted.rs",
        line=4,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([finding], ["deleted.rs"], {}) == []


def test_filter_findings_drops_file_level_block_finding() -> None:
    mod = load_module()
    block = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    warn = mod.Finding(
        id="w",
        severity="warn",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([block, warn], ["foo.rs"], {"foo.rs": {1}}) == [warn]


def test_filter_findings_drops_global_llm_finding_when_disallowed() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="",
        line=None,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings(
        [finding], ["foo.rs"], {"foo.rs": {1}}, allow_global=False
    )
    assert kept == []


def test_reconcile_verdict_only_blocks_fail() -> None:
    mod = load_module()
    warn = mod.Finding("w", "warn", "s", "f", 1, "s", "d")
    block = mod.Finding("b", "block", "s", "f", 1, "s", "d")
    assert mod.reconcile_verdict([warn]) == "pass"
    assert mod.reconcile_verdict([warn, block]) == "fail"


def test_merge_findings_prefers_block_over_warn() -> None:
    mod = load_module()
    warn = mod.Finding("dup", "warn", "s", "f.rs", 3, "same", "d")
    block = mod.Finding("dup", "block", "s", "f.rs", 3, "same", "d")
    merged = mod.merge_findings([warn, block])
    assert len(merged) == 1
    assert merged[0].severity == "block"


# --- rule section routing ----------------------------------------------------


def test_select_rule_sections_always_includes_surface_and_boundary() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["Cargo.toml"])
    assert "Public Surface Discipline" in sections
    assert "Public Boundary Safety Audits" in sections
    assert "Work Logs And Design Records" in sections
    assert "No Ad Hoc Fixes" in sections


def test_select_rule_sections_routes_capi() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tensor4all-capi/src/tensor.rs"])
    assert "C API And Language-Binding ABI" in sections
    assert "Language Bindings" in sections
    assert "Index Identity Semantics" in sections


def test_select_rule_sections_routes_tensorbackend_to_dense_layout() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(
        ["crates/tensor4all-tensorbackend/src/matrix.rs"]
    )
    assert "Dense Layout And Linear Algebra" in sections
    assert "Unsafe Code Boundary" in sections


def test_select_rule_sections_routes_network_stack_to_materialization() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tensor4all-treetn/src/treetn/ops.rs"])
    assert "No Hidden Dense Materialization In Production Paths" in sections
    assert "Tensor Network Test Comparisons" in sections


def test_select_rule_sections_routes_tci_stack() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(
        ["crates/tensor4all-quanticstci/src/quantics_tci.rs"]
    )
    assert "No Hidden Dense Materialization In Production Paths" in sections
    assert "Cache Ownership" in sections


def test_select_rule_sections_routes_graph_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(
        ["crates/tensor4all-treetn/src/treetn/named_graph.rs"]
    )
    assert "Graph Algorithms" in sections


def test_select_rule_sections_routes_tutorials() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["docs/book/src/tutorials/intro.md"])
    assert "Online Tutorial Synchronization" in sections
    assert "Documentation Examples" in sections


def test_select_rule_sections_routes_cargo_manifest() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tensor4all-core/Cargo.toml"])
    assert "Dependencies And Numeric Conventions" in sections


def test_select_rule_sections_excludes_human_only_sections() -> None:
    mod = load_module()
    sections = set(mod.select_rule_sections(["crates/tensor4all-core/src/lib.rs"]))
    assert sections.isdisjoint(mod.HUMAN_ONLY_SECTIONS)
    assert "Base Branch Synchronization" in mod.HUMAN_ONLY_SECTIONS


def test_every_rule_section_is_reachable() -> None:
    """A new REPOSITORY_RULES section must be routed, always-on, or human-only.

    Without this, adding a section silently makes it invisible to the reviewer.
    """
    mod = load_module()
    documented = set(mod.parse_repository_rules_sections())
    routed = set(mod.ALWAYS_SECTIONS) | set(mod.HUMAN_ONLY_SECTIONS)
    for _pattern, names in mod.SECTION_TRIGGERS:
        routed |= set(names)
    assert documented <= routed, f"unrouted rule sections: {sorted(documented - routed)}"
    assert routed <= documented, f"routing names a missing section: {sorted(routed - documented)}"


def test_build_rules_payload_returns_requested_section_bodies() -> None:
    mod = load_module()
    payload = mod.build_rules_payload(["Graph Algorithms"])
    assert payload.startswith("## Graph Algorithms")
    assert "Public Surface Discipline" not in payload


# --- tenferro route boundary --------------------------------------------------


CARGO_WITH_DEP = """[package]
name = "tensor4all-core"

[features]
tenferro-cpu-faer = ["dep:tenferro-cpu"]

[dependencies]
tenferro-linalg.workspace = true
serde = "1"

[dev-dependencies]
tenferro-tensor.workspace = true
"""


def check_cargo(mod, path, added):
    """Run deterministic_checks with the manifest text stubbed in."""
    original = mod.file_text_at
    mod.file_text_at = lambda p, *, ref, worktree: CARGO_WITH_DEP
    try:
        return mod.deterministic_checks([path], added=added, ref="head")
    finally:
        mod.file_text_at = original


def test_tenferro_dependency_lines_ignores_feature_names() -> None:
    mod = load_module()
    found = mod.tenferro_dependency_lines(CARGO_WITH_DEP)
    # Line 8 is the real [dependencies] entry; line 5 is a feature name and
    # line 12 is a dev-dependency, both out of scope.
    assert found == {8: "tenferro-linalg"}


def test_tenferro_route_blocks_new_dependency() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/Cargo.toml"
    findings = check_cargo(
        mod, path, {path: [(8, "tenferro-linalg.workspace = true")]}
    )
    assert len(findings) == 1
    assert findings[0].id == "tenferro-route"
    assert findings[0].severity == "block"
    assert f"{path}:8" in findings[0].detail


def test_tenferro_route_ignores_feature_line_edit() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/Cargo.toml"
    findings = check_cargo(
        mod, path, {path: [(5, 'tenferro-cpu-faer = ["dep:tenferro-cpu"]')]}
    )
    assert findings == []


def test_tenferro_route_allows_tensorbackend() -> None:
    mod = load_module()
    path = "crates/tensor4all-tensorbackend/Cargo.toml"
    findings = check_cargo(
        mod, path, {path: [(8, "tenferro-linalg.workspace = true")]}
    )
    assert findings == []


def test_tenferro_route_ignores_untouched_existing_dependency() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/Cargo.toml"
    # An unrelated line changes; the pre-existing tenferro dep on line 8 is not
    # in the added set, so the #566 backlog stays out of the way.
    findings = check_cargo(mod, path, {path: [(9, 'serde = "1"')]})
    assert findings == []


def test_crate_of_extracts_crate_name() -> None:
    mod = load_module()
    assert mod.crate_of("crates/tensor4all-core/Cargo.toml") == "tensor4all-core"
    assert mod.crate_of("Cargo.toml") is None
    assert mod.crate_of("internal/foo/Cargo.toml") is None


# --- prohibited doctests ------------------------------------------------------


def test_prohibited_doctest_blocks_no_run() -> None:
    mod = load_module()
    path = "crates/tensor4all-treetn/src/lib.rs"
    diff = make_diff(path, ["/// ```no_run", "/// let x = 1;", "/// ```"])
    findings = mod.deterministic_checks(
        [path], added=mod.added_lines_with_text(diff)
    )
    assert len(findings) == 1
    assert findings[0].id == "prohibited-doctest"
    assert "no_run" in findings[0].detail


def test_prohibited_doctest_accepts_plain_and_annotated_fences() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/src/lib.rs"
    diff = make_diff(
        path, ["/// ```", "/// ```rust", "/// ```text", "/// ```should_panic"]
    )
    findings = mod.deterministic_checks(
        [path], added=mod.added_lines_with_text(diff)
    )
    assert findings == []


def test_prohibited_doctest_covers_markdown_and_comma_lists() -> None:
    mod = load_module()
    path = "docs/book/src/guide.md"
    diff = make_diff(path, ["```rust,ignore"])
    findings = mod.deterministic_checks(
        [path], added=mod.added_lines_with_text(diff)
    )
    assert len(findings) == 1
    assert "ignore" in findings[0].detail


# --- model response handling -------------------------------------------------


def test_extract_json_payload_strips_fence() -> None:
    mod = load_module()
    payload = mod.extract_json_payload('```json\n{"verdict": "pass"}\n```')
    assert payload == {"verdict": "pass"}


def test_extract_json_payload_reports_malformed_embedded_object() -> None:
    mod = load_module()
    try:
        mod.extract_json_payload("noise {not json} tail")
    except ValueError as err:
        assert "not valid JSON" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_parse_findings_caps_model_output() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {
                "id": f"f{index}",
                "severity": "block",
                "rule_section": "Public Boundary Safety",
                "file": "crates/tensor4all-core/src/lib.rs",
                "line": index + 1,
                "summary": "s",
                "detail": "d",
            }
            for index in range(mod.MAX_FINDINGS_PER_CHUNK + 5)
        ],
    }
    verdict, findings = mod.parse_findings(raw)
    assert verdict == "fail"
    assert len(findings) == mod.MAX_FINDINGS_PER_CHUNK


def test_parse_findings_normalizes_common_severity_aliases() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {"id": "a", "severity": "CRITICAL", "file": "f.rs", "line": 1},
            {"id": "b", "severity": "info", "file": "f.rs", "line": 2},
            {"id": "c", "severity": "nonsense", "file": "f.rs", "line": 3},
        ],
    }
    _, findings = mod.parse_findings(raw)
    assert [item.severity for item in findings] == ["block", "warn", "warn"]


def test_parse_findings_rejects_non_integer_line() -> None:
    mod = load_module()
    raw = {"verdict": "pass", "findings": [{"id": "a", "line": "3"}]}
    try:
        mod.parse_findings(raw)
    except ValueError as err:
        assert "line must be an integer" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_parse_findings_rejects_unknown_verdict() -> None:
    mod = load_module()
    try:
        mod.parse_findings({"verdict": "maybe", "findings": []})
    except ValueError as err:
        assert "verdict" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_llm_response_error_finding_blocks_with_diagnostic() -> None:
    mod = load_module()
    finding = mod.llm_response_error_finding(ValueError("bad json"))
    assert finding.severity == "block"
    assert "ValueError: bad json" in finding.detail


# --- diff chunking -----------------------------------------------------------


def test_split_diff_chunks_respects_limit() -> None:
    mod = load_module()
    # Each file stays under the per-file limit, so only the aggregate limit splits.
    piece = "x" * (mod.MAX_FILE_DIFF_CHARS - 10)
    per_chunk = mod.MAX_DIFF_CHARS // len(piece)
    files = {f"f{index}.rs": piece for index in range(per_chunk + 1)}
    chunks = mod.split_diff_chunks(files)
    assert len(chunks) == 2
    assert all(len(chunk) <= mod.MAX_DIFF_CHARS for chunk in chunks)


def test_split_large_file_diff_preserves_file_header() -> None:
    mod = load_module()
    header = [
        "diff --git a/big.rs b/big.rs",
        "--- a/big.rs",
        "+++ b/big.rs",
    ]
    body = "\n".join(
        f"@@ -{index},1 +{index},1 @@\n+{'y' * 1000}"
        for index in range(1, 120)
    )
    chunks = mod.split_large_file_diff("\n".join([*header, body]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("diff --git a/big.rs b/big.rs")
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS


def test_split_large_file_diff_splits_single_overlong_line() -> None:
    mod = load_module()
    header = [
        "diff --git a/big.rs b/big.rs",
        "--- a/big.rs",
        "+++ b/big.rs",
    ]
    line = "+" + "z" * (mod.MAX_FILE_DIFF_CHARS * 2)
    chunks = mod.split_large_file_diff("\n".join([*header, "@@ -1,1 +1,1 @@", line]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("diff --git a/big.rs b/big.rs")
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS


def test_transport_errors_cover_every_below_json_failure() -> None:
    """socket.timeout only aliases TimeoutError from Python 3.10 on."""
    import http.client
    import socket
    import urllib.error

    mod = load_module()
    for exc_type in (
        socket.timeout,
        TimeoutError,
        ConnectionResetError,
        urllib.error.URLError,
        http.client.IncompleteRead,
    ):
        assert issubclass(exc_type, mod.TRANSPORT_ERRORS), exc_type


def test_call_deepseek_retries_transient_network_errors() -> None:
    import socket
    import urllib.request

    mod = load_module()
    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"verdict\\":\\"pass\\"}"}}]}'

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("The read operation timed out")
        return FakeResponse()

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = fake_urlopen
    mod.time.sleep = lambda _seconds: None
    try:
        payload = mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep

    assert calls["n"] == 2
    assert payload == {"verdict": "pass"}


def test_call_deepseek_reraises_after_retries_exhausted() -> None:
    import socket
    import urllib.request

    mod = load_module()

    def always_timeout(request, timeout=None):
        raise socket.timeout("nope")

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = always_timeout
    mod.time.sleep = lambda _seconds: None
    try:
        mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    except mod.TRANSPORT_ERRORS:
        pass
    else:
        raise AssertionError("expected the timeout to propagate")
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep


def test_contains_sensitive_text_flags_typed_declaration() -> None:
    """A type annotation used to hide the literal from the pre-upload guard."""
    mod = load_module()
    for line in (
        f'const API_KEY: &str = "{VALUE}";',
        f'let api_key: String = "{VALUE}".into();',
        f'client_secret : &\'static str = "{VALUE}"',
        f'PASSWORD: str = "{VALUE}"',
    ):
        assert mod.contains_sensitive_text(line), line


def test_redact_sensitive_text_masks_typed_declaration() -> None:
    mod = load_module()
    redacted = mod.redact_sensitive_text(f'const API_KEY: &str = "{VALUE}";')
    assert VALUE not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_typed_declaration_guard_keeps_env_lookups_quiet() -> None:
    mod = load_module()
    for line in (
        'let key = std::env::var("DEEPSEEK_API_KEY")?;',
        "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}",
        "api_key: Option<String>,",
    ):
        assert not mod.contains_sensitive_text(line), line


# --- hunk header renumbering --------------------------------------------------


def test_split_oversized_hunk_renumbers_each_chunk() -> None:
    mod = load_module()
    header = ["diff --git a/big.rs b/big.rs", "--- a/big.rs", "+++ b/big.rs"]
    body = [f"+line {index}" + "y" * 900 for index in range(120)]
    chunks = mod.split_oversized_hunk(header, ["@@ -1,0 +1,120 @@ fn ctx()", *body])
    assert len(chunks) > 1

    starts = []
    for chunk in chunks:
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS
        hunk_line = [line for line in chunk.splitlines() if line.startswith("@@")][0]
        parsed = mod.HUNK_HEADER.match(hunk_line)
        assert parsed is not None
        assert parsed.group(5) == " fn ctx()"
        starts.append((int(parsed.group(3)), int(parsed.group(4))))

    # Every chunk starts where the previous one ended, and the counts sum to
    # the original 120 added lines.
    assert starts[0][0] == 1
    for (start, count), (next_start, _) in zip(starts, starts[1:]):
        assert start + count == next_start
    assert sum(count for _, count in starts) == 120


def test_split_oversized_hunk_counts_context_and_removals() -> None:
    mod = load_module()
    header = ["diff --git a/a.rs b/a.rs", "--- a/a.rs", "+++ b/a.rs"]
    hunk = ["@@ -10,3 +20,3 @@", " ctx", "-gone", "+added"]
    chunks = mod.split_oversized_hunk(header, hunk)
    assert len(chunks) == 1
    hunk_line = [line for line in chunks[0].splitlines() if line.startswith("@@")][0]
    parsed = mod.HUNK_HEADER.match(hunk_line)
    # context + removal advance old; context + addition advance new.
    assert (int(parsed.group(1)), int(parsed.group(2))) == (10, 2)
    assert (int(parsed.group(3)), int(parsed.group(4))) == (20, 2)


def test_split_oversized_hunk_falls_back_on_unparseable_header() -> None:
    mod = load_module()
    header = ["diff --git a/a.rs b/a.rs", "--- a/a.rs", "+++ b/a.rs"]
    chunks = mod.split_oversized_hunk(header, ["@@ garbage @@", "+one"])
    assert len(chunks) == 1
    assert "@@ garbage @@" in chunks[0]


def test_line_deltas_classifies_diff_lines() -> None:
    mod = load_module()
    assert mod.line_deltas("+added") == (0, 1)
    assert mod.line_deltas("-removed") == (1, 0)
    assert mod.line_deltas(" context") == (1, 1)
    assert mod.line_deltas("\\ No newline at end of file") == (0, 0)


# --- API key validation -------------------------------------------------------


def test_api_key_problem_detects_non_ascii_without_echoing_it() -> None:
    mod = load_module()
    key = SK[:29] + "\u2026" + "tail"
    problem = mod.api_key_problem(key)
    assert problem is not None
    assert "non-ASCII" in problem
    assert "29" in problem
    assert key not in problem


def test_api_key_problem_detects_empty_and_whitespace() -> None:
    mod = load_module()
    assert "empty" in mod.api_key_problem("")
    assert "whitespace" in mod.api_key_problem("sk-abc def")


def test_api_key_problem_accepts_a_normal_key() -> None:
    mod = load_module()
    assert mod.api_key_problem(SK) is None


def test_api_key_error_finding_blocks_and_names_the_secret() -> None:
    mod = load_module()
    finding = mod.api_key_error_finding("The secret is empty.")
    assert finding.severity == "block"
    assert finding.id == "llm-api-key-invalid"
    assert "DEEPSEEK_API_KEY" in finding.summary


def test_contains_sensitive_text_flags_passphrase_with_spaces() -> None:
    """A quoted credential may legitimately contain spaces."""
    mod = load_module()
    assert mod.contains_sensitive_text(f'password = "{PW}"')


def test_redact_sensitive_text_masks_whole_quoted_value() -> None:
    mod = load_module()
    redacted = mod.redact_sensitive_text(f'password = "{PW}"')
    assert "horse" not in redacted
    assert "battery" not in redacted
    assert redacted == "password = [REDACTED_SECRET]"


def test_contains_sensitive_text_flags_unterminated_quote() -> None:
    """A value that closes on a later line hides from single-line patterns."""
    mod = load_module()
    assert mod.contains_sensitive_text('secret = "opens here')
    assert mod.contains_sensitive_text("token: &str = 'starts")


def test_prohibited_doctest_recognizes_tilde_fences() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/src/lib.rs"
    diff = make_diff(path, ["/// ~~~rust,no_run", "/// let x = 1;", "/// ~~~"])
    findings = mod.deterministic_checks(
        [path], added=mod.added_lines_with_text(diff)
    )
    assert len(findings) == 1
    assert "no_run" in findings[0].detail


# --- Cargo dependency forms ---------------------------------------------------


CARGO_ALIASES = """[features]
tenferro-cpu-faer = ["dep:tenferro-cpu"]

[dependencies]
linalg = { package = "tenferro-linalg", workspace = true }
"tenferro-tensor" = { workspace = true }

[dependencies.tenferro-einsum]
workspace = true

[target.'cfg(unix)'.dependencies]
renamed = { package = "tenferro-ad" }

[dev-dependencies]
tenferro-cpu.workspace = true
"""


def test_tenferro_dependency_lines_sees_every_declaration_form() -> None:
    """Cargo renaming, quoted keys, and subtables all declare a dependency."""
    mod = load_module()
    found = mod.tenferro_dependency_lines(CARGO_ALIASES)
    assert found == {
        5: "tenferro-linalg",   # package = rename
        6: "tenferro-tensor",   # quoted key
        8: "tenferro-einsum",   # [dependencies.<name>] subtable
        12: "tenferro-ad",      # rename under a target table
    }
    # The [features] entry and the dev-dependency stay out of scope.
    assert 2 not in found
    assert 15 not in found


def test_dependency_table_of_classifies_headers() -> None:
    mod = load_module()
    assert mod.dependency_table_of("dependencies") == ("dependencies", None)
    assert mod.dependency_table_of("dependencies.tenferro-linalg") == (
        "dependencies",
        "tenferro-linalg",
    )
    assert mod.dependency_table_of("target.'cfg(unix)'.dependencies") == (
        "dependencies",
        None,
    )
    assert mod.dependency_table_of("features") is None
    assert mod.dependency_table_of("package") is None


def test_tenferro_route_blocks_a_renamed_dependency() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/Cargo.toml"
    original = mod.file_text_at
    mod.file_text_at = lambda p, *, ref, worktree: CARGO_ALIASES
    try:
        findings = mod.deterministic_checks(
            [path],
            added={path: [(5, 'linalg = { package = "tenferro-linalg" }')]},
            ref="head",
        )
    finally:
        mod.file_text_at = original
    assert len(findings) == 1
    assert "tenferro-linalg" in findings[0].detail


# --- content-based routing ----------------------------------------------------


def test_select_rule_sections_routes_on_changed_content() -> None:
    """A generic path gaining an unsafe block must still supply the rule."""
    mod = load_module()
    path = "crates/tensor4all-core/src/lib.rs"
    assert "Unsafe Code Boundary" not in mod.select_rule_sections([path])
    added = {path: [(10, "    unsafe { ptr.read() }")]}
    assert "Unsafe Code Boundary" in mod.select_rule_sections([path], added)


def test_content_triggers_cover_the_generic_path_cases() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/src/lib.rs"
    cases = [
        ("pub fn f() -> anyhow::Result<()> { Ok(()) }", "Error Handling"),
        ("let d = tt.to_dense()?;",
         "No Hidden Dense Materialization In Production Paths"),
        ("use tenferro_linalg::svd;", "Dense Layout And Linear Algebra"),
        ("if a.id() == b.id() {", "Index Identity Semantics"),
        ("static CACHE: OnceLock<u8> = OnceLock::new();", "Cache Ownership"),
    ]
    for line, section in cases:
        sections = mod.select_rule_sections([path], {path: [(1, line)]})
        assert section in sections, line


def test_content_triggers_never_select_human_only_sections() -> None:
    mod = load_module()
    path = "crates/tensor4all-core/src/lib.rs"
    added = {path: [(1, "unsafe { }"), (2, "anyhow::Result<()>")]}
    sections = set(mod.select_rule_sections([path], added))
    assert sections.isdisjoint(mod.HUMAN_ONLY_SECTIONS)


def test_content_triggers_name_only_documented_sections() -> None:
    mod = load_module()
    documented = set(mod.parse_repository_rules_sections())
    for _pattern, names in mod.CONTENT_TRIGGERS:
        assert set(names) <= documented, names


def test_budget_is_smaller_than_the_workflow_timeout() -> None:
    """The script must finish before the job is killed, or no report is posted."""
    mod = load_module()
    workflow = (mod.ROOT / ".github" / "workflows" / "review_bot.yml").read_text()
    minutes = [
        int(line.split(":")[1].strip())
        for line in workflow.splitlines()
        if line.strip().startswith("timeout-minutes:")
    ]
    assert minutes, "review_bot.yml lost its job timeout"
    assert mod.DEFAULT_BUDGET_SECONDS < min(minutes) * 60


def test_call_deepseek_does_not_retry_past_the_deadline() -> None:
    """Retrying into the job deadline loses the report to a kill signal."""
    import socket
    import urllib.request

    mod = load_module()
    calls = {"n": 0}

    def always_timeout(request, timeout=None):
        calls["n"] += 1
        raise socket.timeout("nope")

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = always_timeout
    mod.time.sleep = lambda _seconds: None
    try:
        mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
            deadline=mod.time.monotonic(),  # already expired
        )
    except mod.TRANSPORT_ERRORS:
        pass
    else:
        raise AssertionError("expected the timeout to propagate")
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep

    # One attempt, not two: the retry would have run past the deadline.
    assert calls["n"] == 1


def test_budget_exhausted_finding_warns_without_blocking() -> None:
    mod = load_module()
    finding = mod.budget_exhausted_finding(2, 5)
    # A partial review must not fail the gate; the deterministic checks still
    # covered the whole diff.
    assert finding.severity == "warn"
    assert "2 of 5" in finding.detail


# --- secret handling ---------------------------------------------------------


def test_redact_sensitive_text_masks_common_secret_forms() -> None:
    mod = load_module()
    text = "\n".join([PAT, AWS, "api_key = supersecretvalue", BEARER])
    redacted = mod.redact_sensitive_text(text)
    assert PAT not in redacted
    assert AWS not in redacted
    assert "supersecretvalue" not in redacted
    assert redacted.count("[REDACTED_SECRET]") >= 4


def test_contains_sensitive_text_ignores_env_lookup_code() -> None:
    mod = load_module()
    assert not mod.contains_sensitive_text(
        'let key = std::env::var("DEEPSEEK_API_KEY")?;'
    )
    assert not mod.contains_sensitive_text("DEEPSEEK_API_KEY: ${{ secrets.KEY }}")


def test_contains_sensitive_text_flags_quoted_credential() -> None:
    mod = load_module()
    assert mod.contains_sensitive_text(f'let api_key = "{VALUE}";')


def test_sensitive_diff_finding_checks_added_lines_only() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,2 +1,2 @@",
            f" let token = {PAT};",
            "+let clean = 1;",
        ]
    )
    assert mod.sensitive_diff_finding(diff) is None


def test_sensitive_diff_finding_reports_added_match_location() -> None:
    mod = load_module()
    diff = make_diff(
        "a.rs", ["let clean = 1;", f"let t = {PAT};"]
    )
    finding = mod.sensitive_diff_finding(diff)
    assert finding is not None
    assert finding.severity == "block"
    assert (finding.file, finding.line) == ("a.rs", 3)


# --- reporting ---------------------------------------------------------------


def test_summarize_llm_review_computes_dropped_count() -> None:
    mod = load_module()
    summary = mod.summarize_llm_review(
        chunk_sizes=[100, 200], elapsed_seconds=1.25, returned_count=5, kept_count=2
    )
    assert "2 chunk(s)" in summary
    assert "3 dropped" in summary


def test_format_report_includes_llm_summary_line() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base",
        head="head",
        verdict="pass",
        findings=[],
        waived=False,
        llm_summary="LLM review: 1 chunk(s)",
    )
    assert "LLM review: 1 chunk(s)" in report
    assert "No findings." in report


def test_format_report_omits_llm_summary_when_absent() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base", head="head", verdict="pass", findings=[], waived=False
    )
    assert "LLM review:" not in report


def test_format_report_lists_findings_with_location() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Boundary Safety",
        file="crates/tensor4all-core/src/lib.rs",
        line=42,
        summary="missing validation",
        detail="detail line",
    )
    report = mod.format_report(
        base="base", head="head", verdict="fail", findings=[finding], waived=False
    )
    assert "[block] x (Public Boundary Safety) crates/tensor4all-core/src/lib.rs:42" in report
    assert "detail line" in report


# --- prompt and rules files --------------------------------------------------


def test_prompt_file_exists_and_requires_json_only() -> None:
    mod = load_module()
    assert mod.PROMPT_PATH.is_file()
    text = mod.PROMPT_PATH.read_text(encoding="utf-8")
    assert "JSON only" in text
    assert "untrusted data" in text


def test_rules_file_states_the_checked_rules() -> None:
    mod = load_module()
    sections = mod.parse_repository_rules_sections()
    assert mod.TENFERRO_ROUTE_CRATE in sections["Dense Layout And Linear Algebra"]
    docs = sections["Documentation Examples"]
    assert "no_run" in docs and "ignore" in docs


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"repository-rules-review: {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
