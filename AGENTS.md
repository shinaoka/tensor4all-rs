# Agent Guidelines for tensor4all-rs

Before acting, read the latest shared tensor4all agent rules from the
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
repository. Start from:

- `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`

If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout:

- `../tensor4all-agent-rules/rules/index.md`

Load only the common, Rust, performance, numerical, docs, or benchmark rule
files relevant to the task. If neither online access nor the sibling checkout
is available, continue from this repository's local rules and state the shared
rules were unavailable when creating a PR.

Then read `README.md` and `REPOSITORY_RULES.md` before starting work.

## Development Stage

**Early development** - no backward compatibility required. Remove deprecated code immediately.

## General Guidelines

- Use same language as past conversations (Japanese if previous was Japanese)
- Source code and docs in English
- Each crate in `crates/` is independent with own `Cargo.toml`, `src/`, `tests/`
- **Bug fixing**: When a bug is discovered, always check related files for similar bugs and propose to the user to inspect them

### API Reference (Check First)

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Read `docs/api/*.md` before source files. Only read source when API doc is insufficient.

## Context-Efficient Exploration

- Use Task tool with `subagent_type=Explore` for open-ended exploration
- Use Grep for structure: `pub fn`, `impl.*for`, `^pub (struct|enum|type)`
- Read specific lines with `offset`/`limit` parameters
- Prefer API docs over full source files

## Code Style

`cargo fmt` for formatting, `cargo clippy` for linting. Avoid `unwrap()`/`expect()` in library code.

**Always run `cargo fmt --all` before committing changes.**

## Documentation Requirements

### Rustdoc Standards

Every public type, trait, and function **must** have doc comments with the following:

**Types (struct/enum/trait):**
- Summary: what it represents, when to use it (1-2 sentences)
- Related types: relationship to similar types (e.g., "`TensorTrain` is the simple chain version; `TreeTN` is the general tree version")
- `# Examples` section with runnable code and assertions

**Functions/methods:**
- Summary: what it does (1 sentence)
- Arguments: meaning, constraints, typical values for each parameter (especially for `Options` types)
- Returns: what is returned, how to use it
- `# Panics` or `# Errors`: under what conditions it fails
- `# Examples` section with runnable code and assertions

**Options/Config types (critical for usability):**
- Each field: meaning, recommended values, and default behavior
- Field relationships and trade-offs (e.g., `rtol` vs `max_bond_dim`)
- "When in doubt" defaults

### Code Example Rules

- All doc examples **must** be runnable (`ignore` and `no_run` attributes are **prohibited**)
- All doc examples **must** include assertions verifying correctness (not just compilation/execution)
  - Use `assert!`, `assert_eq!`, `approx::assert_abs_diff_eq!`, etc.
  - Weak smoke checks such as non-zero, non-empty, finite, shape-only, or
    "rank is positive" are not enough unless that exact property is the
    documented behavior. Prefer checking numerical values against known
    answers, algebraic identities, reconstruction error, or structural
    invariants that would catch an incorrect implementation.
- mdBook guide code blocks follow the same rules: runnable with assertions
- mdBook code blocks use hidden lines (`# ` prefix) for `use` statements and `fn main()` wrappers

### CI Verification

- `cargo test --doc --release --workspace` must pass (rustdoc examples)
- `./scripts/test-mdbook.sh` must pass (mdBook guide examples; raw `mdbook test docs/book` does not receive the resolved `--extern` flags that the guide snippets need)

### Public Surface Drift

- `README.md`, rustdoc, examples, and `skills/use-tensor4all-rs/` must not
  claim more than the current public surface actually provides.
- When changing public APIs, documented capabilities, or user-facing examples, check for stale names, stale capability claims, and references to removed paths or workflows.
- Keep documentation slightly behind reality if validation is incomplete; do not advertise partially landed surfaces as stable or fully supported.

### Online Tutorial Synchronization

- The live online tutorials are in `docs/book/src/tutorials/`.
- Runnable tutorial demos live in `docs/tutorial-code/src/bin/` with shared helpers in `docs/tutorial-code/src/`.
- When changing public APIs, tutorial code, generated tutorial CSV/PNG artifacts, or examples quoted by the online tutorials, update the live mdBook tutorial page in the same branch.
- Treat `docs/tutorial-code/docs/tutorials/` as legacy/reference material unless this policy is changed explicitly.

## Error Handling

- `anyhow` for internal error handling and context
- `thiserror` for public API error types

### C API Error Handling

The C API (`tensor4all-capi`) preserves error details through
`t4a_last_error_message`. See `docs/CAPI_DESIGN.md` for the detailed ABI rules.

- Fallible C API functions return `enum t4a_status_code`, not a bare `int`
  or generic `StatusCode` typedef.
- Use `run_catching`, `unwrap_catch`, `capi_error`, and `err_status` so
  `Err(e)` values and panic payloads reach `t4a_last_error_message`.
- **Do not add new `catch_unwind` / `Err(_) => T4A_INTERNAL_ERROR` patterns**
  without preserving error messages.
- Ordinary release builds omit Rust debug information. When debugging through
  Tensor4all.jl or the C API requires source-level Rust backtraces, build with
  `cargo build --profile release-debug -p tensor4all-capi` and set
  `RUST_BACKTRACE=1` when running the caller.

## Testing

```bash
cargo nextest run --release --workspace          # Full suite
cargo nextest run --release --test test_name     # Specific test
cargo nextest run --release -p crate_name        # Single crate
```

**Always use `--release` mode for tests** to avoid excessive execution time in debug builds.

- Private functions: `#[cfg(test)]` module in source file
- Integration tests: `tests/` directory
- Dense whole-result comparisons: avoid per-element re-contraction loops in tests. Materialize once to a dense tensor/matrix, then compare via tensor subtraction and `maxabs()`. `TensorDynLen` subtraction aligns indices by index semantics, so explicit axis reordering is usually unnecessary.
- **Test tolerance changes**: When relaxing test tolerances (unit tests, codecov targets, etc.), always seek explicit user approval before making changes.
- **Dense whole-result comparisons**: When comparing full tensors/operators in tests, do not recompute contractions element-by-element. Materialize once to a dense tensor/matrix, then compare via tensor subtraction and `maxabs()`. `TensorDynLen` subtraction aligns indices by index semantics, so explicit axis reordering is usually unnecessary.

### Coverage and Path Exercise

- Each distinct control-flow path (error branches, layout variants, boundary conditions) needs a test. Happy-path only is insufficient.
- When removing code, check whether the removed tests were the sole exerciser of any shared helper; add replacement coverage if so.
- Before pushing a deletion PR, run the CI coverage check locally:

```bash
cargo llvm-cov --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Fix drops by adding tests, not by lowering thresholds (threshold changes need explicit approval).

## API Design

Only make functions `pub` when truly public API.

### Layering and Maintainability

**Respect crate boundaries and abstraction layers.**

- **Never access low-level APIs or internal data structures from downstream crates.** Use high-level public methods instead of directly manipulating internal representations.
- **Use high-level APIs.** If downstream code needs low-level access, create appropriate high-level APIs rather than exposing internal details.
- **Prefer type-generic Rust APIs.** If a public Rust library API can be expressed generically, do not introduce scalar-specific entry points such as `*_f64` / `*_c64`. Add or use a generic API instead, and prefer the generic form in library code, tests, examples, and docs. This rule does not apply to the C API / FFI boundary, where scalar-specific names may be appropriate.
- **Examples:**
  - Instead of `match scalar { AnyScalar::F64(x) => ... }`, use `scalar.real()`, `scalar.is_complex()`, `scalar.is_zero()`
  - Instead of `AnyScalar::F64(1.0)`, use `AnyScalar::new_real(1.0)`
  - Instead of `AnyScalar::C64(z)`, use `AnyScalar::new_complex(re, im)`

**This applies to both library code and test code.** Tests should also use public APIs to maintain consistency and reduce maintenance burden when internal representations change.

**No ad hoc fixes:**

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated implementations, or downstream reach-through into lower-level internals when a higher-level seam should exist instead.
- If a needed behavior does not fit the current abstraction cleanly, add or refine the appropriate seam instead of patching around it locally.

### Code Deduplication

- **Avoid duplicate test code.** Use macros, functions, or generic functions to share test logic.
- **Example pattern for testing f64/Complex64:**

```rust
fn test_op_generic<T: Scalar + From<f64>>() { /* test */ }

#[test]
fn test_op_f64() { test_op_generic::<f64>(); }
#[test]
fn test_op_c64() { test_op_generic::<Complex64>(); }
```

### Dense Layout And Linear Algebra

- Dense flat-buffer APIs use **column-major** linearization. New public
  constructors, exports, examples, FFI contracts, and docs must state or
  preserve column-major semantics.
- Do not add row-major compatibility shims or hidden row-major round-trips in
  library code. If external data is naturally row-shaped, convert privately at
  that explicit boundary.
- Reuse `tensor4all-tensorbackend::Matrix` for dense matrix values crossing crate
  boundaries. Do not introduce duplicate public matrix containers in downstream
  crates.
- Use tenferro-backed `tensor4all-tensorbackend` / existing tensor4all
  abstractions for SVD, QR, einsum, and dense tensor linear algebra instead of
  local reimplementations.
- Do not instantiate `CpuBackend` directly outside `tensor4all-tensorbackend`.
  Use tensorbackend wrappers or `with_default_backend` so CPU execution shares
  the repository's process-global tenferro `CpuContext`.
- `Tensor3Ops::slice_site`, `Tensor4Ops::slice_site`, and dense/full-tensor
  exports return column-major flat data.

## C API & Language Bindings

See `docs/CAPI_DESIGN.md` for C API patterns. Bindings: [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl) (separate repo), `python/tensor4all/`.

Truncation tolerance: support both `cutoff` (ITensors) and `rtol` (tensor4all-rs). Conversion: `rtol = √cutoff`.

### Cross-repo development with Tensor4all.jl

When a Tensor4all.jl feature requires new C API functions:

1. Develop both sides locally. Tensor4all.jl's `deps/build.jl` can use a local
   Rust build via `TENSOR4ALL_RS_PATH`.
2. Test both sides locally until all tests pass.
3. Create and merge the tensor4all-rs PR **first**.
4. Then update the pin hash in Tensor4all.jl `deps/build.jl` to the merged
   commit on remote and create the Tensor4all.jl PR.

Issue templates are in `.github/ISSUE_TEMPLATE/`. Feature requests from
Tensor4all.jl should link the related Julia-side issue.

## Dependencies

- Prefer existing tensor4all core/tcicore/simplett abstractions and
  tenferro-backed `tensor4all-tensorbackend` operations for arrays and linear
  algebra. Add direct array/linalg dependencies only when the established
  abstraction is genuinely insufficient.
- SVD singular values: `s[[0, i]]` not `s[[i, i]]`

## Git Workflow

**Never push/create PR without user approval.**

### Base Branch Synchronization

- At the start of work, run `git fetch origin` and branch from `origin/main`
  rather than a stale local `main`.
- Before treating PR checks as final, fetch `origin` and verify that the PR
  branch contains the current `origin/main`.
- If a PR is behind `main`, update the PR branch from `origin/main` before
  relying on checks, enabling auto-merge, or declaring it ready to merge.
- After updating from `origin/main`, re-monitor CI; earlier green checks do not
  prove the synchronized branch is green.

### Pre-PR Checks (matches CI)

Run before pushing. CI runs the same checks, but local = faster feedback:

```bash
cargo fmt --all                        # Auto-fix formatting
cargo fmt --all -- --check             # Dry-run (matches CI)
cargo clippy --workspace --all-targets -- -D warnings   # Matches CI
cargo test -p <changed-crate>          # Quick check first
cargo nextest run --release --workspace # Full test suite
cargo doc --workspace --no-deps        # Build rustdoc
```

### Repository Rules Review Bot

`.github/workflows/review_bot.yml` reviews every PR diff against
`REPOSITORY_RULES.md`. It runs from the trusted base revision and treats PR
contents as data: the PR head is fetched for `git diff` only, never checked out
or executed. Findings post as a single updating PR comment; only
`block`-severity findings fail the check.

Preview the review locally before pushing:

```bash
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
python3 scripts/test-repository-rules-review.py   # the script's own tests
```

Drop `--dry-run` to include the LLM pass; it needs `DEEPSEEK_API_KEY` in the
environment or in a repo-root `.env` (`pip install -r scripts/requirements-dev.txt`).

The system prompt lives in `ai/prompts/repository-rules-review.md`. Three
deterministic checks run before the LLM and independently of it:

| Check | Rule | Baseline |
|-------|------|----------|
| Secret-shaped text in added lines | — | Blocks the upload before anything reaches the API |
| New direct `tenferro-*` dependency outside `tensor4all-tensorbackend` | Dense Layout And Linear Algebra | #566 Phase 3 backlog: core, tcicore, simplett, treetci |
| Added `ignore` / `no_run` doctest fence | Documentation Examples | #566 Phase 1 backlog: 2 `no_run` sites |

Both rule checks are delta-scoped: they reject new violations without failing on
the recorded backlog. The tenferro check parses Cargo.toml table context, so
`tenferro-*` **feature** names are not mistaken for dependencies, and
`dev-dependencies` are out of scope.

Maintainer escape hatches, both requiring the `maintain`/`admin` role and
reapplication after the latest push:

| Label | Effect |
|-------|--------|
| `rules-review:no-llm` | Skips the LLM pass; deterministic checks still run |
| `rules-review:waive` | Waives the review entirely |

When adding a `## ` section to `REPOSITORY_RULES.md`, also route it in
`SECTION_TRIGGERS` (or `ALWAYS_SECTIONS` / `HUMAN_ONLY_SECTIONS`); an unrouted
section is never shown to the reviewer, and
`test_every_rule_section_is_reachable` fails.

| Change Type | Workflow |
|-------------|----------|
| Minor fixes | Branch + PR with auto-merge |
| Large features | Worktree + PR with auto-merge |

```bash
# Minor: branch workflow
git checkout -b fix-name && git add -A && git commit -m "msg"
cargo fmt --all && cargo clippy --workspace  # Lint before push
git push -u origin fix-name
gh pr create --base main --title "Title" --body "Desc"
gh pr merge --auto --squash --delete-branch

# Large: worktree workflow
git worktree add ../tensor4all-rs-feature -b feature

# Check PR before update
gh pr view <NUM> --json state  # Never push to merged PR

# Monitor CI
gh pr checks <NUM>
gh run view <RUN_ID> --log-failed
```

**Before creating PR**: Verify README.md is accurate (project structure, examples).
