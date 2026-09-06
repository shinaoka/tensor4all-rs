# Agent Guidelines for tensor4all-rs

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(offline fallback: `../tensor4all-agent-rules/rules/index.md`). Load only the
common, Rust, performance, numerical, docs, or benchmark rule files relevant to
the task. If neither is available, continue from the rules in this file and
state so when creating a PR.

Then read `README.md` and `REPOSITORY_RULES.md`. Read `PERFORMANCE_TIPS.md`
(performance audit catalog) during performance-sensitive implementation (TT
evaluation, interpolation, caches, contraction hot paths) and when reviewing
any PR touching those areas; `skills/audit-performance/` launches that audit.

## Development Stage

**Early development**: no backward compatibility. Remove deprecated code
immediately.

## General Guidelines

- Reply in the language of the conversation (Japanese if it was Japanese).
  Source code and docs are English.
- Each crate in `crates/` is independent with its own `Cargo.toml`, `src/`,
  `tests/`.
- **Bug fixing**: when a bug is found, check related files for similar bugs and
  propose inspecting them to the user.

### API Reference (Check First)

```bash
cargo run -p xtask --release -- api-dump
```

Generates a temporary inventory under `target/api-dump/` (ignored; never commit
or hand-edit) and verifies every public crate under `crates/` appears exactly
once. Read the relevant `target/api-dump/*.md` before source; read source only
when the inventory is insufficient.

## Context-Efficient Exploration

- Task tool with `subagent_type=Explore` for open-ended exploration.
- Grep for structure: `pub fn`, `impl.*for`, `^pub (struct|enum|type)`.
- Read specific lines with `offset`/`limit`; prefer API docs over full source.

## Code Style

`cargo fmt` and `cargo clippy`. Avoid `unwrap()`/`expect()` in library code.
**Always run `cargo fmt --all` before committing.**

## Documentation Requirements

### Rustdoc Standards

Every public type, trait, and function **must** have doc comments:

- **Types**: 1-2 sentence summary (what, when to use); relationship to similar
  types (e.g. "`TensorTrain` is the simple chain version; `TreeTN` is the
  general tree version"); `# Examples` with runnable asserted code.
- **Functions/methods**: 1 sentence summary; arguments (meaning, constraints,
  typical values, especially for `Options` types); returns (what is returned,
  how to use it); `# Panics`/`# Errors` (under what conditions it fails);
  `# Examples` with runnable asserted code.
- **Options/Config types**: each field's meaning, recommended values, default
  behavior; field relationships and trade-offs (e.g. `rtol` vs
  `max_bond_dim`); "when in doubt" defaults.

### Code Example Rules

- All doc examples **must** be runnable (`ignore` and `no_run` are
  **prohibited**) and **must** assert correctness (`assert!`, `assert_eq!`,
  `approx::assert_abs_diff_eq!`, ...). Non-zero, non-empty, finite,
  shape-only, or positive-rank checks are insufficient unless that property is
  the documented behavior; check known values, algebraic identities,
  reconstruction error, or structural invariants.
- mdBook guide code blocks follow the same rules, with hidden lines (`# `
  prefix) for `use` statements and `fn main()` wrappers.

### CI Verification

- `cargo test --doc --release --workspace` must pass.
- `./scripts/test-mdbook.sh` must pass (raw `mdbook test docs/book` lacks the
  resolved `--extern` flags the guide snippets need).

### Public Surface Drift

- `README.md`, rustdoc, examples, and `skills/use-tensor4all-rs/` must not
  claim more than the current public surface provides.
- When changing public APIs, documented capabilities, or user-facing examples,
  check for stale names, stale capability claims, and references to removed
  paths or workflows.
- Keep documentation slightly behind reality if validation is incomplete; do
  not advertise partially landed surfaces as stable.

### Online Tutorial Synchronization

- Live tutorials: `docs/book/src/tutorials/`. Runnable demos:
  `docs/tutorial-code/src/bin/`, shared helpers in `docs/tutorial-code/src/`.
- When changing public APIs, tutorial code, generated tutorial CSV/PNG
  artifacts, or examples quoted by the tutorials, update the live mdBook
  tutorial page in the same branch.
- `docs/tutorial-code/docs/tutorials/` is legacy/reference material unless this
  policy is changed explicitly.

## Error Handling

- `anyhow` for internal errors and context; `thiserror` for public API error
  types.

### C API Error Handling

`tensor4all-capi` preserves error details through `t4a_last_error_message`; ABI
rules in `docs/CAPI_DESIGN.md`.

- Fallible C API functions return `enum t4a_status_code`, not a bare `int` or
  generic `StatusCode` typedef.
- Use `run_catching`, `unwrap_catch`, `capi_error`, and `err_status` so `Err(e)`
  values and panic payloads reach `t4a_last_error_message`.
- **No new `catch_unwind` / `Err(_) => T4A_INTERNAL_ERROR` patterns** that drop
  error messages.
- Release builds omit Rust debug info. For source-level backtraces through
  Tensor4all.jl or the C API, build with
  `cargo build --profile release-debug -p tensor4all-capi` and run the caller
  with `RUST_BACKTRACE=1`.

## Testing

```bash
cargo nextest run --release --workspace          # Full suite
cargo nextest run --release --test test_name     # Specific test
cargo nextest run --release -p crate_name        # Single crate
```

**Always use `--release` for tests**; debug builds are too slow.

- Private functions: `#[cfg(test)]` module in the source file. Integration
  tests: `tests/`.
- **Dense whole-result comparisons**: do not recompute contractions
  element-by-element. Materialize once to a dense tensor/matrix, then compare
  via tensor subtraction and `maxabs()`. `IdxTensor` subtraction aligns indices
  by semantics, so explicit axis reordering is usually unnecessary.
- **Test tolerance changes** (unit tests, codecov targets, ...) require explicit
  user approval.

### Coverage and Path Exercise

- Every distinct control-flow path (error branches, layout variants, boundary
  conditions) needs a test; happy-path only is insufficient.
- When removing code, check whether the removed tests were the sole exerciser
  of a shared helper and add replacement coverage.
- Before pushing a deletion PR, attest in the PR body that removed code paths
  were reviewed for coverage impact (shared rules `common/docs-and-tests.md`:
  coverage is CI-owned, the local gate is attestation-based). CI coverage is
  authoritative; a local run is optional:

```bash
cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Fix drops by adding tests, not by lowering thresholds (threshold changes need
explicit approval).

## API Design

Only make functions `pub` when truly public API.

### Layering and Maintainability

**Respect crate boundaries and abstraction layers**, in library and test code
alike.

- **Never access low-level APIs or internal data structures from downstream
  crates.** If downstream code needs low-level access, add a high-level API
  instead of exposing internals.
- **Prefer type-generic Rust APIs.** No scalar-specific entry points such as
  `*_f64` / `*_c64` when a generic form is possible; prefer the generic form in
  library code, tests, examples, and docs. The C API / FFI boundary is exempt.
- Examples: `scalar.real()`, `scalar.is_complex()`, `scalar.is_zero()` instead
  of `match scalar { AnyScalar::F64(x) => ... }`; `AnyScalar::new_real(1.0)`
  instead of `AnyScalar::F64(1.0)`; `AnyScalar::new_complex(re, im)` instead
  of `AnyScalar::C64(z)`.

**No ad hoc fixes:** nothing that violates DRY, KISS, or layering; no
compatibility shims, duplicated implementations, or downstream reach-through
where a higher-level seam should exist. If a behavior does not fit the current
abstraction, add or refine the seam.

### Code Deduplication

Avoid duplicate test code: share via macros, functions, or generics.

```rust
fn test_op_generic<T: Scalar + From<f64>>() { /* test */ }

#[test]
fn test_op_f64() { test_op_generic::<f64>(); }
#[test]
fn test_op_c64() { test_op_generic::<Complex64>(); }
```

### Dense Layout And Linear Algebra

- Dense flat-buffer APIs are **column-major**. New public constructors,
  exports, examples, FFI contracts, and docs must state or preserve this.
- No row-major compatibility shims or hidden round-trips in library code;
  convert row-shaped external data privately at that explicit boundary.
- Reuse `tensor4all-tensorbackend::Matrix` for dense matrices crossing crate
  boundaries; no duplicate public matrix containers downstream.
- Use tenferro-backed `tensor4all-tensorbackend` / existing tensor4all
  abstractions for SVD, QR, einsum, and dense linear algebra, not local
  reimplementations.
- Canonical integrations supply a configured `CpuBackend` to
  `tensor4all_tensorbackend::CpuExecutionContext`; legacy convenience APIs may
  use `with_default_backend` behind the `global-defaults` feature. Never create
  an unconfigured fallback backend in downstream operation code.
- `Tensor3Ops::slice_site`, `Tensor4Ops::slice_site`, and dense/full-tensor
  exports return column-major flat data.

## C API & Language Bindings

The C API is the binding boundary; patterns in `docs/CAPI_DESIGN.md`. Bindings:
[Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl) (separate repo).

Truncation tolerance: support both `cutoff` (ITensors) and `rtol`
(tensor4all-rs); `rtol = √cutoff`.

### Cross-repo development with Tensor4all.jl

When a Tensor4all.jl feature needs new C API functions:

1. Develop both sides locally; Tensor4all.jl's `deps/build.jl` uses a local
   Rust build via `TENSOR4ALL_RS_PATH`.
2. Test both sides locally until all tests pass.
3. Create and merge the tensor4all-rs PR **first**.
4. Update the pin hash in Tensor4all.jl `deps/build.jl` to the merged remote
   commit and create the Tensor4all.jl PR.

Issue templates: `.github/ISSUE_TEMPLATE/`. Feature requests from Tensor4all.jl
link the related Julia-side issue.

## Dependencies

- Prefer existing tensor4all core/tcicore/simplett abstractions and
  tenferro-backed `tensor4all-tensorbackend` for arrays and linear algebra. Add
  direct array/linalg dependencies only when those are genuinely insufficient.
- SVD singular values: `s[[0, i]]`, not `s[[i, i]]`.

## Git Workflow

**Never push/create PR without user approval.**

### Base Branch Synchronization

- Start work with `git fetch origin` and branch from `origin/main`, not a stale
  local `main`.
- Before treating PR checks as final, fetch `origin` and verify the PR branch
  contains current `origin/main`. If behind, update from `origin/main` before
  relying on checks, enabling auto-merge, or declaring it ready.
- After updating, re-monitor CI; earlier green checks do not cover the
  synchronized branch.

### Pre-PR Checks (matches CI)

Run before pushing.

```bash
cargo fmt --all                        # Auto-fix formatting
cargo fmt --all -- --check             # Dry-run (matches CI)
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo test -p <changed-crate> --release # Quick check first
cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5
cargo test --profile ci -p tensor4all-hdf5
cargo test --doc --profile ci --workspace -j 8
cargo doc --workspace --no-deps        # Build rustdoc
```

### Repository Rules Review Bot

`.github/workflows/review_bot.yml` reviews every PR diff against
`REPOSITORY_RULES.md` and `PERFORMANCE_TIPS.md` from the trusted base
revision; the PR head is fetched for `git diff` only, never checked out or
executed. Findings post as one updating PR comment; only `block`-severity
findings fail the check.

Preview locally before pushing:

```bash
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
python3 scripts/test-repository-rules-review.py   # the script's own tests
```

Drop `--dry-run` for the LLM pass; it needs `DEEPSEEK_API_KEY` in the
environment or a repo-root `.env` (`pip install -r scripts/requirements-dev.txt`).
System prompt: `ai/prompts/repository-rules-review.md`. Three deterministic
checks run before and independently of the LLM:

| Check | Rule | Baseline |
|-------|------|----------|
| Secret-shaped text in added lines | (none) | Blocks the upload before anything reaches the API |
| New direct `tenferro-*` dependency outside `tensor4all-tensorbackend` | Dense Layout And Linear Algebra | #566 Phase 3 backlog: core, tcicore, simplett, treetci |
| Added `ignore` / `no_run` doctest fence | Documentation Examples | #566 Phase 1 backlog: 2 `no_run` sites |

Both rule checks are delta-scoped: new violations fail, the recorded backlog
does not. The tenferro check parses Cargo.toml table context, so `tenferro-*`
**feature** names are not mistaken for dependencies and `dev-dependencies` are
out of scope.

Maintainer escape hatches (`maintain`/`admin` role, reapply after the latest
push):

| Label | Effect |
|-------|--------|
| `rules-review:no-llm` | Skips the LLM pass; deterministic checks still run |
| `rules-review:waive` | Waives the review entirely |

When adding a `## ` section to `REPOSITORY_RULES.md` or `PERFORMANCE_TIPS.md`,
route it in `SECTION_TRIGGERS` (or `ALWAYS_SECTIONS` / `HUMAN_ONLY_SECTIONS`);
an unrouted section is never shown to the reviewer and
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

### New Public Crate Checklist

Before a PR that adds a public workspace crate, complete every applicable item;
mark inapplicable items explicitly in the PR body.

- [ ] Add the crate to the workspace and the root `README.md` crate map.
- [ ] Add the crate and its primary use cases to `llms.txt` with a direct link
      to the most useful guide for a new user or coding agent.
- [ ] Update `docs/book/src/architecture.md` (layer diagram, crate table,
      goal-to-crate selection table).
- [ ] Add or update an mdBook guide and register it in
      `docs/book/src/SUMMARY.md`.
- [ ] Link the new guide from related guides so users need not know the crate
      name in advance.
- [ ] Add a crate `README.md` or an equally clear docs.rs landing page.
- [ ] Provide a runnable, asserted example exercising the crate's primary
      nontrivial code path; a degenerate shortcut or smoke test is not enough.
- [ ] Remove developer-local absolute paths and references to inaccessible
      design material from committed documentation.
- [ ] Run workspace doctests and `./scripts/test-mdbook.sh`.

For every PR, verify the root `README.md` remains accurate, including project
structure and examples.
