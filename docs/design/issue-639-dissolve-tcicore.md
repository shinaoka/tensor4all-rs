# Design: dissolve tensor4all-tcicore into tensor4all-core (#639)

Status: user-approved direction (2026-08, "A1 + tcicore 廃止 — no compatibility
phase, delete the crate"). Review by luna (read-only): pre-implementation design
review + post-implementation diff review.

## Problem

#566 Phase 5 claimed "fix the inverted layer (core depends on tcicore)" but
core still depends on tcicore. The dependency is not just the Scalar trait:
`core::factorize` (LU/CI paths) uses tcicore's LU/LUCI algorithms, and core
re-exports `tcicore::Scalar as CommonScalar`.

Per the user: this is the no-backward-compatibility phase (AGENTS.md "Early
development - remove deprecated code immediately"). Instead of moving the
needed closure into core and keeping tcicore as a re-export shim, **dissolve
tcicore entirely**: all its modules move into core, the crate is deleted, and
every user switches to `tensor4all_core::X`.

## Layer result

```
tensorbackend ← core ← {simplett, treetci, aci, quanticstci, tensorci, ...}
```

No tcicore. core absorbs the TCI substrate (consistent with the workspace
charter: "tensor4all-rs — Rust core: tensor networks, TCI, quantics
transforms").

## Module map (tcicore → core)

Mirror the tcicore layout inside core so the public-name swap is 1:1:

| tcicore path | moves to | notes |
|---|---|---|
| `src/scalar.rs` (Scalar, scalar_tests!) | `core/src/scalar.rs` | replaces the 232-byte re-export; core lib exports `Scalar` AND keeps `CommonScalar` alias |
| `src/error.rs` (MatrixCIError, Result) | `core/src/error.rs` | new module; no name clash in core |
| `src/matrixlu.rs` (rrlu, rrlu_inplace, RrLU, RrLUOptions) | `core/src/matrixlu.rs` | |
| `src/matrixluci/` (MatrixLUCI, MatrixLuciScalar, block_rook/dense/factors/kernel/source/types + tests) | `core/src/matrixluci/` | |
| `src/matrix_luci.rs` (MatrixLuciFactors, matrix_luci_factors_from_*) | `core/src/matrix_luci.rs` | stays a private `mod` with `pub use` of its items, exactly as tcicore does (1:1 mirror, no new public module) |
| `src/matrixaca.rs` (MatrixACA) | `core/src/matrixaca.rs` | implements AbstractMatrixCI; no cached_function/indexset dep |
| `src/traits.rs` (AbstractMatrixCI) | `core/src/traits.rs` | new module; core has no `traits` module |
| `src/cached_function/` (CachedFunction, CacheKey, CacheKeyError, IndexInt) | `core/src/cached_function/` | uses bnum (core must add the dep) |
| `src/indexset.rs` (IndexSet, LocalIndex, MultiIndex) | `core/src/indexset.rs` | |
| `benches/` (rrlu_bench, cached_function, dense_vs_tenferro, lazy_block_rook) | `core/benches/` | core gains its first benches; add `[[bench]]` + criterion dev-dep |
| `examples/benchmark_matrix_lu.rs` | `core/examples/` (or fold into benches) | |

All rustdoc examples inside the moved code that reference `tensor4all_tcicore`
must be rewritten to `tensor4all_core` (doctests run in CI).

Internal `crate::` imports inside the moved code become core module paths
(`crate::error::{MatrixCIError, Result}`, `crate::scalar::Scalar`,
`crate::matrixlu::RrLUOptions`, ...). No cfg(feature) gates exist in the moved
code (verified), so no feature-porting is needed beyond core's existing
tensorbackend features.

## core lib.rs public surface (mirror tcicore's re-exports 1:1)

```
pub use self::matrixluci::Scalar as MatrixLuciScalar;
pub use cached_function::{CachedFunction, cache_key::CacheKey, error::CacheKeyError, index_int::IndexInt};
pub use error::{MatrixCIError, Result};
pub use indexset::{IndexSet, LocalIndex, MultiIndex};
pub use matrix_luci::{matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix,
    matrix_luci_factors_from_matrix_owned, MatrixLUCI, MatrixLuciFactors};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_inplace, RrLU, RrLUOptions};
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
// existing alias kept:
pub use scalar::Scalar as CommonScalar;
```

Every downstream `tensor4all_tcicore::X` import then maps to
`tensor4all_core::X` with no semantic change.

## core Cargo.toml changes

- Add deps: `bnum` (cached_function keys), `paste` (scalar_tests! macro).
- Add dev-deps: `approx` (moved tests), `criterion` (moved benches).
- NOT needed: `uint` (not used by moved code/tests/benches — luna verified),
  `tenferro-einsum` (not imported by moved tcicore source). `tenferro-linalg`
  and `tenferro-tensor` are already normal core dependencies.
- Remove the `tensor4all-tcicore` dependency and the feature-forwarding lines
  (`tensor4all-tcicore/tenferro-*`) — the same backend/CPU/BLAS/provider/
  einsum/linalg features are already forwarded by core directly (luna
  verified).
- Add `[[bench]]` entries for the moved benches (and move or fold
  `examples/benchmark_matrix_lu.rs`).
- Downstream manifests (aci, simplett, etc.) that forward
  `tensor4all-tcicore/tenferro-*` features must drop those forwarding lines.

## Downstream updates (import swap + Cargo.toml dep removal)

| crate | tcicore surface used |
|---|---|
| tensor4all-aci | MatrixCIError, MatrixLuciScalar, matrix_luci_factors_from_matrix_owned, RrLUOptions — **adds a direct tensor4all-core dep** (currently none) |
| tensor4all-partitionedtt | MultiIndex, MatrixLuciScalar, Scalar |
| tensor4all-quanticstci | MatrixLuciScalar |
| tensor4all-simplett | MatrixCIError, MatrixLuciScalar, Scalar, AbstractMatrixCI, MatrixLUCI, RrLUOptions, rrlu |
| tensor4all-tensorci | AbstractMatrixCI, IndexSet, MatrixACA, RrLUOptions, matrix-LUCI fns, MatrixLUCI, MatrixCIError, MatrixLuciScalar, MultiIndex, Scalar (CachedFunction only in docs) — **adds a direct tensor4all-core dep** (currently none); also migrate its example/benchmark that reference tcicore |
| tensor4all-treetci | MatrixLuciScalar (as Scalar), RrLUOptions |
| tensor4all-interpolativeqtt | none in src — drop the unused dep |
| docs/book-tests | drop the dep (workspace member) |
| tensor4all-capi | no tcicore dep (verified) |

Additional files to update (luna-verified):
- workspace root `Cargo.toml`: remove `tensor4all-tcicore` member + any
  workspace-dep entry; regenerate `Cargo.lock`.
- `scripts/library-panics-baseline.json`: **move** the 7 tcicore entries
  (`matrixlu.rs:741:debug_assert_eq`, `matrixluci/source.rs` x4, ...) to their
  new core paths — the assertions move with the code, they are not deleted.
- `scripts/check-crate-boundaries.py`, `scripts/test-check-crate-boundaries.py`,
  `scripts/repository-rules-review.py`: drop/rename tcicore references
  (boundary fixtures may become generic cycle fixtures).
- `benchmarks/rust/benchmark_matrix_lu.rs`, `benchmarks/results/2026-05-22-matrix-lu-hilbert.md`,
  `benchmarks/README.md`: tcicore references.
- `docs/api/tensor4all_tcicore.md`: remove/regenerate via api-dump.
- `docs/book/src/architecture.md` (crate diagram) and
  `docs/book/src/guides/tci-advanced.md` (live guide), plus
  `docs/PROVENANCE_AND_CITATION_POLICY.md`: tcicore references.
- `skills/use-tensor4all-rs/references/{crates.md,recipes.md}` and `SKILL.md`,
  and `ai/prompts/` if it names tcicore.

## Verification

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace`
- `cargo test --doc --release --workspace`, `./scripts/test-mdbook.sh`
- `cargo doc --workspace --no-deps`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree`
- No-reference check for `tensor4all-tcicore`/`tensor4all_tcicore`, with
  explicit exclusions: `.worktrees/` (foreign worktrees), generated
  `docs/book/book/` (mdbook build artifacts), and the deliberately-kept
  historical records (`docs/plans/2026-03-27-tcicore-extraction-*`,
  `docs/design/*` that document the pre-dissolution state).

## Review verdicts

- Design (pre-implementation), luna: NEEDS-FIX (round 1: deps/downstream/panics
  baseline/additional files/matrix_luci mirror; round 2: tensorci row) → all
  findings fixed per luna's prescriptions → design approved.
- Diff (post-implementation), luna: TBD.
- Diff (post-implementation), luna: TBD.
