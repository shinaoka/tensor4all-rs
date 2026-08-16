# Design: #616 — complete the Phase-4 rename sweep (7 names + scalar-suffixed constructors)

Status: user-approved (2026-08, "y" to proceed; no-compatibility phase).
Review by luna (read-only): design pre-review + diff post-review.

## Problem

Umbrella #566 Phase 4 claimed the Julia-style concatenated names were renamed
to snake_case (item 32) and scalar-suffixed Rust entry points removed (item 34),
but 7 names and 2 scalar-suffixed constructors remain public and unrenamed.
This task completes the sweep and corrects the ledger.

## Renames (snake_case primary + `#[doc(alias = "old")]`; old names removed from API)

| Old | New | Location | Call sites to update |
|---|---|---|---|
| `hasinds` | `has_inds` | core/src/index_ops.rs:417 | index_ops doc examples, core lib.rs re-export (:91), tests/common_index_ops.rs |
| `hascommoninds` | `has_common_inds` | core/src/index_ops.rs:449 | same files; **plus itensorlike/src/tensortrain.rs:16 (import) and :953 (internal use)** |
| `isortho` | `is_ortho` | itensorlike/src/tensortrain.rs:450 | prelude.rs doctest, tensortrain.rs internal calls (:387/:399) + doctests + tests, itensorlike README.md:54 |
| `orthocenter` | `ortho_center` | itensorlike/src/tensortrain.rs:458 | same |
| `maxiter` | `max_iter` | quanticstci/src/options.rs:83 (QtciOptions field) | options.rs default impl + doc examples + **builder internals (:194 `self.maxiter = maxiter`) + to_treetci_options (:286 `max_iter: self.maxiter`)**, quantics_tci.rs, options/tests, feature_test_physicist.rs |
| `nrandominitpivot` | `n_random_init_pivot` | quanticstci/src/options.rs:93 | quantics_tci.rs (:463/:787), tests |
| `unfoldingscheme` | `unfolding_scheme` | quanticstci/src/options.rs:102 | quantics_tci.rs (:628/:730), tests |

Builder methods (`with_maxiter`, `with_nrandominitpivot`, `with_unfoldingscheme`)
are NOT in the #616 enumeration and are left unchanged (their parameter names
are internal; only the field assignments inside them are renamed). The
singular `hasind` is not in the enumeration; left unchanged. treetn `maxiter`
occurrences are GMRES/KrylovKit names — unrelated to QtciOptions.

## Removal (scalar-suffixed constructors)

`Storage::from_dense_f64_col_major` (storage.rs:1268) and
`Storage::from_dense_c64_col_major` (:1293) are removed; the generic
`Storage::from_dense_col_major<T>` (storage.rs:315) already exists and the two
tests in storage/tests/mod.rs already call the generic form (only their test
names mention the old fns — rename the test names). The removed methods' doc
examples disappear with them. No `#[doc(alias)]` (the generic is the
replacement; scalar-suffixed names belong at the FFI boundary only).

## Ledger

`docs/superpowers/ledgers/2026-08-11-issue-566-pr4-ledger.md` items 32 (:65) and
34 (:67) claim DONE. Append a dated residual-completion note to each item
stating the 7 names and 2 constructors were completed by this PR, so the ledger
matches the tree.

## Verified scope boundaries (grep on origin/main)

- No capi/treetn/simplett/treetci/aci usage of the 7 names (treetn `maxiter`
  hits are GMRES/KrylovKit names).
- docs/tutorial-code uses QtciOptions only via builders (`.with_*`); its
  `config.maxiter` hits are local config structs, not QtciOptions fields —
  out of scope.
- No `#[doc(alias)]` exists anywhere in crates/ yet — first introduction.
- `hascommoninds`/`hasinds` are re-exported at core lib.rs:91 — update the
  list to the new names.
- **Rust-facing docs to update**: itensorlike README.md:54 (isortho),
  docs/book/src/guides/tensor-train.md:180-181 (isortho/orthocenter),
  docs/book/src/guides/tci.md:126 (nrandominitpivot), skills
  references/crates.md:47 (isortho/orthocenter), references/recipes.md:235
  (isortho), references lines 86/328 (nrandominitpivot),
  docs/tutorial-code/docs/tutorials/qtt_function_tutorial.md:137
  (nrandominitpivot/unfoldingscheme).
- **Out of scope (Julia-side API)**: docs/examples/julia/core.jl:30
  (`hascommoninds`) and quanticstci.jl:14 (`unfoldingscheme=:fused`) call
  the Tensor4all.jl package API, a separate repo; the Rust rename does not
  change the Julia surface and those examples must NOT be edited here.

## Verification

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace`
- `cargo test --doc --release --workspace`, `./scripts/test-mdbook.sh`
- `cargo doc --workspace --no-deps` (doc(alias) renders)
- `python3 scripts/repository-rules-review.py --base origin/main --worktree`
- grep: no remaining old names except `#[doc(alias)]` attributes and the
  ledger note.

## Review verdicts

- Design (pre-implementation), luna: TBD.
- Diff (post-implementation), luna: TBD.
