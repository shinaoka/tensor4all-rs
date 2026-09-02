# SRC Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the remaining items from the SRC provenance audit's
recommendations, and unblock PR #694's failing Coverage CI check, on branch
`feature/treetn-src`.

**Architecture:** Seven sequential tasks against real Rust source (not
documentation) in `tensor4all-rs`. Each task is TDD: write/extend a failing
test, implement the minimal fix, verify, commit. Tasks 1 and 2 unblock the
open PR's failing CI gate; the rest close out audit recommendations that
were never addressed. Task 7 is explicitly measurement-gated — it may
produce a worklog instead of a code change.

**Tech Stack:** Rust, `cargo test`/`cargo llvm-cov`/`cargo fmt`/`cargo
clippy`, the existing `tensor4all-treetn`/`tensor4all-tensorbackend`/
`tensor4all-core` crates.

**Spec:** `docs/plans/2026-08-28-src-provenance-audit-report.md` (§12
"Recommended next steps" is the binding list these tasks implement; §1, §8,
and the `F-1`/`F-4`/`F-5`/`F10` findings in the WS-backend/WS-tree-probe
sections give the exact reasoning behind tasks 5, 7, and part of 1).

## Global Constraints

- All work happens directly on branch `feature/treetn-src` in the existing
  worktree `/root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src`
  (already pushed to `origin`, PR #694 open against `main`). No new
  branch/worktree.
- `cargo fmt --all -- --check`, `cargo clippy` with warnings denied for any
  crate you touch, and the full test suite for any crate you touch must pass
  before a task's commit.
- Commit messages follow the branch's existing style: `type(scope): summary`
  (e.g. `perf(backend): use backend QR for incremental SRC`,
  `test(backend): cover incremental QR validation errors`).
- Do not translate the reference Python/C++ at
  `/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/` — the
  audit's `LICENSE-RISK` clearance (§8 of the spec) depends on the Rust
  staying an independent derivation, not a port.
- Coverage gate: `scripts/check-coverage.py` requires 75% per-file line
  coverage (`cargo llvm-cov --release --workspace --exclude
  tensor4all-hdf5 --json`, doctests NOT counted). Two files are currently
  under it: `crates/tensor4all-tensorbackend/src/incremental_qr.rs` (73.8%)
  and `crates/tensor4all-core/src/tensor_like.rs` (41.5%).

---

## Task 1: Cover `incremental_qr.rs`'s validation errors and scalar impls

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/incremental_qr.rs:150-234`
  (`from_factors`, `new` — the guard clauses)
- Modify: `crates/tensor4all-tensorbackend/src/incremental_qr/tests.rs`
  (add new `#[test]` functions near the existing
  `incremental_qr_rejects_invalid_shapes` at line 289)

**Interfaces:**
- Consumes: `IncrementalQr::<T>::new(Matrix<T>) -> Result<Self,
  BackendLinalgError>`, `IncrementalQr::<T>::from_factors(Matrix<T>,
  Matrix<T>) -> Result<Self, BackendLinalgError>` (existing, unchanged
  signatures).
- Produces: nothing new consumed by later tasks.

The existing test `incremental_qr_rejects_invalid_shapes` (line 289) only
covers `append`'s shape validation. `new` (lines 215-234) and `from_factors`
(lines 150-185) have their own guard clauses that are never exercised:

- `new`: empty matrix (`nrows == 0 || ncols == 0`, line 216) and wide matrix
  (`nrows < ncols`, line 219).
- `from_factors`: empty `q` (line 154), thin-`Q` violation (`q.nrows() <
  q.ncols()`, line 157), and incompatible `Q`/`R` dimensions (`r.nrows() !=
  q.ncols() || r.ncols() < q.ncols()`, line 165).

Separately, `IncrementalQrScalar`'s `f32`/`Complex32` impls (lines 42-69)
are dead: the core `IncrementalQrState` enum only has `F64`/`C64` variants
(confirmed by the audit's WS-backend finding F10), so no production code
path ever instantiates `IncrementalQr<f32>` or `IncrementalQr<Complex32>`.
Since they are genuinely unreachable rather than merely untested, do not
write throwaway tests to exercise them — remove them, matching the audit's
recommendation to treat premature/dead code as a deletion candidate rather
than a coverage target.

- [ ] **Step 1: Write the failing tests for `new`'s guards**

Add to `crates/tensor4all-tensorbackend/src/incremental_qr/tests.rs`:

```rust
#[test]
fn incremental_qr_new_rejects_empty_matrix() {
    let empty_rows = Matrix::from_col_major_vec(0, 1, vec![]);
    assert!(IncrementalQr::new(empty_rows).is_err());

    let empty_cols = Matrix::from_col_major_vec(3, 0, vec![]);
    assert!(IncrementalQr::new(empty_cols).is_err());
}

#[test]
fn incremental_qr_new_rejects_wide_matrix() {
    let wide = Matrix::from_col_major_vec(2, 3, vec![1.0; 6]);
    assert!(IncrementalQr::new(wide).is_err());
}
```

- [ ] **Step 2: Run to verify they fail or already pass**

Run: `cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml
incremental_qr_new_rejects -- --nocapture`

Expected: both tests should already PASS (the guards exist), since this
step is establishing coverage, not fixing a bug. If either test panics with
something other than a clean `Err` (e.g. the code accepts a 0x0 or wide
matrix), that is a real bug — stop and report it before continuing, since
the audit did not flag this as a correctness issue and a passing-but-wrong
guard would be a new finding.

- [ ] **Step 3: Write the failing tests for `from_factors`'s guards**

```rust
#[test]
fn incremental_qr_from_factors_rejects_empty_q() {
    let empty_q = Matrix::from_col_major_vec(0, 0, vec![]);
    let r = Matrix::from_col_major_vec(1, 1, vec![1.0]);
    assert!(IncrementalQr::from_factors(empty_q, r).is_err());
}

#[test]
fn incremental_qr_from_factors_rejects_non_thin_q() {
    // Q must be tall-or-square: nrows >= ncols. 2x3 violates that.
    let wide_q = Matrix::from_col_major_vec(2, 3, vec![1.0; 6]);
    let r = Matrix::from_col_major_vec(3, 3, vec![1.0; 9]);
    assert!(IncrementalQr::from_factors(wide_q, r).is_err());
}

#[test]
fn incremental_qr_from_factors_rejects_incompatible_r() {
    let q = Matrix::from_col_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    // R must have r.nrows() == q.ncols() == 2, and r.ncols() >= 2.
    let bad_r = Matrix::from_col_major_vec(1, 2, vec![1.0, 1.0]);
    assert!(IncrementalQr::from_factors(q, bad_r).is_err());
}
```

- [ ] **Step 4: Run to verify all four new tests pass**

Run: `cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml
incremental_qr_new_rejects incremental_qr_from_factors_rejects --
--nocapture`
Expected: 4 passed; 0 failed.

- [ ] **Step 5: Delete the dead `f32`/`Complex32` `IncrementalQrScalar` impls**

In `crates/tensor4all-tensorbackend/src/incremental_qr.rs`, delete the
`impl IncrementalQrScalar for f32` block (lines 42-50) and the `impl
IncrementalQrScalar for Complex32` block (lines 62-70). Leave the `f64` and
`Complex64` impls untouched. Check the file still compiles — if anything
else in the crate references the `f32`/`Complex32` impls (it shouldn't,
per the audit's F10 finding, but verify), stop and report instead of
force-deleting.

- [ ] **Step 6: Run the full crate test suite and coverage check**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml
cargo llvm-cov --release -p tensor4all-tensorbackend --json --output-path /tmp/cov_backend_recheck.json
python3 -c "
import json
with open('/tmp/cov_backend_recheck.json') as f:
    d = json.load(f)
for e in d['data'][0]['files']:
    if e['filename'].endswith('incremental_qr.rs'):
        print(e['summary']['lines'])
"
```

Expected: all tests pass, and the printed `percent` is >= 75.0.

- [ ] **Step 7: Format, lint, commit**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-tensorbackend/src/incremental_qr.rs crates/tensor4all-tensorbackend/src/incremental_qr/tests.rs
git commit -m "test(backend): cover incremental QR validation errors, drop dead f32/Complex32 impls"
```

---

## Task 2: Cover `tensor_like.rs`'s trait-default methods

**Files:**
- Modify: `crates/tensor4all-core/src/tensor_like/tests/mod.rs`
- Read (do not modify): `crates/tensor4all-core/src/tensor_like.rs:1000-1520`
  (the six default-method bodies), `crates/tensor4all-core/src/defaults/idx_tensor.rs`
  (confirms `IdxTensor` overrides all six — lines 5647, 5656, 6000, 6041,
  6048, 6056 — so the *only* concrete type in this codebase never reaches
  these default bodies)

**Interfaces:**
- Consumes: `TensorLike` and its supertraits (`TensorVectorSpace +
  TensorFactorizationLike + TensorConstructionLike`,
  `crates/tensor4all-core/src/tensor_like.rs:624,780,927,1113,1602`).
- Produces: nothing consumed by later tasks.

The uncovered lines are the default bodies of `factorize_probe_columns_incremental`
(~1060-1074), `src_error_estimate` (~1103+), `from_dense_any` (~1207+),
`stack_along_new_index` (~1283+), `concatenate_along_new_index` (~1395+),
and `select_indices` (~1476+) inside `TensorFactorizationLike`/
`TensorConstructionLike`. These are intentional extension points for future
`TensorLike` implementors (the module doc at `stack_along_new_index`
explicitly frames them as a "correctness fallback"), not dead/premature
code like Task 1's deleted impls — do not delete them.

- [ ] **Step 1: Assess whether a minimal test-double is proportionate**

Count the total non-default (required) method surface across
`TensorVectorSpace`, `TensorContractionLike`, `TensorFactorizationLike`, and
`TensorConstructionLike` (46 `fn` signatures total across the file, as of
this plan's writing — re-count, since the exact number matters for this
decision). If implementing a minimal struct that satisfies every required
method (returning trivial/degenerate values where the test doesn't care) is
achievable in something close to 150-250 lines, build it (Step 2). If the
required surface turns out to need substantially more scaffolding than
that to compile, stop and use the coverage-threshold override path instead
(Step 2-alt) — do not spend more than one focused attempt on the
test-double before falling back.

- [ ] **Step 2: Build a minimal `TensorLike` test-double and exercise the six defaults**

In `crates/tensor4all-core/src/tensor_like/tests/mod.rs`, add a private
struct (e.g. `DefaultOnlyTensor`) that implements `TensorVectorSpace +
TensorContractionLike + TensorFactorizationLike + TensorConstructionLike`
(and therefore `TensorLike`) by delegating every REQUIRED method to a
trivial inner representation (a `Vec<f64>` with a fixed shape is enough —
the test doesn't need real tensor semantics, only enough structure for the
six default bodies to run without panicking) and does NOT override
`factorize_probe_columns_incremental`, `src_error_estimate`,
`from_dense_any`, `stack_along_new_index`, `concatenate_along_new_index`,
or `select_indices`. Write one `#[test]` per default method that
constructs a `DefaultOnlyTensor`, calls the default method directly (e.g.
`DefaultOnlyTensor::stack_along_new_index(&[...], index, axis)`), and
asserts on the result — mirroring the assertions already present in each
method's rustdoc example (e.g. `src_error_estimate`'s doctest at
`tensor_like.rs:1090-1101` asserts `estimate.error.is_finite()` and
`estimate.norm.is_finite()`; reuse that shape for the unit test).

- [ ] **Step 2-alt (only if Step 1 ruled out the test-double): document a threshold override**

Add an entry to `coverage-thresholds.json`'s `"files"` map for
`crates/tensor4all-core/src/tensor_like.rs` at a value that reflects the
file's actual reachable-by-`IdxTensor` coverage (measure it first — do not
guess a number), with a one-line comment file
(`docs/worklogs/2026-08-29-tensor-like-coverage-exception.md` or similar,
matching the repo's existing worklog convention) explaining that the
six default bodies are unreachable through the only production
implementor and are covered by doctests instead, citing the specific
doctest line numbers.

- [ ] **Step 3: Run the new tests**

Run: `cargo test --manifest-path crates/tensor4all-core/Cargo.toml
tensor_like::tests -- --nocapture`
Expected: all new tests pass.

- [ ] **Step 4: Run the coverage check**

```bash
cargo llvm-cov --release -p tensor4all-core --json --output-path /tmp/cov_core_recheck.json
python3 -c "
import json
with open('/tmp/cov_core_recheck.json') as f:
    d = json.load(f)
for e in d['data'][0]['files']:
    if e['filename'].endswith('tensor_like.rs'):
        print(e['summary']['lines'])
"
```

Expected: `percent` >= 75.0 (via Step 2's real coverage, or Step 2-alt's
documented, justified threshold).

- [ ] **Step 5: Format, lint, commit**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-core/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-core/src/tensor_like/tests/mod.rs coverage-thresholds.json docs/worklogs/
git commit -m "test(core): cover TensorLike default-method bodies"
```

---

## Task 3: Fix the fabricated citation and the LICENSE-RISK wording

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:4`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:4`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction.rs:11,15`

**Interfaces:** None — doc-comment-only change, no code/API surface
affected.

Three module docs cite "Algorithm 1 and Sections 2.3--2.5 of
Camaño--Epperly--Tropp" (or "Sections 2.3--2.5" alone). The paper's actual
§2 (`/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`)
has only §2.1 and §2.2; the algorithmic content these files implement is
§3.1-§3.5. Separately, `contraction.rs:15` reads "a line-by-line
cross-check" against the unlicensed reference repo — the audit cleared this
of an actual `LICENSE-RISK` finding (spec §8) but flagged the phrase itself
as worth rewording since it's the exact trigger language the policy warns
about.

- [ ] **Step 1: Fix the three citations**

In each of the three files, replace "Sections 2.3--2.5" (or "Algorithm 1
and Sections 2.3--2.5") with "Algorithm 1 and Sections 3.1-3.5" — read each
file's exact current sentence first (they are not worded identically) and
edit in place rather than a blind find-and-replace, since the surrounding
grammar differs per file.

- [ ] **Step 2: Reword the LICENSE-RISK trigger phrase**

In `crates/tensor4all-treetn/src/treetn/contraction.rs:15`, reword "The
author implementation used for a line-by-line cross-check is
`chriscamano/RandomMPOMPS`..." to something that preserves the factual
content (which reference implementation was used to validate numerical
behavior) without the phrase "line-by-line cross-check" — e.g. "The author
implementation used to validate numerical behavior and parameter
conventions (not translated) is `chriscamano/RandomMPOMPS`...".

- [ ] **Step 3: Verify no other file repeats either issue**

```bash
grep -rn "2\.3.*2\.5\|Sections 2\.3\|line-by-line cross-check" crates/tensor4all-treetn/src/
```

Expected: no output.

- [ ] **Step 4: Run doctests for the touched files (doc comments can contain doctests)**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --doc
```

Expected: all pass (this change should not touch any runnable doctest code,
only prose, but verify nothing else was disturbed).

- [ ] **Step 5: Format, commit**

```bash
cargo fmt --all -- --check
git add crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs
git commit -m "docs(treetn): correct fabricated paper citation and license-risk wording"
```

---

## Task 4: Delete the dead `site_probe` function family

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs:180-224`
  (delete `site_probe`, `site_probe_batch`, `site_probe_batch_range`)
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs`
  (test module, around lines 813-816, 870, 1090 — remove or adapt the
  tests that call these functions)

**Interfaces:** None — these functions have zero production call sites
(confirmed: `grep -rn "site_probe\b|site_probe_batch\b|site_probe_batch_range\b"
crates/tensor4all-treetn/src/` returns only the definitions and their own
`#[cfg(test)]` usages).

- [ ] **Step 1: Confirm zero production callers, one more time, on current HEAD**

```bash
grep -rn "site_probe\b" crates/tensor4all-treetn/src/ --include="*.rs" | grep -v "#\[cfg(test)\]"
```

Read the output. If anything outside a `#[cfg(test)] mod tests` block
appears, stop — the audit's finding may be stale and you need to
investigate before deleting.

- [ ] **Step 2: Read the two test functions that use these helpers**

Read `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs` around
lines 860-880 and 1080-1100 (the tests using `site_probe` and
`site_probe_batch`). Determine whether each test exists *only* to exercise
the dead helper (in which case delete the whole test) or whether it also
exercises other, still-relevant behavior via these helpers as setup
scaffolding (in which case rewrite the setup to use the production
probe-construction path instead of deleting the test's actual assertions).

- [ ] **Step 3: Delete `site_probe`, `site_probe_batch`, `site_probe_batch_range`**

Delete the three function definitions (lines 180-224 in the current file;
re-check line numbers after Step 2's edits, since removing/rewriting tests
above them shifts nothing since tests are below, but confirm before
editing).

- [ ] **Step 4: Remove the now-dead import**

Fix the `#[cfg(test)]` import list around line 816 that imports
`site_probe, site_probe_batch` — remove the now-unused names, keep
`probed_site_pair_batch_range` and `ProbeBank` if still used elsewhere in
the test module.

- [ ] **Step 5: Run the crate test suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml
```

Expected: all pass, no "unused import"/"unused function" warnings (check
with `cargo clippy` in Step 6).

- [ ] **Step 6: Format, lint, commit**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs
git commit -m "refactor(treetn): remove dead site_probe function family"
```

---

## Task 5: Remove the unreachable-and-dangerous `.or_else` fallback

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:564`
  (`directed_messages`)
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:626`
  (`directed_messages_batched`)
- Read: `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs:708`
  (existing regression coverage, confirmed sufficient in Step 1 — not
  modified)

**Interfaces:** None — internal function, no public API change. Error
message text changes are not part of any documented public contract.

Both `directed_messages` and `directed_messages_batched` look up a message
with `.get(&(neighbor.clone(), parent.clone()))
.or_else(|| messages.get(&(parent.clone(), neighbor.clone())))
.ok_or_else(|| anyhow::anyhow!(...))`. The audit (spec finding F-1, WS-tree-probe
section) proved the primary `.get(...)` always succeeds given the two-pass
postorder/reverse-postorder traversal (D-1 step 2), so the `.or_else` arm
is dead — and if it were ever reached (e.g. a future refactor breaks the
traversal invariant), it would silently substitute a message flowing in
the *opposite* direction, producing a wrong-but-shape-compatible result
instead of the existing `.ok_or_else` error. Fix: delete the `.or_else`
arm so a broken invariant fails loudly via the existing error path instead
of failing silently.

- [ ] **Step 1: Confirm the existing regression coverage, don't duplicate it**

You cannot directly unit-test "the `.or_else` never fires" without
controlling the internal `messages` map — the honest regression lock is a
test that exercises the postorder/reverse-postorder traversal on an
interior center, so that if a future change ever broke the ordering
invariant, the dense-oracle comparison would catch the resulting wrong
answer (silent substitution produces a shape-compatible but numerically
wrong result). This test already exists:
`src_fixed_matches_naive_on_a_branched_tree_when_probe_cap_is_full`
(`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs:708`) uses
`make_branched_pair()`, whose node `"C"` is the branch/hub (interior, not a
leaf — confirmed by `"C"` being the center-node argument in the
construction that starts the `names` list), and compares the SRC result
against `tn_a.contract_naive(&tn_b)`. Run it now to confirm it passes and
genuinely exercises the interior-center path:

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml src_fixed_matches_naive_on_a_branched_tree_when_probe_cap_is_full -- --nocapture
```

Expected: PASS. Do not add a new test for this — Step 2's fix is validated
by this existing test continuing to pass, since the fix is behavior-neutral
on any traversal ordering that isn't already broken.

- [ ] **Step 2: Delete the `.or_else` arm in both functions**

In `directed_messages` (around line 564), change:

```rust
let message = messages
    .get(&(neighbor.clone(), parent.clone()))
    .or_else(|| messages.get(&(parent.clone(), neighbor.clone())))
    .ok_or_else(|| {
        anyhow::anyhow!(
            "contract_src: side message is missing for {:?} around {:?}",
            neighbor,
            parent
        )
    })?;
```

to:

```rust
let message = messages
    .get(&(neighbor.clone(), parent.clone()))
    .ok_or_else(|| {
        anyhow::anyhow!(
            "contract_src: side message is missing for {:?} around {:?}",
            neighbor,
            parent
        )
    })?;
```

Apply the identical change in `directed_messages_batched` (around line
626), preserving that function's own error message text
("side batched message is missing...").

- [ ] **Step 3: Run the full test suite, including the interior-center regression test from Step 1**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml
```

Expected: all pass, including every existing SRC/tree test — this change
should be behavior-neutral for every currently-passing case, by the
audit's own proof that the arm was dead.

- [ ] **Step 4: Format, lint, commit**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs
git commit -m "fix(treetn): remove unreachable wrong-direction message fallback in directed_messages"
```

---

## Task 6: Add the adaptive dense-oracle test and a genuine isometry check

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`

**Interfaces:**
- Consumes: `contract`, `ContractionOptions::src().with_src_options(SrcOptions::adaptive(...))`,
  `TreeTN::contract_naive`, `IdxTensor::distance`/`norm` (all existing,
  used elsewhere in this same test file — follow the established pattern,
  e.g. the fixed-rank tests the audit's WS-tests section already
  catalogued).

The audit (WS-tests §5c) found `validate_ortho_consistency` — the helper
the existing tests use to check canonical/isometric structure — only
checks connectivity/direction *metadata*, never actual tensor values, so
no test in the suite currently proves a result tensor is numerically
unitary/isometric. Separately, the existing dense-oracle comparisons
(`< contract_naive`) cover fixed-rank SRC but not adaptive-rank SRC.

- [ ] **Step 1: Write the failing adaptive dense-oracle test**

Add to `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`,
mirroring the existing fixed-rank test `src_fixed_matches_exact_contraction_when_probe_cap_is_full`
(around line 518-537), which uses the `make_three_node_chain_pair()`
fixture (nodes `"A"`-`"B"`-`"C"`, every bond `dim(2)`), endpoint center
`"C"`, and the `actual.sub(&expected).unwrap().maxabs().unwrap()` residual
pattern already established in this file — use the exact same pattern,
only swapping `SrcOptions::fixed()` for `SrcOptions::adaptive()`:

```rust
#[test]
fn src_adaptive_matches_exact_contraction_on_a_small_chain() {
    let (tn_a, tn_b) = make_three_node_chain_pair();
    let expected = tn_a.contract_naive(&tn_b).unwrap();
    let options = ContractionOptions::src().with_max_bond_dim(4).with_src_options(
        SrcOptions::adaptive(1.0e-8, 4)
            .with_min_rank(1)
            .with_rank_increment(1)
            .with_seed(123)
            .with_final_svd(false),
    );
    let actual = contract(&tn_a, &tn_b, &"C".to_string(), options)
        .unwrap()
        .to_dense()
        .unwrap();

    let error = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(error < 1.0e-8, "adaptive SRC residual is {error}");
}
```

- [ ] **Step 2: Run it to verify it fails only if genuinely broken**

Run: `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml src_adaptive_matches_exact_contraction_on_a_small_chain -- --nocapture`
Expected: PASS. Adaptive SRC's correctness is not in question (the audit
independently re-derived and confirmed it); this test closes a coverage
gap, not a bug fix. If it fails, stop and investigate before continuing —
that would be a new, unexpected finding.

- [ ] **Step 3: Write the genuine isometry check**

Add a second test that, after an SRC contraction (fixed-rank is fine),
verifies the actual numerical isometry property of a non-root result
tensor: reshape it (via whatever `IdxTensor` method exposes matricization
along its canonical-center axis — check `to_dense`/existing matricization
helpers used elsewhere in the crate) into a `bond x (other indices)`
matrix `M` and assert `M^† M` is close to the identity (`< 1e-8` per
entry, or a Frobenius-norm bound consistent with this file's existing
tolerance conventions). Name it something like
`src_result_tensor_is_numerically_isometric`.

- [ ] **Step 4: Run it, iterate on the matricization helper if needed**

Run: `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml src_result_tensor_is_numerically_isometric -- --nocapture`
Expected: PASS. If no existing helper exposes the right matricization,
check `IdxTensor`'s public API (`crates/tensor4all-core/src/defaults/idx_tensor.rs`)
for the closest primitive (e.g. a reshape/unfold method) rather than adding
new production code for a test-only need.

- [ ] **Step 5: Run the full crate test suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml
```

- [ ] **Step 6: Format, lint, commit**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "test(treetn): add adaptive SRC dense-oracle test and a numerical isometry check"
```

---

## Task 7: Measure F-4/F-5 before touching `src_tree.rs`'s tree-path performance

**Files:**
- Read: `crates/tensor4all-treetn/examples/benchmark_src.rs` (check if
  still present in the worktree — it was uncommitted scratch from an
  earlier session; if absent, recreate a minimal timing harness, or reuse
  whatever benchmark example this PR's own commit `2395ec5` ("bench(treetn):
  add deterministic SRC comparison") added)
- Read: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:314-398`
  (`ensure_width`, `batch` — the functions F-4/F-5 concern)
- Create (conditionally): `docs/worklogs/2026-08-29-src-tree-path-performance.md`
  (if measurement does not justify a code change)
- Modify (conditionally): `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`
  (only if measurement justifies it)

**Interfaces:** Unknown until Step 1's measurement determines whether this
task produces a code change at all.

The spec is explicit: "**Measure first:** both are established by
index-counting arguments from a single workstream (WS-tree-probe), not
profiled measurements, so confirm the actual cost before spending effort
fixing them." Do not implement a fix before this step produces numbers.

- [ ] **Step 1: Build and run a profiling comparison**

```bash
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml --example benchmark_src
RAYON_NUM_THREADS=1 T4A_BENCH_CENTER=<an interior node name for the fixture> \
  ./target/release/examples/benchmark_src 10 8 5 mpo-mpo 3 false
```

Run it once with an endpoint center (default, no `T4A_BENCH_CENTER`) and
once with an interior center, at at least two bond dimensions (e.g. 4 and
8), for `mpo-mpo` mode (the case that exercises the general tree path
`ensure_width`/`batch` rather than the chain-specialized fast path).
Record the `src-adaptive` and `src-fixed` per-run timings for each
configuration.

- [ ] **Step 2: Decide whether the measured cost justifies the fix**

If the interior-center `mpo-mpo` case shows a clear (multiple-times, not
noise-level) slowdown attributable to `ensure_width`/`batch`'s repeated
local-pair materialization (F-4) or the width-keyed cache's behavior (F-5)
specifically — not just general adaptive-vs-fixed overhead, which the
audit already explained separately — proceed to Step 3. If the measured
cost is modest, or dominated by something else (e.g. the general
wrapper/planner overhead an earlier session's benchmark comparison against
the Python reference already identified as the dominant cost at these
problem sizes), stop here and go to Step 3-alt.

- [ ] **Step 3: Implement the fix, if justified**

Follow the direction the spec names: "F-4's fix direction is already
demonstrated by the chain path's own
`contract_prefix_with_probed_site_pair_batch_range`" — read that function
in `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` and
`src_probe.rs` for the pattern it uses to avoid repeated local-pair
materialization, and adapt the equivalent tree-path functions
(`ensure_width`/`batch` in `src_tree.rs`) to the same pattern. Write a
regression test comparing before/after timing at a fixed problem size
(not asserting an exact number — assert the new path produces the same
numeric result as `contract_naive`, and separately record the timing
improvement in the commit message, not as a test assertion, since CI
hardware timing is not deterministic enough to gate on).

- [ ] **Step 3-alt: Document the finding instead, if not justified**

Write `docs/worklogs/2026-08-29-src-tree-path-performance.md`, matching the
style of `docs/worklogs/2026-08-29-backend-incremental-qr-refactor.md`
(context/sources, measurement methodology, the actual numbers from Step 1,
and the conclusion — F-4/F-5 are real code-quality observations but not
the dominant cost at measured problem sizes, so no code change is made in
this task). This is a legitimate, complete deliverable for this task — do
not force a code change to have "done something."

- [ ] **Step 4: If Step 3 ran, verify and commit**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
git add crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "perf(treetn): reduce redundant local-pair materialization in tree-path SRC"
```

- [ ] **Step 4-alt: If Step 3-alt ran, commit the worklog**

```bash
git add docs/worklogs/2026-08-29-src-tree-path-performance.md
git commit -m "docs(treetn): record tree-path SRC performance measurement (no fix warranted)"
```
