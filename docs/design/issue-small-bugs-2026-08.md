# Design: small bug fixes from bot issues (#622, #629, #638, #630, #615)

Status: approved for implementation by user (2026-08, review by luna).
Scope decision: #639 (Scalar trait move) and #616 (rename sweep) are structural
API work, excluded. Only the five concrete bot-reported bugs are in scope.

Review gate (luna, read-only):
- Pre-implementation design review 2026-08: verdict NEEDS-FIX — one finding:
  #638 right-side length check must be `indices.len() >= len` (the recursion
  reads `indices[n]` .. `indices[len-1]`, so `len - n` under-requires, e.g.
  len=3, n=1, two pairs: check passes but recursion panics). Design updated
  accordingly (see §3). All other findings APPROVED with test additions:
  #622 direct macro regression, #638 asymmetric dims + short right side with
  n>0, #615 ensure the hashed path itself is exercised.

## 1. #622 — tensorbackend unchecked shape product (panic on 2^64 wrap)

**Bug**: `tensor_element.rs` `dense_diagonal_values` computes
`dims.iter().product::<usize>()` unchecked; for `diag_len^logical_rank == 2^64`
this wraps to 0, allocates an empty buffer, and the diagonal-fill loop panics.
`dense_native_tensor_from_col_major` has the same unchecked product feeding an
`ensure!` length check.

**Fix**: add a crate-local `checked_product(dims: &[usize]) -> Result<usize>`
(try_fold with checked_mul, same shape as
`idx_tensor.rs::checked_product`), use it in both sites. Since the crate uses
`anyhow`, `checked_product(...)?` composes directly.

**Tests**: regression in `tensor_element.rs` tests (or the tenferro_bridge
integration test) with `data.len() = 2^16`, `logical_rank = 4` expecting a
typed error, plus a zero-size/empty edge case.

## 2. #629 — svd bond legs do not compose (public API only)

**Bug**: `svd_with` returns `U=[left..., bond]`, `S=[bond, bond.sim()]`,
`V=[right..., bond]`; `U·S·Vᴴ` fails (`S`'s second leg `sim` never meets `V`'s
`bond` leg).

**Fix (A案, approved)**: give `vh`/`V` the `sim` leg — `vh_indices =
vec![bond_index.sim()]` — matching the ITensors convention `S: [l, l'], V: l'`.
Reconstruction `U·S` contracts on `bond`, then contracts with `Vᴴ` on `sim`.

**Scope**: public `svd_with` only. Internal `svd_for_factorize` + `factorize.rs`
are left untouched (they work via an internal `replaceind` compensation); add a
one-line comment in `factorize.rs` documenting the internal sim-scaffolding
convention.

**Tests**: update `linalg_svd.rs` — flip the two `s.indices[0].id == v...id`
assertions to `s.indices[1].id == v...id`; remove the `replaceind` step from
the `reconstruct_from_svd` helper so the existing reconstruction tests exercise
plain `U·S·Vᴴ`; keep all existing reconstruction tests (they become the
no-workaround regression).

## 3. #638 — simplett Contraction evaluation panics on caller indices

**Bug**: `Contraction::{evaluate, evaluate_left, evaluate_right}` index into
tensors/slices with unvalidated caller-supplied `(i_k, j_k)` pairs and slice
lengths → Rust panic instead of `MPOError`.

**Fix**: mirror the sibling `MPO::evaluate` validation:
- private helper `validate_indices(&self, range)` checking every `(i_k, j_k)`
  in the range against `self.site_dims[k]` (`i_k < s1_a`, `j_k < s2_b`),
  returning `MPOError::IndexOutOfBounds { site, index, max }`;
- `evaluate`: validate all sites 0..len;
- `evaluate_left(n)`: add `indices.len() >= n` length check (InvalidOperation),
  validate sites 0..n;
- `evaluate_right(n)`: add `indices.len() >= len` length check (the recursion
  reads `indices[n]` .. `indices[len-1]`; `len - n` would under-require,
  per luna design review), validate sites n..len.

**Tests**: in `mpo/contraction.rs` tests — out-of-range pair values for both i
and j on asymmetric physical dims, short slice for `evaluate_left` (n=2 on a
1-element slice) and `evaluate_right` (short input with n>0), valid path still
works.

## 4. #630 — ACI default-path guard recovery has no regression test

**Bug (gap)**: the only guard-recovery assertion hardcodes
`nsearch_global_pivots = 30`; the default caller receives 5. lingrui96 measured
the default path and it recovers (seeds 0–5, machine-zero), so this is
test-only.

**Fix**: in `global_guard/tests.rs`, change the guard-on arm of
`global_guard_recovers_missed_near_degenerate_feature` to **not override**
`nsearch_global_pivots` (per lingrui96: assert the default by not setting it,
so a default change breaks the test). Implement via a `run_case` variant whose
options leave `nsearch_global_pivots` at `AciOptions::default()`. Keep the
guard-off arm as-is.

**Tests**: the modified test itself.

## 5. #615 — hashed common_ind_positions diverges for directed indices

**Bug**: `common_ind_positions_hashed` buckets `indices_b` by `IndexLike::Id`
and looks up `idx_a.id()`; the linear path matches on full `is_contractable`.
For a directed `IndexLike` whose `conj()` changes the id, a Ket matches a Bra
partner with a different id → hashed path silently drops the pair.

**Fix**: bucket by full index value and look up `conj(idx_a)`:
- `positions_by_value: HashMap<I, SmallVec<[usize; 2]>>` keyed by `idx_b`
  (owned; `IndexLike: Clone + Eq + Hash`);
- for each `idx_a`, probe candidates under key `idx_a.conj()` with
  `is_contractable`.

Equivalence to the linear path: linear accepts exactly `b == conj(a)`
(undirected: `conj(a) == a`; directed: `conj(a) == b` by `is_contractable`).
This deviates from the issue's suggested "key by full value + probe same-value
candidates", which would still drop directed pairs (Ket vs Bra differ as full
values); the conj-lookup form is the exact linear equivalent.

**Tests**: add a minimal directed test `IndexLike` impl (Ket/Bra, `conj()`
switches state **and** id — this is the case where id-keying fails) in the
`index_ops` test module, calling `common_ind_positions_hashed` directly (it is
private but same-module tests can reach it) and asserting it equals
`common_ind_positions_linear` on directed pairs (per luna: ensure the hashed
path itself is exercised, not only the dispatch); plus a same-ID/different-tags
case and a dispatch-level case with scan_work > 64 forcing the hashed path.

## Verification

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release -p tensor4all-core -p tensor4all-tensorbackend -p tensor4all-simplett -p tensor4all-aci` (+ tests dirs)
- `cargo doc --workspace --no-deps`
- Review gate: luna (read-only) reviews design (pre) and diff (post).

## Review verdicts

- Design (pre-implementation), luna: NEEDS-FIX → fixed (#638 right length
  check) → re-approved per findings; all other sections APPROVED.
- Diff (post-implementation), luna: TBD (record verdict here).
