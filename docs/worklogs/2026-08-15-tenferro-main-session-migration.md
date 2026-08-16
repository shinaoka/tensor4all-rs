> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Tenferro Main Session-API Migration Work Log

## Scope

Issue [#623](https://github.com/tensor4all/tensor4all-rs/issues/623), implementation-order item 2 only:

- update all tenferro pins to final merged revision `b5a106be3133979d78832a0ca3f4d6b57613b3d7`;
- migrate existing CPU internal calls to tenferro #1680's final receiver-first session APIs;
- preserve tensor4all's public backend-implicit eager API.

CUDA, placement-aware session routing, public explicit sessions, and tracing are deferred to later reviewed phases.

## Documents and references

- `docs/design/tenferro-main-session-migration.md`
- `docs/design/index.md`
- tensor4all-rs #623
- tenferro-rs #1680 and #1673
- repository and shared Rust/performance/test rules referenced by `AGENTS.md`

## Frontier review gates

### Pre-implementation design review

- Reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Date: 2026-08-15
- Reviewed artifact: `docs/design/tenferro-main-session-migration.md` and its `docs/design/index.md` entry
- Verdict: **Correct-to-merge**
- Blocking findings: none
- Confirmed: exact synchronized pin, mechanical scope, receiver-first session APIs, no nested session entry, AD/graph/einsum preservation, and verification coverage

### Post-implementation diff review

- Round 1 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed findings: mixed-dtype non-contiguous borrowed einsum promotion; unconditional dense operand copies in ordinary contraction; engine reset retaining the backend buffer pool; panic-based simplett host extraction on fallible paths; lost `Clone` on public linalg result types.
- Round 2 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed finding: ordinary and batched matrix multiplication still called the
  low-level session-receiver `TensorDot::dot_general`. Rank-2 multiplication now
  uses receiver-first `TensorSessionOpsExt::matmul`; batched multiplication uses
  one compiled einsum execution because the receiver-first surface has no
  arbitrary batched-dot method.
- Round 3 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed finding: the new public `From<tenferro_tensor::Error> for BridgeError`
  implementation expanded the downstream-visible conversion surface. It was
  removed; native constructor errors now pass through a private local helper.
- Round 4 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Correct-to-merge**
- Initial CI then exposed stricter repository gates not included in the local
  command set: `clippy::missing_panics_doc` on simplett's infallible export,
  and the compiler-backed panic audit after source-line migration. The public
  docs now state the host-invariant panic contract. Raw invariant
  `panic!`/`unreachable!` sites were consolidated behind private validated-state
  assertion helpers, while the existing public-assertion baseline was relocated
  without adding entries (23 before and after). The route review failure comes
  from the trusted base script parsing root `[workspace.dependencies]`; this PR
  contains the parser regression fix, so the maintainer waiver is required once
  for this same PR and deterministic checks remain active in all other gates.
- Round 5 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed findings: added checked, typed-error `try_from_fn` / `try_from_elem`
  constructors while retaining the existing infallible convenience signatures;
  removed safe mutable access to simplett's wrapped tenferro tensor so the
  host/rank invariant is enforceable; made typed linalg result fields private,
  validated host ownership at construction, and added read accessors plus
  consuming `into_parts` methods so `Clone` is valid for every publicly
  constructible result.
- Round 6 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Correct-to-merge**
- The second CI run passed lint, tests, doctests, panic audit, and route review,
  then found two remaining repository gates: the new fallible constructors had
  non-specific `# Errors` wording, and coverage fell in the two files that
  gained invariant/error plumbing. The docs now name
  `TensorTrainError::InvalidOperation`; focused tests cover both branches of
  `require_invariant` and the shared inner-product read-error mapper, without
  changing coverage thresholds.
- Round 7 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed finding: non-overflowing but unallocatable shapes could still panic in
  the new fallible simplett constructors. Both now call `try_reserve_exact`, map
  capacity failures to `TensorTrainError::InvalidOperation`, and test
  `[usize::MAX, 1]` without evaluating the element closure.
- Round 8 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Findings**
- Fixed finding: the retained infallible constructor docs now list allocation
  failure and tenferro shape/data rejection in addition to shape overflow.
- Round 9 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Correct-to-merge**
- The third CI run passed every gate except coverage generation, where the
  unrelated `tensor_train_accessor` test randomly selected only the function's
  zero-valued point and rejected all initial pivots before coverage JSON was
  produced. Its accessor-only fixture is now strictly positive, removing that
  random invalid initialization; the focused test passed 50 consecutive runs.
- Round 10 reviewer: `reviewer-gpt` (GPT-5.6 Sol)
- Verdict: **Correct-to-merge**

## Implementation

Implemented in the dedicated issue #623 worktree against the synchronized tenferro revision.

The implementation files are:

- `Cargo.toml`
- `crates/tensor4all-core/src/` conversion, contraction, factorization, QR, SVD, and structured-contraction call sites (including SVD tests)
- `crates/tensor4all-simplett/src/` tensor, type, einsum, and MPO call sites (including naive-contraction tests)
- `crates/tensor4all-tensorbackend/src/` backend, context, matrix, scalar, tensor-element, and tenferro-bridge call sites (including backend, matrix, scalar, and bridge tests)
- `crates/tensor4all-tcicore/benches/dense_vs_tenferro.rs`
- `crates/tensor4all-tcicore/benches/rrlu_bench.rs`
- `benchmarks/rust/benchmark_tt_ops.rs`
- `crates/tensor4all-quanticstci/tests/qft_2d_test.rs` and `docs/book/src/guides/qft.md` (canonical quantics-grid decoding)
- `crates/tensor4all-core/tests/tensor_native_ad.rs` (tracked mixed real-coefficient/complex-tensor `axpby` regression)
- `crates/tensor4all-treetn/tests/gse.rs` (differentiate an expanded-bond output row and verify every original state/reference leaf)
- migrated rustdoc examples in `tensor4all-simplett` and `tensor4all-tensorbackend` now propagate tenferro constructor/host-access errors
- `scripts/repository-rules-review.py` and its tests: workspace dependency declarations are no longer misclassified as new direct crate dependencies, allowing synchronized pin-only updates while preserving the downstream routing check
- `docs/design/index.md` and `docs/design/tenferro-main-session-migration.md`

The final localized benchmark pass made eager tensor construction, output
snapshot reads, scalar reads, dot-general configurations, and CPU runtime setup
fallible without changing the measured operation sequence or snapshot behavior.

Tracked mixed-dtype `IdxTensor::scale` / `axpby` now stay on tenferro's eager
AD path through explicit casts and eager operations. The former
`duplicate_value()` plus native arithmetic fallback detached tracked graphs and
was removed rather than retained as a compatibility workaround.

Review fixes moved mixed-dtype promotion into the compiled native-einsum graph,
so non-contiguous borrowed views remain borrowed through execution and convert
inside the same runtime program. Ordinary untracked N-ary contraction now uses
`NativeTensorReadInput` instead of duplicating every operand; tracked N-ary
contraction remains on eager einsum for both same and mixed dtypes. Engine reset
now drops the old runtime and then clears its backend pool. Fallible simplett
host extraction propagates typed helper errors, while its fixed-rank public
wrapper validates and documents the host-backed invariant. Manual `Clone`
implementations preserve the existing public linalg result contract. Rank-2
matrix multiplication now uses receiver-first `TensorSessionOpsExt::matmul`.
The batched path uses the existing compiled einsum runtime instead of the
low-level `TensorDot` SPI, preserving one backend execution without introducing
a public or local duplicate batched-dot vocabulary.

## Verification

- RED migration baseline: `cargo check --workspace --all-targets` failed with
  the expected removed-API call sites, including 9 final benchmark errors.
- RED final-pin baseline: the unfiltered release workspace suite failed only
  `tracked_negative_real_analytic_ops_match_principal_complex_values_and_backward`
  with a missing temporary-input metadata registration. The minimal upstream
  reproducer became tenferro #1700.
- `cargo fmt --all` and `cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- `cargo check --workspace --all-targets` — passed.
- `cargo clippy --workspace --all-targets -- -D warnings` — passed.
- `cargo test --release -p tensor4all-core --test tensor_any_scalar
  tracked_negative_real_analytic_ops_match_principal_complex_values_and_backward`
  — passed.
- `cargo test --release -p tensor4all-core --test tensor_native_ad
  tracked_complex_axpby_with_real_coefficients_preserves_gradients` — passed.
- `cargo test --release -p tensor4all-treetn --test gse
  global_subspace_expand_preserves_ad_tracking_through_local_density_path` — passed.
- `cargo test --release -p tensor4all-tensorbackend
  einsum_native_tensor_reads_promotes_non_contiguous_borrowed_view` — passed.
- `cargo test --release -p tensor4all-tensorbackend
  reset_default_engine_releases_retained_backend_buffers` — passed.
- `cargo test --release -p tensor4all-core --test tensor_contraction` — passed:
  **28 passed**.
- `cargo test --release -p tensor4all-simplett` — passed: **298 passed**.
- `cargo test --release -p tensor4all-tensorbackend` — passed: **320 passed,
  2 ignored**.
- `cargo test --release -p tensor4all-tensorbackend
  matrix::tests::test_mat_mul` — passed: **2 passed**.
- `cargo test --release -p tensor4all-tensorbackend
  matrix::tests::batched_mat_mul_same_shape` — passed: **2 passed**.
- `CARGO_BUILD_JOBS=4 cargo nextest run --release --workspace --no-fail-fast`
  — passed: **2804 passed, 14 skipped, 0 failed**.
- `cargo test --doc --release --workspace` — passed: **867 passed**.
- `cargo doc --workspace --no-deps` — passed (pre-existing rustdoc warnings remain).
- `./scripts/test-mdbook.sh` — passed after synchronizing the 2D QFT guide's
  quantics decoding with the canonical inherent-grid API.
- `python3 scripts/test-repository-rules-review.py` — passed: **90 tests**.
- `cargo clippy --workspace --all-targets -- -D warnings -D
  clippy::missing_errors_doc -D clippy::missing_panics_doc` — passed.
- `python3 scripts/audit-library-panics.py` — passed: **0 unbaselined,
  0 stale**.
- `python3 scripts/check-public-error-docs.py` — passed.
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run`
  — passed with no findings.

## Follow-up test correction

The native einsum cache regression test was updated for tenferro #1680: it now
checks the process-global runtime's bounded prepared-plan cache via
`Runtime::cache_stats()?.prepared_plans.entries`, rather than the unrelated
compiler extension cache. The helper and test names now identify the
prepared-plan cache.

Verification:

- RED baseline: the old assertion failed because the compiler cache had no
  entries after native einsum execution.
- `cargo fmt --all` — passed.
- `CARGO_BUILD_JOBS=4 cargo nextest run --release -p tensor4all-tensorbackend` —
  passed: 182 tests passed, 2 skipped.

## Migration regressions and upstream fixes

Validation found and corrected two stale downstream assumptions:

- Real-input/complex-output eager AD projects cotangents back to the real leaf
  dtype. The scalar test checks a zero default-seed cotangent and a nonzero
  `imag_part()` derivative, matching tenferro's Hermitian real-inner-product
  convention.
- The two-dimensional QFT test and guide decoded interleaved quantics
  coordinates with reversed bit significance during TreeTCI materialization.
  Both now use the inherent grid's canonical `quantics_to_grididx` conversion;
  no tolerance changed.

Three upstream eager-AD defects were fixed rather than hidden downstream:

- [tenferro-rs #1692](https://github.com/tensor4all/tenferro-rs/issues/1692),
  merged by [PR #1696](https://github.com/tensor4all/tenferro-rs/pull/1696)
  as `dd9c8a742aa7bf889382a315ff0a1eb2ac6fdd39`: eager concatenate/stack now
  records exact semantic input shapes before deferred AD replay.
- [tenferro-rs #1698](https://github.com/tensor4all/tenferro-rs/issues/1698),
  merged by [PR #1699](https://github.com/tensor4all/tenferro-rs/pull/1699)
  as `8a8196a95363158f147b1feff2bc3b2d4bc4d267`: deferred eager semantics now
  mirror execution-time mixed-dtype promotion, including the mixed
  `Add -> Eigh` regression.
- [tenferro-rs #1700](https://github.com/tensor4all/tenferro-rs/issues/1700),
  merged by [PR #1701](https://github.com/tensor4all/tenferro-rs/pull/1701)
  as `b5a106be3133979d78832a0ca3f4d6b57613b3d7`: raw eager carriers retain
  metadata scopes introduced by temporary promotion/exactification helpers.

[tenferro-rs #1693](https://github.com/tensor4all/tenferro-rs/issues/1693)
tracks Faer's non-scale-invariant full-pivot LU singularity threshold
separately. It is not a migration blocker: the unit-scale input that exposed it
came from the corrected downstream QFT decoding, and the final migration suite
is green without a tolerance or solver change.

## Coverage impact

All removed API call paths were replaced through the same public tensorbackend
and high-level wrappers. Existing coverage remains, and new regressions cover
mixed real/complex `axpby`, negative-real analytic AD, the GSE local-density
path through an expanded bond row, canonical 2D QFT decoding, mixed-dtype
non-contiguous borrowed einsum, and backend-pool release on engine reset. No
test was removed, skipped, or weakened, and no coverage threshold changed.

## Remaining risks

The exact final pin passes every local migration gate. The only remaining gate
for this implementation item is the pending full-diff `reviewer-gpt` verdict.
Hosted CI remains authoritative for workspace coverage and platform-specific
jobs.

Later issue #623 work remains intentionally separate: the placement-aware
session seam, CUDA routing, public explicit-session APIs, and tracing phases.
