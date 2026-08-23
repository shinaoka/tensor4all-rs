# TreeACI audit and remediation

## Scope

This pass audited the complete production surface of `tensor4all-treeaci` for
correctness failures, unbounded retention, redundant work, pathological
scaling, stale code, and misleading documentation. The starting point was
commit `9c4508b`, which already made guard input evaluators persistent across
scans and reused directed messages.

The audit used the generated API inventory first, then reviewed every
production module and its related tests. A forced dead-code build and targeted
release tests supplemented the source review.

## Findings and changes

### Guard scans and global-pivot injection

- A floating-zone callback expanded the same logical points once for input
  evaluators and again for the output evaluator. It now expands once and shares
  the column-major coordinates.
- `SampleArena::inject_global_point_impl` cloned all retained component records
  for every proposed point despite already having an append-only
  checkpoint/rollback journal. Injection now mutates through that journal and
  rolls back both failures and no-op projections.
- Global injection rebuilt every input frame from scratch after growth. It now
  calls `InputFrameStore::extend`, preserving unchanged `Rc` frames and
  computing only newly interned samples.
- Guard working-byte checks covered only the final input-value buffer. They now
  account for the simultaneously retained expanded coordinates, input values,
  target and approximation buffers, and evaluator/error transient.
- The message cache was unbounded by default and its option applied once per
  evaluator. `message_cache_max_bytes` is now a 256 MiB aggregate guard budget,
  divided across every input evaluator and the output evaluator. A zero budget
  still disables retention without disabling Guard or changing results.

### Frame storage and local updates

- Directed frames stored a column-major `sample_count x bond_dim` matrix, so
  reading one sample copied a strided row. They now retain sample-major flat
  buffers, matching how contractions produce and consume frame vectors.
- The redundant `sample_ids = 0..sample_count` vector was removed.
- `max_frame_bytes` was checked against mandatory directed frames when built and
  against candidate frames when inserted, but an arena extension could grow
  mandatory frames while an older shared candidate cache remained. The
  combined retained payload could therefore exceed the documented limit.
  Extension now reclaims the optional candidate cache before publishing such a
  store, and diagnostics include both frame families.
- Local updates retained candidate frames for all inputs simultaneously. They
  now process one input at a time.
- Packing row candidates built a matrix and called a backend transpose. The
  row matrix is now written directly in column-major layout.
- Production cloned the entire sampled local matrix only to retain a test-only
  field. The clone and field now exist only in tests; production moves the
  buffer into LUCI.
- The local working-set estimate now matches streamed frames and coexisting BLAS
  buffers, and duplicate local resource-limit code was removed.

### Initialization and wrappers

- Random-output construction searched every prepared edge for every bond axis.
  It now builds one bond-replacement map and performs direct lookups.
- Algebraic cut dimensions previously walked every directed component
  recursively for every edge. They now use one dependency-ordered dynamic
  program. The remaining component-node enumeration is iterative, avoiding
  recursion depth failures on long chains.
- Scalar `tree_elementwise` and `hadamard_many` performed checked batch indexing
  for every input at every point; the already validated column-major slice is
  now consumed in contiguous point chunks.
- The single-site path charged only its input buffer even though the output or
  conversion buffer coexisted with it. Its working-byte check now includes both.

### Cleanup and stale claims

- Blanket production `allow(dead_code)` attributes and genuinely unused fields,
  methods, and parameters were removed or restricted to tests.
- The old guard “evaluation counter” was removed: it counted only initially
  cold evaluator instances, not cache work, so its performance test did not
  establish the property its name claimed.
- Cache architecture, branch batching, and maturity documentation now describe
  the current implementation.
- The chain parity benchmark no longer carries the stale unseeded workaround;
  both arms use the same first input as their initial guess.

## Second audit pass

The follow-up pass concentrated on allocation peaks, cache accounting, error
atomicity, partial-cut Guard updates, and standalone crate configurations.

- Global-pivot padding replaced every output bond and rebuilt every core even
  when Guard activated only one cut. It now creates replacement indices only
  for growing cuts and rebuilds only their incident cores. Inactive bond
  identities remain unchanged.
- Masked global injection projected a point onto every directed cut before
  consulting the mask. It now projects only active cuts and the recursive
  dependencies needed to materialize them.
- Padding's working-set estimate counted destination cores but omitted the
  source buffer materialized by `to_vec`. The peak now includes the largest
  simultaneously live source core.
- Candidate-cache byte accounting charged duplicate candidates more than once
  when equal keys appeared in one batched call. A single-entry insertion path
  now hashes, stores, and charges each key once.
- Single- and two-incoming frame batching allocated core, input, stage-one, and
  Cartesian output matrices without enforcing `max_working_bytes`. Both frame
  construction and candidate contraction now check the actual per-group peak
  before allocating. Local updates add this scratch to their other live
  buffers. The normal local-update path derives exact group sizes in O(1) from
  the complete Cartesian candidate-set invariant, avoiding a second candidate
  regrouping pass solely for resource accounting.
- `TreeTN::canonicalize_mut` used `mem::take` and left the caller with an empty
  default network on any validation, factorization, or backend error. It now
  restores a pre-operation snapshot on failure.
- `tensor4all-treeaci`'s default and provider-injection feature sets selected a
  numerical provider without enabling the legacy backend surface that its
  current dependencies use. Standalone package builds now enable that surface
  explicitly instead of relying on unrelated workspace feature unification.

## Numerical issue review

Open issue #666 reports rejection of a full-rank chain initial guess. A direct
16-site, local-dimension-2, bond-dimension-128 regression using the benchmark's
deterministic fully mixed core construction succeeds on the current code and
preserves the dense tensor. Because no failing case was reproduced, this pass
does not add a speculative QR/SVD fallback that could mask a real numerical
failure. The acceptance regression remains to prevent recurrence.

## Validation

All commands below completed successfully unless noted:

- `cargo fmt --all -- --check`
- `cargo test --release -p tensor4all-treeaci --features tensor4all-tensorbackend/global-defaults --no-fail-fast`
  - 119 unit tests passed, one opt-in performance test ignored
  - 8 integration tests passed
  - 18 doctests passed
- `cargo test --release -p tensor4all-aci --features tensor4all-tensorbackend/global-defaults --no-fail-fast`
  - 85 unit tests passed, one timing test ignored
  - 5 integration tests passed
  - 19 doctests passed
- `cargo clippy --release -p tensor4all-treeaci -p tensor4all-aci --all-targets --features tensor4all-tensorbackend/global-defaults -- -D warnings`
- `cargo doc --workspace --no-deps --features tensor4all-tensorbackend/global-defaults`
  completed; it reported only existing warnings in other crates.
- `cargo run -p xtask --release -- api-dump`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run`
- `python3 scripts/test-repository-rules-review.py` (90 tests passed)
- The opt-in branch-frame comparison measured 2.39 ms batched versus 31.08
  ms scalar (13.0x) on this host.
- The seeded `treeaci_parity` preflight completed for bond dimensions 16, 32,
  64, and 128. TreeACI converged in two sweeps in all four cases with dense
  relative errors from `9.99e-9` to `1.95e-8`. The filtered chi=16 timing was
  about 43.4 ms; Criterion compared it with a pre-existing local baseline and
  reported a 9.8% midpoint improvement.

Second-pass validation added:

- Standalone `cargo test --release -p tensor4all-treeaci`: 122 unit tests
  passed, one opt-in timing test ignored; eight integration tests and 18
  doctests passed.
- The full TreeTN release suite passed after the canonicalization fix: 453 unit
  tests, all integration tests, and 124 doctests.
- Both TreeACI default and `tenferro-provider-inject` standalone feature
  configurations passed `cargo check --release`.
- The final filtered chi=16 TreeACI timing was 44.48 ms on this host. An
  intermediate implementation that re-hashed every candidate for resource
  accounting measured 45.20-45.78 ms and was replaced; the final single-entry
  cache path and O(1) local scratch calculation recovered that regression.

## Third audit pass

The third pass re-read the public API inventory and every production module,
then used phase telemetry at chi=128 to distinguish setup, local proposals,
frame extension, LUCI, callback, and Guard costs.

- Guard evaluators were constructed even when Guard was disabled. They are now
  created lazily only when an eligible Guard scan actually runs.
- Directed sample dependency order was recomputed during every frame extension,
  and all-cut seed projection recursively walked the same subtrees once per
  cut. The order is now prepared once; all-cut projection is one iterative
  dependency pass, while single-cut bootstrap visits only its dependency
  subtree without recursion.
- Algebraic component dimensions rejected long valid chains when the exact
  physical-space product exceeded `usize`. Algebraic ceilings now saturate,
  and bootstrap products stop at the finite target rank they are used to test.
- Initial-guess scalar validation copied every core payload only to discard it.
  It now checks the scalar conversion contract with a representative value.
- Explicit guesses alone underwent a full-rank CI canonicalization before the
  first pass, although bootstrap uses only their validated ranks and the first
  pass replaces every core. They now follow the generated-output path: defer
  canonicalization until the updated, lower-rank first-pass result exists.
- Single-incoming frame kernels issued one small matrix multiplication per
  local physical coordinate. They now stack all physical slices and contract
  them in one backend multiplication during both frame construction and local
  candidate evaluation.
- Candidate-cache keys retained cloned variable-length incoming-ID vectors.
  Common leaf, chain, and trivalent cases now use compact fixed-size keys;
  higher-degree scalar fallbacks skip optional retention. Cache accounting now
  includes key bytes.
- Guard's nested start vectors and accumulated candidate points were absent
  from its working-memory estimate. They are charged before allocation and as
  candidates accumulate. Expanded local coordinates are bounds checked instead
  of silently wrapping through mixed-radix modulo.
- A Guard scan could offer several pivots after a cut was only one rank below
  its configured or algebraic limit. Injection now consumes a per-cut remaining
  capacity and disables each cut as soon as it is full.
- The one-node exact entry initialized the complete general TreeACI state and
  then prepared the same problem again. It now branches before state/bootstrap/
  frame construction while preserving explicit-guess validation.
- LUCI pivot indices are checked before indexing candidate arrays, global-pivot
  padding uses precomputed checked strides, and a redundant bond-index clone was
  removed.

The forced dead-code build found no production dead code. Experimental order,
skeleton, and validation modules remain `cfg(test)` reference oracles rather
than shipping in the library. Extended Clippy reported only soft function-length
warnings in TreeACI; it found no cognitive-complexity or redundant-clone warning.

### Third-pass performance and validation

- The isolated chi=128 Criterion midpoint moved from 102.69 ms to 92.15 ms
  after deferred initial-guess canonicalization, then to 75.15 ms after fused
  all-physical single-incoming contractions. Accuracy and evaluated-point counts
  were unchanged.
- The final full paired run reported midpoint TreeACI/SimpleTT ratios of 0.82,
  1.12, 0.83, and 1.01 at chi 16, 32, 64, and 128 respectively. The chi=16
  SimpleTT sample had two severe high outliers, so that ratio is indicative
  only. TreeACI dense relative errors were `9.99e-9` to `1.95e-8`.
- `RUSTFLAGS='-D dead_code' cargo check --release -p tensor4all-treeaci
  --all-targets --features tensor4all-tensorbackend/global-defaults` passed.
- Standard release Clippy with `-D warnings` passed for both changed ACI crates.
- The full TreeACI release suite passed with 131 unit tests, 7 public API tests,
  1 rank-scaling integration test, and 18 doctests; two opt-in timing tests were
  ignored by default.

`cargo nextest` is not installed in this environment, so the documented
nextest command could not run; the corresponding release-mode Cargo test
commands above were used for both changed crates.
