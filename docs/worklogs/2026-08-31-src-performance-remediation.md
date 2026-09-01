# SRC Tree Message-First Performance Remediation

## Summary

The general-tree SRC path materialized a probed local product before it
contracted incoming branch messages. At a degree-`d` node with operand bond
dimension `chi`, that temporary retained both operands' bonds and scaled as
`batch_width * chi^(2d)`. The implementation now keeps the factorized probe
tensors separate and gives raw A, raw B, probes, and incoming messages to one
planner-visible retained-index contraction.

The primary performance gate is **PASS**. On the measured four-leaf star at
bond dimension 8, fixed SRC improved from 2.253 seconds to 8.625 milliseconds
per run, adaptive SRC improved from 2.211 seconds to 10.162 milliseconds, and
peak process RSS fell from 1.07 GiB to 30.3 MiB. The complete TreeTN release
test suite and doctests pass without changing a test tolerance.

On the 10-site chain gate, aligning adaptive cache segments removed another
7.5--10.6%, then routing statically binary contractions through the existing
pairwise fast path removed 24.9--30.9% from adaptive SRC and 41.4--46.5% from
fixed SRC. Fixed SRC is now faster than zip-up at measured bond dimensions 4,
8, and 16; at bond dimension 32 it remains about 30% slower.

## Scope and sources read

- Baseline commit: `2daf5e368bbecb812a06084c9034e7f12fbe432a`.
- Candidate: uncommitted worktree diff based on that commit.
- `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`.
- `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs`.
- `crates/tensor4all-treetn/src/treetn/contraction.rs` zip-up implementation.
- `crates/tensor4all-treetn/examples/benchmark_src.rs`.
- `docs/worklogs/2026-08-29-src-tree-path-performance.md`.
- `docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md`.
- Python RandomMPOMPS reference commit
  `fe6ad494fc6f3605fc3963360f626d83f47bc2ce`, used for algorithm/data-flow
  comparison only because that checkout has no detected license file.

## Root cause

`EnvironmentCache::batch` and `grow_segment` formerly called
`probed_site_pair_batch_range` for every node. For the four-leaf bond-8 star,
the center's pre-paired tensor had shape
`[8, 8, 8, 8, 8, 8, 8, 8, 4]`: 67,108,864 `f64` values, or 512 MiB for one
temporary. The downward pass then contracted that object separately for each
leave-one-neighbor-out message.

Zip-up did not have this failure mode: it supplied local operands and child
remainders together, allowing the contraction planner to eliminate branch
bonds before joining the full local operands.

The new `probe_batch_tensors` helper constructs only the small factorized probe
tensors. `directed_messages_batched_from_factors` supplies those probes, raw
local operands, and incoming messages together. A numerical unit test compares
every new directed message with the old pre-paired reference on a deterministic
small network.

## Measurement protocol

- CPU: AMD Ryzen 9 6900HX under WSL2.
- Rust: `rustc 1.98.0 (88d9e12ae 2026-08-18)`.
- Release profile.
- `RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`.
- `T4A_BENCH_SKIP_EXACT=1` for timing; correctness was checked separately by
  dense-oracle tests.
- Deterministic seeds from `benchmark_src`: network seed 13, SRC seed 1234.
- Five repetitions for the recorded after cases. Baseline values came from the
  same host and command family immediately before the change.

## Results

### Degree scaling at bond dimension 4

| Star degree | Method | Before ms/run | After ms/run | Change |
|---:|---|---:|---:|---:|
| 3 | zip-up | 2.102 | 2.242 | reference |
| 3 | SRC fixed | 3.001 | 2.441 | -18.7% |
| 3 | SRC adaptive | 4.686 | 3.997 | -14.7% |
| 4 | zip-up | 2.605 | 2.593 | reference |
| 4 | SRC fixed | 9.935 | 3.863 | -61.1% |
| 4 | SRC adaptive | 8.870 | 5.873 | -33.8% |
| 5 | zip-up | 4.005 | 4.355 | reference |
| 5 | SRC fixed | 175.747 | 6.787 | -96.1% (25.9x) |
| 5 | SRC adaptive | 121.588 | 9.679 | -92.0% (12.6x) |

The old degree-four-to-five jump was 17.7x for fixed SRC and 13.7x for
adaptive SRC. After the change it is 1.76x and 1.65x respectively.

### Bond scaling on a four-leaf star

| Bond dimension | zip-up ms/run | SRC fixed ms/run | SRC adaptive ms/run |
|---:|---:|---:|---:|
| 4 | 2.593 | 3.863 | 5.873 |
| 6 | 3.173 | 4.625 | 7.103 |
| 8 | 6.348 | 8.625 | 10.162 |

Before the change, fixed SRC at bond dimensions 4, 6, and 8 took 11.8 ms,
250 ms, and 2.253 s, closely following the predicted `chi^8` temporary.

### Contraction profile and memory

Before the change the four dominant downward contractions each accepted the
pre-paired `[8; 8] + [4]` center tensor and took roughly 0.30--0.39 seconds.
After the change no profile signature contains that tensor. The planner sees
the two `[2, 2, 8, 8, 8, 8]` local operands, two `[2, 4]` probes, and three
`[8, 8, 4]` incoming messages together. Peak process RSS for the complete
three-algorithm harness fell from 1.07 GiB to 30.3 MiB.

### Adaptive chain segment alignment

The adaptive chain prefix cache previously let any site's narrower
`maximum_width` terminate a segment early. Later sites then repeatedly crossed
that ragged boundary and rebuilt each requested increment by selecting and
restacking individual columns. The cache now permits a narrow initial segment
but grows every later segment by the full `rank_increment`; at most
`rank_increment - 1` extra columns are computed when a site caps a request.

An immediate A/B comparison used 20 repetitions, a 10-site MPO--MPS chain,
`max_rank = 32`, `rank_increment = 3`, and the single-thread environment listed
above:

| Input bond | Ragged ms/run | Aligned ms/run | Change |
|---:|---:|---:|---:|
| 4 | 29.194 | 26.520 | -9.2% |
| 8 | 44.548 | 40.076 | -10.0% |
| 16 | 47.333 | 42.323 | -10.6% |
| 32 | 56.652 | 52.389 | -7.5% |

Exploratory runs without fixed BLAS/Rayon thread counts were discarded: on
these small contractions, thread-pool overhead slowed every algorithm by about
fivefold and made non-adjacent absolute timings misleading.

### Chain binary-contraction dispatch

The chain probe helpers also sent every two-tensor contraction through the
generic N-ary `TensorLike::contract` entry point. `IdxTensor` already overrides
`contract_pair` with a direct dot-general implementation, but SRC never reached
it. An adaptive bond-32 profile showed two repeated generic signatures alone
accounting for 175 calls per run. Replacing only statically binary calls with
the existing pairwise seam avoids repeated einsum planning without changing
the generic tensor abstraction.

The same 20-repetition A/B protocol produced:

| Input bond | Adaptive N-ary ms/run | Adaptive pairwise ms/run | Change |
|---:|---:|---:|---:|
| 4 | 26.520 | 19.922 | -24.9% |
| 8 | 40.076 | 27.690 | -30.9% |
| 16 | 42.323 | 29.764 | -29.7% |
| 32 | 52.389 | 38.322 | -26.9% |

Fixed SRC benefits even more because its contraction work is concentrated in
one batch:

| Input bond | Fixed N-ary ms/run | Fixed pairwise ms/run | Change |
|---:|---:|---:|---:|
| 4 | 13.343 | 7.144 | -46.5% |
| 8 | 17.828 | 9.922 | -44.3% |
| 16 | 21.806 | 12.482 | -42.8% |
| 32 | 39.856 | 23.347 | -41.4% |

The corresponding unchanged zip-up times were 11.383, 12.747, 14.701, and
17.955 ms/run respectively.

### Rust/Python chain benchmark rerun

The final Rust worktree and the local Python reference were rerun with
`n_sites=10`, physical dimension 2, `rank_increment=3`, `rtol/cutoff=1e-4`,
`min_rank/mindim=2`, and `max_rank/maxdim=bond_dim^2`. Rust used the release
`benchmark_src` harness with five measured repetitions through bond 32 and three
at bonds 64 and 128, with one fixed-thread process per algorithm. Python used
the reference checkout's C++ incremental-QR extension, one warm-up call, and
five measured repetitions. Input tensors were generated independently (Rust
`StdRng` real `f64`; Python NumPy real-valued tensors, with complex128 SRC
sketch intermediates), so
the correctness columns are each relative to that implementation's own exact
dense MPO--MPS product rather than cross-language element equality.

| Bond | Implementation/method | ms/run | output max bond | relative error |
|---:|---|---:|---:|---:|
| 4 | Rust zip-up | 10.776 | 16 | 9.745e-18 |
| 4 | Rust SRC fixed | 6.421 | 16 | 1.320e-18 |
| 4 | Rust SRC adaptive | 18.968 | 16 | 1.293e-18 |
| 4 | Python zip-up | 0.661 | 16 | 6.643e-15 |
| 4 | Python SRC fixed | 1.621 | 16 | 4.335e-15 |
| 4 | Python SRC adaptive (incremental) | 2.828 | 16 | 5.681e-15 |
| 8 | Rust zip-up | 12.218 | 32 | 1.361e-20 |
| 8 | Rust SRC fixed | 13.180 | 64 | 6.794e-22 |
| 8 | Rust SRC adaptive | 25.845 | 32 | 1.543e-21 |
| 8 | Python zip-up | 3.713 | 64 | 4.358e-15 |
| 8 | Python SRC fixed | 7.655 | 64 | 5.251e-15 |
| 8 | Python SRC adaptive (incremental) | 4.216 | 32 | 5.271e-15 |
| 16 | Rust zip-up | 15.832 | 32 | 4.710e-23 |
| 16 | Rust SRC fixed | 56.963 | 256 | 1.942e-24 |
| 16 | Rust SRC adaptive | 28.859 | 32 | 1.103e-23 |
| 16 | Python zip-up | 36.431 | 254 | 6.347e-5 |
| 16 | Python SRC fixed | 90.999 | 256 | 2.985e-15 |
| 16 | Python SRC adaptive (incremental) | 6.118 | 32 | 5.617e-15 |
| 32 | Rust zip-up | 19.297 | 32 | 7.610e-26 |
| 32 | Rust SRC fixed | 421.317 | 512 | 5.877e-27 |
| 32 | Rust SRC adaptive | 36.397 | 32 | 6.125e-26 |
| 32 | Python zip-up | 165.715 | 256 | 5.838e-15 |
| 32 | Python SRC fixed | 341.301 | 512 | 3.956e-15 |
| 32 | Python SRC adaptive (incremental) | 16.487 | 32 | 9.668e-15 |
| 64 | Rust zip-up | 32.027 | 32 | 6.804e-29 |
| 64 | Rust SRC fixed | 2163.576 | 512 | 1.074e-29 |
| 64 | Rust SRC adaptive | 86.799 | 32 | 9.551e-29 |
| 64 | Python zip-up | 313.214 | 256 | 6.256e-15 |
| 64 | Python SRC fixed | 993.502 | 512 | 2.420e-15 |
| 64 | Python SRC adaptive (incremental) | 50.467 | 32 | 9.213e-15 |
| 128 | Rust zip-up | 57.108 | 32 | 1.309e-31 |
| 128 | Rust SRC fixed | 16031.691 | 512 | 1.772e-32 |
| 128 | Rust SRC adaptive | 466.728 | 32 | 2.235e-31 |
| 128 | Python zip-up | 2686.134 | 256 | 6.045e-15 |
| 128 | Python SRC fixed | 4857.656 | 512 | 2.972e-15 |
| 128 | Python SRC adaptive (incremental) | 351.171 | 32 | 1.277e-14 |

The Python `src-old` (`random_contraction`) adaptive implementation was also
measured at 2.951, 4.663, 6.684, and 17.659 ms/run for bonds 4, 8, 16, and 32;
its corresponding relative errors were 4.271e-15, 3.722e-15, 5.225e-15, and
7.980e-15. Python has no general-tree SRC implementation in this reference
checkout, so no tree cross-language row is available.

For the bond-64 and bond-128 Python correctness runs, the exact oracle was
constructed by materializing the MPO and MPS separately to dense objects and
then multiplying them. This avoids allocating the uncompressed product MPS
whose intermediate bond would be `bond_dim^2`; it is mathematically the same
exact dense MPO--MPS result used for the error columns.

### High-bond complexity regression against issue #563

Issue #563 gives the simplified arithmetic cost
`O(n D chi chibar^2)` for both zip-up and SRC, where `D` is the MPO bond
dimension, `chi` is the MPS bond dimension, and `chibar` is the output bond
dimension. The earlier version of this worklog incorrectly treated `D` as the
fixed physical dimension. The paper states that the simplified expression
assumes `d = O(1)` and `D <= chi <= chibar`; the benchmark violates that
assumption once its common input bond exceeds the adaptive output bond 32.

The paper's full SRC count is
`O(n d D chi chibar (chi + chibar + d D))`. The benchmark varied both input
bonds together (`D = chi = b`) with `d = 2` and high-bond adaptive output
`chibar = 32`. Substitution gives a leading cubic dependence on `b`, not a
linear one. The paper's full zip-up count,
`O(n (d D chi chibar^2 + d^2 D^2 chi chibar))`, has the same leading cubic
dependence in this regime. For bond 64 to 128 the full SRC expression predicts
a 7.43x increase (local log2 slope 2.89). The corresponding profiled Rust
pairwise-contraction kernel time increased from 22.650 ms to 171.412 ms: 7.57x,
or slope 2.92. That kernel therefore follows the full paper count almost
exactly; it does not contain an additional asymptotic bond factor.

To remove the unrelated `max_rank = bond^2` fixed-rank stress configuration,
Rust and Python were also rerun with `chibar/max_rank = 32` at bonds 32, 64,
and 128:

| Implementation/method | bond 32 ms | bond 64 ms | bond 128 ms | local 64-to-128 slope |
|---|---:|---:|---:|---:|
| Rust zip-up | 19.601 | 28.881 | 56.723 | 0.97 |
| Rust SRC fixed | 24.871 | 105.534 | 638.834 | 2.60 |
| Rust SRC adaptive | 36.614 | 86.736 | 440.018 | 2.34 |
| Python zip-up | 13.418 | 78.543 | 489.767 | 2.64 |
| Python SRC fixed | 17.504 | 107.301 | 633.977 | 2.56 |
| Python SRC adaptive (incremental) | 11.358 | 56.101 | 348.851 | 2.64 |

Both SRC implementations trend toward the full formula's cubic high-bond
regime, and their fixed-rank bond-128 times are nearly identical (639 ms Rust,
634 ms Python). Rust adaptive is about 26% slower than Python at that point,
which is a constant-factor gap rather than evidence of a wrong asymptotic
complexity. Rust zip-up is substantially faster than the full-count trend in
this measured range; an upper operation-count bound does not require every
implementation/backend to attain that exponent.

The benchmark now reports the MPO and MPS bond dimensions separately and
accepts `T4A_BENCH_MPS_BOND_DIM`, preventing future regressions from silently
varying `D` and `chi` together while interpreting one as fixed.

The corrected benchmark was then used to vary one input bond at a time with
adaptive `chibar/max_rank = 32` (five repetitions):

| Sweep | MPO bond D | MPS bond chi | Rust adaptive ms/run |
|---|---:|---:|---:|
| vary D | 32 | 32 | 40.217 |
| vary D | 64 | 32 | 60.555 |
| vary D | 128 | 32 | 126.072 |
| vary chi | 32 | 32 | 39.782 |
| vary chi | 32 | 64 | 58.348 |
| vary chi | 32 | 128 | 99.289 |

Changing either input bond alone is much milder than changing both together.
This confirms that the apparent extra scaling came from the old benchmark's
coupled `D = chi = input_bond` axis, not from a Rust-only repeated-work loop.

### Why the fixed Python environment order regresses in Rust

The first Rust A/B port of the Python cap update forced the reference order
`Q^* x right_environment -> MPO -> MPS` at every adaptive step. That version
was about 10--11% slower at bonds 64 and 128, so it was reverted. A native
einsum path trace explains the reversal: the reference order is optimal only
for some relations between the old right-cap rank and the newly selected
rank, while the Rust N-ary planner changes the tree with the actual dimensions.

For a representative bond-64 growth step, the factor has dimensions
`[physical=2, old_cap=16, new_rank=32]`, the right environment has
`[MPO_left=64, MPS_left=64, old_cap=16]`, and the MPO/MPS bonds are 64. The
time-optimized Rust tree is:

1. `MPS x environment`: flop-index product 8,388,608, intermediate 131,072
   elements;
2. `MPO x partial`: 16,777,216, intermediate 131,072 elements;
3. `Q^* x partial`: 4,194,304, intermediate 131,072 elements.

The total is 29,360,128 with a 131,072-element peak. Forcing the Python order
on the same tensors gives 4,194,304 + 33,554,432 + 16,777,216 = 54,525,952
and a 262,144-element peak: 1.86x the arithmetic proxy and 2x the temporary.
When the ranks reverse (`old_cap=32`, `new_rank=16`), the Rust planner itself
selects the Python order; the issue is therefore forcing that order across all
adaptive rank configurations, not the order in isolation.

A second A/B used an explicit rank-aware pairwise implementation that follows
those two planner choices. With five repetitions it measured 93.679 ms versus
94.434 ms for the N-ary planner at bond 64 (within run noise), and 481.046 ms
versus 466.733 ms at bond 128 (3.1% slower). Under pairwise profiling at bond
128, the explicit form issued 200 instead of 176 pairwise calls and produced
211,025,920 instead of 167,772,160 output bytes across those calls. Splitting
one N-ary compiled graph into three independently dispatched pairwise graphs
therefore loses another constant factor even after matching its arithmetic
tree. The explicit candidates were reverted; retaining the adaptive N-ary
planner is the measured implementation choice.

Because the scalar type, RNG, warm-up policy, and backend differ, these rows
are diagnostic rather than a claim of a normalized language benchmark. The
stable within-language conclusions are: Rust fixed/adaptive correctness is at
machine precision on these cases; Python adaptive correctness is likewise at
machine precision; Python zip-up's bond-16 error (`6.347e-5`) is consistent
with its requested `1e-4` cutoff; and Rust's final pairwise/tree changes do not
alter the expected dense result.

### Lazy adaptive prefixes and optimized probe order

A matched operation-count trace found that the adaptive chain cache propagated
every new probe segment through all eight interior prefix sites. At bond 128
this produced 256 site-columns, while the reference sweep needed 158 because
high-rank columns first requested near the center never need propagation back
towards the right boundary. `PrefixCache` now creates each segment at site zero
and extends it through later sites only when requested. A non-monotonic cache
test covers a ragged site requesting a later segment before revisiting the
missing middle segment.

The prefix helper now gives the incoming prefix, local operand, and physical
probes to the retained-index N-ary optimizer together. This lets the existing
cost model eliminate the dimension-two probe before an expensive bond
contraction when that order is cheaper, without hard-coding an MPO--MPS matrix
layout or creating the full local product.

A clean old-pin A/B used ten paired one-thread processes with five repetitions
per process. The bond-128 median improved from 462.204 ms to 390.939 ms
(-15.4%). A five-pair rerun after the ragged-cache review fix measured 459.950
to 373.704 ms (-18.8%). The exact dense check reported relative error
`2.091e-31`. With the separately tested latest-tenferro adapter, the same
algorithmic change reduced the median from 466.794 to 376.059 ms (-19.4%) and
the prefix trace fell to 160 site-columns, close to the reference count.

### Fresh one-thread crossover rerun

A subsequent comparison matched the warm-up policy as well as the algorithm
parameters: every fresh process ran one untimed contraction followed by five
timed repetitions. Ten paired processes were pinned to one CPU for bonds 32,
64, and 128; five paired processes narrowed the crossover. Python used NumPy
2.0.0 with scipy-openblas 0.3.27 and its pure-Python incremental-QR fallback.
The tensor values and dense backends still differ, so these are matched
configuration and shape measurements, not identical-input kernel timings.

| Input bond | Rust (ms) | Python (ms) | Rust / Python |
| ---: | ---: | ---: | ---: |
| 32 | 57.911 | 19.678 | 2.943 |
| 64 | 98.665 | 75.970 | 1.299 |
| 72 | 108.387 | 97.258 | 1.114 |
| 80 | 128.834 | 126.600 | 1.018 |
| 96 | 181.041 | 196.715 | 0.920 |
| 128 | 343.313 | 448.202 | 0.766 |

The crossover is near bond 80; at bond 128 Rust is 23.4% faster. The earlier
Python bond-128 diagnostic of 348.851 ms was not reproducible in this rerun: an
additional unpinned run was also about 450 ms. The current paired result
therefore supersedes that absolute-time comparison. Rust dense-oracle relative
errors were `6.255e-26`, `1.052e-28`, and `2.099e-31` at bonds 32, 64, and 128.
Python selected `[5, 5, 11, 17, 32, 16, 8, 4, 2]` at every input bond.

The remaining low-bond difference is fixed contraction cost rather than excess
adaptive work. Rust now creates 160 prefix site-columns versus Python's 158.
At bond 32, Rust stage timing assigned 46.184 ms to sketch/prefix contractions
and 9.440 ms to incremental QR plus environment projection; setup and result
assembly were below 0.1 ms. Instrumented Python spent 13.185 ms in 683 matrix
multiplications and about 3 ms in incremental-QR routines. Generic retained and
N-ary lowering, dispatch, and intermediate tensor materialization therefore
dominate Rust while the matrices are small. Once arithmetic dominates, the
Rust/faer path scales better and overtakes Python/OpenBLAS. A skinny
`N = 1, K <= 4` lowering is a separate micro-kernel follow-up, not another SRC
algorithm correction.

## Verification

- `cargo test --release -p tensor4all-treetn 'treetn::contraction::' --lib`:
  74 passed, including four prefix-cache tests.
- `cargo test --release -p tensor4all-treetn --lib`:
  498 library tests passed; prior integration tests and 140 doctests passed;
  one pre-existing diagnostic test remained ignored.
- `cargo clippy -p tensor4all-treetn --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.

One cache test formerly compared independently planned floating-point results
bit-for-bit. The new multi-factor order exposed last-bit planner-order
variation. No tolerance was relaxed: that test now checks bitwise replay of the
same cached misaligned range, while a separate new test checks the new directed
messages against the old reference numerically at the existing `1e-10` scale.

## Remaining risks and follow-ups

- Directed messages still repeat one planner contraction per outgoing
  direction. Prefix/suffix partial contractions may reduce the remaining
  degree-squared work, but the need must be measured after this fix.
- Adaptive chain SRC still pays many small retained-index contractions and QR
  appends. A flattened-batch owner-level primitive remains a possible chain
  follow-up, but the generic N-ary calls on its ordinary binary path are gone.
- Tree segment caches can still create ragged boundaries and fall back to
  per-column selection plus restacking. Applying the chain policy there may
  be more expensive because one tree segment computes all directed messages;
  it needs its own A/B gate.
- Fixed-rank callers can request a rank far above the effective opposite-side
  support; benchmark configuration and API semantics need a separate review.
- CUDA and alternative CPU providers were not measured.
- The existing nine-leaf star benchmark is not a valid scaling gate at bond 8:
  each input center tensor alone is about 4 GiB. Future large-tree gates should
  use bounded-degree binary or comb topologies.
