# Remaining SRC performance items (#701–#704)

## Protocol

All measurements used the #706 deterministic release harness on the AMD EPYC
7713P host with Rayon, BLAS, OpenBLAS, OMP, and MKL thread counts fixed to one.
Temporary section instrumentation was enabled only for one timed run after the
harness's untimed warm-up and was removed afterward. A candidate subpath needed
at least 10% end-to-end share before new production machinery was allowed.

## #701 — remaining fixed contraction overhead

A fresh bond-32 adaptive chain run took 38.084 ms under section instrumentation
and submitted 254 N-ary contractions. Totals were:

| Section | ms | share |
|---|---:|---:|
| core plan | 1.372 | 3.60% |
| size validation | 0.115 | 0.30% |
| native operand preparation | 0.683 | 1.79% |
| result wrapping | 0.747 | 1.96% |
| backend execution | 17.237 | 45.26% |

No measured non-backend fixed N-ary contraction section reached the 10% need
gate; together they were about 7.7%. These section totals intentionally do not
account for the complete SRC runtime: the remainder includes QR, probe/cache
work, pairwise contractions, and other algorithm stages and is not classified
here as backend work. #701 concerns the named fixed per-N-ary sections above;
none justified another production optimization. Previously rejected
planner-map, linear cache, label-grouping, and fusion candidates were not
repeated.

## #702 — directed-message planner repetition

The complete directed-message routine occupied 39–60% of timed adaptive tree
runs for binary trees with 3/7/10 nodes and bonds 4/8, establishing that message
execution is important. That first measurement includes required contraction
kernels, however, while the proposed prefix/suffix change targets repeated
*planning*.

A second profile isolated all core planning on bond-8 trees:

| Nodes | plan calls | plan ms | run ms | share |
|---:|---:|---:|---:|---:|
| 3 | 15 | 0.123 | 2.455 | 5.00% |
| 7 | 169 | 1.546 | 31.231 | 4.95% |
| 10 | 257 | 2.489 | 57.031 | 4.36% |

The targeted overhead is below the 10% need gate. Prefix/suffix products would
also materialize products of leave-one-out branch spaces and risk replacing a
small planner cost with payload growth. No production change was made.

## #703 — aligned tree environment segments

Temporary cache instrumentation found misaligned split/restack fallback calls in
bounded binary-tree gates: 4 calls per two runs at n=7 bond 4/8, 8 at n=10 bond
4, and 10 at n=10 bond 8; n=3 and bond-2 cases had none.

The promoted change permits only the first segment to be narrow and grows later
segments by the full rank increment, with at most `rank_increment - 1` extra
columns. Range and accumulated-width arithmetic is checked. Tests cover a narrow
initial segment, bounded overgeneration, aligned direct reuse, true misaligned
fallback, exact replay, and zero-width rejection.

Predeclared primary gate: #706 quick, tree n=7 bond=4 adaptive, at least 3%
improvement; other cases at most 5% regression. The first 10-pair/30-repetition
run had a passing primary ratio `0.9454` (5.46% improvement, CI upper `0.9589`)
but failed overall because the unchanged 4.6-ms tree-fixed case had a noisy CI
upper of `1.0695` despite an improving median. The complete stronger rerun used
10 pairs and 60 repetitions with unchanged thresholds and passed all cases.
Additional 10-pair/20-repetition bond-8 diagnostics improved by 3.89% at n=7
and 4.88% at n=10.

The final independent review verdict was `Correct-to-merge`; all four Minor
findings were also fixed.

## #704 — flattened adaptive batch primitive

A fresh bond-32 adaptive chain profile separated backend cached-plan lookup,
session execution, and session shell over 254 calls. The 36.116-ms run spent
0.233 ms in plan lookup and 0.470 ms in the session shell: 1.95% combined.
Required kernel execution was 16.856 ms. The candidate boundary therefore fails
the 10% need gate. The earlier experiment that fused 57 of 230 boundaries was
also neutral. Adding a new flattened-batch API would not be justified; no code
was added.

## Verification

- Temporary instrumentation strings were removed from core, tensorbackend,
  TreeTN, and benchmark sources.
- `cargo test --release -p tensor4all-treetn 'treetn::contraction::' --lib`:
  75 passed before final Minor fixes; rerun in integration verification.
- #703 focused request/replay test passed.
- #706 paired/raw JSON records are retained in `/tmp` for the active session;
  curated results are recorded above.
- No tolerance was changed.
