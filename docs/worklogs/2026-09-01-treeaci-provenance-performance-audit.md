# TreeACI algorithm provenance and performance audit

Date: 2026-09-01

Audited revision: `fd61f082c7d2234db2e7b88e73e5fd1f5a0c4228` (`origin/main`)

Audit branch: `investigate/aci-provenance-performance`

## Scope and source policy

This is a static, line-range-complete audit of the production code in
`crates/tensor4all-treeaci/src/*.rs`. The three modules compiled only under
`#[cfg(test)]` (`order_experiment.rs`, `skeleton.rs`, and `validate.rs`) and the
`*/tests/` trees are validation evidence, not production algorithm code, and
are listed separately below.

Only these algorithm and performance authorities were allowed:

1. Ritter, *Fast elementwise operations on tensor trains with alternating
   cross interpolation*, arXiv:2604.00037v2 (23 Apr 2026), including Algorithms
   1--5 and equations (4)--(11).
2. The chain implementation in `crates/tensor4all-aci/src/` at the audited
   revision (called “simplett ACI” below).
3. Hiroshi Shinaoka's review comments on the merged original TreeACI
   [PR #646](https://github.com/tensor4all/tensor4all-rs/pull/646), specifically
   the [performance-blocking review](https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5313040237),
   the [representative-rank comparison protocol](https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5316892012),
   and the [packed directed-message cache design](https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5316916280).
4. Hiroshi Shinaoka's topology-cost caveat in large
   [issue #671](https://github.com/tensor4all/tensor4all-rs/issues/671), especially
   the [coordination-number/bond-dimension relation](https://github.com/tensor4all/tensor4all-rs/issues/671#issuecomment-5391376991).

No other cited TCI paper, previous TreeACI worklog, issue, pull request, or
TreeTN implementation was used as algorithm authority. TreeTN source was
inspected only to determine the runtime semantics and costs of calls made by
TreeACI. Tests and benchmarks are evidence only.

The provenance labels used below are:

- **Paper**: a direct implementation of a cited equation or pseudocode step.
- **simplett ACI**: a direct analogue of the cited chain implementation.
- **Hiroshi review**: an authorized performance expectation, measurement
  protocol, or cache design from PR #646 or issue #671; it is not silently
  promoted to an algorithm derivation.
- **Tree generalization — re-derived**: absent from the paper's equations and
  pseudocode; derived from the cut-message equations in this document.
- **[AI Supplied]**: no authority in either allowed source. This includes
  ordinary Rust API glue, validation, resource policy, caches, traversal policy,
  diagnostics, rollback machinery, and other engineering choices. The label
  does not imply that a block is incorrect.

The paper's conclusion says a tree generalization is straightforward but gives
no tree equations or tree pseudocode. That sentence is not sufficient
provenance for any concrete tree implementation below.

## Re-derivation of the missing tree algorithm

Let `T = (V, E)` be a tree. Node `u` has a flattened physical coordinate
`s_u in {0, ..., d_u - 1}`. Cutting an undirected edge `{u,v}` splits the tree
into two components. For the orientation `e = u -> v`, write `C_e` for the
component containing `u`, and

```text
in(e) = { w -> u : w is adjacent to u and w != v }.
```

A nested component sample is recursively

```text
i_e = (s_u, (i_f) for f in in(e)).                       (T1)
```

This is the unique tree analogue of a TT prefix or suffix: because the graph is
a tree, the incoming components are disjoint and their union with node `u` is
exactly `C_e`. On a chain, `in(e)` has zero or one element, and (T1) reduces to
the paper's left/right multi-indices in equation (4).

For input tensor network `X^(n)`, define its cut frame/message by

```text
F_e^(n)(i_e)[a_e]
  = sum over all internal bonds in C_e
      X_u^(n)[s_u, a_e, (a_f) for f in in(e)]
      * product over f in in(e) F_f^(n)(i_f)[a_f].       (T2)
```

The leaf case is the empty product. Equation (T2) is well-founded in a
dependency order from leaves toward the directed cut. On a chain it is exactly
the left/right-frame contraction of paper equations (7)--(9), and the frame
updates in paper Algorithm 5.

The two component samples on opposite orientations cover `V` without overlap.
Therefore the input value at a local cross entry is

```text
X^(n)(i_e union i_reverse(e))
  = sum over a_e F_e^(n)(i_e)[a_e]
                 * F_reverse(e)^(n)(i_reverse(e))[a_e].  (T3)
```

For an elementwise function `f`, the edge-local cross matrix is

```text
M_e[i,j] = f(X^(1)(i union j), ..., X^(N)(i union j)).   (T4)
```

Applying LUCI to (T4), selecting row and column component samples, and reshaping
the two factors into the endpoint cores is the tree-cut version of paper
equations (10)--(11) and Algorithms 2--4. Nesting follows by construction from
(T1): every selected parent sample contains only already-active child samples.

If every incoming candidate set has size `r_f`, the number of candidates on a
directed cut is

```text
C_(u->v) = d_u * product over f in in(u->v) r_f.         (T5)
```

The edge-local matrix has `C_(u->v) * C_(v->u)` entries. This product is not an
implementation accident: it is inherent in materializing (T4). It explains why
branching can be intrinsically more expensive than a chain. For example, a
degree-three endpoint contributes `d*r^2`; an edge joining two degree-three
nodes produces `d^2*r^4` local points, while a uniform chain edge produces
`d^2*r^2`. The implementation can still avoid unnecessary allocations,
repacking, hash tables, and scalar contractions around this unavoidable count.

## 2026-09-02 performance-suspect follow-up

Previous worklogs were read only to locate prior experiments. They were not
treated as authority. In particular, the rejected two-GEMM branch experiment
in `2026-08-23-treeaci-branched-hotpaths.md` changed the floating-point
reduction order and moved downstream pivot paths; it is not evidence that the
same rewrite is safe.

Fresh release-mode diagnostics found three distinct effects.

### Ephemeral matrices use the wrong existing upstream seam

`tensor4all-tensorbackend::mat_mul_owned` has existed since 2026-05-22 and its
contract explicitly says that it reuses consumed matrix buffers when building
tenferro tensors. TreeACI's chain paths instead create fresh `Matrix` values,
call borrowed `mat_mul`, and immediately discard those matrices in:

- the stored-frame single-incoming batch;
- the candidate-frame single-incoming batch; and
- the final row-frame by column-frame local materialization.

This is not a new algorithm. It is an existing upstream API used with the same
matrix product and reduction order. A test-only `[AI Supplied]` switch changed
only those three eligible calls. On a 32-site chain with two inputs and a
genuine high-rank plateau, three interleaved release runs gave these medians:

| maximum input bond | borrowed init | owned init | borrowed 2 sweeps | owned 2 sweeps |
|---:|---:|---:|---:|---:|
| 256 | 200 ms | 159 ms | 315 ms | 265 ms |

The combined median fell by about 17.6%. With the switch enabled, the complete
TreeACI release suite passed: 136 unit tests, 7 public-API integration tests, 1
rank-scaling test, and 18 doctests (4 opt-in diagnostics remained ignored).
This proves a material copy/ownership overhead in the current chain path; it
does not yet constitute a production fix or a full numerical regression gate.

The same-shape batched upstream API was also tested for two large, compatible
input GEMMs. Unlike the owned-vs-borrowed result, it was unstable and was
usually slower at batch size two. Therefore batching across operands is not
currently accepted as a remedy merely because simplett ACI uses it.

### Branch frame kernel misses an existing upstream batch capability

The two-incoming TreeACI kernel dispatches `incoming_dim_2 + 1` borrowed
matrix multiplications. Keeping exactly the existing per-`incoming_dim_2`
reduction decomposition but submitting its same-shaped first-stage products
through `tensor4all-tensorbackend::batched_mat_mul_same_shape_owned` produced
bit-identical results in the diagnostic fixtures and reduced the representative
32-by-256-by-256, 40-column kernel median from 115 ms to 21 ms (5.4x).

That direct prototype repeats the shared right-hand matrix once per job (about
20 MiB in this fixture), so it cannot replace the production path without
respecting `max_working_bytes`. The pinned tenferro revision already has
`GroupedGemmJob`/`grouped_gemm_cached`, whose offset descriptors permit every
job to refer to the same RHS buffer. TreeACI must not depend on tenferro
directly; the appropriate production direction is a budgeted
tensorbackend-level grouped/shared-operand seam. This seam and its policy are
**[AI Supplied]** until separately designed and tested.

### A vertex-centered warm evaluator repeats avoidable work

Fresh phase timing of an identical, fully warmed 64-point batch on a native
16-site TreeTN chain showed zero message-cache misses and zero message
contractions, yet `contract_center_for_points` still consumed 87--95% of each
call at bond dimensions 64, 128, and 256. The current raw center kernel visits
`d * product(incident bond dimensions)` core entries per distinct center
evaluation. Hiroshi's issue #671 comment is authority for that node-local work
count, but it does not establish that every warm whole-tree evaluation must end
at a vertex.

The following alternative tree extension has no direct source in the paper or
simplett implementation and is therefore **[AI Supplied]**. It is re-derived
here rather than assumed. For any tree edge `e`, deleting `e` partitions the
vertices into `L_e` and `R_e`. Leave only the cut index `a_e` uncontracted and
define the two directed component messages

```text
L_e(x_L, a_e) = contraction of every tensor and internal bond in L_e,
R_e(x_R, a_e) = contraction of every tensor and internal bond in R_e.
```

Because a tree has no second connection between these components,

```text
X(x) = sum over a_e L_e(x_L, a_e) * R_e(x_R, a_e).       (T6)
```

Equation (T6) is the arbitrary-tree form of simplett `TTCache::evaluate_many`'s
left/right split. Once both component messages hit the cache, final assembly is
`O(chi_e)` per point, not `O(d_u * product(incident chi))`. A branch node's
high-coordination contraction has not disappeared: it is paid when a component
message containing that node is first built, then can be cached per component
assignment instead of being repeated as an uncached center contraction.

Consequently the existing deterministic `d*chi^z` work-count test correctly
describes `contract_raw_center`, but it must not be interpreted as the desired
warm-evaluation complexity regression. An edge-centered cached evaluator and a
test that gates its warm final assembly at `O(chi_e)` require a separate design.

## Exhaustive production-code provenance ledger

Line numbers refer to the audited revision. Imports, derives, error conversion,
checked arithmetic, documentation, and module/public re-exports in a listed
range inherit the range's label unless a narrower row says otherwise.

### Public surface and non-algorithm support

| File and lines | Block | Provenance |
|---|---|---|
| `lib.rs:1-81` | Module layout, maturity notes, re-exports | **[AI Supplied]**; the phrase “tree ACI” is motivated by the paper, but this API and its performance claims are not specified there. |
| `prelude.rs:1-7` | Convenience re-exports | **[AI Supplied]** |
| `batch.rs:1-167` | Column-major callback view and checked access | simplett ACI `batch.rs:1-174`, adapted by name only; all validation/API details are **[AI Supplied]** |
| `error.rs:1-195` | Error type and result alias | **[AI Supplied]** |
| `scalar.rs:1-99` | Supported scalar/node traits and conversions | simplett ACI `scalar.rs:1-94` for scalar bounds; TreeTN node abstraction and conversions are **[AI Supplied]** |
| `options.rs:1-125` | Options and defaults | simplett ACI `options.rs:1-185` for tolerance, rank, sweep, guard, and seed concepts; tree root, traversal, candidate/frame/core/working budgets, and exact defaults are **[AI Supplied]** |
| `result.rs:1-106` | Termination, diagnostics, result | simplett ACI `result.rs:1-73` for histories/termination; tree diagnostics and layout are **[AI Supplied]** |
| `traversal.rs:1-29` | Public traversal strategy | **[AI Supplied]** |
| `branch_diagnostics.rs:1-11` | Diagnostics registry re-export | **[AI Supplied]** |

### Problem preparation and topology

| File and lines | Block | Provenance |
|---|---|---|
| `problem.rs:1-46` | Directed-edge and physical/problem records | **Tree generalization — re-derived**, definitions (T1)--(T2); Rust representation and budgets are **[AI Supplied]** |
| `problem.rs:47-251` | Validate common topology/indices, flatten physical axes, build schedule | Common-input checks correspond to simplett ACI `validation.rs:47-117`; directed cuts and incoming branches are **Tree generalization — re-derived**; ordering, root choice, limits, and schedule construction are **[AI Supplied]** |
| `problem.rs:252-307` | Directed-frame dependency order | **Tree generalization — re-derived** from the recursion in (T2); the particular DFS/stack implementation is **[AI Supplied]** |
| `problem.rs:308-381` | Option/limit/overflow validation | **[AI Supplied]** |
| `problem.rs:382-423` | Create both orientations and ordered incoming arcs | **Tree generalization — re-derived**, `in(e)` in (T1)--(T2); deterministic sorting is **[AI Supplied]** |

### Initialization

| File and lines | Block | Provenance |
|---|---|---|
| `initialize.rs:1-67` | Initial rank selection | simplett ACI `random_tt.rs:15-137` for guess/rank construction; per-edge tree ranks are **Tree generalization — re-derived**; caps and validation are **[AI Supplied]** |
| `initialize.rs:68-80` | Algebraic cut bounds | **Tree generalization — re-derived**: rank is bounded by the smaller component's physical dimension; checked implementation is **[AI Supplied]** |
| `initialize.rs:81-140` | Initial-guess topology, ranks, resources, scalar kind | simplett ACI `random_tt.rs:41-83`, generalized to all tree edges; TreeTN-specific checks are **[AI Supplied]** |
| `initialize.rs:141-215` | Random TreeTN output | simplett ACI `random_tt.rs:85-150` for random rank-compatible cores; arbitrary-degree core construction is **Tree generalization — re-derived**; RNG/layout details are **[AI Supplied]** |
| `initialize.rs:216-280` | Bootstrap component/candidate/pivot samples | Paper Algorithm 1 says to initialize index sets but specifies no tree seed construction. Recursive projection is **Tree generalization — re-derived** from (T1); digit-reversal-like global seeds and pairing policy are **[AI Supplied]** |
| `initialize.rs:281-345` | Component dimensions/nodes and checked product | **Tree generalization — re-derived** for cut components; implementation is **[AI Supplied]** |
| `state.rs:1-53` | Test-only timing counters compiled out of production | **[AI Supplied]** instrumentation |
| `state.rs:55-69` | Owned sweep state | Paper Algorithm 1 state (`A`, `P`, `T`, frames, errors), plus **Tree generalization — re-derived** directed samples/frames; ownership layout is **[AI Supplied]** |
| `state.rs:71-149` | Prepare problem/output, bootstrap samples, build frames | Paper Algorithm 1 initialization; simplett ACI `state.rs:68-109`; tree portions follow (T1)--(T2). Deferring numerical CI canonicalization until after the first pass is **[AI Supplied]** |

### Recursive samples

| File and lines | Block | Provenance |
|---|---|---|
| `samples.rs:1-76` | IDs, recursive component record, arenas, candidate/pivot sets | **Tree generalization — re-derived**, equation (T1); arenas, IDs, hash keys, generations, checkpoints are **[AI Supplied]** |
| `samples.rs:77-140` | Candidate uniqueness and pivot-pair accessors | Paper equation (4) index sets and Algorithm 4 selected indices; tree orientation is **Tree generalization — re-derived**; linear dedup/accessor design is **[AI Supplied]** |
| `samples.rs:141-197` | Injection report and checkpoint/rollback | Global injection concept follows simplett ACI `state.rs:551-644`; transaction machinery is **[AI Supplied]** |
| `samples.rs:198-300` | Build/project recursive component samples from global seeds | **Tree generalization — re-derived**, equation (T1) and component partition; stack, memo, and interning mechanics are **[AI Supplied]** |
| `samples.rs:301-422` | Inject a global point into selected directed cuts | simplett ACI `state.rs:565-640`, generalized via (T1); activation mask and rollback behavior are **[AI Supplied]** |
| `samples.rs:423-477` | Reassemble a global point and record access | Inverse of **Tree generalization — re-derived** (T1); validation is **[AI Supplied]** |
| `samples.rs:478-684` | Validate/intern/project/write recursive records | **Tree generalization — re-derived**, equation (T1); append-only hash interning and byte accounting are **[AI Supplied]** |
| `samples.rs:685-700` | Logical record-byte accounting | **[AI Supplied]** |

### Input cut frames

| File and lines | Block | Provenance |
|---|---|---|
| `frames.rs:1-80` | Checked scratch/resource helpers | **[AI Supplied]** |
| `frames.rs:82-174` | Test counters | **[AI Supplied]** instrumentation |
| `frames.rs:176-193` | Sample-major directed frame | **Tree generalization — re-derived**, `F_e(i_e)` in (T2); storage layout/copying are **[AI Supplied]** |
| `frames.rs:195-260` | Candidate cache keys, store, prepared cores | Frame meaning follows (T2); persistent cache, `Rc`, compact key cutoff at two incoming edges, and copied prepared-core representation are **[AI Supplied]** |
| `frames.rs:262-518` | Build or extend every input/directed frame | **Tree generalization — re-derived**, evaluate (T2) in dependency order. Append-only reuse is analogous to simplett ACI retaining left/right frames, but the global rebuild/`Rc`/memo-spine design is **[AI Supplied]** |
| `frames.rs:520-605` | Accounting and candidate cache | **[AI Supplied]** |
| `frames.rs:608-692` | Predict candidate contraction scratch | Formula follows **Tree generalization — re-derived** (T2)/(T5); resource accounting policy is **[AI Supplied]** |
| `frames.rs:694-734` | Bond/frame lookup | **Tree generalization — re-derived** data access; checks/copies are **[AI Supplied]** |
| `frames.rs:735-958` | Batch zero/one-incoming candidate frames | Zero/one-incoming specialization of **Tree generalization — re-derived** (T2), reducing on a chain to paper equations (8)--(9). Packing, cache probing, hash grouping, and the chosen GEMM layout are **[AI Supplied]**; simplett ACI `state.rs:215-529` precomputes core matrices and batches frame updates instead. |
| `frames.rs:960-1173` | Batch exactly-two-incoming candidate frames | **Tree generalization — re-derived** (T2). The full gathered Cartesian product, BTreeMap/HashMap grouping, cache, allocation, and kernel decomposition are **[AI Supplied]** |
| `frames.rs:1175-1259` | Scalar candidate frame | **Tree generalization — re-derived** (T2); persistent caching and per-call owned `Vec` are **[AI Supplied]** |
| `frames.rs:1261-1368` | Recursive memoized frame builder | **Tree generalization — re-derived** (T2); memo shape, lazy pull-through, and owned-row copies are **[AI Supplied]** |
| `frames.rs:1369-1546` | Batch stored-frame construction for one incoming edge | One-incoming specialization of (T2); batch/packing strategy is **[AI Supplied]** and analogous to simplett ACI `state.rs:301-529` |
| `frames.rs:1547-1715` | Batch stored-frame construction for two incoming edges | **Tree generalization — re-derived** (T2); grouping and kernel strategy are **[AI Supplied]** |
| `frames.rs:1717-1818` | Resolve cut bond and scalar arbitrary-degree contraction | **Tree generalization — re-derived**, direct evaluation of (T2) |
| `frames.rs:1820-1845` | Recursive contraction over incoming axes | **Tree generalization — re-derived**, product/sum in (T2); recursive scalar implementation is **[AI Supplied]** |
| `frames.rs:1847-1933` | Pack one-incoming core slices and GEMM | Algebra follows one-incoming (T2); packing and GEMM decomposition are **[AI Supplied]**. Chain analogue: simplett ACI precomputed `InputCoreMatrices`, `state.rs:40-65,77-87`. |
| `frames.rs:1935-1977` | Two-incoming kernel (`incoming_dim_2 + 1` GEMMs) | Algebra follows two-incoming (T2); this decomposition is **[AI Supplied]** |
| `frames.rs:1979-2036` | Optional branch diagnostics | **[AI Supplied]** |
| `frames.rs:2037-2114` | Bond lookup, one-time core copying/strides, axis lookup | TreeTN adapter and optimization machinery: **[AI Supplied]** |

### Local edge update and commit

| File and lines | Block | Provenance |
|---|---|---|
| `local_update.rs:1-30` | Local-factor result | Paper Algorithms 3--4 and equation (11), with tree samples from (T1); diagnostic fields are **[AI Supplied]** |
| `local_update.rs:32-84` | Enumerate both sides and size local matrix | **Tree generalization — re-derived**, equations (T4)--(T5); limits are **[AI Supplied]** |
| `local_update.rs:85-152` | Peak/core resource accounting | **[AI Supplied]** |
| `local_update.rs:153-218` | Produce every input value by row/column cut-frame GEMM | **Tree generalization — re-derived**, equation (T3). The nested `Vec<Vec<T>>`, repacking, product allocation, and scatter are **[AI Supplied]**. Simplett ACI counterpart: `local.rs:304-374`, using prebuilt local factors and same-shape batched GEMM when possible. |
| `local_update.rs:223-293` | Callback, sampled scale, owned LUCI, select factors | Paper equations (10)--(11), Algorithms 2--4; direct chain counterpart simplett ACI `state.rs:774-841` |
| `local_update.rs:295-310` | Map pivot indices to component samples | Paper Algorithm 4 selected row/column indices plus **Tree generalization — re-derived** (T1) |
| `local_update.rs:312-364` | Materialize Cartesian-product candidates | **Tree generalization — re-derived**, equation (T5). One heap `Vec` of incoming pairs per candidate and mixed-radix enumeration are **[AI Supplied]** |
| `transaction.rs:1-51` | Propose then commit edge update | Paper Algorithm 3 local update; transaction split and profiling are **[AI Supplied]** |
| `transaction.rs:53-189` | Clone/stage output, intern pivots, extend every frame store, commit state | Pivot replacement corresponds to paper Algorithm 3 and simplett ACI `state.rs:843-857`; tree samples follow (T1). Whole-network metadata clone, rollback checkpoint, global frame-store extension, generation tracking, and atomic staging are **[AI Supplied]** |
| `transaction.rs:191-237` | Reshape factors into endpoint cores and move center | Paper equation (11)/Algorithm 3 on a chain; arbitrary incident axes are **Tree generalization — re-derived**. TreeTN replacement/canonical metadata operations are **[AI Supplied]** |
| `transaction.rs:239-274` | Construct factor index order | **Tree generalization — re-derived** from (T1)/(T5); exact index ordering is **[AI Supplied]** |

### Traversal and convergence

| File and lines | Block | Provenance |
|---|---|---|
| `path_cover.rs:1-119` | Minimum-retracing continuous walk and reverse spine plan | Paper Algorithm 1 only defines forward/backward chain sweeps. Need for a connected center walk follows the local two-core update, but the diameter/minimum-retracing policy is **[AI Supplied]** |
| `path_cover.rs:120-271` | Adjacency, farthest/path, DFS walk, reverse spine | **[AI Supplied]** graph algorithm |
| `path_cover.rs:272-434` | Validate directed coverage/tree, union-find | **[AI Supplied]** |
| `schedule.rs:1-72` | Pass/history records | Paper Algorithm 1 sweep/error history and simplett ACI `elementwise.rs:126-150`; tree update traces are **[AI Supplied]** |
| `schedule.rs:73-174` | Alternate directional passes, guard, stopping | Paper Algorithm 1; simplett ACI `elementwise.rs:132-196`; tree traversal and per-cut capacities are **Tree generalization — re-derived** where mathematical, otherwise **[AI Supplied]** |
| `schedule.rs:176-189` | Global-injection capacity | simplett ACI `state.rs:576-634`, generalized by cut bounds; implementation is **[AI Supplied]** |
| `schedule.rs:191-248` | Clone phases, run one pass, aggregate errors | Paper Algorithm 1 and simplett ACI `elementwise.rs:132-150`; phase cloning/aggregation details are **[AI Supplied]** |
| `schedule.rs:250-266` | One-time deferred full-tree CI canonicalization | Paper Algorithm 1 right-canonicalizes the initial chain before sweeps; simplett ACI `state.rs:862-948`. Performing it after the first tree pass and relying on retained canonical metadata are **[AI Supplied]** |
| `schedule.rs:268-316` | Execute every continuous path step serially | Local update follows Paper Algorithm 3; continuous tree order is **[AI Supplied]** |
| `schedule.rs:318-412` | Convergence/rank-limit/error helpers and step lookup | Paper Algorithm 1 convergence plus simplett ACI `elementwise.rs:340-449`; tree-wide maximum/cap policy and lookup mechanics are **[AI Supplied]** |

### Global guard and entry points

| File and lines | Block | Provenance |
|---|---|---|
| `global_guard.rs:1-194` | Floating-zone global error search | Direct counterpart of simplett ACI `global_guard.rs:1-193`; not in paper Algorithms 1--5, therefore authority is **simplett ACI**. Tree cached evaluators, memory accounting, hint selection, and candidate dedup are **[AI Supplied]** |
| `global_guard.rs:195-290` | Inject pivots into all eligible cuts | simplett ACI `state.rs:551-644`; component projection is **Tree generalization — re-derived** (T1), while masks/transaction behavior are **[AI Supplied]** |
| `global_guard.rs:291-528` | Pad grown output bonds | simplett ACI `state.rs:674-727`; arbitrary affected tree cores are **Tree generalization — re-derived**. Planning, selective rebuild, byte budget, and whole-network metadata clone are **[AI Supplied]** |
| `global_guard.rs:529-782` | Persistent input evaluators, batch expansion/budgets | simplett ACI persistent guard cache (`state.rs:32-37`, `global_guard.rs`); TreeTN cached-evaluator integration and all budget/hint policy are **[AI Supplied]** |
| `global_guard.rs:783-855` | Output evaluator and checked counts | simplett ACI guard solution evaluation; TreeTN adapter is **[AI Supplied]** |
| `single_site.rs:1-115` | Exact one-site evaluation | simplett ACI `elementwise.rs:220-255`; TreeTN construction/resources are **[AI Supplied]** |
| `elementwise.rs:1-193` | Validate, one-site shortcut, initialize, sweep, cleanup, result | Paper Algorithm 1 and simplett ACI `elementwise.rs:108-218`; tree state/traversal are the re-derived blocks above; cleanup/diagnostics/error glue are **[AI Supplied]** |
| `elementwise.rs:194-220` | Scalar callback wrapper | simplett ACI `elementwise.rs:311-338`; adapter details are **[AI Supplied]** |
| `hadamard.rs:1-79` | Elementwise product convenience API | simplett ACI elementwise API; product wrapper/docs are **[AI Supplied]** |

### Test-only source (not algorithm authority)

| Files | Purpose | Status |
|---|---|---|
| `order_experiment.rs`, `skeleton.rs`, `validate.rs` | Edge-order experiments and independent nesting/interpolation/gauge checks | **[AI Supplied]** validation evidence; excluded from production ledger because `lib.rs:67-72` compiles them only for tests |
| `problem.rs:425-614`, `options.rs:127-137`, `scalar.rs:101-194` | Inline unit-test modules | Evidence only; excluded from production ranges |
| `src/*/tests/mod.rs`, `tests/*.rs` | Correctness, error-path, resource, and timing checks | Evidence only; never used as provenance |
| `tensor4all-treetn/benches/cached_evaluator.rs` additions | Same-input cold/warm evaluator parity and `d * product(chi_e)` scaling fixtures | Measurement protocol and representative `chi` values: **Hiroshi review**; star/chain fixtures, tolerances, Criterion configuration, and benchmark plumbing: **[AI Supplied]** |

## Correctness fix: preserve exact 32-bit cached-evaluator dtypes

TreeACI's public `TreeAciScalar` contract includes `f32`, `f64`, `Complex32`,
and `Complex64`.  `TreeTNCachedEvaluator::can_use_raw_messages` distinguishes
only real versus complex, while all real raw leaf/chain/branch kernels read
`f64` and all complex kernels read `Complex64`.  The generic fallback helper
`tensor_values_any` makes the same two-way choice.  Consequently 32-bit tensors
are accepted into a 64-bit reader rather than dispatched by their exact dtype.

An **[AI Supplied]** numerical regression fixture evaluates the same
two-node contraction through ordinary `TreeTN::evaluate` and the cached
evaluator.  The ordinary evaluator produces the asserted values 23 and 46 for
both 32-bit scalar kinds.  Before the fix, the cached evaluator failed before
arithmetic:

```text
f32: expected F64, actual F32
c32: expected C64, actual C32
```

No new dtype API is required to correct the dispatch.  Upstream `IdxTensor`
already exposes the backend-neutral predicates `is_f32`, `is_f64`, `is_c32`,
and `is_c64`.  `tensor4all-partitionedtreetn` already demonstrates the exact
four-way pattern with its internal `ScalarKind`; the private
`IdxTensor::scalar_dtype` and CUDA-only `cuda_dtype` are therefore not blockers.
The cached evaluator ignored those existing high-level capabilities and
collapsed the type to a real/complex bit.  A correction must use exact typed
dispatch rather than probing through failed `to_vec` calls and must not add a
TreeACI-local tenferro dependency.  Every raw leaf, chain, branch, leaf-center,
internal-center, and generic tensor-to-scalar path must be checked together
when this is fixed.

The implemented small correction now classifies tensors with those four
upstream predicates, preserves all four variants in `CachedScalar`, materializes
generic message/final-result tensors through the matching typed reader, and
keeps the existing specialized raw kernels restricted to their actual
`f64`/`Complex64` contract.  The formerly ignored regression now passes for
both cold computation and typed cache reconstruction, alongside explicit
`f32`/`Complex32` cache-payload tests.  This restores correctness without
claiming performance parity: 32-bit evaluation currently uses the generic
contraction path, and extending the specialized raw kernels generically is a
separate optimization task.  The dispatch structure follows the existing
upstream `tensor4all-partitionedtreetn::ScalarKind` pattern; the new regression
fixture and the choice to defer a 32-bit raw fast path are **[AI Supplied]**.

## Performance findings

### P0: warm chain evaluation does not reuse the center contraction

The representative comparison requested in Hiroshi's PR #646 review was added
to `crates/tensor4all-treetn/benches/cached_evaluator.rs`. It uses the same
16-site `f64` chain and 8-by-8 coordinate batch for both evaluators, fixes both
at the midpoint, validates every result before timing, constructs cold
evaluators outside the timed region, and measures both cold and repeated warm
calls at `chi = 64, 128, 256`.

The APIs have an important, unavoidable comparison detail: `TTCache` splits on
a bond, whereas `TreeTNCachedEvaluator` centers on a node. The latter is the
closest representable midpoint, not an identical contraction object. That
difference is itself causal on warm calls.

| chi | TTCache cold | TreeTN cold | Tree/TT cold | TTCache warm | TreeTN warm | Tree/TT warm |
|---:|---:|---:|---:|---:|---:|
| 64 | 2.189 ms | 4.429 ms | 2.02x | 10.58 us | 3.318 ms | 314x |
| 128 | 9.058 ms | 9.860 ms | 1.09x | 13.49 us | 4.516 ms | 335x |
| 256 | 37.79 ms | 25.04 ms | 0.66x | 19.17 us | 8.265 ms | 431x |

Cold results rule out a general high-bond defect in TreeTN's chain arithmetic:
the gap closes and reverses as `chi` grows. Warm results reproduce the severe
chain regression and identify it as a reuse-granularity problem:

- `cached_evaluator.rs:1151-1208` eagerly walks every non-center node in
  postorder on every call. An all-hit parent message does not short-circuit its
  descendants as `TTCache::evaluate_left/right` recursion does.
- The all-hit path at `cached_evaluator.rs:2772-2817` allocates a new packed
  result and copies every cached column for every visited directed message.
- `cached_evaluator.rs:1230-1310` rebuilds local keys and compact subtree
  assignment batches from scratch on every call.
- Most importantly, the cache excludes the center node.
  `cached_evaluator.rs:3387-3552,3718-3834` contracts the center tensor again
  on every call. For an internal chain center this is
  `O(n_points * d * chi^2)`. A warm `TTCache` has already cached environments
  that include every site tensor and only takes an `O(n_points * chi)` inner
  product at its split.

A two-point floating-zone scan, which is closer to a Guard coordinate walk,
confirms that the 64-point batch is not creating the effect. On an 8-site
chain, TreeTN/TTCache was 147.3 us/0.870 us (169x) at `chi=128` and
258.4 us/1.127 us (229x) at `chi=256`.

The existing “cache was hit” tests prove correctness and nonzero reuse, but do
not assert that a hit skips descendant reconstruction or center arithmetic.
The correct fix seam is therefore an edge-centered or lazily recursive cached
evaluation plan, not a faster hash function. Such a plan is **[AI Supplied]**
unless separately re-derived and reviewed; neither the paper nor simplett ACI
specifies its tree form.

### P0: distinguish topology-required branch cost from branch overhead

Hiroshi's issue #671 comment gives the correct normalization. For physical
dimension `d`, coordination number `z`, and incident bond dimensions
`chi_1,...,chi_z`, the local tensor has

```text
d * product_e chi_e
```

elements; only for uniform bonds is this `d * chi^z`. The new
`treetn_warm_center_coordination_vs_bond` benchmark places this actual product
in every benchmark ID and compares the same two-point warm center scan on a
degree-2 chain node and degree-3 star hub.

| chi | z=2 local elements | z=2 time | z=3 local elements | z=3 time | z3/z2 time |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2,048 | 105.2 us | 65,536 | 188.4 us | 1.79x |
| 64 | 8,192 | 112.0 us | 524,288 | 823.2 us | 7.35x |

The z=3 local work proxy is 32x and 64x larger respectively, yet elapsed time
is only 1.79x and 7.35x larger. Equivalently, time per local tensor element is
lower at the branch hub in these fixtures. This does **not** prove all branch
paths efficient, but it does show that the raw center kernel's absolute branch
cost is presently within the topology-required envelope; a wall-time ratio
without `d * product(chi_e)` would misdiagnose it.

The existing ignored 16-site chain-versus-comb diagnostic was also run at
`d=2`, `chi=128`, with one degree-3 hub as a non-center branch-message node:

```text
chain: 48.421641 ms
comb:  499.832829 ms
comb / chain wall time: 10.32x
```

Summing `d * product(chi_e)` over every node gives 459,264 local elements for
the chain and 4,588,288 for the comb, a 9.9905x work ratio. The wall-time ratio
divided by this topology-required ratio is 1.033. In this matched-bond fixture,
therefore, nearly the entire apparent branch slowdown is explained by the
larger dense tensors, including the non-center branch-message path. It is not
evidence of a large residual Guard branch-kernel regression.

The topology-normalized total does not make the implementation overhead
innocent.  A diagnostic split of the same run found that the degree-3 hub's
non-center branch-message path executed 60 one-point/one-physical-value BLAS
groups and spent:

| phase | copied/processed values | elapsed |
|---|---:|---:|
| decode two cached child messages | 21,376 | 0.038 ms |
| gather the requested child columns | 15,360 | 0.022 ms |
| repack the immutable hub tensor slice into `left` | 125,829,120 | 384.408 ms |
| first-child matrix multiplication | - | 72.239 ms |
| second-child accumulation | - | 0.243 ms |

At `chi=128`, every `left` contains `128^3 = 2,097,152` `f64` values,
or 16 MiB.  Rebuilding it 60 times writes 960 MiB.  The local physical
dimension is two and the rooted orientation and tensor are fixed, so those 60
groups can refer to at most two distinct physical slices: each distinct slice
is repacked at least 30 times on average.  Setup alone is 75.6% of the measured
508.409 ms comb walk and 5.32x the measured matrix-multiplication time.

This repacking has the same `product(chi_e)` element count as one necessary
local contraction, but it is data rearrangement, not one of the multiplications
required by Hiroshi's `d * product(chi_e)` arithmetic bound.  Therefore the
earlier 1.033 topology-normalized total concealed a confirmed large constant
factor: the necessary dense branch arithmetic is accompanied by repeated
same-order full-slice traffic.

The unresolved branch-specific implementation costs are therefore:

- TreeACI candidate frames still fall from a batched kernel to per-candidate
  scalar recursion at three or more incoming components (next section).
- A non-center degree-3 node uses the grouped branch-message path described
  above, which rebuilds the physical-slice `left` packing on every call even
  though the local core and rooted orientation are immutable.  Reusing a
  prepared orientation/slice, or exposing a borrowed strided/permuted matrix
  view from the tensor backend, is **[AI Supplied]** until the replacement is
  derived and reviewed.  The current tensorbackend `Matrix` API has owned
  matrices and borrowed multiplication, but no borrowed matrix-view seam that
  can express this layout without packing.  Tensorbackend's lower-level
  `einsum_native_tensor_reads` can accept non-contiguous native views, and
  `tensor4all-core::contract_with_options` already reaches that path.  What is
  missing is a high-level TreeTN/tensorbackend seam that combines a fixed
  physical slice, the required axis permutation, and batched/shared operands
  without rebuilding the packed matrix.  Reaching directly into native
  tenferro reads from TreeTN would violate the repository layering rule; the
  new high-level facility belongs upstream.

  A test-only A/B independently checked whether the existing generic
  `contract_with_options` pipeline is already that facility.  Disabling all
  specialized raw message kernels changed the same `chi=128` walk from about
  48 ms to 3.458 s on the chain and from about 508 ms to 4.676 s on the comb:
  approximately 71x and 9.2x regressions.  The generic comb/chain ratio looks
  smaller only because the chain was made drastically slower.  Thus the
  existing generic path is not a usable direct replacement; the missing piece
  is specifically a low-copy high-level seam usable by the specialized
  contraction, not merely a call to the generic contraction API.  The A/B
  switch is **[AI Supplied]**.

The child decode and gather copies are also implementation overhead and remain
valid suspects despite being only about 0.060 ms in this trace.  Their small
share here does not justify retaining them: it only establishes ordering for
future changes.

#### Deterministic bond-exponent regression gate

`raw_center_work_scales_with_coordination_number_and_actual_bond_dimensions`
is a release unit test for the exponent itself. Test-only instrumentation
increments a local counter in the actual innermost dense-core multiplication
loop and publishes it through a thread-local counter after the contraction;
production builds contain neither the counter nor its increments.

The test evaluates every physical value once and asserts:

- uniform z=2 at `chi = 64, 128, 256`: each doubling grows visits by `2^2`;
- uniform z=3 at `chi = 64, 128, 256`: each doubling grows visits by `2^3`;
- unequal bonds `[64, 128, 256]` at `d=3`: exactly
  `3 * 64 * 128 * 256 = 6,291,456` visits;
- real `f64` arithmetic at z=3, `chi=128` (about 32 MiB of dense core payload):
  the all-one result equals `128^3` at both physical values.

The exponent-only calls use a zero-sized test scalar, so the actual loop nest
executes through z=3, `chi=256` without allocating the otherwise 268 MiB `f64`
core. The separate `f64`, `chi=128` case prevents a counter-only change from
satisfying the test while still exercising representative dense storage.

The complexity expectation is **Hiroshi review**. The thread-local
instrumentation, zero-sized scalar, and fixture are **[AI Supplied]**.
This is deliberately a deterministic work-count gate rather than a flaky
wall-clock assertion. Criterion remains responsible for detecting constant-
factor and cache-management regressions that preserve the exponent.

### P1: the packed message cache still violates part of the review protocol

The persistent cache implements the broad shape of Hiroshi's PR #646 design,
but several details remain avoidable:

- `get_or_compute_node_message` maps each requested missing key back to its
  column with `missing_keys.iter().position(...)` and then `to_vec()`. For `m`
  misses this is `O(m^2)` key comparison plus one allocation/copy per message,
  before the columns are appended to the packed buffer.
- Every node/call clones `assignment_batch.first_points`; all-hit lookup
  allocates a vector of positions; a partial-hit path performs a second
  `get_all_cached(&hit_keys)` lookup and allocates positions only to discard
  them; and result reconstruction copies every requested packed column into a
  new flat vector.
- The miss path separately allocates `missing_points`, `missing_keys`, a
  `HashSet` for deduplication, a temporary `HashMap` for over-budget columns,
  and a `Vec<CacheSlot<_>>`.  An over-budget column is cloned out of the
  temporary map before being copied again into reconstructed output.
- `PackedMessageCache::retained_bytes` counts only
  `columns.len() * size_of::<T>()`. Hiroshi explicitly required real cache
  memory to include key/HashMap/capacity overhead, not payload alone.

A release representation census on this target measured:

```text
f64 = 8 bytes
CachedScalar = 24 bytes
IndexKey = 48 bytes
AnyScalar = 96 bytes
```

The packed cache stores `CachedScalar`, so an all-real message consumes three
times the scalar payload before accounting for `HashMap` buckets, the 48-byte
inline `IndexKey`, boxed limbs for wide keys, and unused capacities.  The
configured byte budget is therefore not a numerical-payload budget, and the
current `retained_bytes` name still understates the allocator-visible retained
memory.  A typed TreeTN evaluation seam like SimpleTT's `TTCache<T>` would
avoid the real/complex enum inflation, but that API/generalization is
**[AI Supplied]** and must be added upstream in TreeTN rather than by making
TreeACI depend on SimpleTT.

This is not the primary cost in the measured 16-site, `chi=128` floating-zone
walk: test-only phase timing reported contraction 82.8%, insert 7.7%, lookup
5.5%, reconstruction 4.0%. It should be fixed after the reuse-granularity
defect, not mistaken for the 300x warm-call cause.

### P0: arbitrary-degree branch contraction has a scalar efficiency cliff

Evidence:

- `frames.rs:735-790` dispatches exactly one incoming edge to the chain-batched
  path, exactly two to a special batched path, but zero or three-or-more to
  `candidate_frame` once per candidate.
- `frames.rs:1760-1845` recursively visits the Cartesian product of every
  incoming bond dimension for each scalar candidate and output-bond element.
- Candidates with three-or-more incoming branches are deliberately not cached
  (`frames.rs:590-599`).
- The existing release microbenchmark at a two-incoming hub (`m=40`, input bond
  `d=32`) measured the batched path at 2.49/3.62/3.98 ms and the scalar path
  still used for three-or-more incoming branches at 30.81/32.25/31.52 ms.
  Medians are 3.62 ms and 31.52 ms, about 8.7x. The test's per-run ratios were
  12.36x, 8.92x, and 7.92x.

The rank/product factor in (T2)/(T5) is mathematically real. The dispatch cliff,
repeated recursive calls, absence of caching, and owned result per candidate are
not required by that derivation and are **[AI Supplied]** implementation costs.

### P0: branch candidates and local matrices grow multiplicatively

`local_update.rs:312-364` implements (T5) literally and allocates one owned
`Vec<(edge,id)>` for every candidate. This is correct for the chosen two-site
tree-cut cross, but it means:

- hub-to-leaf at a degree-`q` hub: `C_hub = d*r^(q-1)`;
- degree-three to degree-three edge: `C_left*C_right = d^2*r^4`;
- callback count, local-matrix storage, and LUCI input size all inherit this
  product.

This is an algorithmic limitation of materializing the full edge cross (T4),
not merely a slow loop. Removing it requires a newly derived lazy/block/pivot
search formulation, not a mechanical optimization.

### P1: every edge commit repeats whole-tree metadata work

Two independent paths have chain-length quadratic shape:

1. `transaction.rs:98` clones `state.output` for every directed-edge commit.
   TreeTN `Clone` was checked as runtime evidence: `IdxTensor` numerical payloads
   are reference counted, so this is not a deep numerical copy, but the complete
   graph, canonical metadata, site-index network, link-index network, and
   orthogonality map are cloned. With `E` updates per pass and `V=E+1` on a
   chain, this is `O(E*(V+E)) = O(E^2)` metadata work.
2. `transaction.rs:130-133` calls `InputFrameStore::extend` after every edge.
   `frames.rs:293-518` rescans every input and every directed edge, recomputes
   counts/dimensions/accounting, allocates a memo spine across all retained
   samples, and rebuilds the outer store. Unchanged numerical frame payloads are
   shared with `Rc`, which is good, but the global metadata pass remains. Across
   `E` commits this is again at least `O(N_inputs*E^2)` metadata, plus work for
   newly interned dependencies.

Neither cost appears in paper Algorithm 3 nor simplett ACI. The chain reference
mutates only the two solution cores (`state.rs:843-849`) and updates only the
one directional frame required by the sweep (`state.rs:851-855`). These are
high-confidence scaling defects even though they need not dominate a small or
high-arithmetic-intensity case.

### P1: TreeACI repeatedly allocates and repacks local input data

For every input and edge, `local_update.rs:169-216` obtains
`Vec<Vec<T>>` row/column frames, builds `row_flat`, builds `col_flat`, allocates
the matrix product, then scatters it into an interleaved `input_values` buffer.
Cache hits in `frames.rs` also clone the cached `Vec<T>`. Candidate construction
itself allocates an incoming-pair `Vec` per candidate.

The simplett reference builds compact local factors once per update and, for
same-shaped inputs, materializes all inputs with one batched GEMM
(`local.rs:304-374`). Its core matrices are prepared once for the whole run
(`state.rs:40-65,77-87`). TreeACI prepares raw cores once, but repacks
single-/two-incoming core matrices in the candidate routines on every call.

These data movements are **[AI Supplied]**, not consequences of (T2)--(T4).

### P2: the two-incoming batched kernel is still fragmented

`frames.rs:972-1173` builds BTreeMap/HashMap groups and gathers input frames for
each physical coordinate. `frames.rs:1948-1977` then repacks one core slice and
launches one GEMM for every value of the second incoming bond, followed by a
final GEMM: `incoming_dim_2 + 1` calls per group. It is much faster than the
scalar fallback in the measured fixture, but its launch/allocation structure is
not prescribed by the tree derivation and remains a likely branch hot spot.

### Not a root cause: traversal does not add extra edge updates per round

The minimum-retracing forward walk has `2E - diameter` steps and the reverse
spine has `diameter` steps. Together they visit every directed orientation once,
so a full forward/reverse round performs `2E` local updates, the same count as
two chain sweeps. Branch edges are retraced geometrically in the forward walk,
but not updated more than twice per round overall. Traversal order may affect
rank trajectories and cache locality, but update count alone does not explain
the branch slowdown.

### Not a repeated full-sweep cost: deferred canonicalization

`schedule.rs:250-266` calls full-tree CI canonicalization only while
`canonical_form()` is `None`. Edge replacement moves the canonical region but
does not clear the form, so after the first finalization this is not repeated on
every pass. It is a one-time analogue of simplett ACI's initial right
canonicalization, though the post-first-pass timing is **[AI Supplied]**.

## 2026-09-02 cumulative-overhead follow-up

The following items remain suspects even where their individual measured share
is small.  The classification is deliberately per mechanism rather than an
attempt to name one exclusive root cause.

### Confirmed: an owned local-matrix path was available upstream but unused

The one-incoming candidate-frame path, one-incoming stored-frame path, and
final local row-by-column materialization passed freshly allocated matrices to
borrowed `tensor4all_tensorbackend::mat_mul`.  That function must clone its
inputs while the already-existing `mat_mul_owned` consumes them and reuses the
buffers when constructing tenferro tensors.  An isolated test-only switch over
exactly those three call sites reduced the median combined initialization plus
two-sweep time of the 32-site, two-input, `chi=256` chain by about 17.6% across
three interleaved runs.  The full TreeACI release suite passed with the switch.

This is direct **upstream implementation evidence**.  It is not a tree
derivation and does not use simplett as a dependency.

### Confirmed: immutable input cores are repacked repeatedly

At `chi=256`, the two one-incoming paths performed 372 oriented-core packs but
only 120 distinct `(input, directed edge)` identities existed.  They copied
30,233,856 scalar values in total versus 9,087,616 distinct oriented values:
21,146,240 redundant `f64` copies, or about 161.3 MiB.  Candidate packs took
38.3--39.2 ms and stored-frame packs 48.4--49.4 ms in representative runs.

The packed matrix is a pure function of an immutable prepared input core and a
directed edge's axis order.  Caching that orientation follows from code
inspection and is **[AI Supplied]**.  As a comparison-only legal reference,
simplett ACI prepares `InputCoreMatrices` once in
`tensor4all-aci/src/state.rs`; TreeACI must not depend on it.

### Confirmed: candidate cache is useful but has an expensive miss layout

Disabling the candidate cache made the `chi=256` candidate row/column phase
about 9% slower and increased candidate core packs from 160 to 240, so removing
the cache is not a remedy.  However, the enabled cache retained 20,580 entries
and 38,731,712 payload bytes for only 1,004 hits and 20,580 misses in the
two-sweep fixture.  Its payload slightly exceeded the base-frame payload.

On a miss, the batched `Matrix` result is extracted into many `Vec<T>` values,
cloned again into the cache, returned as `Vec<Vec<T>>`, and immediately packed
back into a `Matrix` in `local_update.rs`.  At `chi=256`, result extraction plus
cache insertion cost about 17 ms and the later local repack about 6--8 ms.
This representation cycle and a future packed cache/result seam are
**[AI Supplied]**; the measured timings and allocation sites are runtime/code
evidence.

### Confirmed small cost: frame growth recopies old prefixes

Across 62 commits of the no-Guard `chi=256` run, `InputFrameStore::extend`
initialized 1,217,338 memo slots, copied 4,153,180 old values, and copied only
75,924 newly computed values.  Old-prefix copies outnumbered new values by
54.7x.  Fixed scanning/setup was only about 1.8 ms and rebuilding about
3.6--3.8 ms in this fixture, so neither dominates here; both remain scaling
suspects because the operation is repeated after every edge commit.  A
persistent/chunked frame representation is **[AI Supplied]**.

### Confirmed upstream constructor mismatch during output commits

`TreeTN::clone` shares every `IdxTensor` numerical payload through `Arc`; it is
not a deep tensor copy.  For the no-Guard `chi=256` run, 62 whole-tree metadata
clones cost about 0.95--1.06 ms.  The rest of output staging was dominated by
constructing the two replacement tensors: about 8.07 ms of an 8.53 ms
`replace_edge_cores` total.  Bond replacement was about 0.09 ms and both tensor
replacements about 0.19 ms.

The cause is an upstream ownership mismatch:
`IdxTensor::from_dense(indices, data: Vec<T>)` consumes a vector, but calls the
slice-based `dense_native_tensor_from_col_major(&data, ...)`; the
`TensorElement` implementation then calls `data.to_vec()` before constructing
the native tensor.  Thus every left/right factor payload is copied once and
the original allocation is discarded.  No public generic owned constructor
exists in the API inventory.  `IdxTensor::from_storage` can preserve ownership
for the storage-supported scalar kinds: composing it with
`Storage::from_dense_col_major` already gives the common `f64`/`Complex64`
paths an ownership-preserving public route.  That two-step compact-storage API
does not support the also-public `f32`/`Complex32` TreeACI contract, so it is
not a generic TreeACI fix; whether downstream use of the explicit storage
representation is the desired abstraction also needs layering review.  The
complete remedy is a generic owned core/tensorbackend constructor seam, not a
TreeACI-to-tenferro reach-through.  This conclusion is direct **upstream API
and implementation evidence**.

### Confirmed related ownership losses in full-rank CI/LU factorization

The one-time deferred CI canonicalization measured about 14.6--16.2 ms in the
no-Guard chain profile (roughly 4--9% of the measured sweep phase, depending on
the input bond dimension).  Its implementation contains several independent
copies.  They remain findings even though this canonicalization is not repeated
after `canonical_form()` becomes `Some(CI)`:

1. `eager_tensor_to_matrix` and `native_tensor_to_matrix` first extract a
   column-major `Vec<T>`.  `matrix_from_col_major_values` then allocates
   `Matrix::zeros(m, n)` and copies the same column-major values into it with a
   nested loop.  The existing upstream
   `Matrix::try_from_col_major_vec(m, n, data)` accepts that allocation
   directly.  The hand-written loop therefore also performs an unnecessary
   zero-fill before overwriting every entry.
2. `factorize_ci_with_options` passes that newly owned matrix to
   `matrix_luci_factors_from_matrix(&a_matrix, ...)`.  The borrowed facade calls
   `rrlu`, whose documented and implemented behavior clones the entire matrix.
   The already-existing `matrix_luci_factors_from_matrix_owned(a_matrix, ...)`
   calls `rrlu_mut` on the consumed buffer instead.  TreeACI's edge-local LUCI
   path already uses this owned facade correctly.
3. The related `factorize_lu_with_options` path has the same input-side issue:
   it creates an owned matrix and immediately calls `rrlu(&a_matrix, ...)`
   despite the existing `rrlu_mut` seam.
4. Both factorization paths convert owned output matrices with the local
   `matrix_to_vec(&matrix)`, which iterates and clones every scalar.  The
   upstream `Matrix::into_col_major_vec` consumes the matrix allocation
   directly.  The resulting `Vec` is then passed through the separately
   confirmed copying `IdxTensor::from_dense` constructor.

These statements are direct **upstream API and implementation evidence**, not
an inferred tree algorithm.  Fixes belong in `tensor4all-core` and
`tensor4all-tensorbackend`; TreeACI must not grow a parallel conversion or a
direct tenferro dependency.  Whether the entire deferred canonicalization can
be removed after a continuous TreeACI pass has not been proved: that would
require a separate tree-CI invariant derivation and remains **[AI Supplied]**.

### Confirmed upstream borrowed-read seam is missed in smaller hot paths

`IdxTensor::to_vec` always duplicates/materializes the tensor value and then
copies its slice into a new `Vec`.  The existing upstream
`IdxTensor::with_dense_slice` explicitly provides the same column-major values
without a new vector for an ordinary host-contiguous tensor, falling back to
materialization only when borrowing is impossible.  The following TreeTN/
TreeACI paths call `to_vec` even though their values do not escape the function:

- real and complex raw leaf-message construction;
- real and complex leaf-center contraction from a raw cached environment;
- the tensor-backed leaf-center fallback, for both the center and environment;
- TreeACI's post-Guard output-core padding before it fills the larger buffer.

The same TreeTN file already uses `with_dense_slice` for internal chain/branch
cores and internal-center contraction, so this is an existing upstream seam,
not a proposed dependency or a simplett reuse.  The leaf tensors contain only
`d*chi` values and output padding measured about 7--8 ms in the earlier Guard
profile, so these copies are not promoted to the sole root cause; they remain
independent defects.  Rewriting the loop bodies inside nested borrowed-read
closures is a mechanical implementation task, but it has not yet been applied
to production.

### Confirmed P0 for default Guard: every result creates a rank-zero tensor

`TreeTNCachedEvaluator::evaluate_batched_with_hint` returns
`Vec<AnyScalar>`.  `AnyScalar::new_real`/`new_complex` calls
`AnyScalar::from_value`, which eagerly initializes `IdxTensor::scalar(value)`;
it is not a lightweight numeric enum.  TreeACI Guard immediately converts each
result back to its generic `T` through `TreeAciScalar::from_evaluated_scalar`.

The fully warm 16-site, 64-point chain fixture measured only the final output
wrapping as 2.61, 2.98, and 3.03 ms/call at `chi=64,128,256`, respectively --
roughly 41--47 microseconds per scalar and independent of the bond contraction.
The actual raw center contractions were 0.26, 1.04, and 4.30 ms/call.

The older, non-authoritative
`2026-08-18-treeaci-message-cache-prototype.md` is a useful clue: its Update 13
already found the same rank-zero construction inside the message cache and
replaced cached values with private lightweight `CachedScalar`.  The current
source confirms that fix stopped at internal cache storage; the final public
batch result is still converted back into one `AnyScalar` per point.  The old
worklog is not used as proof for the present claim; the source path and timings
above independently reproduce it.

The legal simplett comparison makes the boundary difference concrete:
`TTCache<T>::evaluate_many` returns `Vec<T>`, and simplett ACI's Guard consumes
those typed values directly.  TreeTN exposes only `Vec<AnyScalar>` from its
batched evaluator in the current API inventory.  Its private `CachedScalar`
keeps internal messages lightweight, but final raw center results are converted
back to `AnyScalar` before TreeACI immediately converts them to `T`.  Thus a
typed TreeTN result seam is currently missing rather than an existing simplett
function that TreeACI is permitted to call.

The 32-site TreeACI fixture with the default Guard enabled measured:

| input bond | Guard search | pivot injection | complete sweep time | combined Guard share | input evaluation | output evaluation |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.096 s | 0.137 s | 1.531 s | 80.5% | 0.750 s / 8,460 results | 0.341 s / 4,195 results |
| 128 | 1.344 s | 0.223 s | 2.027 s | 77.3% | 0.979 s / 8,588 results | 0.359 s / 4,259 results |
| 256 | 2.265 s | 0.366 s | 3.241 s | 81.2% | 1.856 s / 8,844 results | 0.402 s / 4,387 results |

Each run completed seven sweeps because Guard injected pivots; the no-Guard
fixture completed two.  These timings therefore compare phase shares within
each run, not total algorithm parity between Guard modes.  They establish that
Guard dominates the default path and that thousands of dynamic-scalar tensor
initializations are real work.  A generic typed TreeTN batch-evaluation API is
the **[AI Supplied]** proposed upstream seam.

Injection was measured separately because it is outside
`find_global_pivots`.  Its dominant subphase was the one post-injection
`InputFrameStore::extend`: 127.7, 213.9, and 355.6 ms at
`chi=64,128,256`, respectively.  Candidate-set cloning cost less than 0.9 ms,
global-point projection less than 0.7 ms, and output padding about 7--8 ms.
Thus the injection extension is another material contributor, while the
whole-candidate clone is currently only a scaling suspect.

A direct complete-ACI chain comparison now separates the local-sweep and
default-Guard cases at input bond 256.  Both arms used the same deterministic
16-site inputs and first-input initial guess.  In each comparison they
completed two sweeps with maximum output rank 17, and dense numerical checks
passed:

| mode | simplett ACI | TreeACI | Tree/simplett | evaluated points (simple/tree) |
|---|---:|---:|---:|---:|
| Guard disabled | 87.274 ms | 77.003 ms | 0.88x | 46,732 / 36,516 |
| Guard enabled | 151.85 ms | 374.37 ms | 2.47x | 47,904 / 37,752 |

The relative dense max errors in the Guard run were `1.442e-8` and
`1.343e-8`, respectively.  With Guard disabled, TreeACI is about 11.8% faster;
with the default Guard it is about 147% slower despite requesting fewer
operator points.  Subtracting the matched no-Guard medians gives approximately
64.6 ms of added simplett work versus 297.4 ms of added TreeACI work, a 4.6x
larger Guard increment.  This subtraction is a diagnostic attribution rather
than an algorithmic identity, but the equal sweep/rank outcomes and nearly
equal additional point counts (`1,172` versus `1,236`) make it substantially
more specific than an unmatched end-to-end ratio.

The opt-in benchmark mode selected by
`T4A_TREEACI_PARITY_ENABLE_GUARD=1` and the subtraction above are
**[AI Supplied]**.  They reproduce the reported chain regression on the default
path while showing that the local chain sweep itself is not currently slower
in this representative case.  The separately matched evaluator benchmark and
source paths remain the causal evidence for the warm-center and `AnyScalar`
costs; a whole-ACI timing alone cannot distinguish them.

Likewise, the older audit notes say injection was changed from rebuilding every
frame from scratch to `extend`.  That historical statement is only a clue.  The
current counters show the incremental replacement is still substantial because
new global samples grow many directed frames and every grown buffer recopies
its retained prefix.

### Confirmed: warm cache and center work solve different problems

Repeating an identical batch after all directed messages were warm produced
zero message contractions and zero insertions, yet center contraction still
accounted for 86.0%, 87.5%, and 94.8% of calls at `chi=64,128,256`.  At an
internal chain center the current raw kernel visits `P*chi^2` core elements;
at a degree-`z` center its direct generalization visits
`P*d*product(chi_i)` local elements.  The component conversion itself copied
8,192/16,384/32,768 values per call but cost only 0.009/0.017/0.037 ms, so it is
a valid small overhead rather than the main center cost.

Splitting the warm environment phase further retained several smaller
contributors.  On the same 16-site, 64-point batches, rebuilding compact
assignments cost about 0.122--0.123 ms/call, the all-hit message walk and
reconstruction cost 0.189--0.241 ms/call, the immutable raw-path capability
scan cost 0.019--0.024 ms/call, and final component assembly about 0.001
ms/call.  These are below the center cost in this fixture, but Guard invokes
the evaluator thousands of times on small batches, where the fixed work is less
amortized.

The assignment builder explains the fixed allocation pressure.  For the
ordinary one-physical-index-per-node case, every node/point builds one local
coordinate `Vec`, `validate_entry_values` builds another `Vec` of cloned-index
pairs, and `build_compact_assignment_batch` builds a third `Vec` for the local
and child assignment IDs.  The 16-site, 64-point fixture therefore creates at
least 3,072 short-lived vectors per evaluator call before cache-key and result
allocations.  Direct bounds checks and an allocation-free compact assignment
key are **[AI Supplied]** implementation directions.

The actual Guard workload makes those fixed per-call costs more important than
the 64-point evaluator microbenchmark suggests.  An opt-in 32-site, two-input
chain profile made about 2,100--2,200 input-evaluator calls and the same number
of output-evaluator calls per run, with only about two points per call:

| input bond | Guard search | input evaluation | input ms/call | output evaluation | output ms/call |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.064 s | 725.4 ms / 2,122 calls | 0.342 | 333.6 ms / 2,115 calls | 0.158 |
| 128 | 1.326 s | 964.2 ms / 2,154 calls | 0.448 | 356.4 ms / 2,147 calls | 0.166 |
| 256 | 2.197 s | 1.793 s / 2,218 calls | 0.809 | 395.9 ms / 2,211 calls | 0.179 |

Input plus output evaluation accounts for 99.5--99.7% of the measured Guard
search time.  The input cost grows with the deliberately large fixed input
bond, while the learned output remains much lower rank and its cost is nearly
flat.  This is direct evidence that the reproduced chain regression is the
high-rank input evaluator being invoked thousands of times in tiny batches,
not an unexplained cost in the random-walk bookkeeping.  The phase counters
and timing fixture are **[AI Supplied]**; the Guard algorithm itself is the
authorized simplett comparison.

There is another independent lifetime mismatch.  Input evaluators are retained
across Guard invocations, but `find_global_pivots` constructs a fresh output
evaluator on every invocation.  The output tensors change between sweeps, so
their numerical messages cannot simply be reused.  However, the immutable
topology, rooted plans, and directed-component layouts are discarded with the
numerical cache and rebuilt.  Separating those lifetimes or sharing immutable
plans is **[AI Supplied]** and its whole-run benefit has not yet been isolated;
it remains a valid small suspect rather than the primary high-rank cost above.

There is also a concrete existing upstream facility that the current tree
cache does not use.  `tensor4all-core::index_key::KeyBuilder` documents and
implements append-style tree-key composition (`local ++ child_1 ++ ...`) and
normalizes the result to the same opaque `IndexKey` as direct
`FlatIndexer::encode`.  `TreeTNCachedEvaluator` instead stores every rooted
subtree's full physical-position list, gathers that complete coordinate vector
again for each requested message key, and directly re-encodes it.  Reworking
the cache-key pipeline around the upstream composition API requires a careful
ownership/uniqueness design and is not yet measured as a replacement, but the
current full-subtree gather is not justified by the absence of an upstream
primitive.

The provenance is unusually explicit: upstream commit `7f56754` is titled
`core: checked bit-packed index-key encoder for tree cache keys`, and the
`KeyBuilder` rustdoc itself states the tree composition rule.  The existing
upstream Criterion fixture also permits a primitive-only comparison.  On this
machine a 64-bit binary key took about 51.2 ns to encode directly and 29.1 ns
to compose from four already-encoded pieces.  This does **not** predict a 1.76x
evaluator improvement: the current and proposed pipelines allocate and retain
different surrounding data, and a chain still copies successively wider keys.
It does establish that the intended upstream composition operation is both
present and individually cheaper in this representative width.  Applying it
to evaluator ownership and traversal is **[AI Supplied]** until an end-to-end
A/B exists.

The layout ownership is also keyed at the wrong granularity.  Numerical
message caches are correctly keyed by `(from, to)`, but
`message_cache_layouts_by_center` retains a complete node-to-layout map for
every center, and each `RootedMessagePlan` retains every node's full subtree
list solely to build those layouts.  A directed component and its physical-key
layout depend on `(from, to)`, not on which more distant node was selected as
the center.  Guard visits every site as the varying center, so a length-`N`
chain retains `O(N^3)` physical-position/node references rather than the
`O(N^2)` total content of its `2E` distinct directed components.

A moving-center `N=16`, `chi=256` diagnostic confirmed the exact counts after
all 16 centers had been visited: 1,616 retained subtree-node references and
1,616 retained layout-position references versus 240 positions across unique
directed components, a 6.73x duplication before counting the `FlatIndexer`
dimension/offset vectors and `HashMap` capacities.  Fifteen newly visited
centers built 1,536 references of each kind; plan construction took 0.214 ms
and layout construction 0.122 ms in that run.  The first complete center scan
took 8.906 ms and the second 4.373 ms, but their difference also includes
new-message/cache warming and is not attributed wholly to metadata.

The count follows directly from the retained structures and the directed-cut
identity.  Moving immutable topology and physical key layouts to one
directed-edge table is **[AI Supplied]** implementation design.  The existing
TreeTN `CachedTopology` and lazy `FitEnvironment::get_or_compute` demonstrate
that fixed topology and hit-before-recursion patterns already exist upstream,
but their linsolve/fit-specific types do not directly implement an
assignment-keyed evaluator cache and should not be coupled into this module as
an ad hoc fix.

The eager postorder walk also materializes cache hits that cannot be observed.
In the fully warm `N=16`, 64-point chain call, 444 cached message columns were
reconstructed at every bond dimension.  At `chi=256` that copied 113,664
`CachedScalar` values, or about 2.60 MiB at the measured 24-byte representation.
Only the two center-adjacent environments, 32,768 values (0.75 MiB), reached
the center.  The remaining 80,896 values (about 1.85 MiB, 71.2% of the
reconstruction) were descendant messages built before an already-cached parent
message was returned and never consumed.  The same reconstructed/final ratio
appeared at `chi=64` and `128`; wall time was 0.147--0.290 ms/call for the
reconstruction portion in representative runs.

A top-down request for each center-adjacent directed message can test that
message's exact subtree key first and recurse into children only on a miss.
This follows the cache dependency itself and matches the lazy-hit pattern
already used by TreeTN's fit environments, but adapting it to per-assignment
batch compaction is **[AI Supplied]**.  It is independent of the edge-centered
final contraction: lazy message lookup removes dead descendant work even if
the current vertex-center arithmetic remains temporarily unchanged.

Simplett's legal comparison path cuts an edge and combines two cached side
messages with a length-`chi` dot product.  Extending that identity to an
arbitrary tree edge is the previously recorded **[AI Supplied]** edge-cut
derivation; it changes desired warm complexity rather than merely tuning the
current vertex-center loop.

### Measured small/absent pass overheads retained in the ledger

Cloning the forward/reverse schedule for a pass cost less than one microsecond
in the high-rank chain fixture, so it is not a current contributor, although
borrowing the immutable plan would still remove needless work.  The
`FrameBuilder` memo-hit branch, which clones a cached vector when reached,
recorded zero memo-hit clones in the same chain runs (19,888 batched/scalar
computations at `chi=256`).  It is therefore absent from this chain trace, not
proved harmless for branch scalar fallbacks.  Both observations are retained
to avoid conflating “small or not exercised in this fixture” with “not a
performance defect.”

## 2026-09-03 #707 closure: owner-layer, dependency, and source-evidence matrix

This section closes the dependency-side audit requested by #707. The original
line-range ledger above remains a historical audit of `fd61f082c7d2234db2e7b88e73e5fd1f5a0c4228`.
The closure snapshot below was checked after merging current `origin/main` into
`2959e3bb434d82d400652fe4896ef97559d4dfc6`, which contains
`750f7711e4d2cf64528d281d0e2606a7c0afa90e` as its current base. Current API
names and source locations in this section were checked against the generated
`target/api-dump/` inventory and the current worktree source.

### Full-text source record

The cited papers were downloaded as complete PDF and TeX/source archives before
being used. The extracted `paper.txt` files were read through their final page,
including appendices, algorithms, listings, and references. The tenferro
repository was cloned and detached at the revision selected by the current
Cargo dependency. These artifacts are ignored build/audit material under
`target/literature/` and are not source-code dependencies of TreeACI.

| source | local full-text clone | source revision/version | archive SHA-256 | accessed | page checklist |
|---|---|---|---|---|---|
| Ritter, “Fast elementwise operations on tensor trains with alternating cross interpolation” ([arXiv:2604.00037](https://arxiv.org/abs/2604.00037)) | `target/literature/aci-2604.00037v2/{paper.pdf,source.tar,paper.txt,main.tex,algorithms.tex}` | `2604.00037v2` | PDF `1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369`; source `b0b3a61be95737a59a7c98d35915826041bd1837efd60cb081f4f3d2f4acdc0a` | 2026-09-03 | PDF/printed pages `1..17` read; main text, §2.1–§2.8, §3–§4, conclusion, Appendix A, Algorithms 1–5, and references read |
| Núñez Fernández et al., “Learning tensor networks with tensor cross interpolation: new algorithms and libraries” ([SciPost Phys. 18, 104](https://doi.org/10.21468/SciPostPhys.18.3.104)) | `target/literature/tci-2407.02454v3/{paper.pdf,source.tar,paper.txt,xfac_paper.tex}` | `2407.02454v3` | PDF `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf`; source `53280bb88d34de7943ef48767f7f94a6789588bf67da1a17ca6c6050c875494a` | 2026-09-03 | PDF/printed pages `1..75` read; main text, §2–§9, Appendices A–B, Algorithms/listings, and references read |
| tenferro-rs specification and implementation | `target/literature/tenferro/repository` | git `007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4` (`fix eager extension dispatch for runtime placement (#1753)`); Cargo source block is recorded by the resolved lockfile | N/A for git checkout; commit is immutable locator | 2026-09-03 | `docs/spec/index.md`, `backend-contract.md`, `tensor-semantics.md`, `ad-contract.md`, `primitive-catalog.md`, `api-conventions.md`, and the grouped-GEMM implementation/tests were read |

The tenferro specification is an ownership and layout authority only. It does
not provide TreeACI pseudocode. The ACI paper's conclusion says a tree version
is straightforward, but supplies no tree equations or tree pseudocode; all
concrete tree identities below therefore remain explicitly
**Tree generalization — re-derived** or **[AI Supplied]**.

### Concrete source-evidence register

The IDs in the owner matrix refer to these concrete locators. `pdf_page` and
`printed_page` are both recorded even when they are equal, so a later version
or preprint pagination change cannot silently move an algorithm citation.

| claim_id | authority_label | source_url_or_doi | local_archive_or_repo | source_commit_or_version | archive_sha256 | accessed_date | pdf_page | printed_page | section_or_subsection | equation_number_or_algorithm_and_line_range | paragraph_heading_and_short_anchor | supported_claim | validation_test_or_benchmark |
|---|---|---|---|---|---|---|---:|---:|---|---|---|---|---|
| `ACI-C1` | `Paper` | `https://arxiv.org/abs/2604.00037` | `target/literature/aci-2604.00037v2/paper.pdf` | `2604.00037v2` | `1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369` | 2026-09-03 | 3 | 3 | §2.1 Problem statement | Eq. (3) | “In practice, this means y should fulfill” | ACI's stated correctness target is a maximum absolute error bounded by `tau`. | Dense TreeACI result residuals in `crates/tensor4all-treeaci/src/elementwise/tests/mod.rs`; parity harness recorded below. |
| `ACI-C2` | `Paper` | `https://arxiv.org/abs/2604.00037` | `target/literature/aci-2604.00037v2/paper.pdf` | `2604.00037v2` | `1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369` | 2026-09-03 | 4 | 4 | §§2.5–2.6 Local problem and index-set optimization | Eqs. (8a–8b), (9), (10), (11) | “precomputing left and right frame matrices”; “approximately factorize” | Nested frame updates reduce repeated TT evaluation and local updates apply the elementwise function before cross factorization. | `frames.rs` scalar-vs-batched tests and TreeACI elementwise dense parity. |
| `ACI-C3` | `Paper` | `https://arxiv.org/abs/2604.00037` | `target/literature/aci-2604.00037v2/paper.pdf` | `2604.00037v2` | `1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369` | 2026-09-03 | 5 | 5 | §2.7 Complexity analysis | Paragraph immediately after §2.7 heading; Eq. (10) referenced | “most expensive step in each local update” | The chain ACI claim is `O(N_sweep L N d^2 chi^3)` when input/output ranks are comparable; this is not a tree complexity claim. | Release scaling measurements in the audit benchmark; no tree claim promoted from this row. |
| `ACI-C4` | `Paper` | `https://arxiv.org/abs/2604.00037` | `target/literature/aci-2604.00037v2/paper.pdf` | `2604.00037v2` | `1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369` | 2026-09-03 | 11 | 11 | Appendix B, pseudocode | Algorithm 1, lines 1–22 | “Main loop”; forward and reverse sweep blocks | ACI initializes frames, performs forward/reverse local updates, updates frames, and stops on rank/error conditions. | `schedule.rs` update-count tests and TreeACI chain parity benchmark. |
| `TCI-C1` | `Paper` | `https://doi.org/10.21468/SciPostPhys.18.3.104` | `target/literature/tci-2407.02454v3/paper.pdf` | `2407.02454v3` | `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf` | 2026-09-03 | 8 | 8 | §3.1 Matrix cross interpolation | Eqs. (6)–(9) | “pivot matrix”; “CI formula” | A matrix approximation is formed from selected rows, columns, and the inverse pivot matrix; these are the chain CI ingredients only. | `matrix_luci` reconstruction tests and local-update dense residual gate. |
| `TCI-C2` | `Paper` | `https://doi.org/10.21468/SciPostPhys.18.3.104` | `target/literature/tci-2407.02454v3/paper.pdf` | `2407.02454v3` | `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf` | 2026-09-03 | 12–13 | 12–13 | §3.3.1 Default full-search prrLU | Eqs. (27)–(32) | “The main advantage of prrLU over a direct CI” | prrLU avoids explicitly inverting an ill-conditioned pivot matrix and is used as the stable CI-equivalent factorization. | `matrix_luci_factors_from_matrix_owned` tests and numerical `N` gate. |
| `TCI-C3` | `Paper` | `https://doi.org/10.21468/SciPostPhys.18.3.104` | `target/literature/tci-2407.02454v3/paper.pdf` | `2407.02454v3` | `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf` | 2026-09-03 | 14 | 14 | §3.3.2 Alternative pivot search methods | Algorithm 1, lines 1–14 | “Block rook pivoting search” | Block rook updates reuse prior pivots and searches selected rows/columns; this supports a future grouped-pivot optimization, not a tree proof. | #712 grouped-GEMM A/B plus scalar pivot oracle. |
| `TCI-C4` | `Paper` | `https://doi.org/10.21468/SciPostPhys.18.3.104` | `target/literature/tci-2407.02454v3/paper.pdf` | `2407.02454v3` | `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf` | 2026-09-03 | 18–19 | 18–19 | §§4.3.1–4.3.3 2-site TCI and CI/prrLU | Eqs. (39)–(41); basic-algorithm steps (1)–(3) | “partial nesting is sufficient” and “reset mode” | Local two-site error is tied to the global error on a nested slice; reset mode may discard bad pivots; this does not authorize a tree slice identity. | TreeACI local-update residual, pivot/rank parity, and `N` convergence gates. |
| `TCI-C5` | `Paper` | `https://doi.org/10.21468/SciPostPhys.18.3.104` | `target/literature/tci-2407.02454v3/paper.pdf` | `2407.02454v3` | `f8d6c5e1dc19350d896f6a4f2bc9c29213d9752b1f45943819d130d2b7379bdf` | 2026-09-03 | 20 | 20 | §4.3.5 Proposing pivots from outside TCI | Paragraph beginning “Given a list of global pivots” | “split each index sigma” and add it to pivot lists | Global pivots enrich local exploration and preserve nesting in the chain algorithm; TreeACI projection remains separately re-derived. | Global-guard injection and rollback tests. |
| `TEN-C1` | `tenferro-rs specification` | `https://tensor4all.org/tenferro-rs/spec/` | `target/literature/tenferro/repository/docs/spec/backend-contract.md` | `007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4` | `N/A (git)` | 2026-09-03 | `N/A` | `N/A` | §IV Dispatch Categories; §VI Backend Traits; §VII Layout and Device Contract | `backend-contract.md:194–208, 286–313, 317–330` | “eligible for grouped segmented execution”; “dense contiguous column-major tensors” | Backend-session operations are the grouped-execution boundary, while runtime tensors are dense column-major; downstream crates must use a facade. | `tensor4all-tensorbackend/src/matrix.rs` layout tests and #712 provider/fallback gate. |
| `TEN-C2` | `tenferro-rs specification` | `https://tensor4all.org/tenferro-rs/spec/` | `target/literature/tenferro/repository/docs/spec/tensor-semantics.md` | `007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4` | `N/A (git)` | 2026-09-03 | `N/A` | `N/A` | §II Metadata-only views; §III runtime tensors; §VII Linalg Batch Convention | `tensor-semantics.md:31–70, 133–162, 258–270` | “Views may be non-contiguous”; “Owned runtime tensors are compact column-major tensors” | Views are metadata-only until materialization, and trailing batch slices are contiguous in column-major storage; no arbitrary-stride promise may be inferred for a dense facade. | `with_dense_slice`/matrix conversion tests and #714/#716 fallback parity. |
| `TEN-C3` | `tenferro-rs specification` | `https://tensor4all.org/tenferro-rs/spec/` | `target/literature/tenferro/repository/docs/spec/ad-contract.md` | `007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4` | `N/A (git)` | 2026-09-03 | `N/A` | `N/A` | Core Primitive Rule Contract; Mode Interpreters and Cacheability | `ad-contract.md:23–60, 105–165` | “Rules emit graph operations; they do not execute tensors, read runtime caches” | AD graph rules and runtime/cache ownership are separate; TreeACI must not claim trace preservation merely because a dense primal result is numerically correct. | #718 AD/trace smoke and unsupported-path gate; record N/A if the ACI route is primal-only. |

### Owner-layer matrix for every #707 operation

`C` is the correctness oracle, `E` is the efficiency counter/timing gate, and
the secondary gate letters are the global contract (`N`, `M`, `F`, `I`, `D`,
`S`, `P`). A row marked **[AI Supplied]** is an engineering hypothesis or
measurement policy, not a literature claim. A row marked **Tree generalization
— re-derived** is justified only by the explicit cut identities `(T1)`–`(T6)`
above and must retain its differential test.

| #707 operation | owning layer and current seam | planned subissue | tenferro status / dependency decision | authority label and exact locator | scalar/layout scope | `C` gate | `E` gate |
|---|---|---|---|---|---|---|---|
| planning, directed-edge records, and dependency order | `tensor4all-treeaci`: `problem.rs:47–307,382–423`; `path_cover.rs:49–115,272–356`; `schedule.rs:73–174` | #707 / #718 | No tenferro dependency; topology and scheduling stay above the backend. | **Tree generalization — re-derived** from `(T1)`–`(T2)`; chain analogue `ACI-C4`; walk policy is **[AI Supplied]**. | all four ACI scalar kinds; column-major callback batches | every directed orientation exactly once per round; chain has the same update count as two sweeps; error/empty-tree paths | count path steps, edge updates, schedule clones, and peak schedule allocations at 1x/2x/4x nodes (`S`, `D`) |
| key construction and component-key composition | `tensor4all-core`: `index_key::FlatIndexer::encode`, `KeyBuilder` at `index_key/mod.rs:310–329,402–521`; TreeTN call sites in `cached_evaluator.rs:1407–1468,2930–3310` | #710 | `KeyBuilder` is already a core seam; no direct tenferro API is needed. | **[AI Supplied]** key layout; `TEN-C2` establishes metadata/layout boundaries; tree component identity is **Tree generalization — re-derived**. | `usize` site coordinates; opaque bit-packed keys; preserve deterministic append order | composed key equals direct encoding for all widths, duplicates, empty components, and overflow errors | ns/key, temporary key allocations, retained key bytes, and hit/miss counters for 1x/2x/4x component widths (`E`, `I`, `D`, `S`) |
| cold evaluator setup | `tensor4all-treetn`: `TreeTNCachedEvaluator::new` and `evaluate_batched` at `cached_evaluator.rs:1024–1122`; ordinary oracle `evaluator.rs:111–291` | #711, then #708/#709 | No direct tenferro reach-through; the evaluator may use backend operations only through existing TreeTN/core seams. | API behavior **[AI Supplied]**; frame motivation `ACI-C2`; public evaluator oracle is current source at the cited lines. | f32, f64, Complex32, Complex64; values `[n_indices,n_points]`, column-major | compare complete output to `TreeTN::evaluate` in one dense result, including empty/single-node/chain/star | cold wall time, plan builds, message contractions, allocations, and peak cache bytes; paired release medians (`E`, `R`) |
| full-hit batches | `cached_evaluator.rs:2930–3310` (`get_or_compute_node_message`) and `4740–4910` (`PackedMessageCache`) | #708, #710 | Cache is TreeTN-owned; tenferro is not the cache authority. | **[AI Supplied]** warm-hit policy; cut reuse is **Tree generalization — re-derived** via `(T6)`; `TEN-C3` forbids using runtime caches as AD rule state. | all dtypes; reordered and duplicate columns preserve output order | warm result equals ordinary evaluation and cold result bitwise/within existing dtype envelope; no stale generation | zero misses/contractions where policy promises a hit; reconstructed values and retained bytes must not grow unbounded (`E`, `I`, `D`) |
| partial-hit batches | `cached_evaluator.rs:1407–1468,2930–3310` assignment and message-cache paths | #710 | Core `KeyBuilder` may reduce key assembly; no tenferro change required. | **[AI Supplied]** cache-state model; `TCI-C4` only supports chain nesting/error reasoning, not this cache. | all dtypes; arbitrary point order with duplicate columns | compare mixed old/new assignments against a fresh evaluator and ordinary dense result; validate exact output permutation (`M`, `F`) | count hit/miss split, recomputation, key allocations, and latency by hit ratio (0/25/50/100%) |
| cache miss and message contraction | `cached_evaluator.rs:2131–3310,3353–3409`; generic fallback `evaluator.rs:294–347` | #710, #711 | Backend calls remain behind TreeTN/core; #1704 is optional only for an already-contiguous eager-leaf path. | frame recursion `ACI-C2`; arbitrary tree message identity **Tree generalization — re-derived** `(T2)`; implementation policy **[AI Supplied]** | all dtypes; leaves, chain, degree-3, unequal bonds | raw/generic differential tests already at `cached_evaluator.rs:6007–6473`; full dense parity required | contractions per unique assignment, bytes written, raw-vs-generic route time, and failure-path allocations (`N`, `F`, `S`) |
| reconstruction and dead-descendant work | `cached_evaluator.rs:3038–3062,3237–3297` and `build_environment_cache:1232–1367` | #708, #710 | No tenferro primitive supplies cache policy; lazy reconstruction remains TreeTN-owned. | **[AI Supplied]** lazy-hit proposal; `TEN-C3` is the boundary preventing AD/cache conflation. | all dtypes; packed `CachedScalar` and final dynamic result | reconstructed messages and final result equal the scalar path; clear/reuse and failed transaction leave no stale value | `reconstructed_values`, final environment values, reconstruction ms/call, and retained payload bytes; must not regress cold path |
| center contraction | `cached_evaluator.rs:3415–3997` and raw kernels `4095–4329` | #708 | #1704 may lower cost for contiguous eager leaves in tenferro, but does not define the TreeTN edge-cut API. | local center shape `ACI-C2` / `ACI-C4`; warm edge assembly `(T6)` is **Tree generalization — re-derived**; warm reuse design is **[AI Supplied]** | all dtypes; degree 0/1/2/3 and unequal incident bonds; column-major core | compare raw, generic, and ordinary `TreeTN::evaluate` dense outputs; assert `d*product(incident bonds)` raw reference visits | center-contract time, core visits, dynamic-scalar wrapping, and edge-final-assembly cost; chain/star at 1x/2x/4x rank (`E`, `S`) |
| dtype dispatch and dynamic result conversion | `cached_evaluator.rs:4517–4608`, `tensor4all-treeaci/src/scalar.rs:1–204`, result assembly `340–347,4552–4578` | #717 (existing correction; matrix completion later) | tenferro supports runtime F32/F64/C32/C64 tags; typed TreeTN API currently returns `Vec<AnyScalar>`, so no typed batch seam exists yet. | four-way preservation is **[AI Supplied]** with `cached_evaluator_preserves_32_bit_scalar_dtypes` at `cached_evaluator.rs:6848–7005`; dtype boundary `TEN-C2` | f32, f64, Complex32, Complex64; no row-major conversion | cold/warm/partial/miss results match ordinary evaluator with existing dtype tolerances and reject cross-dtype values | per-dtype conversion/wrapping time, copies, and output allocation count; no f64/c64 regression while adding f32/c32 coverage (`N`, `F`, `D`) |
| raw and generic kernels | `cached_evaluator.rs:1472–2890,4095–4329`; scalar TreeACI frame kernels `frames.rs:1919–2245` | #711, #713 | No direct tenferro dependency; backend capability checks remain in TreeTN/tensorbackend. | raw algebra is **Tree generalization — re-derived** `(T2)`; implementation dispatch and degree limit are **[AI Supplied]**; `ACI-C2` is the chain frame analogue. | all four dtypes; non-contiguous/unsupported/provider fallback cases | scalar reference vs optimized for leaf, chain, branch, degree-4 fallback, zero dimensions, and unequal bond dims | kernel wall time, core visits, temporary payload bytes, and route frequency; generalized route must not regress the scalar route |
| dense tensor construction and storage ownership | `tensor4all-core`: `IdxTensor::from_dense` `defaults/idx_tensor.rs:6431–6488`, `from_storage:2660–2710`, `with_dense_slice:6953–7020`; TreeACI consumers `transaction.rs:200–230`, `global_guard.rs:315–553` | #716, with #718 fallback gate | Ownership belongs to core/storage; tenferro runtime is a dense leaf boundary (`TEN-C2`), not a reason for TreeACI to add a direct dependency. | **[AI Supplied]** ownership diagnosis; dense boundary `TEN-C2`; no paper claim | all four dtypes; column-major dense buffers and borrowed contiguous views | dense round-trip, non-contiguous fallback, zero-size/error paths, and AD/trace behavior where supported | copies avoided, allocation count, bytes copied, peak resident logical payload; paired before/after release measurements |
| matrix creation and matrix-to-vector conversion | `tensor4all-tensorbackend`: `Matrix::try_from_col_major_vec` and `into_col_major_vec` at `matrix.rs:417–432,494–511`; typed tensor bridge `560–658` | #716 | Existing high-level owned seams are present; TreeACI must consume them rather than reach into tenferro. | **[AI Supplied]** ownership optimization; `TEN-C1` and `TEN-C2` fix column-major semantics. | all BLAS scalar kinds supported by the backend; no hidden row-major round trip | shape overflow, length mismatch, column-major indexing, and owned bridge round trips | zero-fill avoided, buffer reuse, bytes copied, and conversion time; no numerical change (`F`, `N`) |
| ordinary GEMM and same-shape batched GEMM | `tensor4all-tensorbackend/src/matrix.rs:1564–1675`; TreeACI call sites `local_update.rs:159–254`, `frames.rs:2105–2245` | #714, #712 | `mat_mul_owned` and same-shape batched APIs already exist in tensorbackend; use the configured backend facade. | `TEN-C1`/`TEN-C2` for backend/layout; ACI local work `ACI-C2`/`ACI-C3`; batching policy **[AI Supplied]** | all four scalar kinds supported by TreeACI; column-major `A*B` | compare against scalar nested-loop reference and borrowed GEMM; exact output order/reduction policy retained | paired wall time plus copies/allocations; a small measurable reduction is sufficient only above noise floor |
| grouped GEMM and shared-operand descriptors | future facade in `tensor4all-tensorbackend`; tenferro capability at cloned revision `crates/tenferro-cpu/src/backend.rs:3434–3448`, `exec_session.rs:714–723`, and grouped tests | #712 | `GroupedGemmJob`/`grouped_gemm_cached` exists in tenferro, but no public tensorbackend facade currently exists. Add the facade there; never add `tenferro-*` to TreeACI/TreeTN. | `TEN-C1`; reuse motivation related to `TCI-C3`; exact shared-RHS policy is **[AI Supplied]** | all supported dtypes; offset ranges, shared RHS, over-budget fallback | grouped output equals ordered individual GEMMs, including empty jobs, overlap rejection, provider fallback, and over-budget route | compare launch/packing/copy bytes and wall time at 1/2/4/8 jobs under `max_working_bytes`; retain scalar route as oracle (`F`, `S`, `P`) |
| LUCI and LU factorization | `tensor4all-core/src/matrix_luci.rs:281–460`; public owned seam `matrix_luci_factors_from_matrix_owned:385–438`; TreeACI `local_update.rs:271–329`; deferred canonicalization in `schedule.rs:268–290` | #716 | Core owns factorization policy; tensorbackend owns matrix representation; rectangular LU is not critical because core retains its rectangular fallback. | ACI Algorithms 2–4, PDF/printed pp. 12–13, lines 1–42 (`ACI-C4` family); prrLU stability and Eq. (27)–(32) `TCI-C2`; ownership optimization **[AI Supplied]** | all four scalar kinds where supported; rectangular local matrices and rank-zero cases | reconstruct `P/L/D/U` residual once as dense matrix, pivot/rank/error trajectory, ill-conditioned and rank-deficient fixtures | owned-vs-borrowed copies, factorization time, peak matrix bytes, and pivot search work; no accuracy/rank regression (`N`, `F`) |
| slicing, views, and column-major exports | TreeTN `cached_evaluator.rs:4534–4550`; core `IdxTensor::select_indices:2341–2410` and `with_dense_slice:6953–7020`; backend matrix views `matrix.rs:462–511` | #716, #711 | tenferro supports metadata-only views but dense backend instructions assume contiguous column-major inputs (`TEN-C1`, `TEN-C2`); materialize explicitly at the boundary. | **[AI Supplied]** TreeTN slicing policy; view contract `TEN-C2`; ACI frame order `ACI-C2` | all dtypes; contiguous and non-contiguous/trace-like views; column-major flat exports | selected indices and view fallback equal dense reference, bounds/errors preserved, no hidden axis permutation | copy count/bytes for contiguous vs fallback, slice time, and downstream GEMM layout hit rate (`F`, `M`) |
| cache accounting, retention, and invalidation | TreeTN `CachedEvaluationStats` `cached_evaluator.rs:358–366`, `PackedMessageCache:4740–4910`; TreeACI `InputFrameStore` `frames.rs:229–421`; Guard budget `global_guard.rs:21–83,800–888` | #710, #718 | tenferro AD cache policy is separate (`TEN-C3`); ACI numerical caches must remain bounded by their own documented budget. | **[AI Supplied]** counters/retention policy; cache correctness derives from `(T1)`–`(T2)` and `(T6)` | all dtypes; clear/reuse, generation and failed-transaction states | cache-on/off, clear, failed commit, input mutation, and over-budget fallback equal fresh computation; errors preserve context | hit/miss, retained logical bytes, frame extension copies, cache plateau/release, and RSS-independent logical accounting (`I`, `D`, `S`) |
| AD/trace preservation at the dense boundary | `tensor4all-core` AD methods `defaults/idx_tensor.rs:3125–3227`; tenferro AD boundary in `TEN-C3`; TreeTN/TreeACI evaluator outputs are primal `AnyScalar` values | #718 cross-cutting gate; no ACI AD implementation is implied | Do not route ACI cache or primal evaluator through tenferro AD graph rules. If a typed/trace-aware API is later added, it belongs at the owning core/TreeTN seam. | **[AI Supplied]** ACI has no paper AD claim; `TEN-C3` is the normative AD/cache separation. | supported traced/eager modes, all affected dtypes, and explicitly unsupported modes | tracked inputs either preserve gradients or return the documented typed unsupported error; primal-only path records `N/A` with reason | no retained graph/tensor buffers from numerical cache; repeated transforms obey documented retention (`F`, `I`, `D`) |
| pivot samples, local factor commit, rollback, and output replacement | TreeACI `transaction.rs:22–330`, `samples.rs:141–197,301–477`, `global_guard.rs:195–553` | #715, #718 | No tenferro role; output/core ownership is delegated to TreeTN/core APIs. | local update follows ACI Eq. (11)/Algorithm 3–4 and global pivots `TCI-C5`; transaction/checkpoint policy **[AI Supplied]** | all four scalar kinds; rank zero/growth, all tree degrees, column-major factors | commit and rollback are atomic; pivot/sample/frame sets and output values match pre-change; error messages remain contextual | per-edge metadata clone, frame extension, output copies, and peak staged bytes; compare failed and successful commits (`E`, `I`, `R`) |

### Explicit dependency decisions and #707 closure gates

1. **#712 grouped GEMM:** tenferro's grouped segmented capability is real at
   the pinned commit, but it is not a tensorbackend public API. The fix is a
   generic `tensor4all-tensorbackend` facade with ordered, budget-aware scalar
   fallback. TreeACI and TreeTN remain free of direct `tenferro-*` dependencies.
2. **#716 ownership:** dense construction, matrix conversion, and LU/LUCI
   buffer reuse belong in `tensor4all-core` and
   `tensor4all-tensorbackend`. TreeACI may consume owned high-level seams but
   must not duplicate tenferro conversions or inspect native tensor storage.
3. **#709 evaluator seam:** typed batch output, immutable plan lifetime, and
   warm edge-cut assembly belong in TreeTN. Tenferro #1704 is optional support
   for an already-contiguous eager-leaf optimization; it does not block the
   TreeTN correctness work and cannot be used as a TreeACI proof.
4. **Rectangular LU:** optional upstream support is not on the critical path;
   the current core rectangular rrLU fallback remains the correctness oracle.
5. **Downstream evidence:** `../../gw-rs/sgw` may be used for an ACI-facing
   integration gate when its ACI path is exercised. `tensor4all-benchmark` is
   supplementary only; its current maintained cases are SimpleTT/chain ACI,
   so it is `N/A` for TreeACI-only changes unless a changed shared seam is
   actually covered.

The #707 closure gates are now explicit:

- `C1 PASS`: each row names an owner, scalar/layout scope, provenance label,
  concrete oracle, and failure/boundary behavior.
- `E1 PASS`: each performance-sensitive row names a release paired metric and
  a counter/timing resource gate; no one-off speedup is treated as proof.
- `R1 PASS`: current public names were checked against `target/api-dump/`, no
  direct tenferro dependency is proposed downstream, and the historical audit
  scope plus current closure snapshot are both stated.
- `P PASS`: every paper/spec-dependent claim has a page/equation/algorithm or
  exact specification line locator in the evidence register. Tree-specific
  claims retain their **Tree generalization — re-derived** or **[AI Supplied]**
  label. The only audit limitation is that GitHub review comments and issue
  measurements remain repository artifacts rather than literature authority.

Verification executed after the closure edit:

- `cargo fmt --all -- --check`: PASS.
- `cargo test --release -p tensor4all-aci -p tensor4all-treeaci -p tensor4all-treetn -p tensor4all-core -p tensor4all-tensorbackend --no-fail-fast`: PASS (exit status 0; all selected unit, integration, and doctest suites passed).
- `git diff --check`: PASS.

No production algorithm code was changed for #707. The next implementation
subissue is **#717**, which will complete the four-scalar-kind cache/topology
regression matrix around the already-landed dispatch correction.

## 2026-09-04 #717 closure: four-scalar cached-evaluator matrix

This section closes the #717 matrix-completion scope around the already-landed
dtype dispatch correction in commit `2dfabd6c5f116e0897d72735642c93de8acc5f4d`.
The implementation change in this subissue is regression coverage and
benchmark instrumentation only: no raw f32/Complex32 kernel was added, no
public API changed, and the existing f64/Complex64 raw path remains intact.
All fixture topology, test-policy, and acceptance labels in this section are
**[AI Supplied]** engineering evidence, not claims copied from the cited
papers. The full-text paper/spec clones and concrete page/equation/source-line
locators remain in the `ACI-C*`, `TCI-C*`, and `TEN-C*` evidence register above;
this subissue makes no new algorithmic literature claim.

### #717 changes

- `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` now has one
  parameterized test fixture for `f32`, `f64`, `Complex32`, and `Complex64`.
  It checks the complete dense result against `TreeTN::evaluate`, verifies the
  dynamic result dtype through `AnyScalar`'s debug dtype tag, and exercises
  cold/warm calls, reordered and duplicated columns, a genuine partial hit,
  zero/limited cache budgets, explicit cache clear/reuse, a leaf-centered
  unequal-bond Y-comb, mixed f32/f64 promotion, invalid shapes/coordinates,
  and the degree-four generic fallback.
- `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs` adds the same
  four-kind check through `GuardOutputEvaluator` and
  `TreeAciScalar::from_evaluated_scalar`, comparing its output to one ordinary
  dense TreeTN materialization.
- `crates/tensor4all-treetn/benches/cached_evaluator.rs` adds paired cold/warm
  measurements using the same three-site, unequal physical-dimension fixture
  for all four scalar kinds. The benchmark keeps f32/Complex32 on the generic
  route and f64/Complex64 on the existing raw route; it does not silently
  convert the 32-bit inputs to 64-bit storage.

### #717 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C2` correctness | **PASS** | `cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast`: 60 passed, 2 ignored; all four new scalar/cache/topology tests pass. `cargo test --release -p tensor4all-treeaci global_guard::tests::guard_cached_evaluator_preserves_all_scalar_kinds`: passed. Full values are compared after one dense ordinary-evaluator materialization; f32/c32 use a `1e-5` relative envelope and f64/c64 `1e-12`, without relaxing existing tests. |
| `E2` efficiency | **N/A — test-only scope** | Focused optimized Criterion run (`cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_cached_scalar_kind`, 10 samples per case) recorded cold/warm medians: f32 `4.007/3.152 ms`, f64 `0.749/0.770 ms`, c32 `4.195/3.176 ms`, c64 `0.784/0.781 ms`. Raw files are retained under `target/criterion/treetn_cached_scalar_kind/**/{base,new}/{sample,estimates}.json`; this patch changes no production path, so it cannot honestly claim a before/after speedup or resource delta. The measurement does identify the generic 32-bit fallback as materially slower than raw 64-bit dispatch; deciding whether that warrants duplicated kernels belongs to the later raw-kernel/performance issue, not #717's coverage closure. |
| `R2` release/regression | **PASS** | The cached-evaluator matrix and Guard typed gate pass in optimized builds. Existing raw/generic differential tests also pass; no direct `tenferro-*` dependency or public API was introduced. The full unfiltered bench command was started but stopped at an unrelated legacy `40x40` uncached case estimated at 202.7 s for ten samples; the affected focused group completed successfully. |
| `N` numerical stability | **N/A (scope)** | This is storage/dispatch coverage, not factorization or convergence. Nonzero complex values, f32/c32 precision envelopes, and mixed promotion are covered by `C2`; rank/pivot/convergence claims remain delegated to the relevant TreeACI issues. |
| `M` metamorphic semantics | **PASS** | Reordered and duplicated point batches are compared in original output order; the partial-hit batch uses a third endpoint coordinate not present in the initial cache, and its hit/miss counters are both asserted. |
| `F` fallback parity | **PASS** | f32/c32 are exercised through the generic fallback, f64/c64 through the raw path, degree-four topology explicitly forces the generic route, and existing raw-vs-generic chain/branch tests remain green. Mixed f32/f64 promotion is compared against the ordinary evaluator. |
| `I` invalidation/retention | **PASS** | Zero budget, one-column budget, repeated warm calls, direct unit-test cache clear/reuse, and retained-byte upper bounds are asserted for all four scalar kinds. |
| `D` determinism | **PASS** | The fixture is fixed-seed-free and deterministic; cold/warm/reordered/partial outputs are exact-repeat comparisons. The dedicated correctness test was rerun three times with the same release command before closure. |
| `S` scaling law | **N/A (scope)** | No production loop or complexity claim changed in #717. Existing #707 chain-size measurements remain historical evidence; a new asymptotic claim is deferred to #708/#710. |
| `P` provenance/observability | **PASS** | New engineering assertions are explicitly tagged **[AI Supplied]**. No paper/spec locator is used as authority for the fixture or implementation. The existing full cloned sources and their exact page/equation/spec-line locators remain recorded above, and Criterion command/configuration/raw paths are recorded here. |

### Boundary and downstream decisions

The mixed f32/f64 test confirms the existing core/tenferro promotion behavior
by comparing cached and ordinary evaluation; it is not treated as an error
case. Invalid rank and out-of-range coordinates retain contextual evaluator
errors. Wide-key construction and working-memory overflow are not reimplemented
in TreeTN: the former belongs to the core/#710 key gate, and the latter to the
TreeACI Guard budget gate already covered by
`guard_working_budget_counts_all_coexisting_evaluation_buffers` and related
tests.

The requested `../../gw-rs/sgw` checkout exists but has a heavily dirty working
tree, so it was not modified or used as a candidate patch target. Since #717
changes no production evaluator behavior, an isolated downstream ACI run would
not distinguish this coverage patch from its current base; downstream evidence
is recorded **N/A** for this subissue. The optional
`../../tensor4all-benchmark` checkout is also dirty and contains no maintained
four-kind TreeTN cached-evaluator case, so it is **N/A** rather than an
unrelated benchmark claim.

### Verification commands

```text
cargo fmt --all
cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
cargo test --release -p tensor4all-treeaci global_guard::tests::guard_cached_evaluator_preserves_all_scalar_kinds
cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_cached_scalar_kind
```

The next implementation subissue is **#715** (cut-local frame growth and
failure-atomic commits). Per the execution protocol, stop here after reporting
this #717 closure; do not begin #715 in the same run.

## 2026-09-04 #715 closure: cut-local frame growth and failure-atomic commits

This subissue changes the internal TreeACI frame storage and Guard/edge
transaction publication order. The persistent frame-segment representation,
the `BC3` benchmark-correctness ordering rule, the paired measurement fixture,
and the rollback test scenarios are **[AI Supplied]** engineering design and
validation policy. No new claim is attributed to ACI/TCI literature here; the
full-text source clones and exact page/equation/algorithm/specification
locators already recorded in the evidence register remain the only literature
authority for the earlier algorithm and backend-boundary claims.

### #715 changes

- `InputFrameStore::extend_new_samples` now validates explicit
  `previous_counts`, computes only newly interned sample ranges, and keeps
  unchanged directed frames `Rc`-shared.
- A grown `DirectedFrame` retains its old prefix through an immutable `Rc`
  base and stores only the new sample rows in its segment. `row_slice` keeps
  the existing sample-major row contract, so consumers do not observe a new
  layout or sample-order convention.
- `commit_edge_proposal` validates proposal/state metadata before staging.
  `inject_global_pivots` computes all next edge ranks before publishing output,
  candidates, frames, or generation; an overflow therefore rolls back the
  arena and leaves the logical state unchanged.
- Added chain/branch frame differential tests, new-range work counters,
  factor-shape and incomplete-metadata transaction tests, Guard rank-overflow
  rollback coverage, and a paired release measurement with a non-smoke
  5-node chain fixture.

### #715 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C3` correctness | **PASS** | `extend_new_samples_matches_full_rebuild_on_chain_and_branch` compares every retained frame row against a fresh rebuild and checks unchanged-edge `Rc` identity. `extend_new_samples_computes_only_new_ranges` checks new-range values, no old-prefix row copy, and retained-prefix identity. Transaction and Guard tests cover callback failure, frame-budget/sample rollback, invalid factor shape, incomplete metadata, and rank-overflow publication. |
| `BC3` benchmark correctness | **PASS** | The complete release TreeACI matrix was run before the speed measurement: 142 unit tests passed with 4 existing ignored diagnostics/high-cost cases, 7 public-API integration tests passed, 1 rank-scaling test passed, and 18 doctests passed. Focused tests were not used as a substitute for this full matrix. |
| `E3` efficiency | **PASS** | After `BC3`, the paired release test used 7 timing samples × 128 repetitions: cut-local median `15.359826 ms`, full rebuild median `35.948933 ms`, an observed improvement of approximately `57.3%`. The candidate used 4 genuine contraction calls versus 12 for the full rebuild, copied 0 old-prefix values, materialized 8 new values, reused 4 edges, grew 4 edges, and retained 192 logical frame bytes. The 7 raw sample durations are printed in the test output recorded below; this is a fixture-level paired result, not an asymptotic claim. |
| `R3` release/regression/downstream | **PASS** | The full local TreeACI matrix passed. An isolated copy of the dirty `/root/projects/gw-rs/sgw` checkout was patched to all local tensor4all-rs crates and ran the complete downstream suite: the first run passed 106/108 lib tests and all other targets except two fixed-checkout-path tests; after providing the exact expected provenance path, those two reruns passed. Combined result: 108/108 lib tests and every integration target passed. The original dirty checkout was not modified. |
| `N` numerical stability | **PASS** | Full frame differential parity passed on chain and branched fixtures; all existing TreeACI numerical/convergence tests remained green. No tolerance was relaxed. |
| `M` metamorphic semantics | **PASS** | Append-only sample IDs, duplicate/no-growth behavior, frame row order, and chain/branch extension parity remain unchanged; existing duplicate global-pivot and candidate/pivot separation tests pass. |
| `F` fallback parity | **N/A (scope)** | #715 does not change scalar fallback or batched contraction formulas; their complete existing TreeACI coverage passed. |
| `I` invalidation/retention | **PASS** | Arena checkpoints roll back failed staging; empty/no-growth paths preserve generation; frame-budget and rank-overflow tests verify no partial publication; persistent segments retain only the logical current prefix plus new rows. |
| `D` determinism | **PASS** | Complete release tests and the fixed-seed/fixed-fixture paired measurement are repeatable; output/sample/frame order is compared deterministically. |
| `S` scaling law | **N/A (scope)** | The measurement demonstrates a target-path resource/time delta on a 5-node chain but makes no new complexity claim; larger scaling belongs to the later evaluator/ownership issues. |
| `P` provenance/observability | **PASS** | All new design and measurement-policy claims are tagged **[AI Supplied]**. No new paper/spec claim was introduced. The benchmark correctness ordering and exact raw command/output are recorded here. |

### Verification commands and raw measurement output

```text
cargo test --release -p tensor4all-treeaci --no-fail-fast
142 unit passed, 4 ignored; 7 public_api passed; 1 rank_scaling passed; 18 doctests passed

cargo test --release -p tensor4all-treeaci frames::tests::paired_release_measurement_for_cut_local_extension -- --ignored --nocapture
#715 paired release resources: chain_nodes=5, max_degree=2, directed_edges=8, candidate_compute_calls=4, full_rebuild_compute_calls=12, old_values_copied=0, new_values_copied=8, extension_calls=1, reused_edges=4, grown_edges=4, retained_bytes=192
#715 paired release measurement: repetitions=128, samples=7, cut_local_median=15.359826ms, full_rebuild_median=35.948933ms, cut_local_all=[14.780307ms, 14.920691ms, 15.27676ms, 15.359826ms, 15.68705ms, 16.51782ms, 17.431146ms], full_rebuild_all=[35.438795ms, 35.459784ms, 35.934676ms, 35.948933ms, 36.42147ms, 36.566873ms, 36.619442ms]

cargo test --release --manifest-path /tmp/sgw-treeaci-gate.6RkGZ6/Cargo.toml --no-fail-fast
combined after the exact provenance-path rerun: 108/108 lib tests and every integration target passed
```

The optional `/root/projects/tensor4all-rust/tensor4all-benchmark` checkout is
dirty and has no maintained TreeTN frame-extension case, so it remains
**N/A**, not an unrelated benchmark claim. The temporary isolated downstream
copy and its path symlink are external validation artifacts and are not part
of the repository change.

## 2026-09-04 #716 closure: ownership-preserving dense and factorization paths

This subissue removes avoidable payload copies at the dense tensor and matrix
factorization boundaries. The ownership diagnosis, API seam, conversion choice,
and efficiency interpretation below are **[AI Supplied]** engineering claims.
They are not attributed to an ACI paper or to tenferro pseudocode. The complete
paper/specification clones and their exact literature locators remain in the
evidence register above.

### #716 changes

- `tensor4all_tensorbackend::TensorElement` and its bridge now provide
  `dense_native_tensor_from_col_major_owned(Vec<T>, dims)`. The implementation
  validates the checked dimension product and passes the owned payload to
  `NativeTensor::from_vec_col_major` without cloning it
  (`crates/tensor4all-tensorbackend/src/tensor_element.rs:29–66`,
  `tenferro_bridge.rs:970–991`).
- `IdxTensor::from_dense` uses the owned bridge after its existing index and
  payload validation (`crates/tensor4all-core/src/defaults/idx_tensor.rs:6431–6488`).
- Default LU/CI factorization constructs matrices with
  `Matrix::try_from_col_major_vec`, uses owned rrLU/MatrixLUCI where available,
  and moves factors out with `into_col_major_vec`; the rectangular internal
  rrLU fallback remains in place
  (`crates/tensor4all-core/src/defaults/factorize.rs:682–719, 794–842,
  847–900`).
- Raw TreeTN leaf, leaf-center, tensor-backed leaf-center, chain-message, and
  branch-message paths now borrow contiguous dense payloads with
  `with_dense_slice`; message gathers decode only requested assignment columns
  (`crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1597–1679,
  2175–2365, 2435–2875, 4389–4463`). Values that intentionally escape as a
  `Vec` remain materialized at the explicit output boundary.
- Added all-four-scalar bridge, dense round-trip, rectangular LUCI, shape/error,
  and TreeTN differential coverage. The owned bridge test also checks native
  buffer pointer identity, which is a deterministic no-payload-clone check.

### #716 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C4` correctness | **PASS** | Full release correctness was run before the efficiency measurement. The bridge and `IdxTensor::from_dense` tests cover f32, f64, Complex32, and Complex64 column-major values; shape mismatch and checked dimension overflow; square/rectangular MatrixLUCI rank/factor parity; and TreeTN cached-evaluator leaf/center/chain/branch raw/generic differential behavior. No tolerance was relaxed. |
| `E4` efficiency | **PASS** | The all-four-kind owned bridge test proves the native tensor buffer pointer is the original `Vec` pointer, so the payload clone is removed deterministically. The paired release measurement (`9` samples × `32` repetitions, `[128,128]` f64 payload) reported borrowed median `1.044 ms`, owned median `0.082 ms`, and `92.1%` constructor-path reduction. This is a controlled constructor-boundary result, not a claim of a 92.1% end-to-end TreeACI speedup; no broad workload speed claim is made. |
| `R4` release/regression/API | **PASS** | The complete tensorbackend, core, TreeTN, and TreeACI release suites passed; the new public method has a runnable asserted rustdoc example; `cargo fmt --all -- --check`, API inventory, clippy, and repository-rule preview were run before closure. The isolated SGW suite passed all 108 library tests and every integration target after its two fixed-path provenance cases were rerun with the required temporary path. No downstream crate gained a direct `tenferro-*` dependency. |
| `N` numerical stability | **PASS** | Owned-vs-borrowed MatrixLUCI factors match for all four scalar kinds, rectangular reconstruction preserves column-major values, and the full core/TreeACI numerical suites remain green. |
| `F` fallback parity | **PASS** | Existing `with_dense_slice` materialization/fallback behavior and raw/generic TreeTN routes remain covered by the complete cached-evaluator and TreeACI matrices; non-escaping reads do not change the fallback contract. |
| `I` invalidation/retention | **PASS** | The owned bridge rejects mismatched lengths before backend construction and rejects overflowing dimension products; output factor buffers are moved into `IdxTensor` rather than retained in duplicate matrix/vector owners. |
| `D` determinism | **PASS** | Exact column-major round trips, factor rank/index/value parity, and complete release suites pass deterministically. The paired measurement uses a fixed payload and reports all sample medians. |
| `M` metamorphic semantics | **PASS** | Raw chain/branch message gathers preserve requested point order and duplicate assignments through the existing full evaluator differential tests; changing only ownership does not alter output layout. |
| `P` provenance/observability | **PASS** | All #716 implementation and measurement claims are labelled **[AI Supplied]**. The only backend authority used is the already-cloned tenferro specification: `TEN-C1`, `docs/spec/backend-contract.md:194–208, 286–313, 317–330`, states the dense contiguous column-major/backend-session boundary; `TEN-C2`, `docs/spec/tensor-semantics.md:31–70, 133–162, 258–270`, states view/materialization and owned compact column-major semantics. The full repository clone and complete paper/spec reading record remain above; no page/equation is misrepresented as an ownership optimization source. |

### Verification commands and raw measurement output

```text
cargo test --release -p tensor4all-tensorbackend --no-fail-fast
221 unit tests passed; 2 bench tests ignored; 149 doctests passed

cargo test --release -p tensor4all-core --test linalg_factorize --no-fail-fast
20 passed, 0 failed

cargo test --release -p tensor4all-core --test tensor_basic --no-fail-fast
55 passed, 0 failed

cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
60 passed, 2 existing diagnostics ignored

cargo test --release -p tensor4all-treeaci --no-fail-fast
142 unit passed, 5 ignored; 7 public_api passed; 1 rank_scaling passed; 18 doctests passed

cargo test --release -p tensor4all-tensorbackend owned_dense_native_tensor_paired_release_measurement -- --ignored --nocapture
#716 paired release dense constructor: dims=[128, 128], repetitions=32, samples=9, borrowed_median_ms=1.044, owned_median_ms=0.082, reduction_pct=92.1, borrowed_all_ms=[1.022128, 1.028541, 1.033501, 1.038069, 1.044171, 1.052276, 1.0643390000000001, 1.0652700000000002, 1.348325], owned_all_ms=[0.08079199999999999, 0.081484, 0.081533, 0.081784, 0.081995, 0.082246, 0.083237, 0.083718, 0.083748], checksum=-576

cargo test --release --manifest-path /tmp/sgw-treeaci-gate.6RkGZ6/Cargo.toml --no-fail-fast
all non-provenance targets passed in the complete run; the two fixed-path cases were then rerun exactly and both passed (108/108 library tests and every integration target)
```

The optional `../../tensor4all-benchmark` checkout contains no maintained
semantically comparable dense-ownership case, so its gate is **N/A**. The
downstream `../../gw-rs/sgw` checkout was not modified; its isolated copy was
patched to this worktree and the complete release suite passed all 108 library
tests and every integration target. The first run reported only the two known
fixed-path provenance failures because the temporary `/tensor4all-rust/tensor4all-rs`
path had not yet been created; exact reruns of both cases passed after the path
was supplied. No new tenferro upstream change is needed.

The next implementation subissue is **#714** (TreeACI local-update ownership
and packed batches). Per the execution protocol, stop here after reporting this
#716 closure; do not begin #714 in the same run.

## Measurements and limitations

Commands were run in release mode in the isolated worktree.

1. Baseline correctness:

   ```text
   cargo test --release -p tensor4all-aci -p tensor4all-treeaci
   271 passed, 0 failed, 3 ignored
   ```

2. The high-rank chain phase profile (`16` sites, `d=2`, two inputs,
   `chi=128`) did not finish a single sweep within approximately 2.5 minutes
   after compilation and was interrupted. No number from that run is used.

3. The branch microbenchmark was run three times as described under P0.

4. The existing chain parity benchmark initially failed to compile because its
   package feature set did not enable
   `tensor4all-tensorbackend/global-defaults` (26 unresolved imports). No
   manifest was changed. Running the same benchmark with that feature enabled
   produced, for the filtered `chi=16` case:

   ```text
   simplett ACI: 46.096--46.537 ms
   TreeACI:      40.389--41.680 ms
   evaluated points: simplett 48,584; tree 29,988
   output max rank: both 32
   relative max error: simplett 1.543e-8; tree 9.988e-9
   ```

   Thus this small case does not reproduce the reported chain slowdown. It
   rules out the claim that TreeACI has a universal constant-factor regression;
   it does not refute the `O(E^2)` metadata paths or high-rank scaling concern.
   The benchmark filter still executed untimed correctness setup for `chi=32`,
   `64`, and `128`; those results are not timing samples.

5. The Hiroshi-review evaluator parity benchmark was added and run in release
   mode. The full cold/warm `chi=64,128,256` results are in the first P0 section
   above. Its pre-timing numerical parity checks passed at all three ranks.

6. The two-point floating-zone comparison was run at 8 sites for `chi=128` and
   `256`; its results are in the same P0 section. The coordination-number
   benchmark was run at `chi=32` and `64`; its results and exact local tensor
   products are in the branch-cost section.

7. The existing message-cache phase breakdown was rerun in release mode:

   ```text
   key_and_lookup 1.5 ms (5.5%)
   contract       21.9 ms (82.8%)
   insert          2.0 ms (7.7%)
   reconstruct     1.0 ms (4.0%)
   total          26.4 ms
   ```

8. The ignored same-bond chain-versus-comb diagnostic was run with the
   `diagnostics` feature at 16 sites and `chi=128`. It passed and reported
   48.421641 ms versus 499.832829 ms (10.32x); the actual summed local-element
   ratio is 9.9905x, as detailed in the branch-cost section.

No profiler trace for the user's exact slow workload was available in this
audit. Findings labelled P0/P1 above are based on exact control-flow and
allocation counts plus the isolated branch microbenchmark. A subsequent fix
phase should first add a reproducible long-chain length sweep and representative
branched fixtures with fixed degree/rank before changing production code.

## Recommended fix order (not implemented in this audit)

1. Redesign cached chain evaluation so a warm directed-message hit does not
   eagerly reconstruct descendant messages, and so the varying center's local
   contraction can be reused at an edge-like cut. Preserve the exact parity
   benchmark as the regression measurement.
2. Add the remaining benchmark fixtures that independently scale chain length,
   input bond dimension, and active candidate rank; include callback counts and
   accuracy/rank parity. The degree/bond local-work fixture now exists.
3. Replace per-edge whole-store `extend` with cut-local incremental frame
   insertion and eliminate the per-edge whole-TreeTN metadata transaction,
   while retaining failure atomicity at a smaller seam.
4. Introduce borrowed/packed candidate-frame batches so local update avoids
   `Vec<Vec<T>>`, cache-hit cloning, repacking, and scatter where layouts match.
5. Generalize the batched message contraction to arbitrary incoming degree, or
   explicitly reject/route high degree until such a kernel exists.
6. Separately derive a lazy/block LUCI source for (T4) if full branch cross
   materialization, rather than contraction mechanics, is the dominant limit.

Every item above changes **[AI Supplied]** machinery or requires a new explicit
tree derivation. None should be attributed to unshown pseudocode in the paper.

## 2026-09-04 #714 closure and CI_rs / Maintenance repair

This entry closes #714 after the complete affected-crate correctness matrix,
the paired release measurement, and the isolated downstream ACI gate. The
packed-batch design, ownership choices, resource interpretation, and CI
diagnosis below are **[AI Supplied]** engineering work. No new numerical
algorithm is attributed to a paper, and no literature-derived claim is made
without the full-source locators already recorded in the evidence register.

### CI_rs and Maintenance failures repaired first

The PR Maintenance scripts check initially failed on
`dense_native_tensor_from_col_major_owned` because its `# Errors` rustdoc
described only generic failure classes. The error contract was made concrete:
the docs now name checked shape-length mismatch and backend conversion failure
at `crates/tensor4all-tensorbackend/src/tensor_element.rs:63`. The repair was
validated by the public-error-doc checker and its complete 15-test regression
suite.

The standalone `tensor4all-aci` CI target also exposed a feature-forwarding
failure: its `default-features = false` dependency graph did not enable the
core `backend-tenferro` feature. The three provider features now forward both
`tensor4all-core/backend-tenferro` and their matching core provider feature at
`crates/tensor4all-aci/Cargo.toml:16–33`. Before the repair, the exact
standalone release command failed with 32 unresolved backend imports; after
the repair the complete package matrix passed. This is a manifest/CI
integration repair, not a tenferro functionality change.

After synchronizing the branch with `origin/main` at `2a4fb6b` (PR #723), the
full core release test exposed a parallel first-use race in the shared eager
context. The reproducible failure was
`defaults::contract::tests::test_mixed_dtype_tensordot_result_remains_a_valid_ad_constant`:
`Runtime::run_prepared` rejected a prepared epoch `1` while the current epoch
was `2`; the test passed alone and with `--test-threads=1`. The shared
`CpuExecutionContext` now installs the built-in einsum and linalg extension
modules while constructing the eager runtime, before the context can be
observed by another thread. This is **[AI Supplied]** lifecycle hardening; it
keeps lazy extension registration from invalidating AD derivative plans during
parallel first use. The post-fix core release library matrix passed `446`
tests with `1` existing ignored test, and the full core package target passed.

### #714 implementation

- `PackedCandidateFrames<T>` is an internal packed owner with explicit bond
  dimension, candidate-order mapping, and checked column-major payload at
  `crates/tensor4all-treeaci/src/frames.rs:255–457`. The local row side is
  produced directly in candidate-by-bond layout and the column side directly
  in bond-by-candidate layout, so the common local update no longer extracts
  `Vec<Vec<T>>` and repacks both sides.
- Candidate-cache values are `Rc<[T]>`, allowing a hit to share the computed
  frame payload while the final batch is assembled. The immutable oriented
  core matrix is also shared across candidate lookups and append-only frame
  extension; its preparation identity is checked by a dedicated test.
- The production one-incoming stored-frame path, candidate-frame path, and
  local row-by-column product consume the configured backend's owned GEMM
  seam. The test-only A/B switches retain the old borrowed/`Vec<Vec<T>>`
  routes solely for paired diagnostics.
- The existing checked working-byte contract remains in force for candidate
  scratch, simultaneously retained frame payloads, local input values, and
  local output/product buffers. A boundary test verifies the exact two-node
  working-byte limit and the one-byte-under limit failure.
- Real and Complex64 branch tests compare the complete packed result against
  the scalar frame oracle and cover leaf, one-incoming, two-incoming,
  three-plus-incoming fallback, unequal-bond, alternate-axis, duplicate,
  cache-hit/miss, and over-budget paths.

### #714 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C5` correctness | **PASS** | The full TreeACI release matrix passed: 145 unit tests, 7 public-API tests, 1 rank-scaling test, and 18 doctests; 6 existing high-cost/diagnostic tests remained ignored. The frame tests include real/Complex64 scalar-vs-packed differential checks, candidate order/dimensions, leaf and degree-three-plus fallback, cache-hit/miss, duplicate candidates, unequal bonds, alternate axes, extension, and exact budget boundaries. |
| `BC5` benchmark correctness | **PASS** | The complete TreeACI matrix above and the complete standalone ACI release matrix were run before the ignored paired measurement. The standalone ACI result was 85 unit tests passed with 1 existing ignored test, 4 integration tests passed, 1 rank-scaling test passed, and 19 doctests passed. No smoke benchmark was used as the correctness gate. |
| `E5` efficiency | **PASS** on causal target-path resource reduction | After `BC5`, paired release local-update runs used the same fixtures, seeds, backend, thread settings, and 16 repetitions × 5 samples. `chain-8x16`: legacy 128 extracted vectors/2048 values versus packed 32 batches/2048 values; medians 1.533 ms versus 1.358 ms, 11.4% lower. `branch-chi32`: legacy 96 vectors/3072 values versus packed 32 batches/3072 values; medians 3.437 ms versus 3.397 ms, 1.2% lower. The resource delta is deterministic (4× fewer candidate packing objects and no `Vec<Vec<T>>` production round trip); the small branch time result is reported without claiming a universal speedup. |
| `R5` release/regression/downstream | **PASS** | Full TreeACI and standalone ACI release tests passed. The clean archived SGW copy ran the complete `run_r10_nblock_treeaci_ab.sh 1.0` A/B workflow successfully for its checked-in SimpleTT, TreeACI, and CTTN runs. `isolate_aci_stage` then passed for both `pi_rtau` and `sigma_rtau`; diagnostics reported convergence, 145,968 and 98,210 evaluated points respectively, and wrote both JSON records. The dirty `/root/projects/gw-rs/sgw` checkout was not modified. |
| `N` numerical stability/convergence | **PASS** | Packed frame outputs match the scalar real/Complex64 oracle exactly within the existing dtype behavior; complete TreeACI/ACI rank-scaling, convergence, pivot, and dense-result tests remained green. No tolerance was relaxed and no reduction order was changed. |
| `M` metamorphic semantics | **PASS** | Candidate-order mappings, duplicate candidates, physical-axis permutations, unequal incident bonds, and row/column layout conversion are covered. The complete existing edge-order/topology and candidate-set tests also remain green. |
| `F` fallback parity | **PASS** | Leaf and three-plus-incoming routes retain the scalar fallback; two-incoming routes retain the existing batched contraction; cache-disabled, cache-over-budget, duplicate, and zero-headroom cases pass. The test-only legacy local pack path agrees with the production packed path. |
| `I` invalidation/retention | **PASS** | Oriented-core preparation is shared per input/directed-edge owner and reused through `extend`; candidate payloads use shared `Rc` ownership, cache accounting remains bounded, cache reclamation tests pass, and the exact working-byte boundary rejects an over-limit request. |
| `D` determinism | **PASS** | Fixed-seed scalar/packed differential tests, candidate order, ranks, and full release matrices are deterministic. Paired timing samples and all raw values are recorded below; timing is not used to assert an asymptotic law. |
| `S` scaling law | **N/A** | #714 claims a local allocation/copy reduction, not an asymptotic complexity change. The required 1×/2×/4× scaling study belongs to #718 and is not inferred from these two fixtures. |
| `P` provenance/observability | **PASS** | All new #714 implementation, CI, and performance claims in this section are **[AI Supplied]**. No new paper/specification claim was introduced, so no new literature locator is asserted. Existing full paper/specification clones, hashes, and page/equation/paragraph locators remain the authoritative register. Diagnostic counters, commands, configurations, baseline/candidate identity, and raw timing arrays are retained below. |

### Verification commands and raw #714 measurement output

```text
cargo test --release -p tensor4all-treeaci frames --no-fail-fast
39 passed, 0 failed, 4 ignored in the filtered frame target

cargo test --release -p tensor4all-treeaci --no-fail-fast
145 unit passed, 6 ignored; 7 public_api passed; 1 rank_scaling passed; 18 doctests passed

cargo test --release -p tensor4all-aci --no-fail-fast
85 unit passed, 1 ignored; 4 integration tests passed; 1 rank_scaling passed; 19 doctests passed

cargo test --release -p tensor4all-treeaci local_update::tests::packed_local_update_release_measurement_for_chain_and_branch -- --ignored --nocapture
#714 packed counters: case=chain-8x16, legacy_vectors=128, legacy_values=2048, packed_batches=32, packed_values=2048
#714 paired release measurement: case=chain-8x16, repetitions=16, samples=5, legacy_median_ms=1.533, packed_median_ms=1.358, reduction_pct=11.4, legacy_all_ms=[1.386798, 1.516191, 1.533022, 1.82989, 1.865387], packed_all_ms=[1.299653, 1.342134, 1.357532, 1.583277, 1.98956]
#714 packed counters: case=branch-chi32, legacy_vectors=96, legacy_values=3072, packed_batches=32, packed_values=3072
#714 paired release measurement: case=branch-chi32, repetitions=16, samples=5, legacy_median_ms=3.437, packed_median_ms=3.397, reduction_pct=1.2, legacy_all_ms=[3.339158, 3.3562, 3.436552, 3.450598, 3.605909], packed_all_ms=[3.271111, 3.341142, 3.396967, 3.438645, 3.457211]

python3 scripts/check-public-error-docs.py
public-error-docs-ok
python3 scripts/test-check-public-error-docs.py
15 tests passed
python3 scripts/test-repository-rules-review.py
90 tests passed
python3 scripts/check-crate-boundaries.py
crate-boundary-ok
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
Verdict: pass; No findings.

python3 scripts/check-crate-boundaries.py
crate-boundary-ok
python3 scripts/audit-library-panics.py
Audit passed: 0 unbaselined findings, 0 stale baseline entries

python3 scripts/check-public-error-docs.py
public Result APIs with incomplete error documentation:
- crates/tensor4all-tensorbackend/src/matrix.rs:638: grouped_mat_mul_shared_with_backend: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1373: new: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1714: with_plan: # Errors does not name a concrete variant or condition
(the same three findings reproduce on a clean `git archive a7632cc` tree, so
 they are inherited from #712 and #709 and are not caused by this commit --
 see Open items item 6)

SGW_RUN_TAG=aci-gate SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0
complete checked-in A/B workflow passed in clean archive /tmp/sgw-treeaci-gate.3XdbB5

cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate/treeaci pi_rtau
converged; sweeps=6; evaluated_points=145968; diagnostics JSON written
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate/treeaci sigma_rtau
converged; sweeps=5; evaluated_points=98210; diagnostics JSON written
```

After the eager-context race repair, the same complete downstream workflow was
rerun with `SGW_RUN_TAG=aci-gate-final`; SimpleTT, TreeACI, CTTN, all slice
extraction, and all plotting stages passed. The final isolated TreeACI checks
also converged: `pi_rtau` in 7 sweeps with 177,766 evaluated points and
`sigma_rtau` in 5 sweeps with 99,374 evaluated points. These observed gate
results are **[AI Supplied]** and do not change the literature evidence.

The first SGW attempt stopped before workload execution because the existing
SGW binary requires the fixed path `/tensor4all-rust/tensor4all-rs`; this was
resolved by an explicit temporary symlink to the candidate worktree and the
complete run was repeated. The original dirty SGW checkout remains untouched.
The sibling `../../tensor4all-benchmark` checkout still has no maintained
TreeACI local-update workload, so its gate is **N/A**. No new direct
`tenferro-*` dependency was introduced and no tenferro-layer functionality
change is required for #714.

## 2026-09-04 #711 reusable TreeTN branch-slice preparation

This entry records the #711 implementation and its local/downstream gates.
The cache ownership, packed layout, threshold, and benchmark interpretation
are **[AI Supplied]** engineering decisions. No new numerical algorithm is
attributed to a paper in this entry; the full ACI/TCI and tenferro clones,
hashes, and concrete page/equation/paragraph locators already recorded in the
evidence register remain authoritative. No source-derived claim is made here
without one of those locators.

### Implementation

- `TreeTNCachedEvaluator` now owns separate real and Complex64 prepared-slice
  caches. A key is `(node, parent, physical coordinate, exact scalar kind)`.
  On a miss, `with_dense_slice` prepares the existing parent-fast,
  child-2-major column-major matrix once; subsequent physical-value groups use
  the borrowed tensorbackend `mat_mul` operand. The existing scalar threshold,
  GEMM grouping, and c2-major/parent-minor accumulation order are unchanged.
- `CachedEvaluatorOptions` adds
  `branch_slice_cache_max_bytes`. `0` retains no prepared payload and follows
  the uncached route; a finite budget refuses entries that do not fit; the
  default preserves the historical unbounded evaluator policy. Only matrix
  payload bytes are charged, and retained bytes are checked against the sum of
  owned payloads on drop. No full-network dense materialization or direct
  `tenferro-*` dependency was added.
- Diagnostics now report prepared-slice hits, misses, budget refusals, and
  retained payload bytes in addition to child decode/gather, setup, GEMM, and
  accumulation counters. The f32/Complex32 and unsupported-shape routes keep
  their existing generic fallback; the new cache is only used for the already
  eligible f64/Complex64 raw branch routes.
- Related code was reviewed: the existing scalar branch contraction remains
  the oracle used by direct tests, higher-than-two-incoming branches remain
  on the existing generic/fallback path owned by the later arbitrary-degree
  task, and the chain raw path was left unchanged except for shared test
  coverage.

### #711 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C6` correctness | **PASS** | The complete `tensor4all-treetn` release package passed: 513 unit tests, all integration targets, and 141 doctests, with no failures. The new matrix includes all 24 permutations of physical/parent/child axes for unequal bonds in both f64 and Complex64, scalar-oracle differential checks, evaluator reuse, cache hit/miss, zero-budget, over-budget, and retained-byte assertions. The affected TreeACI and standalone ACI release packages also passed (145 unit, 7 public API, 1 rank-scaling, 18 doctests; and 85 unit, 4 integration, 1 rank-scaling, 19 doctests respectively). No smoke-only result closed this gate. |
| `BC6` benchmark correctness | **PASS** | The full affected-crate release matrices above were completed before accepting timing data. The benchmark uses the same fixed tree, assignments, center, backend, and warm-up for both cases; numerical correctness is established by the complete test/oracle matrix, not by a benchmark smoke sample. |
| `E6` efficiency | **PASS** above the measured noise floor | The paired Criterion release benchmark disables message retention in both cases and changes only prepared-slice retention. Medians were: bond 64 prepared 1.6107 ms vs repacked 2.9664 ms (45.7% lower); bond 128 6.7952 ms vs 56.331 ms (87.9% lower); bond 256 156.11 ms vs 357.95 ms (56.4% lower). After warm-up the prepared route performed zero per-iteration branch-slice writes; the repacked route wrote 524,288, 4,194,304, and 33,554,432 f64 values per iteration respectively. The chain parity repeat at bond 256 was 25.093 ms and remained consistent with the prior ~25.04 ms baseline; warm chain points and the other cold points showed no reproducible regression. The first full-run bond-256 Criterion comparison was noisy, so the exact target was repeated and the repeat, not the noisy first comparison, is the reported regression decision. |
| `R6` release/regression/downstream | **PASS** | Full TreeTN, TreeACI, and ACI release matrices passed. In clean archive `/tmp/sgw-treeaci-gate.3XdbB5`, the complete `SGW_RUN_TAG=aci-gate-711 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0` workflow passed for SimpleTT, TreeACI, CTTN, all four checkpoint stages, five row slices, assembly, and plotting. Isolated TreeACI diagnostics converged: `pi_rtau` in 7 sweeps with 177,706 evaluated points and `sigma_rtau` in 5 sweeps with 99,554 evaluated points. The dirty SGW checkout was not modified. |
| `N` numerical stability/convergence | **PASS** | All prepared-path outputs match the existing scalar/generic contraction within existing dtype behavior across the complete axis permutation and unequal-bond matrix. The downstream ACI stages converged with the configured tolerance; no tolerance was relaxed and reduction order was preserved. |
| `M` metamorphic semantics | **PASS** | Axis permutations, unequal incident bonds, directed orientation keys, repeated warm calls, and cache-disabled/over-budget equivalence are covered. The original point order and duplicate-preserving grouped reduction remain unchanged. |
| `F` fallback parity | **PASS** | Scalar work below the existing threshold, zero budget, insufficient budget, f32/Complex32, unsupported shapes, and higher-degree generic routes retain fallback behavior and pass the relevant release tests. |
| `I` invalidation/retention | **PASS** | The evaluator owns the prepared slices for its lifetime, the key includes direction/coordinate/dtype, zero budget retains none, finite budgets refuse non-fitting payloads, and the retained-byte invariant is checked. No input tensor is mutated and no hidden full-network copy is introduced. |
| `D` determinism | **PASS** | Fixed fixtures, axis permutations, oracle results, cache counters, downstream stages, and release test outcomes are repeatable. Timing outliers are explicitly reported as noise rather than used as correctness evidence. |
| `S` scaling law | **N/A** | #711 is a constant-factor setup/copy reduction, not an asymptotic claim. The 64/128/256 paired benchmark demonstrates the target-path resource reduction; the plan's formal 1x/2x/4x scaling-law gate remains owned by #718. |
| `P` provenance/observability | **PASS** | All #711 design and performance claims are labelled **[AI Supplied]**. No new literature claim was introduced. The API dump was regenerated successfully, the new public option is documented with its budget semantics, and commands, backend settings, raw medians, counters, and clean downstream path are recorded here. `tensor4all-benchmark` remains **N/A** because its maintained checkout has no semantically comparable TreeACI workload. |
| `CI6` remote regression | **PASS** | The preceding failed CI_rs run was inspected before push: run `33820251118`, head `d5de28f0d77b0dde667e0dfddb5c5892c6b78c9a`, failed job `Maintenance scripts`, step `Audit public Result APIs`, with the concrete log anchor `dense_native_tensor_from_col_major_owned: # Errors does not name a concrete variant or condition`; the Test, Coverage, Doctests, and Lint jobs passed and rollup failed because Maintenance failed. The first #711 run `33856026628`, head `11904ad4503d1d3d7ddf41b28853a122faabf907`, reached the same Maintenance job but failed earlier at `Audit library panic paths`, log anchor `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:5181:debug_assert_eq`; the new assertion was test-only scoped and the exact local `python3 scripts/audit-library-panics.py` passes with zero unbaselined findings. Replacement run `33856551145` at head `898548186fe842eb700521ac3328399d43ace7f7` passed Lint, Doctests, Maintenance scripts, Test, Coverage, and `rollup-rs`. The local `python3 scripts/check-public-error-docs.py` is also green with its complete 15-test suite passing. This CI result closes the prior failure regression independently of the in-progress #710 work. |

### #711 verification commands and raw measurements

```text
cargo run -p xtask --release -- api-dump
complete API inventory verified

cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
66 filtered tests: 64 passed, 2 existing ignored

cargo test --release -p tensor4all-treetn --no-fail-fast
513 unit tests and all integration targets passed; 141 doctests passed

cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
passed
cargo bench -p tensor4all-treetn --bench cached_evaluator --no-run
passed

cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_prepared_branch_slice_reuse
prepared/repacked medians (ms): 64=1.6107/2.9664, 128=6.7952/56.331, 256=156.11/357.95

cargo bench -p tensor4all-treetn --bench cached_evaluator -- 'hiroshi_chain_evaluator_parity/treetn_cold/256'
repeat median 25.093 ms; Criterion exact-target comparison [-11.133%, -9.2949%, -7.1677%]

cargo test --release -p tensor4all-treeaci --no-fail-fast
145 unit, 7 public_api, 1 rank_scaling, 18 doctests passed; 6 existing ignored
cargo test --release -p tensor4all-aci --no-fail-fast
85 unit, 4 integration, 1 rank_scaling, 19 doctests passed; 1 existing ignored

SGW_RUN_TAG=aci-gate-711 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0
complete clean-copy A/B workflow passed
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-711/treeaci pi_rtau
Converged; sweeps=7; evaluated_points=177706
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-711/treeaci sigma_rtau
Converged; sweeps=5; evaluated_points=99554

gh run view 33820251118 --json headSha,conclusion,status,jobs
failed CI_rs: Maintenance scripts / Audit public Result APIs; other jobs passed
gh run view 33820251118 --log-failed
concrete failure: dense_native_tensor_from_col_major_owned # Errors documentation incomplete
python3 scripts/check-public-error-docs.py
public-error-docs-ok
python3 scripts/test-check-public-error-docs.py
15 tests passed

gh run view 33856026628 --json headSha,conclusion,status,jobs
first #711 run: Maintenance scripts failed at Audit library panic paths
concrete failure: cached_evaluator.rs:5181:debug_assert_eq was unbaselined
python3 scripts/audit-library-panics.py
Audit passed: 0 unbaselined findings, 0 stale baseline entries
```

The next implementation subissue identified by the #711 report was **#710**
(directed-component keys, layouts, and cache accounting). The execution rule
was subsequently updated so a pending remote CI result does not block this
independent task; its implementation and gates are recorded below.

## 2026-09-04 #710 directed-component keys, layouts, and cache accounting

This entry records the #710 implementation and its local/downstream gates.
The ownership boundary, append ordering, metadata census, cache accounting
formula, and benchmark interpretation are **[AI Supplied]** engineering
decisions. The checked primitive used here is the existing `tensor4all-core`
`KeyBuilder`; its concrete source locator is
`crates/tensor4all-core/src/index_key/mod.rs:436-521`, and the evidence
register above records the audited commit and full-source archive. No new
literature-derived algorithmic claim is introduced by this change.

### Implementation

- `TreeTNCachedEvaluator` now builds one immutable
  `DirectedComponentLayout` per directed edge `(from, to)`. It stores the
  component's physical input positions, checked `FlatIndexer`, and deterministic
  child append order. `RootedMessagePlan` retains only center-specific
  traversal state; its duplicated subtree node lists and center-indexed layout
  maps were removed.
- Per-call assignment batches encode local coordinates once and compose nested
  `IndexKey` values with checked `KeyBuilder` capacity/push/finish operations.
  The message cache now receives exact component keys directly, without
  gathering a full rooted-subtree `Vec<usize>` for every cache lookup. The
  component layout's direct encoder is tested against the composed key for each
  unique assignment, including nested components and wide keys.
- `PackedMessageCache` preserves #626's append-only, no-eviction, over-budget
  uncached-miss policy. Its logical payload bytes remain the admission budget;
  `owned_retained_bytes_estimate` additionally counts vector capacity, key
  storage, map entry/bucket overhead, and fixed metadata. The standard-library
  `HashMap` bucket contribution is explicitly documented as a deterministic
  estimate (`16` control/bucket bytes per allocated slot), because its exact
  allocator layout is not a stable API. Test-only evaluator statistics expose
  key count, logical bytes, and owned-storage estimate without adding a hot-path
  production scan.
- The cache validates both the number and width of computed columns before any
  insertion, so a malformed compute result cannot partially mutate the packed
  cache. Zero-budget, bounded, clear/reuse, partial-hit, and over-budget
  behavior remains covered by the existing four-dtype matrix.

### #710 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C7` correctness | **PASS** | The complete TreeTN release library matrix passed: 519 tests ran, with 517 passed and 2 existing tests remaining ignored; all TreeTN integration targets passed (3, 61, 6, 18, 12, 20, 1, 35, 2, 27, 2, 28, 20, 18, 1, 11, 2, and 2 tests respectively); 141 doctests passed. New tests cover exact direct/composed key equality, empty/singleton/nested/duplicate/reordered/wide/invalid/overflow cases, directed-layout sharing across centers, and 4/8/16-site path, Y, comb, and unequal-bond metadata. `tensor4all-core/common_basic` passed all 9 tests. |
| `BC7` benchmark correctness | **PASS** | The full affected TreeTN release library/integration/doctest matrix passed before accepting timing data. The evaluator benchmark uses a complete fixed topology/assignment scan; numerical correctness is established by the full scalar-oracle and downstream matrices, not by the benchmark's timing loop alone. |
| `E7` efficiency | **PASS** through causal resource reduction with no measured target-path regression | The prior N=16, χ=256 diagnostic recorded 1,616 retained subtree/layout position references; the new same diagnostic retained 240 positions across 30 directed layouts, an 85.15% reduction (6.73x less duplicate position storage). Old/new moving-center timings were `8.906/4.373 ms` versus `8.992/4.318 ms` (cold/second scan): cold delta is within measurement noise and warm scan improved about 1.3%. Pinned evaluator-level Criterion scan (N=32, χ=8, CPU 0, 10 samples) measured cold median `2.3161 ms` and warm median `1.8195 ms`; the primitive 51.2 ns versus 29.1 ns key result is not used as an evaluator-speedup claim. |
| `R7` release/regression/downstream | **PASS** | `tensor4all-treetn`, `tensor4all-core/common_basic`, `tensor4all-treeaci`, and `tensor4all-aci` release matrices passed. Clean archive `/tmp/sgw-treeaci-gate.3XdbB5` with `SGW_RUN_TAG=aci-gate-710 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0` passed SimpleTT/TreeACI/CTTN, all checkpoint stages, slices, assembly, and plotting. Isolated TreeACI diagnostics converged: `pi_rtau` 7 sweeps/177646 evaluated points and `sigma_rtau` 5 sweeps/99554 evaluated points. The original dirty SGW checkout remained untouched. |
| `N` numerical stability/convergence | **PASS** | Complete four-dtype TreeTN tests and scalar/generic message-oracle comparisons passed; no tolerance was relaxed. Downstream `pi_rtau` and `sigma_rtau` retained convergence and configured error bounds. |
| `M` metamorphic semantics | **PASS** | Reordered/duplicate/partial-hit batches, all directed orientations, all visited centers, empty and wide keys, unequal bonds, path/Y/comb topologies, and capacity-zero/over-budget cache policies passed with preserved output ordering. |
| `F` fallback parity | **PASS** | f32/Complex32 generic routes, unsupported raw shapes, zero budget, insufficient budget, and uncached over-budget misses remain valid and pass the release matrix. Malformed column-count/width errors are reported before insertion. |
| `I` invalidation/retention | **PASS** | No center-specific component layout remains; evaluator-owned directed layouts are immutable for evaluator lifetime. Logical payload budget is unchanged, while owned capacity/map storage is measured separately. Clearing the evaluator cache forces misses and no hidden dense full-network copy was added. |
| `D` determinism | **PASS** | Sorted topology traversal and explicit child order make composed keys deterministic. Direct/composed equality, cache counters, topology census, complete release tests, and downstream output checks passed repeatedly; timing outliers were not used as correctness evidence. |
| `S` scaling law | **PASS** for the scoped metadata law | The release test gates 4/8/16-site paths at `N(N-1)` retained one-physical-index positions and 2E directed layouts, plus Y/comb/unequal-bond trees. This establishes the scoped O(N²) chain metadata law versus the previous center × component retention; it makes no claim about full evaluator contraction complexity. |
| `P` provenance/observability | **PASS** | New design and performance statements are labelled **[AI Supplied]**. The checked key operation points to the exact core source range and prior full-source audit register; no new literature claim or direct `tenferro-*` dependency was added. `tensor4all-benchmark` remains N/A because its checkout has no comparable maintained TreeACI workload. |
| `CI` remote regression | **PENDING** | #711's prior CI failure was repaired and independently closed by run `33856551145` at head `898548186fe842eb700521ac3328399d43ace7f7`. The fresh #710 required-check run is `33862884867` at head `68af6cc01b31ed69c725b97e3db249a3330f6862`; as checked, Maintenance scripts and Lint passed, while Doctests, Test, and Coverage were pending. Review bot run `33862882258` passed. A pending CI run is not a pass and is monitored in parallel; it does not block starting #708. Any new failure is a next-round CI-repair task. |

### #710 verification commands and raw measurements

```text
cargo test --release -p tensor4all-treetn --lib --quiet
519 tests ran: 517 passed, 2 existing ignored
cargo test --release -p tensor4all-treetn --tests --quiet
all integration targets passed
cargo test --doc --release -p tensor4all-treetn --quiet
141 doctests passed
cargo test --release -p tensor4all-core --test common_basic --no-fail-fast --quiet
9 tests passed
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
passed
cargo bench -p tensor4all-treetn --bench cached_evaluator directed_component_scan -- --noplot
10 samples per evaluator case; unpinned repeated medians remained within noise
taskset -c 0 cargo bench -p tensor4all-treetn --bench cached_evaluator directed_component_scan -- --noplot
cold [2.2415 ms, 2.3161 ms, 2.3575 ms]; warm [1.8042 ms, 1.8195 ms, 1.8428 ms]
taskset -c 0 cargo bench -p tensor4all-treetn --bench cached_evaluator -- --noplot --sample-size 10 --warm-up-time 0.5 --measurement-time 1
all 101 benchmark IDs in the affected benchmark binary completed; raw Criterion data is retained under ignored target/criterion/
cargo test --release -p tensor4all-treetn --lib diagnostic_same_batch_warm_environment_vs_center_cost -- --ignored --nocapture --test-threads=1
moving-center N=16 chi=256: first=8.992 ms, second=4.318 ms, layouts=30, refs=240
cargo test --release -p tensor4all-treeaci --no-fail-fast --quiet
145 unit, 7 public_api, 1 rank_scaling, 18 doctests passed; 6 existing ignored
cargo test --release -p tensor4all-aci --no-fail-fast --quiet
85 unit, 4 integration, 1 rank_scaling, 19 doctests passed; 1 existing ignored
SGW_RUN_TAG=aci-gate-710 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0
complete clean-copy A/B workflow passed
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-710/treeaci pi_rtau
Converged; sweeps=7; evaluated_points=177646
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-710/treeaci sigma_rtau
Converged; sweeps=5; evaluated_points=99554
```

## 2026-09-04 #708 complete warm edge-cut evaluation

This entry records the implementation and gates for #708. The complete
warm-edge work count, deterministic cut selection, typed final dot product,
and benchmark acceptance interpretation are **[AI Supplied]** engineering
design and evidence policy. The tree-cut identity is explicitly
`Tree generalization — re-derived`: after removing an edge, the two directed
component messages are vectors on that bond and their dot product reconstructs
the scalar tree contraction. It is not attributed to a tree pseudocode section
in the ACI paper. No new literature-derived claim is introduced here. The
full-text ACI/TCI/tenferro source clones and their concrete page/equation/
pseudocode/source-line locators remain the evidence register above; no source
locator is used for the new engineering identity.

### Implementation

- A hinted batch now uses a deterministic top-level edge cut and combines the
  two directed messages with a typed f32/f64/Complex32/Complex64 bond dot
  product. The existing vertex-center raw/generic contraction remains the
  no-hint/default and fallback route.
- `get_or_compute_node_message` checks the exact parent component cache before
  recursively requesting children. A complete parent hit therefore returns
  without descendant reconstruction. Partial/miss requests recurse only as
  needed and preserve the original point order and duplicate semantics.
- The warm reverse-message path builds only the endpoint assignment batch when
  all requested keys are already cached; cold/partial requests retain the full
  rooted assignment construction. The direct encoder uses the immutable
  directed component layout and the same checked `FlatIndexer` contract as the
  composed-key path.
- Final edge assembly reads the cached scalar storage once and increments a
  test-only work counter once per point/bond pair. Checked multiplication and
  addition guard assignment and message offsets; scalar storage kind is never
  silently promoted or converted.
- The differential matrix covers path, unequal-bond Y, and comb fixtures,
  cold/full-hit/partial/reordered/duplicate/repeated batches, and all four
  scalar kinds. The ordinary `TreeTN::evaluate` result is the sole numerical
  oracle.

### #708 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C8` correctness | **PASS** | The full TreeTN release library matrix passed 522 tests: 520 passed and 2 existing tests remained ignored. All TreeTN integration targets passed, and 141 release doctests passed. The new complete edge-cut matrix passed path/Y/comb, unequal bonds, cold/full-hit/partial/reordered/duplicate/repeated batches, and f32/f64/Complex32/Complex64 against ordinary `TreeTN::evaluate`; no tolerance was relaxed. |
| `BC8` benchmark correctness | **PASS** | The full affected TreeTN, TreeACI, and ACI release matrices plus TreeTN doctests completed before accepting timing data. The benchmark's fixed N=16, two-point, five-bond-dimension workload is checked against the ordinary evaluator before each timed family; full-matrix evidence, not a benchmark smoke call, closes correctness. |
| `E8` efficiency | **PASS for the intended warm hinted path; opt-in cold setup disclosed** | The deterministic warm work counter is exact: 7 points × χ=2 gives 14 edge-assembly visits; a parent cache hit makes descendant message contracts zero. Pinned CPU-0 Criterion medians (µs, χ=16/32/64/128/256) were cold vertex `79.455/121.116/231.046/627.671/2699.512`, cold hinted edge `111.850/156.175/275.369/707.205/3048.775`, warm vertex `30.944/38.119/47.615/77.798/189.064`, and warm hinted edge `31.331/35.821/37.510/40.616/43.534`. Warm edge deltas versus vertex were `+1.3%/-6.0%/-21.2%/-47.8%/-77.0%`; the χ=16 case is within small fixed-cost noise and χ≥32 improves. The hinted cold setup is slower by `40.8%/28.9%/19.2%/12.7%/12.9%` because it materializes the reverse orientation; this is reported rather than hidden. The no-hint/default cold route remains the existing vertex-center path and is not replaced by the new edge route. |
| `R8` release/regression/downstream | **PASS** | TreeTN, TreeACI, and ACI full release matrices passed. A clean SGW archive `/tmp/sgw-treeaci-gate-708.bi58t6` patched only with local tensor4all-rs path dependencies ran `SGW_RUN_TAG=aci-gate-708 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0`; SimpleTT, TreeACI, and CTTN completed their full configured pipelines, all four checkpoint stages, five row slices, assembly, and plotting. Isolated TreeACI diagnostics converged: `pi_rtau` in 6 sweeps with 142808 evaluated points and `sigma_rtau` in 5 sweeps with 96618 evaluated points. The dirty `/root/projects/gw-rs/sgw` checkout was not modified. |
| `N` numerical stability/convergence | **PASS** | All four scalar kinds and unequal-bond/topology cases match the ordinary evaluator. Downstream TreeACI stages converge under the configured `1e-4` tolerance; no tolerance or reduction order was changed. |
| `M` metamorphic semantics | **PASS** | Exact full-hit repeats, reordered columns, duplicate columns, a partial/miss batch, repeated partial batches, all output ordering, and all directed message orientations used by the fixtures pass. |
| `F` fallback parity | **PASS** | No-hint calls continue through the existing center route; zero-bond, scalar-kind mismatch, unsupported raw shape, higher-degree generic, f32/c32 generic, zero-budget, and over-budget behavior remains covered by the full TreeTN release matrix. |
| `I` invalidation/retention | **PASS** | Cache lookup is exact by directed component and assignment key; partial misses compute only required dependencies; no input tensor is mutated and no public cache/API contract changes. Existing clear, zero-budget, bounded-budget, and uncached miss tests remain green. |
| `D` determinism | **PASS** | Sorted neighbor selection makes the cut deterministic; checked assignment encodings and stable point-order remapping preserve repeatable keys/results. The complete release matrices and fixed benchmark fixture are repeatable; timing outliers are not correctness evidence. |
| `S` scaling law | **PASS for the scoped warm assembly law** | The test-only counter enforces exactly `points * chi_edge`, independent of descendant size. The benchmark shows the warm edge route becoming increasingly favorable as χ grows; this is a scoped final-assembly/resource claim, not a claim about the full tree evaluator's asymptotic miss cost. |
| `P` provenance/observability | **PASS** | New algorithm/design and gate-policy statements are labelled **[AI Supplied]** or `Tree generalization — re-derived`. No new paper claim was made, no new tenferro dependency was added, and the pre-cloned full-text evidence register remains the source-compliance record. `tensor4all-benchmark` is N/A: its dirty checkout has no maintained comparable TreeTN edge-cut workload. |
| `CI` remote regression | **PENDING for the current docs follow-up; implementation run recorded** | Before #708, #710's CI run `33863186382` at head `dd1fc895c78b5d33de80cbceae1f94c2b094bf07` passed Coverage, Doctests, Lint, Maintenance scripts, Test, and `rollup-rs`; review bot run `33863184261` also passed. The #708 implementation push is run `33867996612` at head `5a083908028f5a8618157340aee1a205c5a8729a`; its review bot `33867994373` passed, while CI_rs Coverage, Doctests, Lint, Maintenance scripts, and Test were pending when this entry was recorded. This worklog-only follow-up will trigger a replacement run; pending remote checks are not treated as passes and do not block starting #712. A newly observed failure is the next-round CI-repair task. |

### #708 verification commands and raw measurements

```text
cargo fmt --all
cargo test --release -p tensor4all-treetn --lib --quiet
522 tests ran: 520 passed, 2 existing ignored
cargo test --release -p tensor4all-treetn --tests --quiet
all integration targets passed
cargo test --doc --release -p tensor4all-treetn --quiet
141 doctests passed
cargo clippy --release -p tensor4all-treetn --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
passed
taskset -c 0 cargo bench --bench cached_evaluator -- treetn_warm_edge_cut_vs_vertex_center
10 samples per case; raw Criterion estimates retained under target/criterion/treetn_warm_edge_cut_vs_vertex_center/**/new/estimates.json
cargo test --release -p tensor4all-treeaci --no-fail-fast --quiet
145 passed, 6 existing ignored; public_api/rank_scaling/doctest targets passed
cargo test --release -p tensor4all-aci --no-fail-fast --quiet
85 passed, 1 existing ignored; integration/rank_scaling/doctest targets passed
SGW_RUN_TAG=aci-gate-708 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 1.0
clean-copy SimpleTT/TreeACI/CTTN workflow passed; all configured extraction and plotting stages passed
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-708/treeaci pi_rtau
Converged; sweeps=6; evaluated_points=142808
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T1.0_mu0.5_aci-gate-708/treeaci sigma_rtau
Converged; sweeps=5; evaluated_points=96618
```

The new implementation has been committed and pushed as the #708 subissue
closure. The next implementation subissue is **#712** (the budgeted
shared-operand grouped-GEMM tensorbackend facade); per the execution protocol,
stop after reporting this #708 closure and do not begin #712 in the same run.

## 2026-09-04 #712 predeclared protocol

This section is recorded before implementing or timing #712. The public
grouped-GEMM facade, validation policy, benchmark workload, and downstream
acceptance interpretation are **[AI Supplied]** engineering design. No new
paper-derived numerical or algorithmic claim is introduced for this subissue.
The upstream tenferro API is software source provenance, not literature: the
full cargo checkout at commit
`007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4` was inspected, especially
`crates/tenferro-cpu/benches/grouped_gemm.rs` lines 1--165
(`GroupedGemmJob::new`, `GroupedGemmConfig::new`, borrowed views, and
`BackendCachedDot::grouped_gemm_cached`) and
`crates/tenferro-cpu/src/dot_runtime.rs` lines 461--490 and 796--940
(validation and configured-provider dispatch). This is an implementation
reference only; it is not cited as a paper result.

The fixed acceptance protocol is:

- correctness: all four supported scalar kinds (f32, f64, Complex32,
  Complex64), empty/singleton/shared-LHS/shared-RHS batches, valid offsets,
  every validation error class, and every tensorbackend release unit,
  integration, and doctest target; valid outputs are compared to sequential
  individual GEMMs before timing is accepted;
- efficiency: pinned CPU 0, one configured backend thread, ten Criterion
  samples per case, complete declared cases consisting of 1, 2, 8, and 32
  jobs with shared RHS/LHS spans and matching duplicated-input baselines;
  report median runtime and exact input-copy/allocation accounting, with no
  post-hoc case or metric selection;
- downstream: `T=1.0` is smoke-only. The mandatory SGW gate is the complete
  `T=0.1` A/B workflow with all configured extraction, checkpoint, row-slice,
  assembly, and plotting stages, followed by isolated `pi_rtau` and
  `sigma_rtau` convergence. T=1 can expose gross failures but never closes
  #712 correctness, efficiency, or downstream readiness;
- regression/provenance: run complete release matrices for tensorbackend,
  TreeACI, and ACI; preserve configured provider/thread behavior; report
  unrelated remote CI failures separately, and do not wait for pending CI to
  start implementation.

## 2026-09-04 #712 completion

The implementation and all #712 gate interpretations in this section are
**[AI Supplied]** engineering evidence. No new literature-derived claim is
made. The upstream tenferro software reference remains the full checkout at
`007e3bb6c1187a2569d237b2bc6e6ad486f2b4f4`; the exact source locations used
were `crates/tenferro-cpu/benches/grouped_gemm.rs:1-165` for the descriptor,
borrowed-view, and cached grouped-call shape, and
`crates/tenferro-cpu/src/dot_runtime.rs:461-490,796-940` for session/provider
validation and dispatch. These are source-code locators, not paper citations.

### Implementation summary

- Added tensorbackend-owned `GroupedGemmJob`, `GroupedGemmOptions`, and
  `GroupedGemmError` APIs. Public jobs contain only offsets and matrix
  dimensions; tenferro descriptors remain private to the bridge.
- Added borrowed and consuming grouped execution functions. Input payloads are
  borrowed directly as one-dimensional column-major views; only the translated
  descriptor metadata is allocated. A configured `CpuBackend` entry point is
  available for caller-owned provider/thread semantics, while the legacy
  convenience entry uses the configured process-global context.
- Validation runs before session entry: checked dimension/offset arithmetic,
  bounds, output disjointness, compatible exact shared LHS/RHS shapes, and
  descriptor translation budget. The output is not mutated on validation error.
- Added all-four-scalar differential tests (f32, f64, Complex32, Complex64),
  shared LHS/RHS cases, empty/singleton/owned cases, layout/order checks,
  overflow/bounds/overlap/shared-shape/budget failures, and configured
  one-thread backend coverage.
- TreeACI production adoption is deferred. The facade's need probe is positive,
  but the current TreeACI frame implementation has no clean grouped-GEMM seam;
  forcing a call-site change would violate the no-regression gate. This is a
  deliberate scope decision, not an end-to-end TreeACI performance claim.

### #712 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C9` correctness | **PASS** | Complete tensorbackend release unit/integration matrix: 227 passed, 1 existing ignored; all 153 tensorbackend release doctests passed. New tests cover all four supported scalar kinds, shared LHS/RHS, column-major offsets, individual-GEMM oracles, empty/singleton/owned behavior, bounds, checked overflow, output overlap, incompatible shared shapes, and budget rejection. |
| `BC9` benchmark correctness | **PASS** | The complete declared matrix (1/2/8/32 jobs × shared LHS/RHS × shared/duplicated) performed an oracle comparison before each timed pair; every case printed `oracle=pass`. No smoke-only benchmark result was accepted. |
| `E9` efficiency | **PASS for the scoped facade; adoption deferred** | Pinned CPU 0, one configured backend thread, ten Criterion samples per timed case. Median µs (shared vs duplicated): shared LHS jobs 1 `9.6069` vs `9.8413` (2.4%), 2 `9.8085` vs `10.179` (3.6%), 8 `11.947` vs `12.379` (3.5%), 32 `19.094` vs `23.089` (17.3%); shared RHS jobs 1 `9.5049` vs `9.7545` (2.6%), 2 `9.7798` vs `10.192` (4.0%), 8 `11.439` vs `12.554` (8.9%), 32 `19.524` vs `23.372` (16.5%). The baseline copied/allocated the shared payload each call; duplicated-input bytes were 2,048/4,096/16,384/65,536 for 1/2/8/32 jobs. This proves a causal facade-level benefit, not a promoted TreeACI end-to-end result. |
| `R9` release/regression | **PASS** | `cargo test --release --workspace --exclude tensor4all-hdf5 --no-fail-fast --quiet` passed every target; separate HDF5 release tests passed 1 active + 4 ignored unit, 46 integration, and 10 additional tests. TreeACI passed 145/6 ignored plus 7 public API, 1 rank-scaling, and 18 doctests; ACI passed 85/1 ignored plus 4 integration, 1 rank-scaling, and 19 doctests. |
| `DS9` downstream T=0.1 | **PASS** | Clean SGW archive `/tmp/sgw-treeaci-gate-712.xUgPMJ` at source HEAD `ba6fbf3e4461bd4b6ba6447d6d22f63151637ac4`, with all tensor4all dependencies patched to this worktree, ran `SGW_RUN_TAG=aci-gate-712-t01 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 0.1`. SimpleTT, TreeACI, SRC, and CTTN completed all configured pipelines, four checkpoint stages, five row slices, assembly, and plotting. Isolated TreeACI `pi_rtau` converged in 7 sweeps with 2,027,526 evaluated points; `sigma_rtau` converged in 6 sweeps with 933,280 evaluated points. |
| `T=1` smoke distinction | **PASS as smoke only** | The prior T=1 run is retained only for gross-breakage diagnosis. It is not used to close `C9`, `BC9`, `E9`, or `DS9`; T=0.1 is the mandatory downstream evidence. |
| `F` fallback/provider | **PASS** | Empty jobs are a no-op; valid jobs dispatch through configured session/provider; invalid requests fail before backend mutation; the explicit `CpuBackend::with_threads(1)` test preserves the configured thread count. |
| `D` determinism | **PASS** | Fixed job order, fixed column-major buffers, deterministic data, pinned CPU, one thread, and fixed ten-sample Criterion protocol were used. Timing outliers are disclosed by Criterion and are not used as correctness evidence. |
| `P` provenance/observability | **PASS** | New engineering statements are labelled **[AI Supplied]**; the tenferro source commit and exact source-line locators are recorded above. No new paper claim, direct downstream tenferro dependency, or tensor4all-benchmark claim was added. `tensor4all-benchmark` was not used because no maintained comparable grouped-GEMM workload was available in scope. |
| `CI` prior-round check | **PASS / pending coverage** | Prior #708 CI run `33868111483` at `afcb350fe41c112d68138e953b1f9b41af52fddf` passed Doctests, Lint, Maintenance scripts, and Test; Coverage remained pending when checked. Review bot run `33868108624` passed. No failure was observed to become a repair task; pending CI did not block #712. |

### #712 verification commands and raw anchors

```text
cargo test --release -p tensor4all-tensorbackend --no-fail-fast --quiet
227 passed; 1 existing ignored; 153 release doctests passed
taskset -c 0 cargo bench --profile release -p tensor4all-tensorbackend --bench grouped_gemm -- --noplot
complete 1/2/8/32 × shared_lhs/shared_rhs × shared/duplicated matrix; 10 samples each; every oracle=pass
cargo test --release --workspace --exclude tensor4all-hdf5 --no-fail-fast --quiet
all targets passed
cargo test --release -p tensor4all-hdf5 --no-fail-fast --quiet
1 active + 4 ignored unit; 46 integration; 10 additional tests passed
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
passed
cargo doc --workspace --no-deps
passed; existing unrelated rustdoc link warnings only
cargo run -p xtask --release -- api-dump
generated complete tensorbackend API inventory; new GroupedGemm surface present
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
pass; python3 scripts/test-repository-rules-review.py: 90 tests passed
SGW_RUN_TAG=aci-gate-712-t01 SGW_ACI_GLOBAL_GUARD=1 ./run_r10_nblock_treeaci_ab.sh 0.1
clean T=0.1 complete workflow passed; all extraction/assembly/plotting stages passed
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T0.1_mu0.5_aci-gate-712-t01/treeaci pi_rtau
Converged; sweeps=7; evaluated_points=2027526
cargo run --release --locked --features isolation-diagnostics --bin isolate_aci_stage -- runs/R10_nblock_T0.1_mu0.5_aci-gate-712-t01/treeaci sigma_rtau
Converged; sweeps=6; evaluated_points=933280
```

## 2026-09-04 #713 closure: arbitrary-degree candidate-frame batching

This entry closes the implementation half of #713: the three-or-more-incoming
candidate-frame contraction no longer falls back to a per-candidate scalar
recursion. Every design choice, routing rule, and measurement interpretation
below is **[AI Supplied]** engineering work. No new paper- or
specification-derived claim is introduced, so no new literature locator is
asserted; the existing full-source register at
`## 2026-09-03 #707 closure` remains the authority for algorithmic claims.
The generalization is labelled **Tree generalization — re-derived**: it
continues the accepted exactly-two-incoming decomposition one incoming
component at a time and is validated against the production scalar
accumulator, not copied from pseudocode.

### #713 changes

- `incoming_batch_matrix(core, outgoing_axis, incoming_axes, physical_offset,
  frame_matrices) -> Result<PackedCandidateBatch<T>>` in
  `crates/tensor4all-treeaci/src/frames.rs` is the degree-generic kernel. It
  handles degree 0 (a gather), 1 (`single_incoming_core_matrix` plus one
  `mat_mul`), 2 (delegates to the untouched
  `two_incoming_core_matrix_batched`), and 3-or-more (`generalized_incoming_batch`).
  All four routes produce the same column-major layout, so the flat offsets
  the existing kernels already read back with are unchanged.
- `PackedCandidateBatch<T>` owns that cross with checked prefix strides
  `strides[k] = outgoing_dim * n_0 * ... * n_{k-1}`; `frame(&coordinates)`
  returns one contiguous `outgoing_dim` slice and rejects a wrong degree or an
  out-of-range column instead of indexing blindly.
- `generalized_incoming_batch` keeps the two-incoming kernel's memory shape:
  step one gathers one `outgoing_dim x d_0` core block per remaining incoming
  coordinate combination, so the complete `outgoing_dim * prod d_k` core cross
  is never materialized. Each later step contracts the next component out of
  the whole running buffer through one shared-operand grouped GEMM
  (`tensor4all_tensorbackend::grouped_mat_mul_shared`, the #712 facade): the
  blocks share one frame matrix and write disjoint output spans, so no block
  is copied into its own matrix. No direct `tenferro-*` dependency is added.
- Contraction proceeds in incoming-edge order — the same order and the same
  association the accepted degree-two kernel uses — so degree three and above
  are the literal continuation of the existing reduction rather than a new
  one. Candidate order, sample order, cache semantics, dtype behavior, pivot
  selection, and evaluated-point accounting are untouched.
- `multi_incoming_scratch_elements` generalizes the working-byte charge;
  `two_incoming_scratch_elements` is now literally its `q = 2` case and a test
  pins the degree-two value against the pre-#713 literal formula, so the
  degree-two contract is unchanged byte for byte.
- `InputFrameStore::candidate_frames_for_edge` dispatches degree >= 3 to the
  new `candidate_frames_for_edge_multi_incoming`; the zero-incoming leaf edge
  keeps the scalar route (its contraction is a plain gather that batching
  cannot improve) and stored-sample materialization
  (`FrameBuilder::compute_batch`) keeps its scalar route at degree >= 3, see
  the boundary note below.

### #713 routing contract

Each `local_coordinate` group takes exactly one of two documented routes:

1. **batched** — when the group's complete Cartesian cross is no larger than
   the number of candidates the caller actually requested *and* every
   simultaneously live buffer of that cross fits `max_working_bytes`
   (`multi_incoming_scratch_elements` plus `grouped_gemm_descriptor_bytes`);
2. **scalar** — the same `contract_prepared_core_slices` contraction
   `candidate_frame` performs, otherwise.

The cross-size condition is what prevents a sparse or diagonal candidate set
from silently materializing the full edge cross: the batched kernel computes
every combination, so it is used only where every combination was requested.
`enumerate_candidates` always emits the complete cross, so production groups
take the batched route whenever their intermediates are affordable. The
budget condition selects a route rather than raising a limit error, so a tight
budget degrades to the previous per-candidate cost instead of failing. The one
intentional accounting change is that
`enumerated_candidate_frame_scratch_elements` now reports the batched charge
for degree >= 3 whenever the batched route will actually run, which is exactly
what #713's acceptance criterion "working memory is checked against the
Cartesian candidate and intermediate sizes" requires; degree 0/1/2 accounting
is unchanged. Both the pre-flight estimate and the kernel use the same
predicate against the same limit, so they cannot disagree about the route.

### #713 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C11` correctness | **PASS** | Degrees 0/1/2/3/4 are compared against the production scalar reduction. `incoming_batch_matrix_matches_the_scalar_accumulator_for_degree_zero_to_four` drives the kernel at every degree, for `f64` and `Complex64` with a genuinely nonzero imaginary part, with unequal bond dimensions, unequal candidate counts, two physical coordinates, and a reversed core axis order (outgoing axis fastest, incoming axes in strictly decreasing axis order), and compares one materialized whole result against `accumulate_incoming` — the exact accumulator `contract_prepared_core` calls. `three_incoming_candidate_batches_match_the_scalar_oracle` and `four_incoming_candidate_batches_match_the_scalar_oracle` do the same through the dispatched `candidate_frames_for_edge` against the `candidate_frame` oracle on 4-arm and 5-arm stars with unequal bonds, two hub physical legs, and permuted hub axis order, asserting candidate order, candidate count, bond dimension, and routing. The pre-existing integer-valued 4-arm fixture still asserts exact equality (`candidate_frames_for_edge_batches_three_incoming_edges`). `tests/branch_degree.rs` runs a full `tree_elementwise` on a hub whose tree coordination number is four (four bonds, i.e. three incoming components per outward arc) and compares one dense materialization against `dense(a).sub(&dense(b))` by `maxabs()`. Complete TreeACI release matrix: 155 unit, 1 branch-degree, 7 public-API, 1 rank-scaling, 18 doctests; 7 ignored diagnostics/measurement tests. |
| `BC11` benchmark correctness | **PASS** | The complete affected-crate matrix above was green before any timing was accepted, and every timed pair inside the measurement itself asserts `max_residual <= 1e-12 * scale` against the scalar oracle before its medians are reported. No smoke case was used to close a gate. |
| `E11` efficiency | **PASS** on wall time far above noise, plus a causal resource reduction | Paired release measurements, same fixture, seeds, backend, and `taskset -c 0`, 5 samples per side per size, 3 independent runs. Medians (batched vs scalar): `m=6` 0.949/0.974/1.134 ms vs 15.796/16.029/18.160 ms (16.65x/16.46x/16.01x); `m=12` 1.169/1.207/2.449 ms vs 126.02/127.24/176.41 ms (107.8x/105.4x/72.0x); `m=24` 3.328/3.279/5.210 ms vs 1.0330/1.0253/1.3171 s (310.4x/312.7x/252.8x). Observed noise floor: within-run 5-sample relative spread on the scalar side is 0.3%-28%, and the worst run-to-run scalar median drift is 40% (`m=12`, loaded machine); the smallest observed speedup, 16.0x, is more than an order of magnitude above that. The causal resource reduction is exact and independent of timing: at `m=12` the scalar route performs `candidates * outgoing_dim * prod(d_k) = 1728 * 4 * 4096 = 28,311,552` core-element reads, while the batched route reads the core exactly `outgoing_dim * prod(d_k) = 16,384` times, a reduction by exactly the candidate count (1728x). The #712 grouped-GEMM facade removes a further 15 backend launches and 21,504 element block copies per group in that shape (16 blocks of 768 elements at step two plus 1 block of 9,216 at step three). |
| `R11` release/regression | **PASS** | The exactly-two-incoming kernel, its scratch formula, its dispatch, and its tests are untouched, and `multi_incoming_scratch_matches_the_two_incoming_specialization` pins the degree-two charge against the literal pre-#713 formula for four shapes. Ranks, pivots, dense residuals, and errors are unchanged: the complete TreeACI release matrix and the `rank_scaling` bound test pass. `cargo clippy --release -p tensor4all-treeaci --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` is clean, `scripts/check-crate-boundaries.py` reports `crate-boundary-ok`, and `scripts/repository-rules-review.py --base main --worktree --dry-run` reports pass with no findings. **Downstream ACI validation is N/A with evidence**: the SGW production runs use `SGW_TOPOLOGY=comb`, whose graph is `(0,1),(0,3),(1,2),(1,4),(2,5)` with maximum coordination number 3, and the repository's `branching_topology.json` fixture also has maximum coordination 3. A coordination-3 node has at most two incoming components on any directed edge, so the downstream tree never reaches the degree >= 3 route this task changes. |
| `CI` prior-round check | **N/A (deferred)** | The branch is committed locally and deliberately not pushed, per this task's instructions; the coordinator owns the push and the required-check comparison. No CI conclusion is claimed here. |
| `N` numerical stability/convergence | **PASS** | The differential tests use non-degenerate, non-power-of-two fixture values with unequal bonds and complex data, and assert a whole-result residual at `1e-13 * scale` (kernel) and `1e-12 * scale` (dispatch) rather than a relaxed tolerance. No tolerance anywhere was relaxed. The degenerate-fixture guards (`scale > 1e-3`, first-two-candidates-differ, `maximum_rank >= 2`) prevent a vacuous pass. Zero/identity and rank-deficient behavior is inherited unchanged from the scalar route, which remains the fallback. The batched association across incoming axes is the one the accepted degree-two kernel already uses; it differs from the scalar nesting order only in the same way degree two has always differed, which is why the residual, not bitwise equality, is the criterion on non-exactly-representable fixtures. |
| `M` metamorphic semantics | **PASS** | `multi_incoming_batch_preserves_order_for_duplicate_and_reordered_candidates` reverses the complete cross and appends duplicates, then compares against the scalar oracle in the caller's order; duplicates do not collapse into one column. The kernel test uses a reversed core axis order so the incoming axes are not in ascending core-axis order, and the dispatch fixtures interleave physical legs between bonds and give the outgoing bond a middle axis position. Local-coordinate grouping is a `BTreeMap`, so group order is independent of candidate order. |
| `F` fallback parity | **PASS** | Both documented routes are exercised and compared: `multi_incoming_batch_falls_back_to_scalar_for_a_sparse_candidate_set` (diagonal set, `batched_groups == 0`, `scalar_groups == 1`, exact equality with the oracle) and `multi_incoming_batch_falls_back_to_scalar_when_the_working_budget_is_tight` (one byte under the batched charge: no error, scalar route taken, exact equality with the oracle, and the pre-flight estimate shrinks back to the scalar charge so kernel and pre-flight agree). Shape and axis error classes are covered by `incoming_batch_matrix_rejects_inconsistent_shapes` and `multi_incoming_scratch_rejects_overflowing_or_inconsistent_shapes`, including a checked `usize` overflow. Leaf edges and the stored-sample path keep their scalar routes and their existing tests. |
| `I` invalidation/retention | **PASS** | Cache semantics are unchanged: three-or-more-incoming candidates are still never cached (their exact identity would need an unbounded vector key), and `multi_incoming_batch_is_deterministic_and_never_caches_candidates` asserts one recorded miss per candidate and zero hits across three repeated calls on the same store. No new retained state is introduced; every intermediate is released at the end of its group. The existing frame-cache bound, reclamation, and extension tests remain green. |
| `D` determinism | **PASS** | Three repeats of the same call on the same store produce bitwise-identical packed payloads (`runs[0] == runs[1] == runs[2]`, exact `Vec<f64>` equality). Routing is a pure function of dimensions, counts, and the configured limit, so it cannot vary between runs. The three independent measurement runs report the same ordering and the same structural counters. |
| `S` scaling law | **PASS** | Independent 1x/8x/64x candidate-product sweep at fixed `chi = [16,16,16]`, `outgoing_dim = 4` (`m = 6, 12, 24`). Scalar time per unit candidate product is flat — 73,130 / 72,930 / 74,727 ns in run 1 — confirming the scalar route costs a constant per candidate and therefore scales as the full candidate product. Batched time per unit candidate product falls 4,392 -> 676 -> 241 ns over the same sweep, i.e. strictly sub-linear in the candidate product. Normalized by the topology-required `d * prod(chi_e) = 16,384`, which is constant across the sweep, the scalar route grows 964 -> 7,692 -> 63,051 ns (proportional to the candidate product) while the batched route grows 58 -> 71 -> 203 ns. Counters, not wall clock, carry the exponent claim: core-element reads are `candidates * outgoing_dim * prod(d_k)` for the scalar route and exactly `outgoing_dim * prod(d_k)` for the batched route, independent of the candidate count. |
| `P` provenance/observability | **PASS** | Every #713 statement in this section is **[AI Supplied]**, and the generalization itself is **Tree generalization — re-derived** from the existing two-incoming kernel plus differential tests against the production scalar accumulator; it is not attributed to any paper. The tensorbackend grouped-GEMM facade is cited to #712's own closure entry and its recorded tenferro source locators, not re-attributed. Raw commands, medians, all five samples per side, relative spreads, structural counters, charged peak bytes, and the fixture shapes are recorded below. |

### #713 verification commands and raw measurement output

```text
cargo test --release -p tensor4all-treeaci frames --no-fail-fast
49 passed, 0 failed, 5 ignored in the filtered frame target

cargo test --release -p tensor4all-treeaci --test rank_scaling --no-fail-fast
1 passed, 0 failed

cargo test --release -p tensor4all-treeaci --no-fail-fast
155 unit passed, 7 ignored; 1 branch_degree passed; 7 public_api passed;
1 rank_scaling passed; 18 doctests passed

cargo fmt --all -- --check
clean

cargo clippy --release -p tensor4all-treeaci --all-targets \
  -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
clean

python3 scripts/check-crate-boundaries.py
crate-boundary-ok

python3 scripts/repository-rules-review.py --base main --worktree --dry-run
Verdict: pass; No findings.

python3 scripts/check-crate-boundaries.py
crate-boundary-ok
python3 scripts/audit-library-panics.py
Audit passed: 0 unbaselined findings, 0 stale baseline entries

python3 scripts/check-public-error-docs.py
public Result APIs with incomplete error documentation:
- crates/tensor4all-tensorbackend/src/matrix.rs:638: grouped_mat_mul_shared_with_backend: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1373: new: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1714: with_plan: # Errors does not name a concrete variant or condition
(the same three findings reproduce on a clean `git archive a7632cc` tree, so
 they are inherited from #712 and #709 and are not caused by this commit --
 see Open items item 6)
```

Test-first record: the differential and routing tests were added and run
before the generalized kernel existed. That run reported
`39 passed; 4 failed; 4 ignored` — the four failures were
`three_incoming_candidate_batches_match_the_scalar_oracle` and
`four_incoming_candidate_batches_match_the_scalar_oracle`
(`batched groups: left 0, right 4` and `left 0, right 2`),
`multi_incoming_batch_preserves_order_for_duplicate_and_reordered_candidates`
(`batched_groups() > 0` false), and
`multi_incoming_batch_falls_back_to_scalar_when_the_working_budget_is_tight`
(the pre-flight still charged the scalar estimate). The residual comparisons
against the scalar oracle passed on the scalar route in that same run, which
is what establishes the oracle harness itself before the route changed.

```text
taskset -c 0 cargo test --release -p tensor4all-treeaci --lib \
  frames::tests::three_incoming_batched_vs_scalar_release_measurement \
  -- --ignored --nocapture

run 1
m=6,  d=16, outgoing_dim=4, chi=[16,16,16], counts=[6,6,6],    candidates=216,   samples=5
  batched_median=948.724us  scalar_median=15.796134ms  speedup=16.65x
  batched_all=[904.732us, 946.069us, 948.724us, 980.113us, 7.212663ms]
  scalar_all=[15.727104ms, 15.75103ms, 15.796134ms, 15.93801ms, 16.085979ms]
  ns per d*prod(chi_e)=16384: batched=57.9055  scalar=964.1195
  ns per candidate product=216: batched=4392.2407  scalar=73130.2500
  full_cross_elements=864    peak_charged_elements=9688    peak_charged_bytes=77504
m=12, d=16, outgoing_dim=4, chi=[16,16,16], counts=[12,12,12], candidates=1728,  samples=5
  batched_median=1.168878ms scalar_median=126.023577ms speedup=107.82x
  batched_all=[1.138431ms, 1.142648ms, 1.168878ms, 1.17018ms, 1.638832ms]
  scalar_all=[125.578942ms, 125.772745ms, 126.023577ms, 126.694108ms, 131.555069ms]
  ns per d*prod(chi_e)=16384: batched=71.3427  scalar=7691.8687
  ns per candidate product=1728: batched=676.4340  scalar=72930.3108
  full_cross_elements=6912   peak_charged_elements=29104   peak_charged_bytes=232832
m=24, d=16, outgoing_dim=4, chi=[16,16,16], counts=[24,24,24], candidates=13824, samples=5
  batched_median=3.328473ms scalar_median=1.033025774s speedup=310.36x
  batched_all=[3.306846ms, 3.311766ms, 3.328473ms, 3.504829ms, 4.117553ms]
  scalar_all=[1.021731925s, 1.028326059s, 1.033025774s, 1.034817035s, 1.080357718s]
  ns per d*prod(chi_e)=16384: batched=203.1539  scalar=63050.8895
  ns per candidate product=13824: batched=240.7750  scalar=74726.9802
  full_cross_elements=55296  peak_charged_elements=118048  peak_charged_bytes=944384

run 2 (medians only)
m=6   batched=974.117us  scalar=16.029238ms  speedup=16.46x
m=12  batched=1.207124ms scalar=127.235003ms speedup=105.40x
m=24  batched=3.279061ms scalar=1.02534917s  speedup=312.70x

run 3 (medians only; loaded machine, both sides equally affected)
m=6   batched=1.134313ms scalar=18.15953ms   speedup=16.01x
m=12  batched=2.449385ms scalar=176.406725ms speedup=72.02x
m=24  batched=5.210035ms scalar=1.317149311s speedup=252.81x
```

### #713 Step 4: full-cross measurement boundary

Issue #713 requires that, once the implementation cliff is removed, the
remaining question — whether materializing the complete edge cross is itself
dominant — be measured and then closed without an algorithm change. It is
dominant, and this task stops here.

Per `local_coordinate` group at incoming degree three with outgoing dimension
`o`, equal incoming bond dimension `d`, and equal candidate count `n`, the
generalized route performs:

- step one (gather + contract component 0): `o * d^3 * n` multiply-adds;
- step two (contract component 1): `o * d^2 * n^2`;
- step three (contract component 2, the step that produces the full cross):
  `o * d * n^3`.

For the measured `o = 4`, `d = 16` sweep this is 98,304 / 36,864 / 13,824 at
`n = 6`; 196,608 / 147,456 / 110,592 at `n = 12`; and 393,216 / 589,824 /
884,736 at `n = 24`. The final, cross-producing step's share of the total
therefore grows 9.3% -> 24.3% -> 47.4% for a 1x/8x/64x candidate product and
tends to 1 as `n^3 / (a + b n^2 + c n^3)`. The measured batched time follows
that shape: 0.949 ms -> 1.169 ms -> 3.328 ms, an accelerating growth
(x1.23 then x2.85) that is converging on linear-in-candidate-product. Peak
charged working memory grows the same way, 77,504 -> 232,832 -> 944,384 bytes,
because the dominant term is the `o * n^3` cross itself.

The transient-memory cost of removing the cliff is explicit: the scalar route
charged only the incoming frame slices (48 elements, 384 bytes at `m = 12`)
while the batched route charges 29,104 elements (232,832 bytes) for the same
group. That is precisely why the routing contract falls back to the scalar
route rather than allocating over budget.

Conclusion: the `[AI Supplied]` implementation cliff identified as
`P0: arbitrary-degree branch contraction has a scalar efficiency cliff` is
removed — the batched route no longer scales with the candidate count in
core-element reads and is 16x-313x faster over the measured sweep. What
remains is the mathematically real `C_(u->v) = d_u * prod(r_f)` full-cross
cost recorded as `P0: branch candidates and local matrices grow
multiplicatively`, which is an algorithmic limitation of materializing the
complete edge cross. Per the plan's Step 4 instruction this task stops at the
measurement boundary; a lazy/block/pivot-search formulation requires a
separate derivation with its own correctness and convergence review.

### #713 deliberate boundaries

- `FrameBuilder::compute_batch` keeps its scalar route at degree >= 3. Stored
  samples are arbitrary interned records rather than a complete cross, so the
  same density guard that protects the candidate route would reject them and
  the batched kernel would compute a superset. The existing
  `compute_batch_keeps_the_scalar_route_on_three_incoming_edges` differential
  test pins that behavior.
- Step one still issues one small GEMM per remaining incoming coordinate
  combination (256 launches in the `m = 12` shape) rather than one grouped
  call, because a single grouped call there would require gathering the
  complete `outgoing_dim * prod(d_k)` core cross — 134 MB at `chi = 64` — which
  is exactly the allocation this kernel is written to avoid. Tiling that
  gather under an explicit memory bound is a possible follow-up, not a
  requirement of #713.
- The zero-incoming leaf route is unchanged: its contraction is a plain
  gather of `outgoing_dim` values with a compact cache key, and batching it
  would change cache semantics for no arithmetic gain.

## 2026-09-04 #709 closure: typed batch evaluation and reusable Guard plans

This entry records the implementation and gates for #709. The typed result
boundary, the plan/message lifetime split, the counting-allocator measurement
method, and the acceptance interpretation below are **[AI Supplied]**
engineering design and evidence policy. No new literature-derived claim is
introduced: the numerical contraction order, candidate order, pivot selection,
tolerance semantics, and evaluated-point accounting are unchanged, and the
existing full-text ACI/TCI/tenferro source clones plus their page/equation/
pseudocode locators remain the evidence register for every algorithmic claim.
The dtype conversion policy is not new either -- it is the dynamic
`AnyScalar` + `TreeAciScalar::from_evaluated_scalar` contract, restated at a
typed seam.

### #709 changes

- `TreeTNCachedEvaluator::evaluate_batched_typed<T: TensorElement>(values,
  hint)` returns `Vec<T>` directly. `evaluate_batched` and
  `evaluate_batched_with_hint` remain the `AnyScalar` compatibility wrappers
  over the same internal result, so no caller loses the dynamic route.
- The evaluator's internal raw leaf/chain/branch centre paths, the
  degree-2/3 raw centre path, the warm edge-cut path, and the generic
  `contract_with_options` centre path now all return `Vec<CachedScalar>`, the
  lightweight typed carrier the message cache already used.
  `AnyScalar::from_value` -- which eagerly builds a rank-zero `IdxTensor` --
  is constructed only inside the compatibility wrapper. The typed route
  constructs no rank-zero tensor for any result.
- The typed route keeps the dynamic wrapper's dtype rules exactly: precision
  conversion within a kind, a real payload widened into a complex request,
  and a complex payload never silently narrowed into a real request. The
  refusal is the new public `EvaluatedScalarKindMismatch`, wrapped in
  `TreeTNOperationError`; `tensor4all-treeaci`'s `scalar::typed_evaluation_error`
  downcasts it back to `TreeAciError::ScalarKind`, so the Guard's error class
  is unchanged and every other evaluator failure still propagates as
  `TreeAciError::TreeTN`.
- `CachedEvaluatorPlan` separates the dtype-independent plan -- topology,
  physical entry table, per-node and per-directed-component key layouts,
  sorted node and neighbour tables, and memoized rooted traversal plans --
  from the numerical message caches, which stay per evaluator.
  `TreeTNCachedEvaluator::with_plan` shares one `Arc` plan; `new` builds a
  private one. `with_plan` validates the node set, the sorted neighbour
  structure, the site-index count, and each physical index's node, and
  deliberately does not compare bond dimensions, values, or dtype: those are
  what a plan is allowed to outlive.
- TreeACI's Guard now uses the typed method for all four scalar kinds on both
  the input and output routes. `InputEvaluators` builds one plan for the
  problem and hands it to every input evaluator and to each per-invocation
  `GuardOutputEvaluator`, so pivot injection rebuilds only numerical
  messages. This is sound because `validate_initial_guess` already requires
  the output to share the inputs' labelled topology and site space; the
  invariant is now also enforced by `with_plan` on every rebuild.
- Assignment allocations: `build_message_assignment_batches` and
  `build_directed_assignment_batch` reuse one coordinate scratch buffer per
  call instead of one vector per node and point, `validate_entry_values`
  bounds-checks borrowed entries instead of cloning index/value pairs into a
  temporary, and the per-call `tree.node_names()` sort and
  `sorted_neighbors(tree)` rebuild are replaced by the plan's immutable
  tables.

### #709 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C10` correctness | **PASS** | `evaluate_batched_typed_matches_any_scalar_wrapper_for_all_scalar_kinds` compares the typed and `AnyScalar` routes for f32/f64/Complex32/Complex64 on cold, warm, and hinted calls: exact equality of value *and* of the wrapper's dtype debug string, preserved batch order, duplicate columns equal, empty batch empty, shape mismatch and unknown-hint errors on both routes, real-to-complex widening, complex-to-real refusal naming the requested dtype, and full `CachedEvaluationStats` equality (message hits/misses, environment count, edge-cut work count, centre-contract count) after every call, which is the evaluated-point accounting. `guard_typed_input_evaluation_matches_the_dense_oracle_for_all_scalar_kinds` repeats the check at the Guard boundary for scan-shaped and general batches against one dense materialization. Release matrices: tensor4all-treetn 523 lib tests passed / 2 pre-existing ignored, all treetn integration targets passed, 152 doctests passed; tensor4all-treeaci 148 lib / 6 pre-existing ignored, 7 public_api, 1 rank_scaling, 18 doctests; tensor4all-aci 85 lib / 1 ignored, 4 elementwise, 1 rank_scaling, 19 doctests. No tolerance was changed. The typed path creates no rank-zero result tensors; that is measured, not asserted from the diff (see `E10`). |
| `BC10` benchmark correctness | **PASS** | The complete affected release matrices above (TreeTN, TreeACI, ACI, plus doctests and clippy with `-D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`) completed before any timing was accepted. Both timed harnesses also check themselves: the microbenchmark asserts wrapper and typed results agree element-wise before timing, and the end-to-end parity benchmark asserts the TreeACI relative error against the chain arm for every bond dimension. |
| `E10` efficiency | **PASS** on both an end-to-end Guard improvement and a causal resource reduction | Primary metric, predeclared as the end-to-end Guard-enabled TreeACI stage time with an MDE of "above the measured run-to-run noise floor". Paired runs, same machine, `taskset -c 0`, release, identical fixture/seed/options, 3 runs per side of 10 Criterion samples each. Baseline `4ec3bcd`, candidate `6cd9c9a`. Medians of the per-run medians (ms), chi=16/32/64/128/256: baseline `145.69/251.75/120.22/143.55/181.50`, candidate `64.11/99.19/50.39/66.99/98.13`, i.e. `2.27x/2.54x/2.39x/2.14x/1.85x` (`-56.0%/-60.6%/-58.1%/-53.3%/-45.9%`). The observed noise floor is the run-to-run median spread, at worst 6.1% (baseline chi=32) and typically 1-3%, so every improvement is an order of magnitude above it. The causal resource reduction is measured by a test-only counting allocator on a warm 16-site chain (the counter is new, so its "before" point is this branch with the typed API present but the assignment-allocation reductions not yet applied, which still carries the pre-change per-result wrapping and per-call assignment work): the `AnyScalar` route needed 3,939 heap blocks per 64-point call at that point and 1,739 after, while the typed route needs 138 -- 28.6x fewer than the pre-change wrapper and far below the audited minimum of 3,072 short-lived assignment vectors per 16-site/64-point call. A warm two-point Guard-shaped call fell from 230 to 138 blocks, and the per-call count is now the same for 2 and 64 points, i.e. no longer proportional to the batch size. No target-path time regression was observed: every arm of the paired microbenchmark and every bond dimension of the end-to-end benchmark improved. |
| `R10` release/regression | **PASS locally; the cross-repo downstream Guard stage was NOT RUN** | The changed-crate release matrices, integration targets, and doctests listed under `C10` all pass, and unchanged chain/simplett behaviour is visible in the end-to-end fixture: the SimpleTT arm's ranks, errors, sweeps, and evaluated points are byte-identical between baseline and candidate, as are TreeACI's (ranks 32/24/19/16/17, sweeps 4/5/2/2/2, errors 9.869e-9/8.622e-9/9.321e-9/9.794e-9/9.416e-9, evaluated points 71,972/66,226/29,452/34,756/37,752, frame records 60, frame retained bytes identical at every chi). Every new public item (`evaluate_batched_typed`, `CachedEvaluatorPlan` and its methods, `TreeTNCachedEvaluator::with_plan`/`plan`, `EvaluatedScalarKindMismatch`) has rustdoc stating shape, column-major layout, dtype policy, ownership, and error contract with a runnable asserted example; no `ignore`/`no_run` fence was added and no `unwrap`/`expect` was added to library code. **Limit:** the `../../gw-rs/sgw` downstream stage isolation was deliberately not run in this task, because it requires a clean cross-repo copy plus a full non-crate-scoped build that this task's build discipline excludes; the whole-workspace and downstream gates are owned by the integrating run. The Guard-enabled end-to-end parity benchmark above exercises the same `find_global_pivots` / `GuardOutputEvaluator` path with identical trajectories, but it is not a substitute for the SGW stage gate, which remains open for #709. |
| `N` numerical stability/convergence | **PASS** | All four scalar kinds are compared against the ordinary evaluator and against a dense oracle at the Guard boundary. The typed conversion is exact for a matching kind, and its only lossy cases (f64 to f32, Complex64 to Complex32, real widened to complex) are exactly the dynamic wrapper's, covered by the existing `TreeAciScalar` precision tests. The end-to-end fixture converges identically at every bond dimension with unchanged errors and ranks, and the parity benchmark's chain-comparison error bound is unchanged. No tolerance and no reduction order was touched. |
| `M` metamorphic semantics | **PASS** | Reordered, duplicated, partially cached, and empty batches, and the same batch answered cold, warm, and hinted, all agree between the typed and `AnyScalar` routes; duplicate columns produce identical results and the hinted and unhinted answers of the same batch are equal. The Guard-level test repeats this for a one-varying-site scan batch and a general five-point batch with a repeat. |
| `F` fallback parity | **PASS** | The `AnyScalar` wrapper is retained and still exercised by the entire pre-existing release matrix, including the f32/Complex32 generic centre route, the f64/Complex64 raw routes, the zero-budget and over-budget message caches, the no-hint vertex-centre route, and the unsupported-raw-shape fallbacks. Both routes were also compared on the same evaluator after a partial-hit batch, and the dtype refusal path was checked on both the TreeTN and TreeACI sides. |
| `I` invalidation/retention | **PASS** | `guard_output_evaluator_reuses_the_plan_when_output_tensors_change` replaces the Guard's output with a same-topology tree of a *different bond dimension* and different values, then rebuilds the output evaluator from the retained plan: the plan handle is asserted identical (`CachedEvaluatorPlan::is_same_as`), the new answers match a dense oracle of the new output, and they differ from the previous answers, so no numerical message survives a value change. Message caches remain keyed by directed component and assignment; the existing clear/reuse, zero-budget, bounded-budget, and subtree-invalidation tests remain green. |
| `D` determinism | **PASS** | Three end-to-end runs per side produced identical ranks, sweeps, errors, evaluated points, and frame counters at every bond dimension, with only wall time varying. The typed conversion is a per-element `match` with no accumulation, so it cannot reorder a reduction; the shared plan's memoized rooted plans are keyed by centre and built from the same sorted neighbour table as before. |
| `S` scaling law | **PASS for the scoped per-call resource law** | The warm per-call allocation count is now 138 for both a 2-point and a 64-point 16-site call, where the audited assignment work was proportional to nodes times points (at least 3,072 vectors at 16 sites and 64 points). The paired microbenchmark separates the same effect from bond dimension: the typed advantage on a warm hinted call is 1.76-2.01x at 2 points and 5.26-6.77x at 64 points across chi=16/64/256, i.e. it grows with batch size and not with chi, which is what a per-result wrapper cost predicts. This is a scoped result-boundary and per-call-overhead claim, not a claim about the evaluator's asymptotic contraction cost. |
| `P` provenance/observability | **PASS** | Every design and gate-policy statement here is labelled **[AI Supplied]**; no new paper claim is made and no source locator is repurposed. No new direct `tenferro-*` dependency was added, and TreeACI reaches TreeTN only through the public typed API, the public plan type, and the public error type. Every performance claim above has its command, configuration, baseline/candidate commit, and raw output recorded below. `tensor4all-benchmark` is N/A: it maintains SimpleTT/chain ACI cases only and cannot be used to claim a TreeACI improvement. |
| `CI` remote regression | **NOT RUN** | The #709 commit is local only; this task was instructed not to push and not to touch a PR, so the required GitHub checks have not been triggered. The gate stays open for the integrating run, which must also compare against the immediately preceding run's failure record. |

### #709 verification commands and raw measurement output

```text
cargo fmt --all -- --check
(clean)

cargo clippy --release -p tensor4all-treetn -p tensor4all-treeaci --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
Finished `release` profile [optimized] target(s)

cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
running 77 tests
test result: ok. 75 passed; 0 failed; 2 ignored; 0 measured; 448 filtered out

cargo test --release -p tensor4all-treetn --lib --no-fail-fast
test result: ok. 523 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out

cargo test --release -p tensor4all-treetn --tests --no-fail-fast
all 22 targets passed (523/3/61/6/0/0/18/12/20/1/35/2/27/2/28/3/20/18/1/11/2/2)

cargo test --doc --release -p tensor4all-treetn
test result: ok. 152 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

cargo test --release -p tensor4all-treeaci global_guard --no-fail-fast
running 23 tests
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 131 filtered out

cargo test --release -p tensor4all-treeaci --no-fail-fast
test result: ok. 148 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out   (lib)
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out     (tests/public_api.rs)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out     (tests/rank_scaling.rs)
test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out    (doctests)

cargo test --release -p tensor4all-aci --no-fail-fast
test result: ok. 85 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out    (lib)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out     (tests/elementwise.rs)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out     (tests/rank_scaling.rs)
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out    (doctests)

cargo test --release -p tensor4all-treetn --lib treetn::cached_evaluator::tests::typed_batch_and_warm -- --nocapture
#709 warm allocation counts: sites=16 points=64 wrapper=1739 typed=138 two_point_typed=138
test result: ok. 1 passed
(the same test, run earlier on this branch with the typed API present but
 before the assignment-allocation reductions -- i.e. with the pre-change
 per-result wrapping and per-call assignment work -- reported
 wrapper=3939 typed=2338 two_point_typed=230)
```

Paired result-boundary microbenchmark, candidate `6cd9c9a`, warm hinted call on
one evaluator per case, 10 samples, pinned to CPU 0:

```text
taskset -c 0 cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_typed_vs_any_scalar_results
any_scalar/chi16_p2      time:   [18.394 µs 18.737 µs 19.123 µs]
typed/chi16_p2           time:   [9.3009 µs 9.3170 µs 9.3480 µs]
any_scalar/chi16_p64     time:   [481.65 µs 499.78 µs 516.89 µs]
typed/chi16_p64          time:   [76.061 µs 76.217 µs 76.406 µs]
any_scalar/chi64_p2      time:   [23.016 µs 23.274 µs 23.579 µs]
typed/chi64_p2           time:   [11.929 µs 12.073 µs 12.216 µs]
any_scalar/chi64_p64     time:   [549.77 µs 573.12 µs 631.71 µs]
typed/chi64_p64          time:   [84.427 µs 84.646 µs 85.040 µs]
any_scalar/chi256_p2     time:   [28.055 µs 28.450 µs 28.771 µs]
typed/chi256_p2          time:   [15.980 µs 16.187 µs 16.379 µs]
any_scalar/chi256_p64    time:   [623.38 µs 632.07 µs 638.42 µs]
typed/chi256_p64         time:   [119.12 µs 120.07 µs 120.99 µs]
```

End-to-end Guard-enabled TreeACI stage, 16-site two-input chain, tree arm only,
identical options and seeds on both sides, `taskset -c 0`, three runs per side:

```text
T4A_TREEACI_PARITY_ENABLE_GUARD=1 taskset -c 0 cargo bench -p tensor4all-aci \
  --bench treeaci_parity -- "aci_vs_treeaci_chain_guard/tree"

baseline 4ec3bcd, per-run Criterion medians (ms)
chi=16    145.69   143.14   146.62
chi=32    259.51   244.48   251.75
chi=64    122.60   119.30   120.22
chi=128   143.58   139.85   143.55
chi=256   181.83   177.89   181.50

candidate 6cd9c9a, per-run Criterion medians (ms)
chi=16     64.212   63.186   64.113
chi=32     99.893   98.619   99.185
chi=64     50.803   50.385   50.297
chi=128    66.994   65.758   67.612
chi=256    96.015  101.54    98.126

median-of-medians ratio (baseline / candidate)
chi=16  2.27x   chi=32  2.54x   chi=64  2.39x   chi=128  2.14x   chi=256  1.85x

identical trajectory on both sides at every chi (candidate lines shown; the
baseline lines are byte-identical apart from wall time)
chi=16   tree: rank 32 err 9.869e-9 sweeps 4 (Converged)  evaluated_points 71972
chi=32   tree: rank 24 err 8.622e-9 sweeps 5 (Converged)  evaluated_points 66226
chi=64   tree: rank 19 err 9.321e-9 sweeps 2 (Converged)  evaluated_points 29452
chi=128  tree: rank 16 err 9.794e-9 sweeps 2 (Converged)  evaluated_points 34756
chi=256  tree: rank 17 err 9.416e-9 sweeps 2 (Converged)  evaluated_points 37752
```

The audit's own Guard-enabled 16-site reference point at chi=256 was 374.37 ms
per run with rank 17, two sweeps, and 37,752 evaluated points. The candidate
reaches the same rank, sweep count, and point count in 98.13 ms. Only the
`181.50 ms` baseline measured here is a paired #709 comparison; the 374.37 ms
figure predates Tasks 6-9 and is quoted only to place this result in the
audit's own trajectory.

Two items are deliberately left open rather than claimed. The `../../gw-rs/sgw`
downstream Guard stage gate and the remote `CI` gate were not run in this task
for the reasons recorded in the ledger. One smaller allocation source also
remains: each component batch still clones its `point_to_assignment` vector out
of the assignment batch (two clones per call in the chain fixture), which the
counting allocator includes in the 138 blocks; removing it needs an ownership
change in the component-batch boundary rather than a scratch buffer, so it was
left for a follow-up instead of being bundled into this closure.


## 2026-09-04 #718 predeclared protocol

This section is written **before** any #718 timing is collected, so that the
primary metric, the minimum detectable effect, and the pass rule of each lane
cannot be chosen after seeing a result. It is committed together with the
benchmark and complexity-test code, ahead of the measurement commit; `git log`
on this file is the record of that ordering. Every policy statement here is
**[AI Supplied]** evidence policy, not a literature claim.

### Terminology fixed for this task

`z` is a node's **tree coordination number**: its number of incident bonds.
The candidate-frame kernel is selected by the number of **incoming
components** of a directed edge, which is `z - 1` (`incoming_to_from` in
`crates/tensor4all-treeaci/src/problem.rs`), because an outward arc excludes
its own target. Therefore:

| `z` | incoming components | candidate-frame route |
|---|---|---|
| 2 | 1 | single-incoming kernel (chain interior) |
| 3 | 2 | exactly-two-incoming kernel (Y junction, comb branch point) |
| 4 | 3 | #713 arbitrary-degree kernel |
| 5 | 4 | #713 arbitrary-degree kernel |

Every number reported below states which of the two conventions it uses. The
#713 route is first reached at `z >= 4`; a coordination-3 branch point does
**not** reach it.

### Lanes, primary metrics, and MDEs

| lane | primary metric | MDE / pass rule |
|---|---|---|
| `L1` reported slow workload | Criterion median of `aci_vs_treeaci_chain_guard/tree/256` and its ratio to the `train/256` arm in the same run | The ratio must be resolved more tightly than the observed run-to-run spread of per-run medians over three whole-binary repetitions. The recorded #699 reference is `2.47x` slower with the default Guard and `0.88x` with it disabled; the lane passes only if the current ratio is measured, paired in the same run, and stated against that reference. No target-path regression versus the reference is allowed. |
| `L2` complexity laws | exact deterministic counters | Equality, not a threshold: raw-centre visits `= d * product(chi_e)`; warm edge-cut assembly visits `= points * chi_edge` and independent of every descendant bond; arbitrary-degree candidate core reads `= outgoing_dim * product(chi_k)` for the batched route and `= candidates * outgoing_dim * product(chi_k)` for the scalar route. A single failing size fails the lane. |
| `L3` independent scaling fixtures | Criterion median per swept point, one dimension varied at a time | Reported as observed growth with the noise floor stated. This lane establishes the gate, so it makes **no** speedup claim; a claim would need a baseline/candidate pair on the same fixture. |
| `L4` #709 downstream Guard stage (carry-over) | isolated `pi_rtau` / `sigma_rtau` stage wall time from identical checkpoint inputs, baseline `4713ba8` versus candidate `a7632cc`, three repetitions per side, `taskset -c 0` | Complete output parity **first**: identical convergence flag, sweep count, evaluated points, and maximum error, plus identical hub/reference node identity, coordination number, and bond dimensions in the stage diagnostics JSON. Only then are paired medians compared, and only a difference above the observed run-to-run spread is reported as a difference. |
| `L5` #713 downstream applicability | structural verification of the downstream topology, not a timing | Pass means the maximum coordination number actually used downstream is read out of the downstream source at its clean `HEAD` and shown to be `< 4`, i.e. the #713 route is unreachable there. Otherwise the gate must be run, not recorded `N/A`. |

### Fixed measurement configuration

- `taskset -c 0`; `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `TENFERRO_NUM_THREADS=1`.
- Release/bench profile; ten Criterion samples per case; `0.5 s` warm-up and a
  `2 s` measurement window in the new scaling benchmark; the existing parity
  benchmark keeps its ten-sample configuration.
- Statistic: Criterion's median with its 95% bootstrap confidence interval.
- Noise floor: the spread of per-run medians across at least three repetitions
  of the whole benchmark binary. This is the number every wall-clock claim is
  compared against; nothing below it is reported as an effect.
- Fixtures are analytic and seed-free apart from one recorded constant, so
  repeated runs build bit-identical inputs.
- Correctness precedes timing in every harness: each case checks one dense
  materialization against one dense reference and aborts on a residual above
  its bound before its timed loop starts.

### Observability limits recorded in advance

Two counters #718 lists are not reachable from a benchmark through the public
surface, and are therefore gated by crate-internal counter tests rather than
invented at the benchmark level:

- the cached evaluator's message-cache hit/miss/eviction counts
  (`CachedEvaluationStats` is private; `TreeAciDiagnostics` exposes retained
  records and logical retained bytes, not hit rates);
- a process peak-byte figure. All byte numbers in this task are the documented
  logical payload accounting, never allocator or RSS measurements.

## 2026-09-04 #718 closure: scaling and end-to-end gates

This entry closes #718, the last subissue of the #699 umbrella. It records the
gate machinery added in commit `5d4bdee`, the measurements taken against the
protocol predeclared in the section above, the two carry-over downstream gates
inherited from #713 and #709, and the `R12` final report covering every
task-level gate of Tasks 1--12.

Every design, gate-policy, and measurement-interpretation statement here is
**[AI Supplied]** engineering evidence. No new literature claim is made and no
existing source locator is repurposed: the full-text ACI/TCI/tenferro clones,
hashes, and page/equation/pseudocode locators recorded in the `ACI-C*`,
`TCI-C*`, and `TEN-C*` register above remain the algorithmic authority.

### #718 changes

Commit `5d4bdee` changes **no production code at all**. Every insertion in a
`src/` file is inside `#[cfg(test)]` (a counter, or a test), and the remaining
files are `benches/`, one `Cargo.toml` target entry, and this worklog.
`git diff a7632cc..5d4bdee` covers:

- `crates/tensor4all-treeaci/benches/aci_scaling.rs` (new) plus its
  `[[bench]]` entry and a `criterion` dev-dependency. It varies chain length,
  input bond dimension, output rank cap, hub coordination number, unequal
  incident bonds at a fixed bond product, and evaluator batch size **one
  dimension at a time**, and covers chain, exactly-two-incoming, three-or-more
  incoming, cold evaluator, warm evaluator, and working-budget cases. Each case
  materializes the interpolated tree once, compares it against one dense
  elementwise reference, and prints residual, sweeps, maximum output rank,
  evaluated points, global pivots, saturated edges, and retained frame/sample/
  candidate accounting before its timed loop starts.
- `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`:
  `warm_edge_cut_assembly_work_is_points_times_edge_bond_only`, which sweeps
  the point count, the cut bond dimension, and the descendant bond dimensions
  independently at 1x/2x/4x and asserts the #708 law exactly.
- `crates/tensor4all-treeaci/src/frames.rs` and `src/frames/tests/mod.rs`: a
  test-only `debug_stats::core_element_reads` counter, recorded once per gather
  and once per scalar contraction call from the shape that call is about to
  walk (so the inner reduction loops keep their exact pre-instrumentation form
  and the #713 timing measurement is not perturbed), plus
  `candidate_product_accounting_separates_the_batched_and_scalar_exponents`.
- `crates/tensor4all-aci/benches/treeaci_parity.rs`: the preserved
  simplett-versus-TreeACI comparison now also reports callback counts for both
  arms, sample-arena and candidate accounting, global pivots, saturated edges,
  the configured budgets, retained-byte compliance assertions, and the
  reproducibility metadata header.

### Terminology used in every number below

`z` is the tree coordination number (incident bonds); the candidate-frame
kernel is chosen by the incoming-component count `z - 1`. `z = 3` (Y junction,
comb branch point) has two incoming components and uses the exactly-two-incoming
kernel; the #713 arbitrary-degree route is first reached at `z = 4`.

### Measurement environment

`AMD Ryzen 9 6900HX with Radeon Graphics`, 8 cores / 16 threads, 19 GiB RAM,
`Linux 6.18.33.2-microsoft-standard-WSL2`. All timings pinned with
`taskset -c 0` and `RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
TENFERRO_NUM_THREADS=1`, release/bench profile, three whole-binary repetitions
per benchmark, Criterion median with a 95% bootstrap confidence interval, ten
samples per case. Noise floor is reported per case as the run-to-run spread of
per-run medians.

### `L2` deterministic complexity gates (counters, not wall clock)

All three laws hold exactly, at every measured size:

| law | gate | measured |
|---|---|---|
| raw-centre topology work `d * product(chi_e)` | `raw_center_work_scales_with_coordination_number_and_actual_bond_dimensions` (kept, unchanged) | `z=2` at `chi=64/128/256`: `2*chi^2`, ratios exactly `4`; `z=3`: `2*chi^3`, ratios exactly `8`; unequal `[64,128,256]` with `d=3`: exactly `3*64*128*256 = 6291456`, not `max(chi)^z` |
| #708 warm edge-cut assembly `points * chi_edge` | `warm_edge_cut_assembly_work_is_points_times_edge_bond_only` (new) | points `2/4/8` at `chi_edge=4`: `8/16/32` visits, exactly linear; `chi_edge` `2/4/8` at `4` points: `8/16/32`, exactly linear; descendant bonds `2/4/8` at fixed `points=4, chi_edge=4`: `16/16/16`, i.e. **no** dependence on descendant size. Each measurement also asserts `message_cache_misses == 0`, so it really is the warm route |
| #713 candidate-product accounting | `candidate_product_accounting_separates_the_batched_and_scalar_exponents` (new) | at `chi=[3,3,3]`, `outgoing_dim=4`, candidates per component `m = 2/4/8` (candidate product `8/64/512`, i.e. 1x/8x/64x, with `outgoing_dim = 4` and `product(chi_k) = 27`): batched core reads `108/108/108` = `outgoing_dim * product(chi_k)` exactly, **constant** in the candidate product; scalar core reads `864/6912/55296` = `candidates * outgoing_dim * product(chi_k)` exactly, growing `1x/8x/64x`; packed full cross `32/256/2048` values, growing `1x/8x/64x`. Both routes are compared against each other with a whole-result residual `<= 1e-12 * scale` before any count is accepted |

The third row is the counter-based statement of the #713 claim that the earlier
closure could only make with a 16x--313x timing ratio: it is the candidate-count
exponent, not a constant factor, that the batched route removes, and the
remaining `outgoing_dim * product(n_k)` full cross is pinned so a future
lazy/block formulation has a number to beat.

### `L1` reported slow workload: the bond-256 default-Guard chain gate

The #699 reference for this lane is "TreeACI is `0.88x` the simplett ACI time
with Guard disabled but `2.47x` with the default Guard" on the deterministic
input-bond-256 16-site chain. Both arms were re-measured in the same binary,
three whole-binary repetitions each.

Default Guard (`T4A_TREEACI_PARITY_ENABLE_GUARD=1`, group
`aci_vs_treeaci_chain_guard`), medians of per-run medians in ms:

| chi | simplett | spread | TreeACI | spread | tree/train | simplett points | tree points |
|---|---|---|---|---|---|---|---|
| 16 | 45.894 | 0.81% | 63.970 | 1.52% | **1.394** | 100472 | 71972 |
| 32 | 20.840 | 0.58% | 100.050 | 1.70% | **4.801** | 30300 | 66226 |
| 64 | 126.010 | 1.04% | 50.942 | 3.00% | **0.404** | 78816 | 29452 |
| 128 | 73.619 | 1.66% | 69.115 | 9.34% | **0.939** | 41536 | 34756 |
| 256 | 117.540 | 1.56% | 97.405 | 2.95% | **0.829** | 47904 | 37752 |

Guard disabled (group `aci_vs_treeaci_chain`):

| chi | simplett | spread | TreeACI | spread | tree/train | simplett points | tree points |
|---|---|---|---|---|---|---|---|
| 16 | 13.629 | 4.53% | 8.320 | 3.74% | **0.610** | 48584 | 29988 |
| 32 | 12.248 | 3.71% | 9.396 | 3.89% | **0.767** | 29096 | 28312 |
| 64 | 20.526 | 2.36% | 11.596 | 3.97% | **0.565** | 37660 | 28312 |
| 128 | 23.897 | 8.08% | 16.525 | 7.66% | **0.692** | 40396 | 33584 |
| 256 | 47.775 | 3.17% | 24.329 | 9.28% | **0.509** | 46732 | 36516 |

At the reference point, bond 256: **`2.47x` slower with the default Guard has
become `0.829x`, and `0.88x` without the Guard has become `0.509x`.** Both are
far outside the observed noise floor (worst per-case run-to-run spread `9.34%`).
Accuracy is unchanged and matched: at `chi=256` the relative dense residual is
`1.442e-8` for simplett and `1.343e-8` for TreeACI against the exact product,
both arms reach rank 17 in 2 sweeps, and TreeACI still evaluates fewer operator
points (37,752 versus 47,904).

Two honest caveats on this table.

1. With the Guard enabled the two implementations no longer follow the same
   trajectory: at `chi=32` TreeACI takes 5 sweeps and 66,226 points against
   simplett's 2 sweeps and 30,300 points, which is why its `4.801` ratio there
   is a *different amount of work*, not a slower implementation; at `chi=64`
   the asymmetry runs the other way (simplett 6 sweeps / 78,816 points versus
   TreeACI 2 sweeps / 29,452 points, ratio `0.404`). The `chi=128` and
   `chi=256` rows are the ones where both arms take the same 2 sweeps at
   comparable rank, and those are the rows the reference point lives on.
2. This is a workload-level comparison of two implementations, not a
   baseline/candidate pair of one implementation, so it states where TreeACI
   now stands against the recorded reference. It is not used to attribute the
   change to any single subissue.

### `L1` reported slow workload: the audit's non-finishing phase profile

`Measurements and limitations` item 2 above records that "the high-rank chain
phase profile ... did not finish a single sweep within approximately 2.5
minutes after compilation and was interrupted. No number from that run is
used." The surviving fixture for that lane is
`state::tests::profile_high_rank_chain_phases_and_candidate_cache`, a 32-site
`d=2` two-input chain at `chi = 64/128/256` (the audit text says 16 sites; the
fixture in the tree is 32, so the comparison below is a conservative lower
bound on the improvement). It now completes both sweeps at every bond
dimension:

```text
Guard disabled (the test's default)
chi=64   initialize=22.951 ms  sweeps=18.897 ms   completed=2  candidate hits/misses 1160/7388
chi=128  initialize=47.393 ms  sweeps=32.774 ms   completed=2  candidate hits/misses 1092/12072
chi=256  initialize=232.399 ms sweeps=151.460 ms  completed=2  candidate hits/misses 1004/20580
whole test: 0.60 s

Guard enabled (T4A_TREEACI_ENABLE_PROFILE_GUARD=1)
chi=64   initialize=23.170 ms  sweeps=313.861 ms   guard search=264.850 ms
chi=128  initialize=45.254 ms  sweeps=507.381 ms   guard search=432.892 ms
chi=256  initialize=215.336 ms sweeps=1.498733 s   guard search=1.231894 s
whole test: 2.70 s
```

The audit's `chi=128` datum was ">= 150 s for less than one sweep"; the same
fixture now completes two sweeps in `80.2 ms` without the Guard and `552.6 ms`
with it. **The reported slow workload reproduces as a fixture and is no longer
slow.** This lane is therefore closed as reproduced-and-resolved rather than
inconclusive.

One residual is visible and is recorded rather than hidden: with the Guard on,
the guard's random-start search still dominates this 32-site fixture --
`1.231894 s` of the `1.498733 s` sweep total at `chi=256`, of which
`1.147752 s` is input evaluation over 2,218 calls returning 8,844 scalars. That
is not a regression (#709's typed path is what makes those calls cheap enough
to finish at all), but it is the next thing to look at on this fixture, and it
is not covered by any #699 subissue. See **Open items** below.

### `L3` independent scaling fixtures

Criterion medians of per-run medians (ms), three whole-binary repetitions,
`taskset -c 0`. Every case's dense residual and diagnostics are printed in the
raw log and reproduced in part here.

**Chain length** at fixed `chi=16`, no rank cap (1x/2x/4x in `N`):

| N | median (ms) | spread | evaluated points | max output rank | sweeps | residual/scale |
|---|---|---|---|---|---|---|
| 4 | 0.645 | 0.66% | 96 | 4 | 2 | 2.220e-16 / 9.969e-1 |
| 8 | 7.806 | 0.90% | 2260 | 15 | 2 | 5.371e-9 / 1.000e0 |
| 16 | 64.857 | 2.45% | 51902 | 29 | 3 | 1.158e-8 / 8.923e-1 |

Growth is `x12.1` then `x8.3`, tracking the evaluated-point count
(`x23.5` then `x23.0`) rather than `N` itself: the cost of doubling a chain is
the output rank it unlocks, and time *per evaluated point* actually falls
(`6.7 -> 3.45 -> 1.25 us`).

**Input bond dimension** at fixed `N=12`, no rank cap:

| chi | median (ms) | spread | evaluated points | max output rank | frame bytes |
|---|---|---|---|---|---|
| 8 | 25.157 | 2.53% | 16182 | 22 | 162272 |
| 16 | 20.620 | 4.35% | 14092 | 26 | 283168 |
| 32 | 21.173 | 1.16% | 17628 | 27 | 468160 |

Flat in `chi` to within noise while retained frame bytes grow `1.7x` per
doubling. This is the useful negative result of the sweep: on this fixture the
input bond dimension buys memory, not time -- the time is set by the output
rank and the evaluated-point count. (The `chi=8` row runs one extra sweep and
five global pivots, which is why it is the slowest.)

**Active output rank** at fixed `N=12`, `chi=16`, cap `4/8/16`:

| cap | median (ms) | spread | evaluated points | residual | termination |
|---|---|---|---|---|---|
| 4 | 3.801 | 2.03% | 1088 | 2.308e-4 | RankLimited |
| 8 | 4.329 | 2.55% | 3392 | 1.460e-5 | RankLimited |
| 16 | 5.175 | 1.26% | 9536 | 3.255e-7 | RankLimited |

Each rank doubling costs about `1.15x` time and `2.8x` evaluated points, and
buys between one and two orders of magnitude of accuracy. The observed rank
equals the cap in every row, so the cap really is the control variable.

**Coordination number** at a fixed 13 sites, fixed hub bond 4, arms rearranged
so only `z` varies:

| z | incoming | arm length | median (ms) | spread | evaluated points | max rank | sweeps |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 6 | 66.332 | 3.15% | 28598 | 14 | 7 |
| 3 | 2 | 4 | 39.820 | 6.82% | 60562 | 14 | 4 |
| 4 | 3 | 3 | 9.613 | 0.46% | 38496 | 8 | 2 |
| 6 | 5 | 2 | 69.727 | 2.48% | 98496 | 4 | 2 |

Time is *not* monotone in `z` at fixed site count, because rearranging the same
13 sites changes the achievable rank (14, 14, 8, 4) and the sweep count
(7, 4, 2, 2) as well as the coordination. Evaluated points do grow with `z`
(`28.6k -> 98.5k`), which is the topology-required part. The `z >= 4` rows are
the ones that exercise the #713 arbitrary-degree route end to end, and they are
the cheapest per evaluated point (`0.25 us` at `z=4` versus `2.32 us` at
`z=2`), so the route is not a cliff at this scale.

**Unequal incident bonds** at coordination 4, 13 sites, hub bond product 256 in
all three layouts -- a new finding:

| hub bonds | median (ms) | spread | evaluated points | us / evaluated point | max rank | residual |
|---|---|---|---|---|---|---|
| `4,4,4,4` | 9.619 | 0.39% | 38496 | 0.25 | 8 | 3.331e-16 |
| `2,4,8,4` | 188.550 | 5.40% | 56784 | 3.32 | 8 | 7.772e-16 |
| `8,4,2,4` | 136.100 | 11.45% | 22446 | 6.06 | 8 | 7.216e-16 |

All three reach the same maximum output rank 8 with an exact residual, and all
three have the same topology-required hub work: on every outward arc of the hub
`outgoing_dim * product(incoming bonds)` equals the bond product 256,
independently of how the 256 is split. The `8,4,2,4` layout nevertheless
evaluates **fewer** points than the equal layout (22,446 versus 38,496) and
still takes `14.2x` the wall time, i.e. `24x` more time per evaluated point.
This is reproducible across all three repetitions (`146.59/136.10/131.00` ms
versus `9.62/9.64/9.60` ms) and is not a convergence-trajectory artefact.

**This is a new, previously unmeasured performance asymmetry**: at equal bond
product, an unequal incident-bond layout costs an order of magnitude more per
unit of required work than an equal one. It is exactly the kind of defect a
single fixed-size fixture cannot see, and it is what this gate was built to
find. #718 does not diagnose it -- diagnosis needs a profile of the routing,
cache, and factorization paths under an unequal layout -- so it is recorded as
an open follow-up rather than explained here. See **Open items**.

**Evaluator batch size**, 16-site chain at `chi=32`, centre site 8, typed
batch evaluation:

| points | cold (ms) | spread | warm (ms) | spread | cold/warm |
|---|---|---|---|---|---|
| 4 | 0.170 | 2.85% | 0.020 | 2.58% | 8.5x |
| 16 | 0.193 | 1.55% | 0.032 | 1.18% | 6.0x |
| 64 | 0.288 | 1.55% | 0.086 | 1.05% | 3.3x |

Cold cost is dominated by fixed setup (`1.7x` for `16x` the points); warm cost
scales with the batch (`4.3x` for `16x` the points), i.e. sub-linear and
consistent with the warm path paying per point and not per node. Every cold and
warm answer is checked against `TreeTN::evaluate` first: residual `1.665e-16`
at every size.

**Working-memory limit**, coordination-4 spider, hub bonds `8,8,8,8`:

```text
ladder: 512 MiB ok | 16 MiB ok | 4 MiB ok | 1 MiB ok | 512 KiB ok
        256 KiB refused: resource="working bytes" requested=429568 limit=262144
1-byte budget refused: resource="working bytes" requested=64 limit=1
generous 512 MiB   21.526 ms (spread 7.46%)  rank 8  66176 points  residual 4.441e-16
tight    512 KiB   21.340 ms (spread 3.71%)  rank 8  66176 points  residual 4.441e-16
```

A budget 1024x smaller changes neither the answer, the rank, the evaluated-point
count nor the time beyond noise, and the first rung below the prepared minimum
is refused with the exact requested/limit pair rather than silently exceeded.
The tight rung is found by a descending ladder at run time rather than
hard-coded, because the charge depends on the candidate counts the run itself
discovers.

### `L4` #709 carry-over: the `../../gw-rs/sgw` downstream Guard stage

This gate was recorded **open, not waived** in the #709 closure ("the
`../../gw-rs/sgw` downstream stage isolation was deliberately not run in this
task"). It is run here.

Method. `git -C /root/projects/gw-rs/sgw archive HEAD` (`ba6fbf3e...`) was
extracted into two clean temporary copies, each given a non-committed
`[patch."https://github.com/tensor4all/tensor4all-rs.git"]` section pointing at
a `git archive` of one tensor4all-rs revision: baseline `4713ba8` (the #713
closure, i.e. #709 absent) and candidate `a7632cc` (the #709 closure). Both
sides therefore contain #713 and differ only by #709. The dirty
`/root/projects/gw-rs/sgw` checkout was never modified: its `HEAD` is still
`ba6fbf3e...`, its 59 pre-existing dirty paths are unchanged, no stash was
created, and no run directory was added to it.

Full workflow, candidate side, `SGW_ACI_GLOBAL_GUARD=1`, `G0_THREADS=1`:

```text
T=1.0  SGW_RUN_TAG=aci-gate-718     simplett real 19.78 s | treeaci real 30.97 s | cttn real 11.18 s
T=0.1  SGW_RUN_TAG=gate718t01       simplett real 29.76 s | treeaci real 26.23 s | cttn real 22.60 s
```

Both completed every configured stage: SimpleTT, TreeACI, and CTTN pipelines,
all four checkpoint stages, five row slices, assembly, and plotting. The
`T=1.0` wall times were taken while the baseline copy was still compiling and
are reported only as completion evidence, not as timings; the `T=0.1` numbers
are from a quiet machine. At `T=0.1` the nblock TreeACI arm is `0.88x` the
SimpleTT arm and the comb CTTN arm `0.76x`, end to end.

Paired isolated stage measurement. Both sides read the **same** candidate-produced
checkpoints, three repetitions per side, `taskset -c 0`, whole-binary wall time
from `/usr/bin/time` plus the stage's own reported ACI time.

Parity first, at `T=1.0`:

| stage | sweeps | max bond | final err | termination | global pivots | evaluated points (base / cand) |
|---|---|---|---|---|---|---|
| `pi_rtau` | 7 / 7 | 38 / 38 | 9.775e-5 / 9.775e-5 | Converged / Converged | `[3,1,2,0,1,0,0]` both | 177526--177586 / 177466--177766 |
| `sigma_rtau` | 5 / 5 | 39 / 39 | 9.304e-5 / 9.304e-5 | Converged / Converged | `[4,3,2,0,0]` both | 99494--99554 / 99554--99614 |

and at `T=0.1`:

| stage | sweeps | max bond | final err | termination | global pivots | evaluated points (base / cand) |
|---|---|---|---|---|---|---|
| `pi_rtau` | 7 / 7 | 169 / 169 | 9.941e-5 / 9.941e-5 | Converged / Converged | `[4,4,3,1,1,0,0]` both | 2152690--2152990 / 2152750--2152810 |
| `sigma_rtau` | 7 / 7 | 123 / 123 | 9.976e-5 / 9.976e-5 | Converged / Converged | `[2,2,2,2,0,0,0]` both | 1209430--1209550 / 1209430--1209670 |

The stage diagnostics JSON also agrees exactly on both sides at both
temperatures: `hub=0:9 (z=2, bond_dims=[4,4])` versus
`chain=0:0 (bond_dims=[6,9])`. The evaluated-point counts differ by at most
`0.17%` **within** each side as well as between sides, because the Guard's
random-start search is not seeded deterministically in this harness; the sweep
count, rank, error, termination, and pivot-per-sweep vector are bit-identical.

Then paired medians:

| stage | T | baseline wall (s) | candidate wall (s) | delta | baseline stage (s) | candidate stage (s) | delta | maxrss base / cand |
|---|---|---|---|---|---|---|---|---|
| `pi_rtau` | 1.0 | 1.44 `[1.41,1.44,1.44]` | 1.14 `[1.14,1.14,1.14]` | **-20.8%** | 1.0 | 0.7 | -30% | 84.9 / 84.6 MiB |
| `sigma_rtau` | 1.0 | 1.33 `[1.32,1.33,1.34]` | 1.15 `[1.11,1.15,1.15]` | **-13.5%** | 0.7 | 0.5 | -29% | 81.0 / 80.9 MiB |
| `pi_rtau` | 0.1 | 3.08 `[3.11,3.05,3.08]` | 2.80 `[2.84,2.80,2.70]` | **-9.1%** | 2.0 | 1.6 | -20% | 125.2 / 124.8 MiB |
| `sigma_rtau` | 0.1 | 2.81 `[2.81,2.81,2.82]` | 2.46 `[2.43,2.47,2.46]` | **-12.5%** | 1.7 | 1.3 | -24% | 125.0 / 125.0 MiB |

Observed noise floor: the within-side per-repetition wall spread is at most
`5.0%` (candidate `pi_rtau` at `T=0.1`) and typically `0--2%`. Every delta is
above it, and the stage-only deltas (`20--30%`) are several times above it. Peak
resident memory is unchanged, so the improvement is time, not a memory trade.

**`R10` for #709 is now PASS.** The gate is closed with parity first, then
paired medians, at two temperatures, on the real production stage.

### `L5` #713 carry-over: downstream applicability, verified not assumed

The #713 closure recorded its downstream gate as `N/A` on the reasoning that
SGW never reaches coordination `>= 4`. That reasoning was re-derived here from
the downstream source at its clean `HEAD` `ba6fbf3e...`, not taken on trust:

- `run_r10_nblock_treeaci_ab.sh:run_one` sets `SGW_LAYOUT=nblock`.
  `src/topology.rs` builds `BuiltinTopology::NBlock` as `nblock_order(...)`
  followed by `order.windows(2)`, i.e. a plain path. Maximum coordination 2,
  so at most **one** incoming component.
- `run_r10_nblock_treeaci_ab.sh:run_cttn` sets `SGW_TOPOLOGY=comb`.
  `src/topology.rs` builds `BuiltinTopology::Comb` as `(0,1)`, `(1,2)`, plus
  `(start + 3*(bit-1), start + 3*bit)` for `start in 0..3` and `bit in 1..r`.
  For any `r`, node 1 carries `(0,1)`, `(1,2)` and `(1,4)` -- degree 3 -- and
  every other node has degree at most 2. Maximum coordination 3, so at most
  **two** incoming components.
- `tests/fixtures/branching_topology.json` has edges
  `[[0,1],[1,2],[1,3],[3,4]]`: degrees `1,3,1,2,1`, maximum coordination 3.

The #713 route needs coordination `>= 4`. It is unreachable in both downstream
arms and in the downstream branching fixture, for any `r`. **`N/A` is
confirmed, with the source evidence, rather than assumed.**

### Sibling `tensor4all-benchmark`: not applicable, with evidence

Checkout `/root/projects/tensor4all-rust/tensor4all-benchmark`, `HEAD`
`f1b139c0...` with 9 pre-existing dirty paths (untouched by this task). Its
maintained binaries are `elementwise_fourier.rs`,
`elementwise_gauss2d_patched.rs`, and `mpo_mpo_aniso_patched.rs`;
`grep -rn "treeaci\|TreeAci" src/ Cargo.toml scripts/` returns nothing and its
manifest has no `tensor4all-treeaci` dependency at all. Independently, this
commit changes no production code, so there is no shared seam that could
regress. The official single-core profile was therefore **not run** and this
gate is **not applicable**, not a waived failure. No SimpleTT/chain ACI run
from that checkout is used anywhere above as a TreeACI efficiency signal.

### #718 gate ledger

| gate | result | evidence and limit |
|---|---|---|
| `C12` correctness | **PASS** | Deterministic work-count gates are asserted independently of timing and all pass exactly (`L2` table). Complete-ACI parity against simplett covers all five bond dimensions with and without the default Guard, comparing sweeps, maximum output rank, last pivot error, callbacks, evaluated points, retained frame/sample/candidate accounting, and the dense residual against the exact elementwise product; at `chi=256` the relative residuals are `1.442e-8` (simplett) and `1.343e-8` (TreeACI). Every scaling case checks one dense materialization against one dense reference before its timed loop and aborts otherwise. Release matrices: `tensor4all-treetn` 524 lib tests passed / 2 pre-existing ignored; `tensor4all-treeaci` 159 lib / 7 pre-existing ignored, 1 `branch_degree`, 7 `public_api`, 1 `rank_scaling`, 18 doctests. No tolerance was relaxed anywhere. **Limit:** the four scalar kinds are not re-swept here; that matrix is #717's and remains green. |
| `E12` efficiency | **PASS** | Every claim above carries a median, a run-to-run noise floor, and a named primary metric fixed in the predeclared protocol before measuring. `L1`: bond-256 default-Guard ratio `2.47x -> 0.829x`, no-Guard `0.88x -> 0.509x`, worst per-case spread `9.34%`. `L4`: baseline/candidate paired medians at two temperatures, deltas `9.1--20.8%` whole-binary and `20--30%` stage-only against a `<= 5.0%` noise floor, with identical outputs. `L2`: exact counter equalities, no threshold involved. No issue is closed on a single favourable run: three whole-binary repetitions per benchmark, three repetitions per side downstream. **Limit:** `L3` reports observed growth only; it establishes the gate and deliberately makes no speedup claim, because a speedup claim would need a baseline/candidate pair on the same new fixture. |
| `R12` regression/integration | **PASS with three named exceptions** | Changed-crate release matrices pass (above). `cargo fmt --all -- --check` clean. `cargo clippy --release -p tensor4all-treeaci -p tensor4all-treetn -p tensor4all-aci --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` clean. `python3 scripts/repository-rules-review.py --base main --worktree --dry-run` pass. The affected downstream isolated stages and the full `run_r10_nblock_treeaci_ab.sh` workflow were run at `T=1.0` and `T=0.1` (`L4`). The sibling benchmark is explicitly not applicable, with evidence. **Exception 1:** the workspace-wide pre-PR gate (`cargo clippy --workspace`, `cargo nextest run --cargo-profile ci --workspace`, `cargo test --doc --profile ci --workspace`, `cargo doc --workspace`) was **not run** in this task and is **pending**; it is owned by the integrating run. **Exception 2:** the remote `CI` gate is **not run** because this branch is deliberately not pushed. **Exception 3:** `python3 scripts/check-public-error-docs.py` **fails**, with three findings that all reproduce on a clean `git archive a7632cc` tree and are therefore inherited from #712 and #709 rather than caused by #718; `audit-library-panics.py` and `check-crate-boundaries.py` pass. This will fail the CI `Maintenance scripts` job and must be fixed before the branch is pushed (Open items item 6). |
| `N` numerical stability | **PASS** | Every parity and scaling case asserts a dense whole-result residual against an exact reference before its timing is accepted; the rank-capped sweep reports a genuine truncation trajectory (`2.308e-4 -> 1.460e-5 -> 3.255e-7`) rather than an accidental exact case; the downstream stages converge to their configured `1e-4` tolerance with identical maximum bond and final error on both sides. No tolerance was relaxed. |
| `M` metamorphic semantics | **PASS** | The unequal-bond group is three permutations of one bond multiset at a fixed product; the coordination group is four rearrangements of one fixed 13-site budget; the counter gates vary points, cut bond, and descendant bond independently and check that only the first two appear in the law. Batch reorder/duplicate/partial-hit equivalence is covered by the existing #708/#709/#710 matrices, which remain green. |
| `F` fallback parity | **PASS** | The working-budget ladder drives the same fixture from a 512 MiB budget down to the smallest feasible 512 KiB rung and asserts identical rank, evaluated points, and residual; the next rung down and a 1-byte budget are both refused with the exact `ResourceLimit` requested/limit pair. Cold and warm evaluator routes are compared against `TreeTN::evaluate` at every batch size. |
| `I` invalidation/retention | **PASS** | Every parity case asserts `frame_retained_bytes <= max_frame_bytes` and `sample_arena_retained_bytes <= max_sample_arena_bytes` against the configured budgets, and prints retained frame records, sample records, candidate entries, and saturated edges. Retained frame bytes grow smoothly with bond dimension (`162 KiB -> 283 KiB -> 468 KiB` over `chi = 8/16/32`) with no plateau failure. **Limit:** these are the documented logical payload figures, not allocator or RSS measurements; the only true peak-memory numbers in this entry are the downstream `maxrss` values. |
| `D` determinism | **PASS** | The scaling fixtures are analytic and seed-free apart from one recorded constant, so repeated runs build bit-identical inputs; the three repetitions of each benchmark reproduce the same ordering and the same diagnostics, with only wall time varying. The counter gates are exact-equality assertions. **Limit:** the downstream Guard's random-start search is not seeded deterministically in the SGW harness, so evaluated points vary by up to `0.17%` between repetitions on *both* sides; every other stage output is bit-identical, and this is stated rather than smoothed over. |
| `S` scaling law | **PASS** | Three exact counter laws at 1x/2x/4x or 1x/8x/64x (`L2`), plus 1x/2x/4x wall-clock sweeps in chain length, input bond, output rank, coordination, and batch size, each with the other dimensions held fixed and each reported with its evaluated-point count so that a trajectory change cannot be mistaken for a complexity change. |
| `P` provenance/observability | **PASS** | Every statement in this entry is **[AI Supplied]**; no new paper or specification claim is made and no existing locator is repurposed. Baseline/candidate commits, hardware, affinity, provider/thread settings, seeds, build profile, warm-up, repetitions, statistic, noise gate, and threshold are printed by both benchmark binaries and reproduced above. The two counters #718 asks for that the public surface cannot supply -- evaluator message-cache hit/miss/eviction, and a process peak-byte figure -- were declared unavailable **before** measuring and are gated by crate-internal counter tests instead of being invented at the benchmark level. |
| `CI` remote regression | **NOT RUN** | The branch is committed locally and deliberately not pushed, per this task's instructions. The required GitHub checks have not been triggered and no CI conclusion is claimed. |

### `R12` final report: every task-level gate, Tasks 1--12

Status as of this entry. "carried" means the gate was closed in its own task's
section above and nothing in #718 disturbs it; #718 changes no production code,
so no earlier `C`/`E` result can have been invalidated by this commit.

| task | issue | gate | status | evidence |
|---|---|---|---|---|
| 1 | #707 | `C1`, `E1`, `R1`, `P` | **PASS** (carried) | `2026-09-03 #707 closure`, owner-layer matrix and source-evidence register |
| 2 | #717 | `C2`, `R2`, `M`, `F`, `I`, `D`, `P` | **PASS** (carried) | `2026-09-04 #717 closure` gate ledger |
| 2 | #717 | `E2`, `N`, `S` | **N/A** (carried) | test-only scope; no production path changed |
| 2 | #717 | downstream, sibling benchmark | **N/A** (carried) | no production evaluator behaviour changed |
| 3 | #715 | `C3`, `BC3`, `E3`, `R3`, `N`, `M`, `I`, `D`, `P` | **PASS** (carried) | `#715 gate ledger`; cut-local median `15.36 ms` versus rebuild `35.95 ms` |
| 3 | #715 | `F`, `S` | **N/A** (carried) | no fallback formula changed; constant-factor claim only |
| 4 | #716 | `C4`, `E4`, `R4`, `N`, `F`, `I`, `D`, `M`, `P` | **PASS** (carried) | `#716 gate ledger`; owned constructor `1.044 -> 0.082 ms` |
| 5 | #714 | `C5`, `BC5`, `E5`, `R5`, `N`, `M`, `F`, `I`, `D`, `P` | **PASS** (carried) | `#714 gate ledger`; 4x fewer candidate packing objects |
| 5 | #714 | `S` | **N/A** (resolved) | #714 deferred its scaling study to #718. #718's chain/bond/rank sweeps exercise the packed local-update path at 1x/2x/4x with no anomalous growth, but no dedicated per-call packing-count law was added, so this stays `N/A` rather than being upgraded to PASS |
| 6 | #711 | `C6`, `BC6`, `E6`, `R6`, `N`, `M`, `F`, `I`, `D`, `P`, `CI6` | **PASS** (carried) | `#711 gate ledger`; prepared versus repacked `45.7%/87.9%/56.4%` lower at bond 64/128/256; `CI6` closed by run `33856551145` |
| 6 | #711 | `S` | **N/A** (resolved the same way as #714) | constant-factor setup/copy reduction; #718 adds the ACI-level sweeps but no dedicated branch-slice scaling law |
| 7 | #710 | `C7`, `BC7`, `E7`, `R7`, `N`, `M`, `F`, `I`, `D`, `S`, `P` | **PASS** (carried) | `#710 gate ledger`; `1616 -> 240` retained positions, scoped `O(N^2)` metadata law |
| 7 | #710 | `CI` | **PENDING** (carried, unresolved here) | recorded pending at run `33862884867`; a later run `33863186382` at `dd1fc89` is recorded green in the #708 entry, but #718 pushed nothing and claims no CI conclusion |
| 8 | #708 | `C8`, `BC8`, `E8`, `R8`, `N`, `M`, `F`, `I`, `D`, `P` | **PASS** (carried) | `#708 gate ledger`; warm edge versus vertex `-21.2%/-47.8%/-77.0%` at chi 64/128/256 |
| 8 | #708 | `S` | **PASS**, strengthened by #718 | the scoped `points * chi_edge` law is now swept in all three factors independently and shown to be independent of descendant size (`L2`) |
| 8 | #708 | `CI` | **PENDING** (carried, unresolved here) | run `33867996612` was pending when recorded; #718 pushed nothing |
| 9 | #712 | `C9`, `BC9`, `E9`, `R9`, `DS9`, `F`, `D`, `P` | **PASS** (carried) | `#712 gate ledger`; shared-operand facade `2.4--17.3%` and `DS9` at `T=0.1` |
| 9 | #712 | `CI` | **PASS / coverage pending** (carried) | run `33868111483`; coverage was pending when recorded |
| 10 | #709 | `C10`, `BC10`, `E10`, `N`, `M`, `F`, `I`, `D`, `S`, `P` | **PASS** (carried) | `#709 gate ledger`; end-to-end Guard `1.85x--2.54x`, typed route 138 versus 1739 heap blocks |
| 10 | #709 | `R10` downstream stage | **PASS**, closed by #718 | `L4` above: parity first, then paired medians at `T=1.0` and `T=0.1` on both `pi_rtau` and `sigma_rtau`, plus the full A/B workflow at both temperatures |
| 10 | #709 | `CI` | **NOT RUN** (carried) | the #709 commit is local only; still owned by the integrating run |
| 11 | #713 | `C11`, `BC11`, `E11`, `R11`, `N`, `M`, `F`, `I`, `D`, `S`, `P` | **PASS** (carried) | `#713 gate ledger`; `16x--313x` and the exact candidate-count core-read reduction |
| 11 | #713 | downstream applicability | **N/A**, verified by #718 | `L5` above: nblock is a path (coordination 2), comb has maximum coordination 3 for any `r`, and the branching fixture has maximum coordination 3; the `z >= 4` route is unreachable downstream |
| 11 | #713 | `S` | **PASS**, strengthened by #718 | the timing-based exponent argument is now a deterministic counter law (`L2` row 3) |
| 11 | #713 | `CI` | **N/A (deferred)** (carried) | not pushed |
| 12 | #718 | `C12`, `E12`, `N`, `M`, `F`, `I`, `D`, `S`, `P` | **PASS** | this entry |
| 12 | #718 | `R12` | **PASS with three named exceptions** | workspace-wide pre-PR gate pending; remote `CI` not run; inherited `check-public-error-docs.py` failure (Open items item 6) |
| 12 | #718 | `CI` | **NOT RUN** | branch not pushed |

Three things are open across the whole umbrella. Two are the same kind: a
remote CI conclusion that only a push can produce (#710, #708, #709, #713,
#718), and the workspace-wide pre-PR gate; neither can be closed by a task that
is instructed not to push. The third is concrete and actionable now: the
`check-public-error-docs.py` Maintenance script fails on the base branch for
three public `Result` APIs introduced by #712 and #709 (Open items item 6).

### Verification commands and raw measurement output

```text
cargo fmt --all -- --check
(clean)

cargo clippy --release -p tensor4all-treeaci -p tensor4all-treetn -p tensor4all-aci --all-targets \
  -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
Finished `release` profile [optimized] target(s) in 32.26s

cargo test --release -p tensor4all-treetn --lib --no-fail-fast
test result: ok. 524 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out

cargo test --release -p tensor4all-treeaci --no-fail-fast
running 166 tests
test result: ok. 159 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out   (lib)
test result: ok. 1 passed  (tests/branch_degree.rs)
test result: ok. 7 passed  (tests/public_api.rs)
test result: ok. 1 passed  (tests/rank_scaling.rs)
test result: ok. 18 passed (doctests)

python3 scripts/repository-rules-review.py --base main --worktree --dry-run
Verdict: pass; No findings.

python3 scripts/check-crate-boundaries.py
crate-boundary-ok
python3 scripts/audit-library-panics.py
Audit passed: 0 unbaselined findings, 0 stale baseline entries

python3 scripts/check-public-error-docs.py
public Result APIs with incomplete error documentation:
- crates/tensor4all-tensorbackend/src/matrix.rs:638: grouped_mat_mul_shared_with_backend: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1373: new: # Errors does not name a concrete variant or condition
- crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:1714: with_plan: # Errors does not name a concrete variant or condition
(the same three findings reproduce on a clean `git archive a7632cc` tree, so
 they are inherited from #712 and #709 and are not caused by this commit --
 see Open items item 6)

RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 T4A_TREEACI_PARITY_ENABLE_GUARD=1 \
  taskset -c 0 ./target/release/deps/treeaci_parity-<hash> --bench --noplot     (x3)
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0 ./target/release/deps/treeaci_parity-<hash> --bench --noplot     (x3)
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0 ./target/release/deps/aci_scaling-<hash> --bench --noplot        (x3)
(equivalently: taskset -c 0 cargo bench -p tensor4all-aci --bench treeaci_parity -- --noplot
 and taskset -c 0 cargo bench -p tensor4all-treeaci --bench aci_scaling -- --noplot)

RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 cargo test --release \
  -p tensor4all-treeaci --lib state::tests::profile_high_rank_chain_phases_and_candidate_cache \
  -- --ignored --nocapture --test-threads=1
test result: ok. 1 passed; finished in 0.60 s   (Guard disabled)
T4A_TREEACI_ENABLE_PROFILE_GUARD=1 (same command)
test result: ok. 1 passed; finished in 2.70 s   (Guard enabled)

# downstream, clean copies only; /root/projects/gw-rs/sgw was not modified
git -C /root/projects/gw-rs/sgw archive HEAD | tar -x -C <tmp>     # HEAD ba6fbf3e
# + a non-committed [patch."https://github.com/tensor4all/tensor4all-rs.git"]
#   pointing at `git archive 4713ba8` (baseline) and `git archive a7632cc` (candidate)
SGW_RUN_TAG=aci-gate-718  SGW_ACI_GLOBAL_GUARD=1 G0_THREADS=1 ./run_r10_nblock_treeaci_ab.sh 1.0
SGW_RUN_TAG=gate718t01    SGW_ACI_GLOBAL_GUARD=1 G0_THREADS=1 ./run_r10_nblock_treeaci_ab.sh 0.1
/usr/bin/time -f "TIMING %e s wall, %M KiB maxrss" taskset -c 0 \
  <side>/target/release/isolate_aci_stage <run_dir> {pi_rtau,sigma_rtau}        (x3 per side)
```

Raw downstream output, `T=0.1`, one repetition per side (the other two agree to
the digits shown except for wall time and the Guard's evaluated-point jitter):

```text
### cand pi_rtau rep 1
  done in 1.7s, max bond=169, final err=9.941e-5
  pi_rtau sweep diagnostics: sweeps=7 evaluated_points=2152810 termination=Converged global_pivots_found=[4, 4, 3, 1, 1, 0, 0]
TIMING 2.84 s wall, 124772 KiB maxrss
### base pi_rtau rep 1
  done in 2.0s, max bond=169, final err=9.941e-5
  pi_rtau sweep diagnostics: sweeps=7 evaluated_points=2152990 termination=Converged global_pivots_found=[4, 4, 3, 1, 1, 0, 0]
TIMING 3.11 s wall, 124960 KiB maxrss
### cand sigma_rtau rep 1
  done in 1.3s, max bond=123, final err=9.976e-5
  sigma_rtau sweep diagnostics: sweeps=7 evaluated_points=1209550 termination=Converged global_pivots_found=[2, 2, 2, 2, 0, 0, 0]
TIMING 2.43 s wall, 124428 KiB maxrss
### base sigma_rtau rep 1
  done in 1.7s, max bond=123, final err=9.976e-5
  sigma_rtau sweep diagnostics: sweeps=7 evaluated_points=1209490 termination=Converged global_pivots_found=[2, 2, 2, 2, 0, 0, 0]
TIMING 2.81 s wall, 125000 KiB maxrss
```

External validation artefacts (clean SGW copies, two `git archive` trees of
tensor4all-rs, two read-only clones used only so the SGW binary's provenance
capture can resolve its fixed `../../tensor4all-rust/tensor4all-rs` path, and
the raw logs) live outside the repository under `/root/downstream718` and are
not part of this change. They are disposable; the numbers above are the record.

### Open items

1. **Unequal incident bonds cost an order of magnitude more per unit of
   required work.** At coordination 4 with a fixed hub bond product of 256, the
   `8,4,2,4` layout takes `6.06 us` per evaluated point against the equal
   layout's `0.25 us`, while the topology-required `outgoing_dim *
   product(incoming bonds)` is identical (256) on every outward arc and the
   unequal layout evaluates *fewer* points. Reproducible across three
   repetitions. Not diagnosed here; it needs a profile of the routing, cache,
   and factorization paths under an unequal layout. This is a new finding of
   the #718 gate and is not covered by any existing #699 subissue.
2. **The Guard's random-start search still dominates the 32-site high-rank
   chain.** At `chi=256` with the Guard enabled it is `1.231894 s` of a
   `1.498733 s` sweep total, `1.147752 s` of which is input evaluation across
   2,218 calls returning 8,844 scalars. Not a regression -- #709 is what makes
   the fixture finish at all -- but it is the largest remaining single cost on
   that fixture and no subissue owns it.
3. **The workspace-wide pre-PR gate is pending.** `cargo clippy --workspace`,
   `cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5`,
   `cargo test --profile ci -p tensor4all-hdf5`,
   `cargo test --doc --profile ci --workspace -j 8`, and
   `cargo doc --workspace --no-deps` were deliberately not run in this task and
   are owned by the integrating run.
4. **Remote `CI` is not run for #710, #708, #709, #713, or #718.** No branch
   was pushed. Each of those entries must have its required-check comparison
   done by the run that pushes, against the immediately preceding failure
   record.
5. **Two observability gaps remain**, declared before measuring rather than
   worked around: the cached evaluator's message-cache hit/miss/eviction counts
   and a process peak-byte figure are not reachable from a benchmark through
   the public surface. Exposing them would be a public-API change with its own
   documentation and review, so it was not bundled into a gates task.
6. **`scripts/check-public-error-docs.py` fails on the base branch and will
   fail the CI `Maintenance scripts` job on push.** Three public `Result` APIs
   have an `# Errors` section that does not name a concrete variant or
   condition: `tensor4all-tensorbackend/src/matrix.rs:638
   grouped_mat_mul_shared_with_backend` (from #712) and
   `tensor4all-treetn/src/treetn/cached_evaluator.rs:1373
   CachedEvaluatorPlan::new` and `:1714 TreeTNCachedEvaluator::with_plan`
   (from #709). All three reproduce on a clean `git archive a7632cc` tree, so
   none is caused by #718; this is exactly the failure class that #711's `CI6`
   record already hit once on a different symbol. It must be fixed before the
   branch is pushed. #718 did not fix it because it is another subissue's
   public API documentation and this task was scoped to gates.
