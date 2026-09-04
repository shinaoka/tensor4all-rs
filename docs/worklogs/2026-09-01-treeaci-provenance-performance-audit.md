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
| `CI6` remote regression | **PENDING PUSH** | The preceding failed CI_rs run was inspected before push: run `33820251118`, head `d5de28f0d77b0dde667e0dfddb5c5892c6b78c9a`, failed job `Maintenance scripts`, step `Audit public Result APIs`, with the concrete log anchor `dense_native_tensor_from_col_major_owned: # Errors does not name a concrete variant or condition`; the Test, Coverage, Doctests, and Lint jobs passed and rollup failed because Maintenance failed. The local reproduction `python3 scripts/check-public-error-docs.py` is green after the earlier repair, with its complete 15-test suite passing. After pushing #711, the new required checks must be inspected and this row updated with the new run ID/head SHA and final conclusion; a pending check is not a pass. |

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
```

The next implementation subissue is **#710** (directed-component keys,
layouts, and cache accounting). Per the execution protocol, stop after this
#711 report; do not begin #710 in the same run.
