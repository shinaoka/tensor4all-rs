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
