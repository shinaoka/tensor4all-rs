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

Only these two algorithm authorities were allowed:

1. Ritter, *Fast elementwise operations on tensor trains with alternating
   cross interpolation*, arXiv:2604.00037v2 (23 Apr 2026), including Algorithms
   1--5 and equations (4)--(11).
2. The chain implementation in `crates/tensor4all-aci/src/` at the audited
   revision (called “simplett ACI” below).

No cited TCI paper, previous TreeACI worklog, issue, pull request, or TreeTN
implementation was used as algorithm authority. TreeTN source was inspected
only to determine the runtime semantics of calls made by TreeACI. Tests and
benchmarks are evidence only.

The provenance labels used below are:

- **Paper**: a direct implementation of a cited equation or pseudocode step.
- **simplett ACI**: a direct analogue of the cited chain implementation.
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

## Performance findings

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

No profiler trace for the user's exact slow workload was available in this
audit. Findings labelled P0/P1 above are based on exact control-flow and
allocation counts plus the isolated branch microbenchmark. A subsequent fix
phase should first add a reproducible long-chain length sweep and representative
branched fixtures with fixed degree/rank before changing production code.

## Recommended fix order (not implemented in this audit)

1. Add benchmark fixtures that independently scale chain length, branch degree,
   input bond dimension, and active candidate rank; include callback counts and
   accuracy/rank parity.
2. Replace per-edge whole-store `extend` with cut-local incremental frame
   insertion and eliminate the per-edge whole-TreeTN metadata transaction,
   while retaining failure atomicity at a smaller seam.
3. Introduce borrowed/packed candidate-frame batches so local update avoids
   `Vec<Vec<T>>`, cache-hit cloning, repacking, and scatter where layouts match.
4. Generalize the batched message contraction to arbitrary incoming degree, or
   explicitly reject/route high degree until such a kernel exists.
5. Separately derive a lazy/block LUCI source for (T4) if full branch cross
   materialization, rather than contraction mechanics, is the dominant limit.

Every item above changes **[AI Supplied]** machinery or requires a new explicit
tree derivation. None should be attributed to unshown pseudocode in the paper.
