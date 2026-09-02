# Repository Rules

> **Shared, repository-neutral rules live in
> [`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
> (`rules/common/*`, `rules/rust/*`, `rules/julia/*`), referenced from
> `AGENTS.md` (URL + sibling checkout `../tensor4all-agent-rules/rules/`).**
>
> The sections below are the **tensor4all-rs-specific residue only**. Public
> surface, error handling, testing/coverage, performance, layering/dedup,
> dense layout, C API ABI, dependencies, and graph/cache policy are covered by
> the shared rules (see the shared `rules/index.md`); do not duplicate them
> here — extend the shared repo instead and keep this file minimal.

## Public Boundary Safety Audits

- User-reachable Rust, tensor-network, C API, and language-binding-facing paths
  must validate input-derived shape, rank, axis, index, dtype/scalar kind,
  topology, truncation, and layout configuration before no-op shortcuts,
  allocation, backend calls, pointer use, or FFI calls.
- Shape products, dense element counts, byte lengths, strides, offsets, bond
  products, launch sizes, and FFI dimensions must use checked arithmetic before
  conversion to `usize`, `i32`, `u32`, pointer offsets, or allocation sizes.
- Publicly reachable library paths must not turn invalid user input into
  `panic`, `unwrap`, `expect`, unchecked indexing, poisoned-lock unwraps, or
  debug-only assertions. Return a crate-local typed error or a C API status with
  preserved diagnostic context instead.
- Validate fast paths and zero-size returns with the same scrutiny as the main
  path. A shortcut must not skip rank, index identity, scalar kind, topology, or
  layout checks that would reject invalid input on the full path.
- Repeated validation shared by Rust, C API, Julia/Python bindings, dense,
  tensor-network, or scalar-specific wrappers should live in shared helpers or
  prepared metadata types. Do not duplicate hand-written checks when a helper
  can enforce the contract before unsafe or performance-sensitive code.
- Validation helpers should return typed prepared metadata when downstream code
  would otherwise repeat rank-minus-one, axis ordering, shape-product, site role,
  edge role, or index mapping calculations.
- Parallel operation surfaces must keep validation and scalar-promotion
  semantics in parity. A bug in one of Rust/C API, f64/Complex64, dense/TN, or
  MPS/MPO-like wrappers should trigger a nearby audit of the corresponding
  surfaces.
- When a bug exposes a public API design mismatch, fix the canonical contract
  at its owner. API compatibility is not a goal in this early-development
  repository unless the task explicitly requires a compatibility window.

## Base Branch Synchronization

- Start new work from the current remote base, not from a stale local checkout:
  run `git fetch origin` and create feature branches or worktrees from
  `origin/main`.
- Before treating PR checks as final, fetch `origin` and verify that the PR
  branch contains the current `origin/main`.
- If GitHub reports the PR as behind the base branch, update the PR branch from
  `origin/main` before relying on checks, enabling auto-merge, or declaring the
  PR ready to merge.
- After any `origin/main` synchronization, rerun or re-monitor CI. Green checks
  from before the synchronization are not sufficient.

## No Hidden Dense Materialization In Production Paths

- Production algorithms must not silently materialize a full dense tensor whose
  memory or time scales as the product of unconstrained site or external index
  dimensions.
- Scalable public paths such as tensor-network contraction, operator apply,
  truncation, fitting, C API entry points, and language-binding-facing APIs
  must avoid hidden calls to full-network dense materialization methods such as
  `contract_to_tensor()` or `to_dense()` unless the API is explicitly a dense
  or reference API.
- Dense/reference implementations are allowed only when they are explicitly
  named as dense, reference, or debug behavior; documented as
  O(product of index dimensions) in memory/time; excluded from default or
  production method dispatch; and covered by small-size tests only.
- Method names must match semantics. For MPO/MPS or `LinearOperator`
  application, `naive` means local exact tensor-network contraction, not full
  dense materialization, unless the API name and documentation explicitly say
  dense/reference.
- If dense materialization is unavoidable for a debugging or reference path,
  prefer an explicit size guard or caller-supplied limit such as
  `max_dense_elements`.
- When adding or changing contraction/application code, check whether any path
  can call `contract_to_tensor()`, `to_dense()`, or equivalent full
  materialization. If yes, justify that the path is explicitly dense/reference
  or replace it with a local tensor-network algorithm.
- Do not replace known structured tensors with dense tensors in production
  paths when the structure is semantically required or already available. This
  includes identity, diagonal, Kronecker delta, copy, one-hot, selector, and
  bridge tensors used only to preserve topology or route bond spaces.
- A tensor that is mathematically a delta/copy/identity must use the appropriate
  compact structured representation when one exists, for example diagonal/copy
  storage or structured axis classes. If only a dense constructor exists, add or
  refine the structured core API first, or reject the unbounded production path.
- Topology-preserving helper tensors must not introduce hidden dense
  `bond_dim^2`, `bond_dim^k`, or site-product payloads. Dimension-1 structural
  links are acceptable; nontrivial bridge deltas must preserve compact
  diagonal/copy structure or remain behind an explicit dense/reference guard.
- Tests for delta/copy/identity paths should check both numerical behavior and
  representation behavior where possible, such as `is_diag()`, `storage_kind()`,
  payload dimensions, axis classes, or a regression whose intended algorithm is
  cheap but accidental dense storage would fail.

## Tensor Network Test Comparisons

- Randomized algorithms must provide both a high-level deterministic seed API
  and a low-level API that accepts a caller-owned `&mut R` where
  `R: rand::Rng + ?Sized`. The high-level API must delegate to the low-level
  implementation, and the low-level implementation must consume the supplied
  stream directly rather than deriving a seed for a hidden RNG.
- Seed-based production entry points must use an explicitly named RNG algorithm,
  not `StdRng`, so dependency upgrades do not silently change the stream.
- Tests of randomized algorithms must use `ChaCha8Rng` with an explicit seed.
  Do not use `StdRng`, `thread_rng`, or an implicit entropy source in those tests.
- Small reference tests may materialize dense tensors. Materialize each full
  result once, subtract tensors, and compare the whole result with `maxabs()`.
  Do not compare by re-contracting or re-evaluating every element one by one.
- Long-TT or long-TreeTN regression tests must be sized so accidental dense
  materialization would fail quickly, but the intended tensor-network algorithm
  remains cheap.
- Long tensor-network tests must not use comparison helpers that dense-materialize
  internally. In particular, do not call `maxabs()` on a TT/TN type unless that
  specific implementation is known to be scalable.
- For long TT/TN equality checks, prefer scalable comparisons:
  - direct-sum difference such as `tt1 - tt2` or `axpby(1, tt1, -1, tt2)`,
    followed by a tensor-network norm;
  - sampled `evaluate()` checks at fixed multi-indices;
  - structural invariants such as node count, site indices, and bounded bond
    dimensions.
- Do not compress a difference TT/TN before using it as an exact test residual;
  compression would make the comparison itself approximate.
- When using norm-based approximate equality, report the residual explicitly,
  for example `diff.norm() / reference.norm()`, so failures are diagnosable.

## Online Tutorial Synchronization

- `docs/book/src/tutorials/` is the live source for online tutorial prose.
- `docs/tutorial-code/src/bin/` and `docs/tutorial-code/src/` are the runnable
  source for tutorial demos and shared tutorial helpers.
- Any change to tutorial APIs, tutorial code, generated tutorial artifacts, or
  public APIs used by tutorials must check and update the corresponding live
  mdBook page before the branch is complete.
- Non-trivial guide snippets should have an executable source of truth, such as
  a checked example, tutorial binary, doctest, or test. If Markdown must copy a
  snippet by hand, add or update a sync/extraction check when practical.
- Diagrams in `README.md`, mdBook, and `docs/design/` are part of the documented
  surface. Crate names, layer assignments, dependency direction, and public
  entry points shown in diagrams must match the current implementation.
- Legacy markdown under `docs/tutorial-code/docs/tutorials/` must not be treated
  as the online source of truth.

## Work Logs And Design Records

- Nontrivial refactors, cleanup streams, AI-assisted implementation batches, and
  changes that make explicit design tradeoffs should leave a curated work log
  under `docs/worklogs/`.
- A work log should record the session summary, code and documents read,
  reference implementations considered, decisions made, alternatives rejected or
  deferred, verification performed, and remaining risks.
- Work logs are reviewer-facing decision records, not raw transcripts and not
  implementation plans. Keep them concise enough to review but specific enough
  that later work can understand the selected abstraction, split, public API, or
  deferral.
- When a PR establishes or changes durable design intent, update the appropriate
  document under `docs/design/` in the same PR. Use work logs for session-level
  rationale and design docs for decisions future implementation should follow.
- When an audit finding is a false positive because of an intentional invariant,
  record the evidence in the issue, PR body, work log, nearby source comment, or
  source-contract test. Do not simply skip it and leave future reviewers to
  rediscover the same non-bug.

## Differentiability And AD Preservation

- Production tensor, tensor-network, linear algebra, and solver paths should
  preserve automatic differentiation metadata whenever the underlying backend
  and operation can do so.
- Unless there is a documented reason not to, dense tensor operations should go
  through `tensor4all-tensorbackend` or existing tensor4all abstractions that
  preserve tenferro AD metadata. Avoid bypassing the backend with local dense
  loops, native-tensor extraction, host scalar conversion, or detached
  reference implementations.
- Do not unnecessarily detach tensors, convert differentiable values through
  plain Rust scalars, force `.real()`/real-only projections, or round-trip
  through dense/reference paths in a way that discards AD metadata.
- When an operation must cross a non-differentiable boundary, such as a scalar
  control-flow decision, diagnostic conversion, FFI boundary, unsupported
  backend routine, or intentionally native/reference implementation, keep that
  boundary explicit in the code and documentation.
- Prefer backend-provided differentiable primitives for contraction, einsum,
  SVD, QR, eigensolvers, exponentials, and scalar operations. If a required
  primitive is missing, add or refine the appropriate backend/core API instead
  of silently implementing a detached local workaround.
- Tests that claim AD support must include oracle or finite-difference
  coverage. Tests for algorithms that are not yet AD-supported should still
  avoid unnecessary AD metadata loss along intermediate tensor paths.

## Index Identity Semantics

- Do not use `index.id()` alone to decide whether two indices are the same
  index object. Full index equality must be used for identity comparisons so
  same-ID indices with different prime levels, tags, directions, or other
  metadata remain distinct.
- Maps and sets that represent index identity must be keyed by the full index
  value, not by `IndexLike::Id`.
- Public and internal APIs that select a concrete tensor leg, TreeTN site,
  edge, topology assignment, replacement target, split/fuse target, or
  restructure target must accept the full `Index` value, not an index ID. Shape
  APIs around index identity so callers can pass the index they mean.
- C API and language-binding-facing APIs must follow the same rule: accept
  `Index` handles for concrete index selection, expose full-index
  equality/hash helpers when bindings need map keys, and do not expose ID-only
  constructors, selectors, or identity getters as public API.
- Pure ID comparisons are allowed only inside implementation details that are
  explicitly about logical-site lookup, compatibility, or contraction pairing.
  Do not expose ID-based public APIs for selecting concrete indices. If a
  temporary internal ID map is unavoidable, name the local variable to make the
  ID-based semantics explicit, such as `logical_site_ids` or
  `contraction_pair_ids`, and reject ambiguous same-ID inputs instead of
  choosing one silently.
- When changing tensor, TreeTN, contraction, direct-sum, replacement, or
  reindexing code, add a regression test with same-ID indices that differ by
  prime level or tags.

## Unsafe Code Boundary

- `unsafe` belongs in FFI, backend, storage, or other leaf modules that own the
  low-level invariant. Do not introduce `unsafe` in high-level tensor-network,
  interpolation, graph/topology, transform, or AD-preserving algorithm code
  when the correct fix is a lower-level helper.
- Count and review `unsafe` by location and purpose, not by raw text search.
  Generated code, comments, tests, and existing backend/FFI seams should be
  separated from production algorithmic code when auditing risk.
- Each new `unsafe` block must have a nearby `// SAFETY:` comment explaining the
  validation or ownership invariant that makes it sound.
- Boundary-condition tests or source-contract tests should cover new unsafe
  indexing, raw pointer, FFI, or backend-native view construction paths.

## File Organization

- Keep source files focused, but do not split files solely to reduce line count.
  Treat roughly 1000 lines as a soft review trigger, not a mechanical limit.
- Split only along clear behavior, abstraction, feature, ownership, or
  public/private API boundaries such as validation, planning, execution,
  topology, contraction, solver, C API bridge, backend glue, or cache ownership.
- Avoid arbitrary `part1`/`part2` splits and tiny files that force readers to
  chase one concept across many modules.
- Use line count to decide where to inspect first. Use responsibility, change
  frequency, public/private API boundaries, and human navigation to decide
  whether and how to split.

## Language Bindings

- New Julia/Python-facing features that need Rust support should land in
  tensor4all-rs first.
- C API changes should account for downstream Tensor4all.jl compatibility and
  pin updates.
