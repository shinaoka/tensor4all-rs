# Repository Rules

> **Shared, repository-neutral rules live in
> [`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
> (`rules/common/*`, `rules/rust/*`, `rules/julia/*`), referenced from
> `AGENTS.md` (URL + sibling checkout `../tensor4all-agent-rules/rules/`).**
>
> The sections below are the **tensor4all-rs-specific residue only**. Public
> surface, error handling, testing/coverage, performance, layering/dedup, dense
> layout, C API ABI, dependencies, and graph/cache policy are covered by the
> shared rules (see the shared `rules/index.md`); do not duplicate them here.
> Extend the shared repo instead and keep this file minimal.

## Public Boundary Safety Audits

- User-reachable Rust, tensor-network, C API, and binding-facing paths validate
  input-derived shape, rank, axis, index, dtype/scalar kind, topology,
  truncation, and layout configuration before no-op shortcuts, allocation,
  backend calls, pointer use, or FFI calls.
- Shape products, dense element counts, byte lengths, strides, offsets, bond
  products, launch sizes, and FFI dimensions use checked arithmetic before
  conversion to `usize`, `i32`, `u32`, pointer offsets, or allocation sizes.
- Publicly reachable library paths never turn invalid user input into `panic`,
  `unwrap`, `expect`, unchecked indexing, poisoned-lock unwraps, or debug-only
  assertions. Return a crate-local typed error or a C API status with preserved
  diagnostic context.
- Fast paths and zero-size returns get the same scrutiny as the main path: a
  shortcut must not skip rank, index identity, scalar kind, topology, or layout
  checks the full path would apply.
- Validation shared by Rust, C API, Julia/Python bindings, dense,
  tensor-network, or scalar-specific wrappers should live in shared helpers or
  prepared metadata types. Do not duplicate hand-written checks when a helper
  can enforce the contract before unsafe or performance-sensitive code. Helpers
  should return typed prepared metadata when downstream code would otherwise
  repeat rank-minus-one, axis ordering, shape-product, site role, edge role, or
  index mapping calculations.
- Parallel operation surfaces keep validation and scalar-promotion semantics in
  parity. A bug in one of Rust/C API, f64/Complex64, dense/TN, or MPS/MPO-like
  wrappers should trigger a nearby audit of the corresponding surfaces.
- When a bug exposes a public API design mismatch, fix the canonical contract at
  its owner. API compatibility is not a goal in this early-development
  repository unless the task explicitly requires a compatibility window.

## Base Branch Synchronization

- Start work from the current remote base: `git fetch origin`, then branch or
  create worktrees from `origin/main`, never a stale local checkout.
- Before treating PR checks as final, fetch `origin` and verify the PR branch
  contains current `origin/main`. If GitHub reports the PR behind the base,
  update from `origin/main` before relying on checks, enabling auto-merge, or
  declaring it ready.
- After any `origin/main` synchronization, rerun or re-monitor CI; earlier
  green checks are insufficient.

## No Hidden Dense Materialization In Production Paths

- Production algorithms never silently materialize a full dense tensor whose
  memory or time scales as the product of unconstrained site or external index
  dimensions.
- Scalable public paths (tensor-network contraction, operator apply,
  truncation, fitting, C API entry points, binding-facing APIs) avoid hidden
  calls to full-network materialization such as `contract_to_tensor()` or
  `to_dense()` unless the API is explicitly dense or reference.
- Dense/reference implementations are allowed only when explicitly named as
  dense, reference, or debug; documented as O(product of index dimensions) in
  memory/time; excluded from default or production dispatch; and covered by
  small-size tests only.
- Method names match semantics. For MPO/MPS or `LinearOperator` application,
  `naive` means local exact tensor-network contraction, not dense
  materialization, unless the name and docs explicitly say dense/reference.
- If dense materialization is unavoidable for a debugging or reference path,
  prefer an explicit size guard or caller-supplied limit such as
  `max_dense_elements`.
- When adding or changing contraction/application code, check whether any path
  can call `contract_to_tensor()`, `to_dense()`, or equivalent. If so, justify
  that the path is explicitly dense/reference or replace it with a local
  tensor-network algorithm.
- Do not replace structured tensors with dense tensors in production paths when
  the structure is semantically required or already available: identity,
  diagonal, Kronecker delta, copy, one-hot, selector, and bridge tensors used to
  preserve topology or route bond spaces.
- A tensor that is mathematically a delta/copy/identity uses the compact
  structured representation when one exists (diagonal/copy storage, structured
  axis classes). If only a dense constructor exists, add or refine the
  structured core API first, or reject the unbounded production path.
- Topology-preserving helper tensors must not introduce hidden dense
  `bond_dim^2`, `bond_dim^k`, or site-product payloads. Dimension-1 structural
  links are fine; nontrivial bridge deltas keep compact diagonal/copy structure
  or stay behind an explicit dense/reference guard.
- Tests for delta/copy/identity paths should check representation as well as
  numerics where possible: `is_diag()`, `storage_kind()`, payload dimensions, axis
  classes, or a regression that is cheap for the intended algorithm but fails
  under accidental dense storage.

## Tensor Network Test Comparisons

- Randomized algorithms provide both a high-level deterministic seed API and a
  low-level API taking a caller-owned `&mut R` where `R: rand::Rng + ?Sized`.
  The high-level API delegates to the low-level one, which consumes the
  supplied stream directly rather than deriving a seed for a hidden RNG.
- Seed-based production entry points use an explicitly named RNG algorithm, not
  `StdRng`, so dependency upgrades do not silently change the stream.
- Tests of randomized algorithms use `ChaCha8Rng` with an explicit seed, never
  `StdRng`, `thread_rng`, or implicit entropy.
- Small reference tests may materialize dense tensors: materialize each full
  result once, subtract, and compare the whole result with `maxabs()`. Never
  compare by re-contracting or re-evaluating element by element.
- Long-TT or long-TreeTN regression tests are sized so accidental dense
  materialization fails quickly while the intended algorithm stays cheap.
- Long tensor-network tests must not use comparison helpers that
  dense-materialize internally; do not call `maxabs()` on a TT/TN type unless
  that implementation is known to be scalable.
- For long TT/TN equality, prefer scalable comparisons: direct-sum difference
  (`tt1 - tt2` or `axpby(1, tt1, -1, tt2)`) followed by a tensor-network norm;
  sampled `evaluate()` at fixed multi-indices; structural invariants such as
  node count, site indices, and bounded bond dimensions.
- Do not compress a difference TT/TN before using it as an exact test residual.
- With norm-based approximate equality, report the residual explicitly, e.g.
  `diff.norm() / reference.norm()`.

## Online Tutorial Synchronization

- `docs/book/src/tutorials/` is the live source for tutorial prose;
  `docs/tutorial-code/src/bin/` and `docs/tutorial-code/src/` are the runnable
  source for demos and shared helpers.
- Any change to tutorial APIs, tutorial code, generated tutorial artifacts, or
  public APIs used by tutorials updates the corresponding live mdBook page
  before the branch is complete.
- Non-trivial guide snippets should have an executable source of truth
  (checked example, tutorial binary, doctest, or test). If Markdown must copy a
  snippet by hand, add or update a sync/extraction check when practical.
- Diagrams in `README.md`, mdBook, and `docs/design/` are documented surface:
  crate names, layer assignments, dependency direction, and public entry points
  must match the implementation.
- Legacy markdown under `docs/tutorial-code/docs/tutorials/` is not the online
  source of truth.

## Work Logs And Design Records

- Nontrivial refactors, cleanup streams, AI-assisted implementation batches,
  and changes with explicit design tradeoffs should leave a curated work log
  under `docs/worklogs/`. A work log should record: session summary, code and
  documents read, reference implementations considered, decisions,
  alternatives rejected or deferred, verification performed, remaining risks.
- Work logs are reviewer-facing decision records, not transcripts or
  implementation plans: concise enough to review, specific enough that later
  work understands the selected abstraction, split, public API, or deferral.
- When a PR establishes or changes durable design intent, update the relevant
  `docs/design/` document in the same PR. Work logs hold session-level
  rationale; design docs hold decisions future implementation should follow.
- When an audit finding is a false positive because of an intentional
  invariant, record the evidence in the issue, PR body, work log, nearby source
  comment, or source-contract test so future reviewers do not rediscover it.

## Differentiability And AD Preservation

- Production tensor, tensor-network, linear algebra, and solver paths should
  preserve automatic differentiation metadata whenever the backend and
  operation allow.
- Unless documented otherwise, dense tensor operations should go through
  `tensor4all-tensorbackend` or existing tensor4all abstractions that preserve
  tenferro AD metadata. Avoid bypassing the backend with local dense loops,
  native-tensor extraction, host scalar conversion, or detached reference
  implementations.
- Do not unnecessarily detach tensors, convert differentiable values through
  plain Rust scalars, force `.real()`/real-only projections, or round-trip
  through dense/reference paths in ways that discard AD metadata.
- Where an operation must cross a non-differentiable boundary (scalar
  control-flow decision, diagnostic conversion, FFI, unsupported backend
  routine, intentionally native/reference implementation), keep that boundary
  explicit in code and documentation.
- Prefer backend-provided differentiable primitives for contraction, einsum,
  SVD, QR, eigensolvers, exponentials, and scalar operations. If one is
  missing, add or refine the backend/core API instead of a detached local
  workaround.
- Tests claiming AD support include oracle or finite-difference coverage. Tests
  for algorithms not yet AD-supported should still avoid unnecessary AD
  metadata loss along intermediate tensor paths.

## Index Identity Semantics

- Never use `index.id()` alone to decide whether two indices are the same
  object. Identity comparisons use full index equality so same-ID indices with
  different prime levels, tags, directions, or other metadata remain distinct.
- Maps and sets representing index identity are keyed by the full index value,
  not `IndexLike::Id`.
- Public and internal APIs that select a concrete tensor leg, TreeTN site,
  edge, topology assignment, replacement target, split/fuse target, or
  restructure target accept the full `Index` value, not an ID.
- C API and binding-facing APIs follow the same rule: accept `Index` handles
  for concrete selection, expose full-index equality/hash helpers when bindings
  need map keys, and expose no ID-only constructors, selectors, or identity
  getters.
- Pure ID comparisons are allowed only inside implementation details explicitly
  about logical-site lookup, compatibility, or contraction pairing. If a
  temporary internal ID map is unavoidable, name it to make the semantics
  explicit (`logical_site_ids`, `contraction_pair_ids`) and reject ambiguous
  same-ID inputs instead of choosing one silently.
- When changing tensor, TreeTN, contraction, direct-sum, replacement, or
  reindexing code, add a regression test with same-ID indices that differ by
  prime level or tags.

## Unsafe Code Boundary

- `unsafe` belongs in FFI, backend, storage, or other leaf modules owning the
  low-level invariant, never in high-level tensor-network, interpolation,
  graph/topology, transform, or AD-preserving algorithm code when the correct
  fix is a lower-level helper.
- Count and review `unsafe` by location and purpose, not raw text search.
  Generated code, comments, tests, and existing backend/FFI seams should be
  separated from production algorithmic code when auditing risk.
- Each new `unsafe` block has a nearby `// SAFETY:` comment explaining the
  validation or ownership invariant that makes it sound.
- Boundary-condition or source-contract tests should cover new unsafe
  indexing, raw pointer, FFI, or backend-native view construction paths.

## File Organization

- Keep source files focused, but never split solely to reduce line count.
  Roughly 1000 lines is a soft review trigger, not a limit.
- Split only along clear behavior, abstraction, feature, ownership, or
  public/private API boundaries: validation, planning, execution, topology,
  contraction, solver, C API bridge, backend glue, cache ownership. No
  `part1`/`part2` splits and no tiny files that scatter one concept.
- Line count decides where to inspect first; responsibility, change frequency,
  API boundaries, and human navigation decide whether and how to split.

## Language Bindings

- New Julia/Python-facing features that need Rust support should land in
  tensor4all-rs first.
- C API changes should account for downstream Tensor4all.jl compatibility and
  pin updates.
