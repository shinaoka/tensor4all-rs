# Issue #566 Remediation Program Design

## Objective

Resolve every item in tensor4all-rs umbrella issue #566, including shared-rule dependencies, correctness and soundness defects, enforcement gaps, typed-error debt, API and layering inconsistencies, architecture documentation, and performance candidates. Close the umbrella only after every item has fresh evidence and every related pull request is merged with green CI.

Moving an item to a follow-up issue is not completion. Each item must end in one of three states:

1. implemented and verified;
2. resolved by an explicit, documented maintainer decision; or
3. measured and demonstrated not to warrant a code change under the performance experiment protocol.

## Constraints

- Preserve existing behavior unless an approved API correction explicitly changes it.
- Early-development policy applies: remove obsolete APIs directly; do not retain compatibility aliases or shims.
- Preserve unrelated user changes. Work from isolated worktrees based on current `origin/main`.
- Luna Max is the sole implementation writer for each active repository worktree.
- Do not run concurrent writers against one worktree.
- Do not lower test tolerances or coverage thresholds without explicit user approval.
- Do not leave TODO placeholders, duplicated implementations, dead code, hidden assumptions, or undocumented behavior changes.
- Use tensorbackend and established tenferro-backed abstractions rather than adding local dense or linear-algebra implementations.
- Keep the network and TCI stacks as deliberately separate historical stacks. Remediate their seam rather than merging them.
- Minimize GitHub and CI load: finish local implementation, review, and validation before the first push of each pull request.
- The user authorizes commit, push, pull-request creation, and auto-merge after green CI. Stop only for a new product, API, architecture, safety, or tolerance decision not covered here.

## Execution Model

Each pull request starts from freshly fetched `origin/main` in a dedicated worktree. Work proceeds serially; a later pull request does not start until its predecessor is merged and the new worktree is based on the resulting `origin/main`.

For each pull request:

1. establish focused failing tests or deterministic audit failures;
2. assign one approved implementation slice at a time to Luna Max;
3. run focused green checks after each slice;
4. obtain an independent specification-compliance review;
5. obtain an independent code-quality and correctness review from a different model family when available;
6. send confirmed findings back to Luna Max for correction;
7. rerun affected checks and the required local PR validation suite;
8. inspect the final diff in the parent session;
9. commit locally, push once when practical, create the pull request, enable auto-merge, and monitor CI;
10. fix real failures rather than weakening gates;
11. after merge, update the #566 evidence ledger and start the next worktree from current `origin/main`.

## Pull Request Program

### Shared-rules prerequisite — at most one tensor4all-agent-rules PR

Finish the shared rules required by tensor4all-agent-rules issues #6, #7, and PR #8. Include the performance experiment protocol, work-log PR-body requirement, rule-inventory meta-rule, final cross-phase audit protocol, and agent-consumer policy needed by tensor4all-rs. Merge this before tensor4all-rs replaces overlapping local rules by reference.

### PR 1 — Soundness, CI enforcement, and housekeeping

Resolve Phase 0 and Phase 1 together:

- restore always-on release workspace testing;
- add checked shape products to Matrix constructors;
- validate the public RRLU path before unchecked access;
- propagate quanticstci coordinate-conversion failures;
- reject Matrix row indices that are out of bounds;
- use checked arithmetic at all cited C API slice and dimension boundaries;
- validate file-derived HDF5 integers;
- reject invalid quanticstransform shift widths;
- wire library-panic auditing into CI and cover public-path assertions;
- enforce missing error and panic documentation incrementally;
- add crate-boundary and dependency-cycle checks;
- replace prohibited `no_run` doctests with runnable examples;
- remove committed debris;
- remove the unused `kryst` dependency and stale claims;
- run coverage in release mode and document all exceptional thresholds;
- retain and validate the repository-rules review bot already merged in PR #568.

Every correctness or soundness defect receives tests for the happy path, error path, boundary conditions, and relevant layout/scalar variants.

### PR 2 — Complete public typed errors, layering, and test hygiene

Burn the public `anyhow::Result` backlog down to zero rather than limiting the work to the three highest-leverage surfaces. Keep the work reviewable through crate-oriented commits within one pull request.

Order the migration from foundations to consumers:

1. define foundational typed error seams and migrate tensor traits;
2. migrate tensorbackend and core;
3. migrate treetci and quanticstci;
4. migrate treetn;
5. migrate quanticstransform and HDF5;
6. sweep every remaining public library surface.

Add a changed-from-base gate first so new public anyhow surfaces cannot enter during migration. At the end, switch the same gate to repository-wide blocking mode.

In the same pull request:

- remove direct tenferro routes outside the sanctioned tensorbackend boundary, or encode only explicitly approved exceptions;
- move the `FullPivLuScalar` seam to the correct foundational layer;
- resolve the `core -> tcicore` dependency inversion;
- document or remove ID-only C API operations according to the approved serialization semantics and make internal union-find plumbing private;
- replace per-element tensor-network comparison loops with materialize-once subtraction and `maxabs()`;
- justify necessary graph traversals, deduplicate equivalent implementations, and remove the quadratic Euler-tour restart scan;
- enforce the retained work-log discipline.

Error enums use `thiserror`, preserve useful source errors, carry structured fields, and name actionable remedies when one is documented. Avoid opaque string payloads for recurring cases.

### PR 3 — API vocabulary, stack seam, architecture, and downstream guidance

Apply the approved API policy:

- an unsuffixed operation returns an owned result;
- `_mut` mutates `self`;
- `_into` writes into caller-provided output;
- retire `_inplace`, `_in_place`, `_owned`, and `scaled` where they conflict with this policy;
- use `max_bond_dim: Option<usize>` for bond caps;
- use `SvdTruncationPolicy` for truncation tolerance;
- use unsuffixed single-point evaluation and `_batched` for batches.

Remove the `evaluate_at` compatibility alias. Replace Julia-style concatenated names with snake_case without retaining compatibility aliases. Unify densification, inner-product, canonicalization, and options vocabulary. Remove scalar-suffixed Rust APIs outside the C API.

Keep the two stacks but remove the public naming trap by renaming the SimpleTT positional type to `SimpleTensorTrain`. Update all examples, documentation, and downstream internal uses in the same pull request.

Sanction `treetn::simplett_bridge` as the only crossing, migrate partitionedtt to it, and reject new ad hoc bridges. State which stack new feature crates target by default. Reduce unnecessary treetn module visibility, but do not perform a speculative full crate split. Document the criteria for any future split.

Adopt shared rules by reference, retain only repository-specific rules locally, add `CONTRIBUTING.md`, finish the downstream consumer skill and guide, add verified `llms.txt`, document error remedies and the resolved TensorTrain naming, and replace the superseded architecture description with a current two-stack/no-facade layer diagram.

### PR 4 — Performance evidence, final audit, and closure

Measure all five candidates in representative end-to-end workloads:

- TreeTNEvaluator reconstruction per point;
- C API cached-evaluator reconstruction per call;
- whole-TT evaluation in global pivot search;
- SimpleTT per-site allocation and slice copying;
- repeated contract profiling environment lookup.

Record workload, hardware, build profile, input sizes, baseline, variance, and end-to-end share. Implement only changes with a material measured contribution. Use the smallest existing abstraction that addresses the measured cause, add regression benchmarks or tests where stable, and remeasure after the change. A candidate with immaterial share is resolved by the recorded evidence rather than speculative code.

Then conduct a fresh cross-phase audit against every #566 item and all repository rules. Update issue #566 with links to merged pull requests, exact validation commands, design decisions, measurements, and evidence. Close it only after current `origin/main` passes the final suite.

## Validation Contract

### Per implementation slice

- establish RED before implementation when behavior changes;
- run focused release-mode tests for the changed crate and path;
- verify the changed path was actually exercised;
- run `cargo fmt --all` before any commit.

### Per tensor4all-rs pull request

At minimum, run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
```

Also run every changed audit script's self-tests and the relevant C API, HDF5, tutorial, benchmark, or generated-artifact checks. Run coverage before any deletion pull request and at final closure:

```bash
cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Coverage execution must use the repository's resolved release-mode policy after PR 1 updates it.

### Final closure validation

Fetch `origin`, verify the closure branch contains current `origin/main`, and rerun all commands above from a clean worktree. Inspect generated docs and `llms.txt`, API dumps where public APIs changed, benchmark evidence, issue state, PR merge state, and the final diff from the original audit basis.

## Disk-Space Policy

Check filesystem free space and build-artifact sizes before and after major workspace builds, coverage runs, and benchmark batches. Reuse build artifacts while space is healthy. At pull-request boundaries, or earlier when free space becomes constrained, remove regenerable `target` directories, coverage profiles, temporary benchmark outputs, and stale managed worktrees. Never remove source changes, committed evidence, active worktrees, or artifacts required for the completion audit. Report current free space when cleanup is performed.

## Evidence Ledger and Completion Audit

Maintain a ledger mapping each of the 39 Phase 0–5 checklist items and five performance candidates to:

- final disposition;
- changed files or explicit decision document;
- tests and commands with results;
- merged pull request and commit;
- residual risk, which must be none for required behavior;
- issue-body checkbox/update.

Before completion, audit every objective requirement against fresh evidence. The program is incomplete if any item is unverified, narrowed, deferred, described as merely "good enough", or left only probably satisfied. Mark the durable goal complete only after #566 itself is closed and all required evidence is present.

## Blocked Stop Condition

If access, tools, external CI, repository permissions, unavailable maintainer input, or an upstream dependency makes completion impossible, do not mark the program complete. Record attempted paths, concrete evidence, the exact blocker, all unmet requirements, and the minimum input or permission needed to continue.
