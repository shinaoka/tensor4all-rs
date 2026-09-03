# Issue #698 final integration audit

## Candidate

Audited implementation commit:
`a0ae762ccb147fb7e8b91d3c185b705d022b674b`.
Base: `origin/main` at `b9636123b239ba3426511311e43f4ac66577b60b`.
A fresh fetch, empty `git status --short`, and merge-base ancestor check confirmed
the candidate contains current `origin/main`.

This file is the only post-audit documentation addition. The final PR-head
refresh must treat it as report-only and reconfirm clean status/base ancestry.

## Lane reports

### Specification and architecture

**PASS.** Item 1, #700, #703, #705, and #706 have implemented deliverables.
#701, #702, and #704 have explicit measured need-gate closures rather than
unowned deferrals. Review findings on benchmark identity/config verification and
mandatory candidate commit provenance were fixed and re-reviewed.

### Safety and resource lifecycle

**PASS.** `PreparedContraction` validates count, explicit dimensions, ordered
full indices, and axis classes before execution and preserves structured/AD
paths. Benchmark timeout kills and reaps the complete process group. Tree memory
preflight uses O(1) checked shape counts; non-finite controls/results are
rejected. Tree cache growth is bounded and checked. No hidden dense/device
transfer or new global cache was introduced.

### Performance and parallelism

**PASS.** The runner sanitizes ambient benchmark/profile variables, validates
selected center/config/build identity, records effective pair count, enforces
5/10-pair minima, fixes all compute-library thread counts to one, and preserves
failure precedence. All negative and initially failed measurements remain in the
worklogs. Prepared-plan claims are limited to N-ary/retained calls; #701's
conclusion is limited to measured fixed N-ary sections.

### Public API and documentation

**PASS.** `PreparedContraction` is exported, documented, and covered by runnable
asserted examples and typed error behavior. The mdBook guide is synchronized.
Malformed literal `/// ///` text in changed public rustdoc was removed. Fixed-rank
and benchmark documentation match implementation and CLI behavior.

### Backend and hardware

**PASS.** Default faer CPU is the required SRC lane. Provider-injection compile
passes but is not misrepresented as a provider-specific SRC run. Nine CUDA
TreeTN tests pass on NVIDIA A100 80GB; CUDA SRC remains accurately marked
unsupported because SRC has no device/context-owning seam. Later changes did not
touch backend/device execution, so this evidence carried forward after explicit
diff-impact review.

### Integration

The first integration verdict was `INCONCLUSIVE` solely because the read-only
auditor could not prove worktree cleanliness or remote freshness. The parent
then fetched origin and supplied exact command evidence: clean status, candidate
HEAD above, current origin/main above, and successful ancestor check. The
independent integration recheck returned **PASS** with no Critical, Important,
or Minor findings.

## Verification evidence

- `cargo fmt --all -- --check`: passed.
- Workspace clippy with `-D warnings`, `missing_errors_doc`, and
  `missing_panics_doc`: passed on the exact implementation tree.
- Non-HDF5 nextest: 3,250 tests passed before the 60-second command ceiling;
  the only two interrupted long tests were rerun with the same workspace feature
  set and both passed, covering all 3,252 tests.
- HDF5: 1 unit passed (4 ignored hardware/library annotations), 46 integration
  tests passed, and 10 doctests passed.
- Workspace doctests: passed; `PreparedContraction`'s three doctests were rerun
  after final review fixes and passed.
- `TENSOR4ALL_CARGO_PROFILE=ci ./scripts/test-mdbook.sh`: passed.
- `cargo doc --workspace --no-deps`: completed; warnings were pre-existing and
  no changed public link remained broken.
- `cargo run -p xtask --release -- api-dump`: complete public inventory passed
  and included `PreparedContraction`.
- `python3 scripts/test-run-src-benchmark-gates.py`: 11 passed.
- Benchmark example estimator tests: 3 passed.
- Changed public error-doc audit, repository-rules dry run, crate-boundary tests,
  and C API header check: passed.
- Provider-injection check: passed.
- CUDA TreeTN test: 9 passed on A100.
- Final exact-candidate quick SRC smoke: PASS, including stale ambient variable
  sanitization.
- Coverage impact attestation: every new branch is covered by focused tests;
  no tolerance or threshold was relaxed. Hosted CI remains authoritative for
  percentage coverage.

## Performance closure

- Automatic binary dispatch is within noise of explicit pairwise calls.
- Prepared N-ary execution improves the representative repeated core contraction
  by 14–17% across five fresh processes.
- Tree segment alignment improves the primary bounded-tree case by 5.46%; the
  final complete quick gate and bond-8 diagnostics pass.
- Fixed N-ary sections (#701), directed-message planning (#702), and flattened
  submission overhead (#704) remain below their predeclared 10% need gates, so
  no speculative machinery was added.
