# CI Build-Artifact Reduction

## Session summary

Extended the repository's debug-info reduction to CI. Comprehensive test and
documentation jobs now use a release-based `ci` profile that disables
incremental output and strips linker symbols. CI also reports target size so
future dependency or target growth is visible.

## Context reviewed

- The build-artifact measurements and profile decision recorded in
  `2026-08-09-release-debug-info-reduction.md`.
- The supplied `strided-rs` build-artifact study.
- Root Cargo profiles, hosted and self-hosted workflows, mdBook test plumbing,
  and Cargo subprocess call sites.
- Shared tensor4all repository, performance, Rust performance, and docs/test
  rules.

## Decisions

- `profile.ci` inherits `release` because tensor4all-rs verification is required
  to run with release semantics. It disables incremental compilation and strips
  symbols, matching the lifecycle of comprehensive CI artifacts.
- Coverage keeps its existing profile because instrumentation owns its artifact
  requirements.
- The mdBook helper accepts `TENSOR4ALL_CARGO_PROFILE`, allowing CI's probe and
  rustdoc library search path to use the same profile without changing the
  local release-mode default.
- Persistent self-hosted targets are not cleaned on every run. Periodic
  profile-specific or full cleanup is documented instead, preserving useful
  dependency reuse.

## Alternatives not selected

- Inheriting `test` for `profile.ci` would lose the repository's required
  release-mode test semantics.
- Applying the CI profile to coverage could interfere with coverage-owned
  compiler instrumentation.
- Consolidating integration-test harnesses was not attempted: the supplied
  study found a small gain relative to the loss of process isolation and
  individually addressable test targets.
- Nested Cargo environment defaults were not added because the repository has
  no independent fixture workspace or trybuild-style Cargo invocation that
  needs them today.
- Adding an artifact-sweeping dependency was unnecessary for the initial
  policy; explicit periodic Cargo cleanup has no new supply-chain or bootstrap
  cost.

## Verification

Cargo metadata/profile parsing, workflow YAML, shell syntax, formatting, and the
deterministic repository-rules review passed. The repository-rules review
script's 89 self-tests also passed.

A fresh-target comparison compiled the `xtask` test harness with four Cargo
jobs, incremental compilation disabled, and no rustc wrapper. The ordinary
release target occupied 64,752 KiB; `profile.ci` occupied 62,032 KiB, a 2,720
KiB (4.20%) reduction. This small representative build validates that symbol
stripping is active; it is not presented as a full-workspace estimate. Both
temporary targets and the generated ignored lockfile were removed afterward.

## Remaining risks

Symbol stripping changes diagnostic richness for CI executables but not runtime
semantics. The full-debug `release-debug` profile remains available for
source-level debugging. Existing historical artifacts remain until explicitly
cleaned.
