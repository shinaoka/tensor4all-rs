# Default Release Debug-Info Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ordinary tensor4all-rs release builds use `debug = 0` while retaining the existing full-debug `release-debug` profile and documenting its C API diagnostic purpose.

**Architecture:** Measure the newly merged `line-tables-only` release baseline against Cargo's `debug=0` override before editing. If disk/ELF evidence confirms the approved reduction, change the one owning workspace profile and update its diagnostic documentation; leave optimization, ABI, and runtime semantics unchanged.

**Tech Stack:** Cargo custom profiles, Rust release builds, ELF `readelf`, GitHub Actions.

## Global Constraints

- Preserve the full-debug `[profile.release-debug]` path.
- Preserve optimization, numerical behavior, ABI, feature selection, assertions, and public APIs.
- Do not touch the user's dirty checkout; work only from fresh `origin/main` isolation.
- Do not add dependencies, shared worktree targets, sccache, or artifact-GC machinery.
- Do not duplicate PR #581; limit the follow-up to the measured `line-tables-only` versus `0` decision and necessary documentation.

---

### Task 1: Capture line-table versus debug-zero evidence

**Files:**
- Produce outside repository: `/tmp/tensor4all-build-artifact-measurements/*`

**Interfaces:**
- Consumes: immutable `origin/main` including merged PR #581.
- Produces: evidence deciding whether a follow-up PR is justified.

- [ ] **Step 1: Record environment**

```bash
mkdir -p /tmp/tensor4all-build-artifact-measurements
rustc -Vv > /tmp/tensor4all-build-artifact-measurements/toolchain.txt
cargo -V >> /tmp/tensor4all-build-artifact-measurements/toolchain.txt
uname -a > /tmp/tensor4all-build-artifact-measurements/system.txt
```

- [ ] **Step 2: Cold-build the line-table baseline**

```bash
rm -rf /tmp/tensor4all-target-line-tables
/usr/bin/time -v -o /tmp/tensor4all-build-artifact-measurements/line-tables-time.txt \
  env CARGO_TARGET_DIR=/tmp/tensor4all-target-line-tables \
      CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  cargo test --locked --workspace --release --no-run
du -s --block-size=1 /tmp/tensor4all-target-line-tables \
  > /tmp/tensor4all-build-artifact-measurements/line-tables-du.txt
```

- [ ] **Step 3: Cold-build the debug-zero candidate without editing source**

```bash
rm -rf /tmp/tensor4all-target-debug-zero
/usr/bin/time -v -o /tmp/tensor4all-build-artifact-measurements/debug-zero-time.txt \
  env CARGO_TARGET_DIR=/tmp/tensor4all-target-debug-zero \
      CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
      CARGO_PROFILE_RELEASE_DEBUG=0 \
  cargo test --locked --workspace --release --no-run
du -s --block-size=1 /tmp/tensor4all-target-debug-zero \
  > /tmp/tensor4all-build-artifact-measurements/debug-zero-du.txt
```

- [ ] **Step 4: Compare C API and representative artifacts**

Build `tensor4all-capi` in both fresh targets if it was not included, record `libtensor4all_capi.so` sizes, inventory the 30 largest files, and compare `.debug_*` sections with `readelf -SW`.

- [ ] **Step 5: Apply the decision gate**

Proceed with Task 2 when `debug=0` deterministically removes line-table sections and yields a meaningful allocated-byte reduction without build failure. If it does not, stop the follow-up PR, document evidence, and treat PR #581 as satisfying the practical objective.

### Task 2: Add the failing profile contract

**Files:**
- Modify: `Cargo.toml:32-52`
- Add or modify the repository's existing configuration-contract test if one exists; otherwise verify through Cargo parsing and diff review rather than adding a one-off parser.

**Interfaces:**
- Produces: ordinary `release.debug = 0`; unchanged `release-debug.debug = true`.

- [ ] **Step 1: Verify the current manifest fails the desired contract**

```bash
python3 - <<'PY'
import tomllib
m = tomllib.load(open('Cargo.toml', 'rb'))
assert m['profile']['release']['debug'] == 0
assert m['profile']['release-debug']['inherits'] == 'release'
assert m['profile']['release-debug']['debug'] is True
PY
```

Expected: assertion failure because ordinary release uses `line-tables-only`.

### Task 3: Implement and document the profile change

**Files:**
- Modify: `Cargo.toml:32-52`
- Modify: `AGENTS.md` C API/debugging guidance
- Modify: `crates/tensor4all-capi/src/lib.rs:86-91`
- Create: `docs/worklogs/2026-08-09-release-debug-info-reduction.md`

**Interfaces:**
- Ordinary command: `cargo ... --release` emits no debuginfo.
- Diagnostic command: `cargo build --profile release-debug -p tensor4all-capi` emits full debuginfo.

- [ ] **Step 1: Change the owning profile**

Set `[profile.release] debug = 0`; retain `[profile.release-debug] inherits = "release"` and `debug = true`. Update the nearby comment to distinguish ordinary backtraces from full diagnostic builds.

- [ ] **Step 2: Document the C API diagnostic command**

Extend the `set_last_error` rustdoc and AGENTS C API guidance with:

```bash
RUST_BACKTRACE=1 cargo build --profile release-debug -p tensor4all-capi
```

Explain that ordinary release omits source-level debug information.

- [ ] **Step 3: Write the work log**

Record PR #581 as the immediate baseline, exact measurement environment/commands, allocated bytes and wall-time deltas, ELF evidence, approved tradeoff, compatibility behavior, rejected alternatives, verification, and residual risks.

- [ ] **Step 4: Run the profile contract**

Rerun the Python TOML assertions from Task 2. Expected: pass.

- [ ] **Step 5: Commit implementation and evidence**

```bash
git add Cargo.toml AGENTS.md crates/tensor4all-capi/src/lib.rs \
  docs/worklogs/2026-08-09-release-debug-info-reduction.md
git commit -m "build: omit line tables from ordinary release builds"
```

### Task 4: Verify ordinary and diagnostic builds

**Files:**
- Review all changed files.

- [ ] **Step 1: Verify Cargo configuration and formatting**

```bash
cargo metadata --locked --format-version 1 >/dev/null
cargo tree --workspace >/dev/null
cargo fmt --all
cargo fmt --all -- --check
```

- [ ] **Step 2: Verify ordinary C API release has no debug sections**

```bash
rm -rf /tmp/tensor4all-capi-release-zero
CARGO_TARGET_DIR=/tmp/tensor4all-capi-release-zero \
  cargo build --locked --release -p tensor4all-capi
! readelf -SW /tmp/tensor4all-capi-release-zero/release/libtensor4all_capi.so \
  | grep -q '\.debug_'
```

- [ ] **Step 3: Verify diagnostic profile retains debug sections**

```bash
rm -rf /tmp/tensor4all-capi-release-debug
CARGO_TARGET_DIR=/tmp/tensor4all-capi-release-debug \
  cargo build --locked --profile release-debug -p tensor4all-capi
readelf -SW \
  /tmp/tensor4all-capi-release-debug/release-debug/libtensor4all_capi.so \
  | grep '\.debug_'
```

- [ ] **Step 4: Run repository-required tests and docs**

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace
cargo test --doc --release --workspace
cargo doc --workspace --no-deps
./scripts/test-mdbook.sh
```

Run coverage only when required by the actual diff/PR gate; never lower thresholds.

### Task 5: Review, deliver, and babysit

**Files:**
- Review all branch changes.

- [ ] **Step 1: Run independent specification and code-quality reviews**

Use fresh read-only reviewers. Fix confirmed findings, rerun invalidated checks, and obtain clean follow-up review.

- [ ] **Step 2: Create or link the GitHub issue and create the PR**

Use issue #566/#9 and PR #581 as context; create a focused follow-up issue only if repository policy requires a separate tracker. Push the branch and create the PR with the repository workflow, including measurements and the work-log link.

- [ ] **Step 3: Enable prescribed auto-merge**

```bash
gh pr merge --auto --squash --delete-branch <PR>
gh pr view <PR> --json autoMergeRequest,mergeStateStatus,state
```

- [ ] **Step 4: Babysit to merge**

Inspect every check and failure log. Fix causes in the isolated worktree, rerun invalidated local evidence, push, and continue. Synchronize with current `origin/main` if behind and re-monitor CI. Finish only when GitHub reports `state=MERGED`; record the merged commit and final issue state.
