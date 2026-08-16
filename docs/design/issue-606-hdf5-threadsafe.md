# Design: #606 — make tensor4all-hdf5 thread-safe by construction (central lock)

Status: user-approved direction (2026-08): "Rust 側で排他するのがいい" (internal
Rust-side serialization) + "一括して管理" (one central mechanism, not ad hoc).
Review by luna (read-only): design pre-review + diff post-review.

## Problem

The HDF5 C library is not thread-safe. The hdf5-metno binding locks each raw
H5* call individually (`h5lock!` → process-wide reentrant mutex), but the
lock is released between calls, so a full save/load operation can interleave
with another thread's operation. Empirically (lingrui96, gw-rs): 2-3 of ~71
parallel tests fail with `H5Fopen(): unable to lock file, errno = 35`
(EAGAIN), and serializing calls alone does NOT cover the same-path
open-after-close lock-release window — `HDF5_USE_FILE_LOCKING=FALSE` was
needed too. Nothing in the crate docs mentions threading.

## Fix (one central mechanism)

### 1. Single choke point: `hdf5_sync` wrapper in `backend.rs`

```rust
// backend.rs
static FILE_LOCKING_INIT: std::sync::Once = std::sync::Once::new();

pub(crate) fn hdf5_sync<T>(f: impl FnOnce() -> T) -> T {
    // Disable HDF5's OS file locking once, before any HDF5 call, unless the
    // caller set HDF5_USE_FILE_LOCKING explicitly. Closes the same-path
    // lock-release window (errno 35) that serialization alone does not cover.
    FILE_LOCKING_INIT.call_once(|| {
        if std::env::var_os("HDF5_USE_FILE_LOCKING").is_none() {
            // safe: edition 2021
            std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
        }
    });
    #[cfg(all(feature = "link", not(feature = "runtime-loading")))]
    { hdf5_metno::sync::sync(f) }
    #[cfg(feature = "runtime-loading")]
    { hdf5_rt::sync::sync(f) }
}
```

- Uses the hdf5 crate's own process-wide REENTRANT lock: nested internal
  calls are safe, and the crate's operations also serialize against any
  direct hdf5-crate usage in the process.
- The lock is taken exactly once per public call and held for the WHOLE
  operation (including handle drops), which is the property the per-call
  `h5lock!` lacks.

### 2. All 9 public entry points wrap in `hdf5_sync`

`save_itensor`, `append_itensor`, `load_itensor`, `save_mps`, `append_mps`,
`load_mps`, `save_treetn`, `append_treetn`, `load_treetn` (all in lib.rs) get
their body wrapped: `hdf5_sync(|| { ... })`. Internal helper functions NEVER
lock — the public boundary is the single choke point, so future public
functions must also wrap (enforced by a doc note + the crate rule).

The crate also re-exports hdf5-rt passthrough items (`hdf5_init`,
`hdf5_is_initialized`, `hdf5_library_path`, sys symbols) for the
runtime-loading backend; those bypass the serialization guarantee and are
documented as outside it (callers who use them directly own their own
thread safety).

### 3. Documentation

- `lib.rs` module doc + crate README: thread-safe by construction — all
  public calls are serialized process-wide (I/O-bound, contention negligible);
  HDF5 OS file locking is disabled by default (`HDF5_USE_FILE_LOCKING=FALSE`),
  set once lazily unless the caller set it explicitly; the explicit-env-var
  note (callers who set it keep their value; multi-process writers should set
  it themselves); rule: every new public function must go through
  `hdf5_sync`.
- `docs/book/src/guides/hdf5-serialization.md`: same notes + the downstream
  story (why serialization alone was insufficient).

### 4. Regression test

A threaded smoke test in `crates/tensor4all-hdf5/tests/` (integration):
- N threads (e.g. 4) each do save+load of distinct temp files, asserting the
  roundtrip equals the input (mirrors the gw-rs reproducer);
- a same-path save-then-load sequence within each thread's iteration (the
  errno-35 window case). The lock serializes whole calls, so each save/load
  is atomic, but the test does NOT assume cross-thread atomicity — each
  thread owns its own path sequence.
Keep iterations modest (I/O-bound; HDF5 tests run in a dedicated CI job).

## Verified facts (origin/main)

- 9 public entry points, all in lib.rs; internal helpers (mps.rs, treetn.rs,
  itensor.rs, index.rs) are pub(crate)/private and reached only from within
  them. (hdf5-rt passthrough re-exports like `hdf5_init` are backend items,
  not serialized entry points.)
- `File::open_as`/`create` lock each raw call via `h5lock!` but not whole
  operations (hdf5-metno-0.12.6 src/hl/file.rs, macros.rs; the sys LOCK is a
  parking_lot ReentrantMutex — reentrancy makes nested `sync::sync` safe).
  hdf5-rt (git tensor4all/hdf5-rt#19fb644) has the identical sync::sync +
  ReentrantMutex LOCK design.
- No FAPL-level file-locking control is exposed by the high-level APIs
  (FileAccessBuilder has no setters) → the env var is the only central
  mechanism. Ordering guarantee: `hdf5_sync` sets HDF5_USE_FILE_LOCKING at
  its very top, BEFORE calling the backend `sync::sync` (whose first act is
  to force library init / H5open), so the variable is in place before any
  HDF5 initialization or file open the crate triggers — regardless of
  whether HDF5 parses it at library init or at first file open. The only
  way to miss it is initializing HDF5 outside the crate (re-exported
  `hdf5_init` passthrough), documented as outside the guarantee.
- Both backends (`link`/hdf5-metno, `runtime-loading`/hdf5-rt) expose
  `sync::sync` (backend.rs already uses it for `attribute_exists`).
- Workspace edition 2021 → `std::env::set_var` is safe. No other workspace
  code sets HDF5_USE_FILE_LOCKING.

## Trade-offs / risks

- Setting `HDF5_USE_FILE_LOCKING=FALSE` is a PROCESS-GLOBAL side effect: it
  affects all HDF5 usage in the process (including direct hdf5-metno/
  hdf5-rt users and other bindings), not just this crate. It weakens
  cross-PROCESS write protection for files opened with HDF5; documented.
  Users who want locking can set the env var themselves (their value wins;
  only set if unset). Explicitly, callers that set it keep their value, and
  pre-initialized HDF5 (via the re-exported `hdf5_init` passthrough) is
  outside the guarantee.
- `sync::sync` silences HDF5 stderr printing while the lock is held; the
  crate's `Hdf5Error` conversion reads the error stack and is unaffected.
- Code that uses hdf5-metno/hdf5-rt directly (not via this crate) is not
  serialized by this crate's lock; the env-var change still applies to it
  process-wide.

## Verification

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace --exclude tensor4all-hdf5` +
  `cargo test --release -p tensor4all-hdf5` (threaded test included)
- `cargo test --doc --release --workspace`, `./scripts/test-mdbook.sh`
- `cargo doc --workspace --no-deps`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree`

## Review verdicts

- Design (pre-implementation), luna: TBD.
- Diff (post-implementation), luna: TBD.
