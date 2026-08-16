# Core Index-Key Encoder Implementation Plan (issue #628)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, reusable mixed-radix integer key encoder to `tensor4all-core`, with fixed-width fast paths, a limb-backed dynamic width beyond 1024 bits, and the benchmark matrix required by issue #628.

**Architecture:** Keys are **bit-packed**, not mixed-radix: dimension `i` occupies `ceil(log2(d_i))` bits at a fixed offset, so encoding is shift-and-OR with no multiplication, tree composition is append at a known bit offset, and injectivity is structural. Values live in an `IndexKey` enum whose large arms are boxed so a `u64` key does not pay `U1024`'s footprint. Beyond 1024 bits the encoder falls back to a `SmallVec` limb representation with the same append semantics.

**Tech Stack:** Rust, `tensor4all-core`, `bnum` (fixed widths above 128 bits), `smallvec` (dynamic limbs), `thiserror`, `criterion` 0.5.

## Global Constraints

- **Branch:** work on `codex/treeaci`, alongside the TreeACI work. A single person is driving both, `#628 -> #626 -> TreeACI guard cache` is one dependency chain, and a broken branch can be rebuilt from history. Split into separate PRs at submission time instead, so the core public-API change can be reviewed on its own.
- **Drift:** `tensor4all-core` is under active change on `main` (three merges in two days, including the 48-file tenferro session migration). Merge `origin/main` before starting and again before the Task 7 gate, and re-run the gate after each merge.
- Issue #628's decisions, verbatim in effect:
  - the new encoder goes in `tensor4all-core`;
  - `simplett`'s private copy is **not** migrated (simplett is expected to be deprecated; the legacy copy stays);
  - `tcicore` keeps its independent implementation — the duplication is accepted rather than introducing a dependency cycle or making TreeTN depend on a TCI algorithm crate;
  - construction and encoding must be checked: dimension/coefficient arithmetic, index length, and per-dimension bounds "must not silently wrap, saturate, or substitute zero";
  - no artificial 1024-bit limit: keep fixed-width fast paths (`u64`, `u128`, `U256`, `U512`, `U1024`) and fall back above that to a dynamically sized limb-backed key, potentially `SmallVec`;
  - the enum layout "should not inflate every fixed-width key merely because the enum has a large inline variant".
- Every public item needs rustdoc with a runnable, asserted `# Examples` block. `ignore` and `no_run` doctest fences are prohibited.
- No `unwrap()`/`expect()` in library code. Test and bench code may use them.
- Run `cargo fmt --all` before every commit; `cargo clippy -p tensor4all-core --all-targets -- -D warnings` must be clean.
- **Build hygiene:** always `--release`, never debug. The two profiles keep separate artifact sets, so a debug run doubles the footprint for no benefit — and `AGENTS.md` mandates `--release` anyway. Iterate with `cargo test --release -p tensor4all-core <filter>`; run the full gate once at Task 7. The first build is cold (10+ minutes, tenferro is fetched and built from git); every later one is incremental. Keep the artifacts for the whole of #628 and `cargo clean` once at Task 7, rather than cleaning per task and paying the cold cost repeatedly. Run at most one build at a time — concurrent cargo invocations serialize on the lock and just duplicate work.

## Design decision: bit-packing, not mixed-radix

Both existing copies compute a width bound of `total_bits = sum(ceil(log2(d_i)))` and then perform *mixed-radix* encoding (`key = sum(value_i * coeff_i)`, `coeff_i = prod(d_j, j<i)`). Mixed-radix needs only `ceil(log2(prod d_i))` bits, which is `<=` that bound, so the existing width check is already sized for bit-packing.

Bit-packing wins here on all three axes #628 cares about:

| | mixed-radix | bit-packed |
|---|---|---|
| Encode | multiply + add per dimension | shift + OR per dimension |
| Overflow risk in encode | real — this is where both copies are unsafe | **none**: no arithmetic that can carry out of the reserved field |
| Tree composition | `parent = local + local_space * child` → general multiprecision multiply, quadratic in limbs | append at a known bit offset → linear in limbs |
| Width | `ceil(log2(prod d_i))` | `sum(ceil(log2(d_i)))`, equal to what the ladder already computes |
| Injectivity | needs the coefficient argument | structural: disjoint bit fields |

The one cost is key density: bit-packed keys are sparse relative to mixed-radix (for example three dimensions of `d = 3` use 6 bits rather than 5). That is irrelevant for hashing, which is the only consumer.

This is the "canonical shift/append or equivalent representation that preserves injectivity" #628 asks for when multiplication scales poorly. If a reviewer wants dense keys instead, the fallback is mixed-radix with a checked multiply-add — but then Task 5's composition cost changes from linear to quadratic in limbs, and Task 6's benchmark must show that is acceptable.

## Width convention

`ceil(log2(d))` for `d >= 1`, computed as `usize::BITS - (d - 1).leading_zeros()` for `d >= 2` and `0` for `d == 1`. A dimension of `0` is rejected.

Note the existing implementations disagree, and this plan follows tcicore's:

- `tcicore/src/cached_function/mod.rs` `total_bits`: `((d - 1) as u64).ilog2() + 1` — correct (`d = 4` gives 2 bits).
- `simplett/src/cache.rs` `compute_total_bits`: `(d as u64).ilog2() + 1` — over-estimates by one at every power of two (`d = 4` gives 3 bits). Safe but wasteful; it selects wider keys than necessary.

## File Structure

**Created:**
- `crates/tensor4all-core/src/index_key/mod.rs` — public surface: `IndexKey`, `FlatIndexer`, `KeyBuilder`, `IndexKeyError`, and the width helper. Re-exported from `lib.rs`.
- `crates/tensor4all-core/src/index_key/fixed.rs` — the fixed-width arms and their shift/OR primitives.
- `crates/tensor4all-core/src/index_key/dynamic.rs` — the limb-backed arm.
- `crates/tensor4all-core/src/index_key/tests/mod.rs` — unit tests.
- `crates/tensor4all-core/benches/index_key.rs` — the #628 benchmark matrix.

**Modified:**
- `crates/tensor4all-core/Cargo.toml` — add `bnum.workspace = true`, add `criterion.workspace = true` to dev-dependencies, add the `[[bench]]` entry. `smallvec` is already a dependency.
- `crates/tensor4all-core/src/lib.rs` — `pub mod index_key;` plus the re-exports.

**Deliberately untouched:** `tensor4all-simplett`, `tensor4all-tcicore`. Both keep their existing copies per #628.

---

## Task 1: Width computation and error type

**Files:**
- Create: `crates/tensor4all-core/src/index_key/mod.rs`
- Create: `crates/tensor4all-core/src/index_key/tests/mod.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub enum IndexKeyError` with variants `ZeroDimension { position: usize }`, `LengthMismatch { expected: usize, actual: usize }`, `IndexOutOfRange { position: usize, value: usize, dim: usize }`, `WidthOverflow { requested_bits: u64 }`
  - `pub fn dimension_bits(dim: usize) -> Result<u32, IndexKeyError>` — `ceil(log2(dim))`, `0` for `dim == 1`, error for `dim == 0`
  - `pub fn total_bits(local_dims: &[usize]) -> Result<u64, IndexKeyError>` — checked sum

- [ ] **Step 1: Write the failing test**

Create `crates/tensor4all-core/src/index_key/tests/mod.rs`:

```rust
use super::*;

#[test]
fn dimension_bits_is_ceil_log2() {
    assert_eq!(dimension_bits(1).unwrap(), 0);
    assert_eq!(dimension_bits(2).unwrap(), 1);
    assert_eq!(dimension_bits(3).unwrap(), 2);
    assert_eq!(dimension_bits(4).unwrap(), 2);
    assert_eq!(dimension_bits(5).unwrap(), 3);
    assert_eq!(dimension_bits(255).unwrap(), 8);
    assert_eq!(dimension_bits(256).unwrap(), 8);
    assert_eq!(dimension_bits(257).unwrap(), 9);
}

#[test]
fn dimension_zero_is_rejected() {
    assert!(matches!(
        dimension_bits(0),
        Err(IndexKeyError::ZeroDimension { position: 0 })
    ));
}

#[test]
fn total_bits_sums_and_reports_the_offending_position() {
    assert_eq!(total_bits(&[2, 2, 2]).unwrap(), 3);
    assert_eq!(total_bits(&[4, 3, 1]).unwrap(), 4);
    assert_eq!(total_bits(&[]).unwrap(), 0);
    assert!(matches!(
        total_bits(&[2, 0, 2]),
        Err(IndexKeyError::ZeroDimension { position: 1 })
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-core index_key::`
Expected: FAIL — `file not found for module index_key` or `cannot find function dimension_bits`.

- [ ] **Step 3: Write minimal implementation**

Create `crates/tensor4all-core/src/index_key/mod.rs`:

```rust
//! Bit-packed integer keys for multi-index maps.
//!
//! A [`FlatIndexer`] turns a multi-index over fixed local dimensions into a
//! single integer [`IndexKey`] suitable for hashing. Dimension `i` occupies
//! `ceil(log2(d_i))` bits at a fixed offset, so encoding is shift-and-OR with
//! no multiplication, and two multi-indices collide only if they are equal.
//!
//! Widths up to 1024 bits use fixed-width fast paths; wider index spaces fall
//! back to a limb-backed representation with the same semantics.

use thiserror::Error;

/// Failures from index-key construction and encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum IndexKeyError {
    /// A local dimension was zero, so the index space is empty.
    #[error("local dimension at position {position} is zero")]
    ZeroDimension {
        /// Position of the offending dimension.
        position: usize,
    },
    /// A multi-index had the wrong number of components.
    #[error("expected a multi-index of length {expected}, got {actual}")]
    LengthMismatch {
        /// Number of local dimensions the indexer was built with.
        expected: usize,
        /// Number of components supplied.
        actual: usize,
    },
    /// A component was not less than its local dimension.
    #[error("index {value} at position {position} is not below dimension {dim}")]
    IndexOutOfRange {
        /// Position of the offending component.
        position: usize,
        /// The supplied value.
        value: usize,
        /// The local dimension at that position.
        dim: usize,
    },
    /// The requested key width exceeds what this build can represent.
    #[error("requested key width {requested_bits} bits is too large")]
    WidthOverflow {
        /// Total bits requested.
        requested_bits: u64,
    },
}

/// Number of bits needed to represent the values `0..dim`.
///
/// # Errors
///
/// Returns [`IndexKeyError::ZeroDimension`] when `dim` is zero.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::dimension_bits;
/// assert_eq!(dimension_bits(1).unwrap(), 0);
/// assert_eq!(dimension_bits(4).unwrap(), 2);
/// assert_eq!(dimension_bits(5).unwrap(), 3);
/// assert!(dimension_bits(0).is_err());
/// ```
pub fn dimension_bits(dim: usize) -> Result<u32, IndexKeyError> {
    match dim {
        0 => Err(IndexKeyError::ZeroDimension { position: 0 }),
        1 => Ok(0),
        _ => Ok(usize::BITS - (dim - 1).leading_zeros()),
    }
}

/// Total bit width of the bit-packed key for `local_dims`.
///
/// # Errors
///
/// Returns [`IndexKeyError::ZeroDimension`] naming the first zero dimension.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::total_bits;
/// assert_eq!(total_bits(&[2, 2, 2]).unwrap(), 3);
/// assert_eq!(total_bits(&[4, 3, 1]).unwrap(), 4);
/// assert_eq!(total_bits(&[]).unwrap(), 0);
/// assert!(total_bits(&[2, 0]).is_err());
/// ```
pub fn total_bits(local_dims: &[usize]) -> Result<u64, IndexKeyError> {
    let mut sum = 0u64;
    for (position, &dim) in local_dims.iter().enumerate() {
        let bits = dimension_bits(dim)
            .map_err(|_| IndexKeyError::ZeroDimension { position })?;
        sum = sum
            .checked_add(u64::from(bits))
            .ok_or(IndexKeyError::WidthOverflow { requested_bits: u64::MAX })?;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests;
```

In `crates/tensor4all-core/src/lib.rs`, add next to the other `pub mod` lines:

```rust
pub mod index_key;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p tensor4all-core index_key::`
Expected: PASS, three tests.

- [ ] **Step 5: Run the doctests**

Run: `cargo test -p tensor4all-core --doc index_key`
Expected: PASS, two doctests.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/src/index_key crates/tensor4all-core/src/lib.rs
git commit -m "feat(core): add index-key width computation and error type"
```

---

## Task 2: Fixed-width encoder for u64 and u128

**Files:**
- Create: `crates/tensor4all-core/src/index_key/fixed.rs`
- Modify: `crates/tensor4all-core/src/index_key/mod.rs`
- Test: `crates/tensor4all-core/src/index_key/tests/mod.rs`

**Interfaces:**
- Consumes: `dimension_bits`, `total_bits`, `IndexKeyError` from Task 1.
- Produces:
  - `pub enum IndexKey { U64(u64), U128(u128) }` — extended in Task 3, so match arms elsewhere must not assume exhaustiveness beyond what exists
  - `pub struct FlatIndexer` with private fields
  - `FlatIndexer::try_new(local_dims: &[usize]) -> Result<FlatIndexer, IndexKeyError>`
  - `FlatIndexer::encode(&self, idx: &[usize]) -> Result<IndexKey, IndexKeyError>`
  - `FlatIndexer::width_bits(&self) -> u64`
  - `FlatIndexer::len(&self) -> usize` and `FlatIndexer::is_empty(&self) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `crates/tensor4all-core/src/index_key/tests/mod.rs`:

```rust
#[test]
fn encoding_is_injective_over_a_small_space() {
    let dims = [3usize, 4, 2];
    let indexer = FlatIndexer::try_new(&dims).unwrap();
    assert_eq!(indexer.width_bits(), 2 + 2 + 1);

    let mut seen = std::collections::HashSet::new();
    for a in 0..dims[0] {
        for b in 0..dims[1] {
            for c in 0..dims[2] {
                let key = indexer.encode(&[a, b, c]).unwrap();
                assert!(seen.insert(key), "collision at {a},{b},{c}");
            }
        }
    }
    assert_eq!(seen.len(), dims[0] * dims[1] * dims[2]);
}

#[test]
fn encoding_rejects_bad_input_instead_of_wrapping() {
    let indexer = FlatIndexer::try_new(&[3, 4]).unwrap();
    assert!(matches!(
        indexer.encode(&[0]),
        Err(IndexKeyError::LengthMismatch { expected: 2, actual: 1 })
    ));
    assert!(matches!(
        indexer.encode(&[3, 0]),
        Err(IndexKeyError::IndexOutOfRange { position: 0, value: 3, dim: 3 })
    ));
    assert!(matches!(
        indexer.encode(&[0, 4]),
        Err(IndexKeyError::IndexOutOfRange { position: 1, value: 4, dim: 4 })
    ));
}

#[test]
fn width_selects_u64_then_u128() {
    let narrow = FlatIndexer::try_new(&[2; 64]).unwrap();
    assert!(matches!(narrow.encode(&[0; 64]).unwrap(), IndexKey::U64(_)));
    let wide = FlatIndexer::try_new(&[2; 65]).unwrap();
    assert!(matches!(wide.encode(&[0; 65]).unwrap(), IndexKey::U128(_)));
}

#[test]
fn an_empty_index_space_encodes_to_zero() {
    let indexer = FlatIndexer::try_new(&[]).unwrap();
    assert_eq!(indexer.width_bits(), 0);
    assert_eq!(indexer.encode(&[]).unwrap(), IndexKey::U64(0));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-core index_key::tests::encoding_is_injective_over_a_small_space`
Expected: FAIL — `cannot find type FlatIndexer`.

- [ ] **Step 3: Write minimal implementation**

Create `crates/tensor4all-core/src/index_key/fixed.rs`:

```rust
//! Fixed-width bit-packed key arms.

/// Places `value` into `key` at `offset` bits. The caller has already checked
/// that `value` fits in the field, so no bits are lost.
pub(super) fn place_u64(key: u64, value: usize, offset: u32) -> u64 {
    key | ((value as u64) << offset)
}

/// Places `value` into `key` at `offset` bits.
pub(super) fn place_u128(key: u128, value: usize, offset: u32) -> u128 {
    key | ((value as u128) << offset)
}
```

In `mod.rs`, add the imports and types:

```rust
mod fixed;

/// A bit-packed multi-index key.
///
/// Arms wider than 128 bits are boxed so that a narrow key does not pay the
/// footprint of the widest arm.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::{FlatIndexer, IndexKey};
/// let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
/// assert_eq!(indexer.encode(&[1, 0]).unwrap(), IndexKey::U64(1));
/// assert_eq!(indexer.encode(&[0, 1]).unwrap(), IndexKey::U64(2));
/// assert_ne!(
///     indexer.encode(&[1, 0]).unwrap(),
///     indexer.encode(&[0, 1]).unwrap()
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexKey {
    /// Keys up to 64 bits.
    U64(u64),
    /// Keys up to 128 bits.
    U128(u128),
}

/// Encodes multi-indices over fixed local dimensions as [`IndexKey`] values.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::FlatIndexer;
/// let indexer = FlatIndexer::try_new(&[3, 4]).unwrap();
/// assert_eq!(indexer.width_bits(), 4);
/// assert_eq!(indexer.len(), 2);
/// assert!(indexer.encode(&[2, 3]).is_ok());
/// assert!(indexer.encode(&[3, 0]).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct FlatIndexer {
    dims: Vec<usize>,
    offsets: Vec<u32>,
    width_bits: u64,
}

impl FlatIndexer {
    /// Builds an indexer for `local_dims`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::ZeroDimension`] when any dimension is zero and
    /// [`IndexKeyError::WidthOverflow`] when the packed width does not fit this
    /// build's key representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::FlatIndexer;
    /// assert!(FlatIndexer::try_new(&[2, 3, 4]).is_ok());
    /// assert!(FlatIndexer::try_new(&[2, 0]).is_err());
    /// ```
    pub fn try_new(local_dims: &[usize]) -> Result<Self, IndexKeyError> {
        let mut offsets = Vec::with_capacity(local_dims.len());
        let mut width_bits = 0u64;
        for (position, &dim) in local_dims.iter().enumerate() {
            let bits = dimension_bits(dim)
                .map_err(|_| IndexKeyError::ZeroDimension { position })?;
            let offset = u32::try_from(width_bits)
                .map_err(|_| IndexKeyError::WidthOverflow { requested_bits: width_bits })?;
            offsets.push(offset);
            width_bits = width_bits
                .checked_add(u64::from(bits))
                .ok_or(IndexKeyError::WidthOverflow { requested_bits: u64::MAX })?;
        }
        if width_bits > 128 {
            return Err(IndexKeyError::WidthOverflow { requested_bits: width_bits });
        }
        Ok(Self { dims: local_dims.to_vec(), offsets, width_bits })
    }

    /// Total packed width in bits.
    pub fn width_bits(&self) -> u64 {
        self.width_bits
    }

    /// Number of local dimensions.
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    /// Whether the indexer has no dimensions.
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Encodes a multi-index.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::LengthMismatch`] when `idx` has the wrong
    /// length and [`IndexKeyError::IndexOutOfRange`] when a component is not
    /// below its dimension. No input produces a silently wrapped key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::FlatIndexer;
    /// let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    /// assert!(indexer.encode(&[1, 1]).is_ok());
    /// assert!(indexer.encode(&[2, 0]).is_err());
    /// assert!(indexer.encode(&[0]).is_err());
    /// ```
    pub fn encode(&self, idx: &[usize]) -> Result<IndexKey, IndexKeyError> {
        if idx.len() != self.dims.len() {
            return Err(IndexKeyError::LengthMismatch {
                expected: self.dims.len(),
                actual: idx.len(),
            });
        }
        for (position, (&value, &dim)) in idx.iter().zip(&self.dims).enumerate() {
            if value >= dim {
                return Err(IndexKeyError::IndexOutOfRange { position, value, dim });
            }
        }
        if self.width_bits <= 64 {
            let mut key = 0u64;
            for ((&value, &offset), _) in idx.iter().zip(&self.offsets).zip(&self.dims) {
                key = fixed::place_u64(key, value, offset);
            }
            Ok(IndexKey::U64(key))
        } else {
            let mut key = 0u128;
            for ((&value, &offset), _) in idx.iter().zip(&self.offsets).zip(&self.dims) {
                key = fixed::place_u128(key, value, offset);
            }
            Ok(IndexKey::U128(key))
        }
    }
}
```

Add `dimension_bits` to the `use` list if the module split requires it; in this layout both live in `mod.rs`, so no import is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tensor4all-core index_key`
Expected: PASS, all unit tests and doctests.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-core --all-targets -- -D warnings
git add crates/tensor4all-core/src/index_key
git commit -m "feat(core): add checked bit-packed encoder for 64- and 128-bit keys"
```

---

## Task 3: Extend to boxed 256/512/1024-bit arms

**Files:**
- Modify: `crates/tensor4all-core/src/index_key/mod.rs`, `crates/tensor4all-core/src/index_key/fixed.rs`
- Modify: `crates/tensor4all-core/Cargo.toml`
- Test: `crates/tensor4all-core/src/index_key/tests/mod.rs`

**Interfaces:**
- Consumes: `IndexKey`, `FlatIndexer` from Task 2.
- Produces: `IndexKey` gains `U256(Box<U256>)`, `U512(Box<U512>)`, `U1024(Box<U1024>)`. `FlatIndexer::try_new` accepts widths up to 1024.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn wide_arms_are_selected_by_width() {
    for (bits, want_u256, want_u512, want_u1024) in [
        (129usize, true, false, false),
        (257, false, true, false),
        (513, false, false, true),
    ] {
        let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
        let key = indexer.encode(&vec![0usize; bits]).unwrap();
        assert_eq!(matches!(key, IndexKey::U256(_)), want_u256, "{bits} bits");
        assert_eq!(matches!(key, IndexKey::U512(_)), want_u512, "{bits} bits");
        assert_eq!(matches!(key, IndexKey::U1024(_)), want_u1024, "{bits} bits");
    }
}

/// #628: "the exact layout should not inflate every fixed-width key merely
/// because the enum has a large inline variant".
///
/// The bound is 24 bytes while the widest arm is `u128`; Task 4 raises it to
/// 40 when the `SmallVec` limb arm lands, which is the last time it may move.
/// A `U1024` inlined here would be 128 bytes on its own.
#[test]
fn the_key_enum_stays_small() {
    assert!(
        std::mem::size_of::<IndexKey>() <= 24,
        "IndexKey is {} bytes; wide arms must be boxed",
        std::mem::size_of::<IndexKey>()
    );
}

#[test]
fn wide_encoding_is_injective_on_the_high_bits() {
    let bits = 300usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    let mut low = vec![0usize; bits];
    low[0] = 1;
    let mut high = vec![0usize; bits];
    high[bits - 1] = 1;
    assert_ne!(
        indexer.encode(&low).unwrap(),
        indexer.encode(&high).unwrap()
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-core index_key::tests::wide_arms_are_selected_by_width`
Expected: FAIL — `no variant named U256`.

- [ ] **Step 3: Add the dependency**

In `crates/tensor4all-core/Cargo.toml`, under `[dependencies]`:

```toml
bnum.workspace = true
```

- [ ] **Step 4: Write minimal implementation**

In `fixed.rs`, add the bnum placement helpers:

```rust
use bnum::types::{U1024, U256, U512};

macro_rules! place_bnum {
    ($name:ident, $ty:ty) => {
        pub(super) fn $name(key: $ty, value: usize, offset: u32) -> $ty {
            key | (<$ty>::from(value as u64) << offset)
        }
    };
}

place_bnum!(place_u256, U256);
place_bnum!(place_u512, U512);
place_bnum!(place_u1024, U1024);
```

In `mod.rs`, extend the enum:

```rust
    /// Keys up to 256 bits.
    U256(Box<bnum::types::U256>),
    /// Keys up to 512 bits.
    U512(Box<bnum::types::U512>),
    /// Keys up to 1024 bits.
    U1024(Box<bnum::types::U1024>),
```

Raise the `try_new` width cap from `128` to `1024`, and extend `encode`'s width dispatch with three more arms following the same shape as the `u128` arm, each folding with the matching `place_*` helper and wrapping the result in `Box::new`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p tensor4all-core index_key`
Expected: PASS. If `the_key_enum_stays_small` fails, an arm is unboxed — box it rather than relaxing the bound.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-core --all-targets -- -D warnings
git add crates/tensor4all-core
git commit -m "feat(core): add boxed 256/512/1024-bit index-key arms"
```

---

## Task 4: Dynamic limb-backed arm beyond 1024 bits

**Files:**
- Create: `crates/tensor4all-core/src/index_key/dynamic.rs`
- Modify: `crates/tensor4all-core/src/index_key/mod.rs`
- Test: `crates/tensor4all-core/src/index_key/tests/mod.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `IndexKey::Limbs(SmallVec<[u64; 2]>)`; `FlatIndexer::try_new` no longer rejects widths above 1024.

The limb representation is little-endian `u64` words. Placing a value at a bit offset writes into at most two adjacent limbs, so encoding stays linear in the number of dimensions and the key is linear in the number of limbs.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn widths_beyond_1024_bits_use_limbs_and_stay_injective() {
    let bits = 2048usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    assert_eq!(indexer.width_bits(), bits as u64);

    let zero = indexer.encode(&vec![0usize; bits]).unwrap();
    assert!(matches!(zero, IndexKey::Limbs(_)));

    let mut seen = std::collections::HashSet::new();
    seen.insert(zero);
    for position in [0usize, 63, 64, 65, 1023, 1024, 1025, 2047] {
        let mut idx = vec![0usize; bits];
        idx[position] = 1;
        let key = indexer.encode(&idx).unwrap();
        assert!(seen.insert(key), "collision with a single bit set at {position}");
    }
}

#[test]
fn a_value_straddling_a_limb_boundary_round_trips() {
    // A radix-64 dimension needs 6 bits; placing several of them puts one
    // value across the 64-bit limb boundary.
    let dims = vec![64usize; 40];
    let indexer = FlatIndexer::try_new(&dims).unwrap();
    let mut a = vec![0usize; 40];
    let mut b = vec![0usize; 40];
    a[10] = 63;
    b[10] = 62;
    assert_ne!(indexer.encode(&a).unwrap(), indexer.encode(&b).unwrap());
}

#[test]
fn dynamic_keys_do_not_inflate_the_enum() {
    assert!(
        std::mem::size_of::<IndexKey>() <= 40,
        "IndexKey is {} bytes",
        std::mem::size_of::<IndexKey>()
    );
}
```

Also update `the_key_enum_stays_small` from Task 3 to the same bound of 40 and delete the duplicate, keeping one assertion. `SmallVec<[u64; 2]>` is 32 bytes on its own, so the 24-byte bound from Task 3 cannot survive this task; 40 is the final bound and must not be relaxed again. If it is exceeded, box the limb arm rather than raising the number.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-core index_key::tests::widths_beyond_1024_bits_use_limbs_and_stay_injective`
Expected: FAIL — `try_new` returns `WidthOverflow`.

- [ ] **Step 3: Write minimal implementation**

Create `crates/tensor4all-core/src/index_key/dynamic.rs`:

```rust
//! Limb-backed keys for index spaces wider than the fixed-width arms.

use smallvec::SmallVec;

/// Little-endian `u64` limbs.
pub(super) type Limbs = SmallVec<[u64; 2]>;

/// Number of limbs needed for `width_bits`.
pub(super) fn limb_count(width_bits: u64) -> usize {
    ((width_bits + 63) / 64) as usize
}

/// Writes `value` into `limbs` starting at bit `offset`.
///
/// The caller has checked that `value` fits its field, so no bits are lost.
/// A field may straddle at most one limb boundary because a single local
/// dimension never exceeds 64 bits.
pub(super) fn place(limbs: &mut Limbs, value: usize, offset: u64) {
    let word = (offset / 64) as usize;
    let shift = (offset % 64) as u32;
    let value = value as u64;
    limbs[word] |= value << shift;
    if shift != 0 {
        let carry_bits = 64 - shift;
        if (value >> carry_bits) != 0 {
            limbs[word + 1] |= value >> carry_bits;
        }
    }
}
```

In `mod.rs`: add `mod dynamic;`, add the enum arm

```rust
    /// Keys wider than 1024 bits, as little-endian 64-bit limbs.
    Limbs(dynamic::Limbs),
```

change `try_new` to stop rejecting widths above 1024 (keep `WidthOverflow` only for the `u32` offset conversion and the `u64` sum), and add the final `encode` arm: allocate `dynamic::Limbs` of `dynamic::limb_count(self.width_bits)` zeros, call `dynamic::place` per dimension using the 64-bit offset, and return `IndexKey::Limbs(limbs)`.

`FlatIndexer::offsets` must become `Vec<u64>` so offsets past 4 GiB of bits are representable; adjust the fixed-width arms to cast down with `u32::try_from(offset)` inside the `<= 1024` branches, where it cannot fail.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tensor4all-core index_key`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-core --all-targets -- -D warnings
git add crates/tensor4all-core
git commit -m "feat(core): add limb-backed dynamic-width index keys"
```

---

## Task 5: Tree-key composition

**Files:**
- Modify: `crates/tensor4all-core/src/index_key/mod.rs`
- Test: `crates/tensor4all-core/src/index_key/tests/mod.rs`

**Interfaces:**
- Consumes: `IndexKey`, `IndexKeyError`.
- Produces:
  - `pub struct KeyBuilder`
  - `KeyBuilder::with_capacity_bits(width_bits: u64) -> Result<KeyBuilder, IndexKeyError>`
  - `KeyBuilder::push(&mut self, key: &IndexKey, key_width_bits: u64) -> Result<(), IndexKeyError>`
  - `KeyBuilder::finish(self) -> IndexKey`
  - `KeyBuilder::width_bits(&self) -> u64`

This is what #626 needs for `key(node) = local_key ++ key(c1) ++ key(c2) ++ ...`. Appending at a known bit offset keeps composition linear in the number of limbs, which is the property #628 asks for when it says to prefer "a canonical shift/append or equivalent representation that preserves injectivity".

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn composition_matches_encoding_the_concatenated_multi_index() {
    let local = FlatIndexer::try_new(&[3, 2]).unwrap();
    let child = FlatIndexer::try_new(&[4, 5]).unwrap();
    let whole = FlatIndexer::try_new(&[3, 2, 4, 5]).unwrap();

    for a in 0..3 {
        for b in 0..2 {
            for c in 0..4 {
                for d in 0..5 {
                    let mut builder =
                        KeyBuilder::with_capacity_bits(whole.width_bits()).unwrap();
                    builder
                        .push(&local.encode(&[a, b]).unwrap(), local.width_bits())
                        .unwrap();
                    builder
                        .push(&child.encode(&[c, d]).unwrap(), child.width_bits())
                        .unwrap();
                    assert_eq!(
                        builder.finish(),
                        whole.encode(&[a, b, c, d]).unwrap(),
                        "composition disagreed at {a},{b},{c},{d}"
                    );
                }
            }
        }
    }
}

#[test]
fn composition_is_injective_across_the_limb_boundary() {
    let child = FlatIndexer::try_new(&[2; 40]).unwrap();
    let mut first = KeyBuilder::with_capacity_bits(80).unwrap();
    first.push(&child.encode(&vec![0; 40]).unwrap(), 40).unwrap();
    first.push(&child.encode(&{ let mut v = vec![0; 40]; v[0] = 1; v }).unwrap(), 40).unwrap();

    let mut second = KeyBuilder::with_capacity_bits(80).unwrap();
    second.push(&child.encode(&{ let mut v = vec![0; 40]; v[0] = 1; v }).unwrap(), 40).unwrap();
    second.push(&child.encode(&vec![0; 40]).unwrap(), 40).unwrap();

    assert_ne!(first.finish(), second.finish());
}

#[test]
fn pushing_past_the_declared_capacity_is_an_error() {
    let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    let mut builder = KeyBuilder::with_capacity_bits(2).unwrap();
    builder.push(&indexer.encode(&[1, 1]).unwrap(), 2).unwrap();
    assert!(matches!(
        builder.push(&indexer.encode(&[1, 1]).unwrap(), 2),
        Err(IndexKeyError::WidthOverflow { .. })
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tensor4all-core index_key::tests::composition_matches_encoding_the_concatenated_multi_index`
Expected: FAIL — `cannot find type KeyBuilder`.

- [ ] **Step 3: Write minimal implementation**

Add to `mod.rs` a `KeyBuilder` that holds `limbs: dynamic::Limbs`, `offset: u64`, and `capacity_bits: u64`. `with_capacity_bits` allocates `dynamic::limb_count(width_bits)` zero limbs. `push` rejects with `WidthOverflow` when `offset + key_width_bits > capacity_bits`, otherwise ORs the pushed key's limbs into `self.limbs` starting at `self.offset` — reading the pushed key's limbs uniformly by converting each arm to limbs — then advances `offset`. `finish` narrows back to the smallest arm that fits `capacity_bits`, so a composed key compares equal to the same value produced by `FlatIndexer::encode`.

Narrowing on `finish` is what makes the first test pass: `encode` on a 9-bit space yields `IndexKey::U64`, so composition must too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tensor4all-core index_key`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-core --all-targets -- -D warnings
git add crates/tensor4all-core
git commit -m "feat(core): add KeyBuilder for append-style tree key composition"
```

---

## Task 6: The #628 benchmark matrix

**Files:**
- Create: `crates/tensor4all-core/benches/index_key.rs`
- Modify: `crates/tensor4all-core/Cargo.toml`

**Interfaces:**
- Consumes: `FlatIndexer`, `IndexKey`, `KeyBuilder`.
- Produces: measurements, and the numbers to paste into #628.

#628 requires benchmarking: indexer construction and encoding; tree-key composition; hashing; `HashMap` hit, miss, and insertion; at widths around 64, 128, 256, 512, 1024, **1025**, 2048, and 4096 bits; with both many binary dimensions and fewer large-radix dimensions.

- [ ] **Step 1: Register the bench**

In `crates/tensor4all-core/Cargo.toml`, add to `[dev-dependencies]`:

```toml
criterion.workspace = true
```

and at the end of the file:

```toml
[[bench]]
name = "index_key"
harness = false
```

- [ ] **Step 2: Write the benchmark**

Create `crates/tensor4all-core/benches/index_key.rs`:

```rust
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher, RandomState};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_core::index_key::{FlatIndexer, IndexKey, KeyBuilder};

/// Target widths in bits, straddling every fixed/dynamic boundary.
const WIDTHS: [u64; 8] = [64, 128, 256, 512, 1024, 1025, 2048, 4096];

/// `(label, dims)` for a target width: many binary dimensions, or fewer
/// large-radix ones. Radix 256 uses 8 bits per dimension.
fn profiles(width: u64) -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("binary", vec![2usize; width as usize]),
        ("radix256", vec![256usize; (width as usize).div_ceil(8)]),
    ]
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/construct");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            group.bench_with_input(
                BenchmarkId::new(label, width),
                &dims,
                |b, dims| b.iter(|| FlatIndexer::try_new(black_box(dims)).unwrap()),
            );
        }
    }
    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/encode");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let idx: Vec<usize> = dims.iter().map(|d| d - 1).collect();
            group.bench_with_input(
                BenchmarkId::new(label, width),
                &idx,
                |b, idx| b.iter(|| indexer.encode(black_box(idx)).unwrap()),
            );
        }
    }
    group.finish();
}

fn bench_compose(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/compose");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            // Split into a local part and three children, as a degree-3 tree node.
            let chunk = dims.len().div_ceil(4).max(1);
            let parts: Vec<Vec<usize>> = dims.chunks(chunk).map(<[usize]>::to_vec).collect();
            let indexers: Vec<FlatIndexer> = parts
                .iter()
                .map(|p| FlatIndexer::try_new(p).unwrap())
                .collect();
            let keys: Vec<(IndexKey, u64)> = indexers
                .iter()
                .zip(&parts)
                .map(|(ix, p)| {
                    let idx: Vec<usize> = p.iter().map(|d| d - 1).collect();
                    (ix.encode(&idx).unwrap(), ix.width_bits())
                })
                .collect();
            let total: u64 = keys.iter().map(|(_, w)| w).sum();
            group.bench_with_input(BenchmarkId::new(label, width), &keys, |b, keys| {
                b.iter(|| {
                    let mut builder = KeyBuilder::with_capacity_bits(total).unwrap();
                    for (key, w) in keys {
                        builder.push(black_box(key), *w).unwrap();
                    }
                    builder.finish()
                })
            });
        }
    }
    group.finish();
}

fn bench_hash(c: &mut Criterion) {
    let state = RandomState::new();
    let mut group = c.benchmark_group("index_key/hash");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let idx: Vec<usize> = dims.iter().map(|d| d - 1).collect();
            let key = indexer.encode(&idx).unwrap();
            group.bench_with_input(BenchmarkId::new(label, width), &key, |b, key| {
                b.iter(|| {
                    let mut hasher = state.build_hasher();
                    std::hash::Hash::hash(black_box(key), &mut hasher);
                    hasher.finish()
                })
            });
        }
    }
    group.finish();
}

fn bench_map(c: &mut Criterion) {
    const ENTRIES: usize = 1024;
    for (op, present) in [("hit", true), ("miss", false)] {
        let mut group = c.benchmark_group(format!("index_key/map_{op}"));
        for width in WIDTHS {
            for (label, dims) in profiles(width) {
                let indexer = FlatIndexer::try_new(&dims).unwrap();
                let mut map: HashMap<IndexKey, usize> = HashMap::new();
                let mut idx = vec![0usize; dims.len()];
                for entry in 0..ENTRIES {
                    idx[0] = entry % dims[0];
                    idx[dims.len() - 1] = entry / dims[0] % dims[dims.len() - 1];
                    map.insert(indexer.encode(&idx).unwrap(), entry);
                }
                let probe = if present {
                    idx[0] = 0;
                    idx[dims.len() - 1] = 0;
                    indexer.encode(&idx).unwrap()
                } else {
                    let far: Vec<usize> = dims.iter().map(|d| d - 1).collect();
                    indexer.encode(&far).unwrap()
                };
                group.bench_with_input(
                    BenchmarkId::new(label, width),
                    &probe,
                    |b, probe| b.iter(|| map.get(black_box(probe)).copied()),
                );
            }
        }
        group.finish();
    }

    let mut group = c.benchmark_group("index_key/map_insert");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let keys: Vec<IndexKey> = (0..ENTRIES)
                .map(|entry| {
                    let mut idx = vec![0usize; dims.len()];
                    idx[0] = entry % dims[0];
                    idx[dims.len() - 1] = entry / dims[0] % dims[dims.len() - 1];
                    indexer.encode(&idx).unwrap()
                })
                .collect();
            group.bench_with_input(BenchmarkId::new(label, width), &keys, |b, keys| {
                b.iter(|| {
                    let mut map: HashMap<IndexKey, usize> = HashMap::with_capacity(ENTRIES);
                    for (entry, key) in keys.iter().enumerate() {
                        map.insert(black_box(key).clone(), entry);
                    }
                    map.len()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_encode,
    bench_compose,
    bench_hash,
    bench_map
);
criterion_main!(benches);
```

- [ ] **Step 3: Run the benchmarks**

Run: `cargo bench -p tensor4all-core --bench index_key`
Expected: completes and writes `target/criterion`.

- [ ] **Step 4: Record the results**

Extract, for each group, the median time at each width and profile, into a table in the commit body and in a comment for #628. The three acceptance questions from the issue, answered with these numbers:

1. Do the fixed-width paths regress materially versus their natural cost? Compare 64/128 against 256/512/1024.
2. Is dynamic-key overhead bounded around the boundary? Compare 1024 against 1025.
3. Beyond the boundary, do encoding, hashing, and append-style composition scale approximately with the number of limbs? Compare 1025, 2048, 4096 — expect roughly linear growth.

If composition is not close to linear, the representation is wrong; report that rather than adjusting the criteria.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/benches crates/tensor4all-core/Cargo.toml
git commit -m "bench(core): add the #628 index-key benchmark matrix"
```

---

## Task 7: Final gate and issue report

**Files:**
- Modify: `crates/tensor4all-core/src/index_key/mod.rs` (module rustdoc only)

- [ ] **Step 1: Run the full gate**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release -p tensor4all-core
cargo doc -p tensor4all-core --no-deps
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
git diff --check
```

Every one must pass. Report any failure rather than working around it.

- [ ] **Step 2: Confirm nothing else moved**

```bash
git diff --stat origin/main..HEAD -- crates/tensor4all-simplett crates/tensor4all-tcicore
```

Expected: empty. #628 explicitly leaves both copies in place.

- [ ] **Step 3: Clean the build output**

```bash
cargo clean
```

Per the workspace's build-hygiene rule: a full release run leaves several GB of artifacts that nothing reclaims.

- [ ] **Step 4: Report on #628**

Post the benchmark table, state which representation was implemented (bit-packed append) and why, and note the two facts found while surveying the existing copies, since both bear on whether either should later adopt this type:

- `simplett`'s `compute_coeffs_*` uses `saturating_mul` and its `flat_index_*` multiplies unchecked, so it can silently produce a wrong key; `tcicore`'s `compute_coeffs` is checked but its `flat_index` substitutes `ZERO` on overflow.
- The two disagree on width: `simplett`'s `compute_total_bits` uses `(d).ilog2() + 1` while `tcicore`'s uses `(d - 1).ilog2() + 1`. The latter is correct; the former over-allocates one bit at every power of two.

Do not push or open a PR without explicit user approval.
