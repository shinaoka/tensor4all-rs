> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Documentation Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure tensor4all-rs documentation around an mdBook user guide published on GitHub Pages, with slim README/crate READMEs, rustdoc examples with assertions, and removal of unmaintained Python bindings.

**Architecture:** mdBook is the central user documentation, published via GitHub Pages (CI already exists in `.github/workflows/deploy-docs.yml`). README.md becomes a minimal entry point linking to the Book. Each crate README follows a unified minimal template linking to the Book. Rustdoc `/// # Examples` with assertions serve as API reference.

**Tech Stack:** mdBook 0.5.2, GitHub Pages, Rust doctests, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-04-08-documentation-improvement-design.md`

---

## Task 1: Remove Python Bindings

**Files:**
- Delete: `python/tensor4all/` (entire directory)
- Delete: `docs/examples/python/` (entire directory)
- Delete: `scripts/run_python_tests.sh`
- Delete: `.github/workflows/CI_py.yml.disabled`

- [ ] **Step 1: Delete Python directories and scripts**

```bash
rm -rf python/tensor4all/
rm -rf docs/examples/python/
rm -f scripts/run_python_tests.sh
rm -f .github/workflows/CI_py.yml.disabled
```

- [ ] **Step 2: Check for remaining Python references**

```bash
grep -r "python" README.md .github/ scripts/ Cargo.toml docs/book/
```

Remove any remaining references found (except generic mentions in design docs).

- [ ] **Step 3: Verify build still works**

```bash
cargo build --workspace --release
```

Expected: SUCCESS (Python bindings are a separate build, not part of Cargo workspace)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unmaintained Python bindings"
```

---

## Task 2: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add Documentation Requirements section**

Add after the "Code Style" section (after line 35):

```markdown
## Documentation Requirements

Every public type, trait, and function **must** include usage examples in its
doc comments (`/// # Examples`). Examples must include assertions to verify
correctness (e.g., `assert!`, `assert_eq!`, `approx::assert_abs_diff_eq!`).
Use `ignore` attribute on examples that cannot run in doctests.
```

- [ ] **Step 2: Add `cargo doc` to Pre-PR Checks**

In the Pre-PR Checks section, add `cargo doc` to the code block (after line 123):

```bash
cargo fmt --all                        # Format all code
cargo clippy --workspace               # Check for common issues
cargo nextest run --release --workspace # Run all tests
cargo doc --workspace --no-deps        # Build rustdoc
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add documentation requirements and cargo doc to pre-PR checks"
```

---

## Task 3: Create Design Documents Index

**Files:**
- Create: `docs/design/index.md`

- [ ] **Step 1: Create index file**

```markdown
# Design Documents

## Architecture & Backend

| Document | Description |
|----------|-------------|
| [t4a_unified_tensor_backend.md](./t4a_unified_tensor_backend.md) | Unified tensor backend design (tenferro-rs integration) |
| [torch_backend.md](./torch_backend.md) | PyTorch backend design exploration |
| [tenferro_ad_scalar_operator_extension_note.md](./tenferro_ad_scalar_operator_extension_note.md) | Tenferro AD scalar operator extension notes |

## Automatic Differentiation

| Document | Description |
|----------|-------------|
| [three_mode_ad_design.md](./three_mode_ad_design.md) | Three-mode automatic differentiation design |

## Julia Compatibility

| Document | Description |
|----------|-------------|
| [quanticstransform_julia_comparison.md](./quanticstransform_julia_comparison.md) | Quantics transform Julia compatibility analysis |
```

- [ ] **Step 2: Commit**

```bash
git add docs/design/index.md
git commit -m "docs: add design documents index"
```

---

## Task 4: mdBook Skeleton

**Files:**
- Modify: `docs/book/src/SUMMARY.md`
- Create: all page stubs under `docs/book/src/`

- [ ] **Step 1: Write SUMMARY.md**

Replace `docs/book/src/SUMMARY.md` with:

```markdown
# Summary

- [Introduction](README.md)
- [Getting Started](getting-started.md)
- [Concepts](concepts.md)
- [Architecture & Crate Guide](architecture.md)
- [Guides]()
  - [Tensor Basics](guides/tensor-basics.md)
  - [Tensor Train](guides/tensor-train.md)
  - [Tensor Cross Interpolation](guides/tci.md)
  - [Quantics Transform](guides/quantics.md)
  - [Tree Tensor Networks](guides/tree-tn.md)
- [Conventions](conventions.md)
- [Julia Bindings](julia-bindings.md)
```

- [ ] **Step 2: Create stub files**

Create empty stub files so `mdbook build` succeeds:

```bash
mkdir -p docs/book/src/guides
touch docs/book/src/getting-started.md
touch docs/book/src/concepts.md
touch docs/book/src/architecture.md
touch docs/book/src/guides/tensor-basics.md
touch docs/book/src/guides/tensor-train.md
touch docs/book/src/guides/tci.md
touch docs/book/src/guides/quantics.md
touch docs/book/src/guides/tree-tn.md
touch docs/book/src/conventions.md
touch docs/book/src/julia-bindings.md
```

- [ ] **Step 3: Verify mdbook builds**

```bash
mdbook build docs/book
```

Expected: SUCCESS with no errors

- [ ] **Step 4: Commit**

```bash
git add docs/book/
git commit -m "docs: add mdBook skeleton with SUMMARY and stub pages"
```

---

## Task 5: mdBook — Introduction Page

**Files:**
- Modify: `docs/book/src/README.md`

- [ ] **Step 1: Write introduction**

Replace `docs/book/src/README.md` with content covering:

- What tensor4all-rs is (Rust tensor network library)
- Target audiences and how to navigate the book:
  - **New to tensor networks?** Start with [Concepts](concepts.md)
  - **Know TCI/QTT, new to this library?** Go to [Getting Started](getting-started.md)
  - **Coming from ITensors.jl?** See [Conventions](conventions.md) for type correspondence
  - **Looking for API details?** See [rustdoc](../rustdoc/tensor4all_core/)
- Feature highlights (bullet list): TCI2, QTT, TreeTN, C API, Julia bindings
- Link to GitHub repository

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/README.md
git commit -m "docs(book): write introduction page"
```

---

## Task 6: mdBook — Getting Started

**Files:**
- Modify: `docs/book/src/getting-started.md`

- [ ] **Step 1: Write getting-started page**

Content to include:

- **Prerequisites**: Rust toolchain (link to rustup.rs)
- **Add to Cargo.toml**: Show which crates to add for common use cases
  - Basic tensor train: `tensor4all-simplett`
  - TCI: `tensor4all-tensorci`
  - Quantics TCI: `tensor4all-quanticstci`
  - Tree tensor networks: `tensor4all-treetn`
- **First example**: Create a constant TT, evaluate, compress. Use the existing simplett example from the current README (lines 110-129) with assertions added.
- **Next steps**: Links to Concepts, Guides

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/getting-started.md
git commit -m "docs(book): write getting started page"
```

---

## Task 7: mdBook — Concepts

**Files:**
- Modify: `docs/book/src/concepts.md`

- [ ] **Step 1: Write concepts page**

Explain in plain language (for readers without deep math background):

- **Tensor Train (TT / MPS)**: Chain of 3-index tensors representing a high-dimensional tensor. Mention bond dimension controls accuracy vs memory.
- **Tensor Cross Interpolation (TCI)**: Approximate a high-dimensional function by sampling adaptively, like matrix CUR decomposition generalized to tensors.
- **Quantics Tensor Train (QTT)**: Binary encoding of continuous variables into a TT. Enables exponential compression for smooth functions.
- **Tree Tensor Network (TreeTN)**: Generalization of TT to tree graphs. Each node holds a tensor, edges represent shared indices.
- **Key terminology**: bond dimension, site index, link index, truncation tolerance (rtol)

Each concept: 1 paragraph explanation + simple ASCII diagram where helpful.

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/concepts.md
git commit -m "docs(book): write concepts page"
```

---

## Task 8: mdBook — Architecture & Crate Guide

**Files:**
- Modify: `docs/book/src/architecture.md`

- [ ] **Step 1: Write architecture page**

Content to include:

- **Crate dependency diagram** (ASCII art):

```
tensor4all-tensorbackend  (scalar types, storage)
        │
tensor4all-core           (Index, Tensor, contraction, SVD/QR)
        │
   ┌────┼──────────┬──────────────┐
   │    │          │              │
treetn  itensorlike simplett   tcicore
   │                  │           │
   │              partitionedtt  tensorci
   │                             │
treetci                    quanticstci
                                 │
                          quanticstransform
```

- **Layer descriptions**: What each layer does, 1-2 sentences per crate
- **Which crate should I use?** Decision guide:
  - "I want to do TCI on a function" → `tensor4all-quanticstci` (high-level) or `tensor4all-tensorci` (low-level)
  - "I want to manipulate tensor trains" → `tensor4all-simplett` (simple) or `tensor4all-itensorlike` (ITensors.jl-like)
  - "I want tree tensor networks" → `tensor4all-treetn`
  - "I want quantics transform operators" → `tensor4all-quanticstransform`
- **Internal crates** (tensorbackend, tcicore): Note that users don't need these directly

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/architecture.md
git commit -m "docs(book): write architecture and crate guide page"
```

---

## Task 9: mdBook — Guides (5 pages)

**Files:**
- Modify: `docs/book/src/guides/tensor-basics.md`
- Modify: `docs/book/src/guides/tensor-train.md`
- Modify: `docs/book/src/guides/tci.md`
- Modify: `docs/book/src/guides/quantics.md`
- Modify: `docs/book/src/guides/tree-tn.md`

Content for each guide is derived from the existing crate READMEs which will be slimmed later. Each guide should include working Rust code examples with assertions.

- [ ] **Step 9a: Write tensor-basics.md**

Source material: `crates/tensor4all-core/README.md`

Content:
- **Index**: Creating indices with `Index::new(dim)`, tags, prime levels
- **Tensor**: Creating tensors with `IdxTensor`, random tensors
- **Contraction**: `contract()`, `contract_multi()`, `AllowedPairs`
- **Factorization**: SVD with truncation, QR
- Code examples from current core README, adapted with assertions

- [ ] **Step 9b: Write tensor-train.md**

Source material: `crates/tensor4all-simplett/README.md`, `crates/tensor4all-itensorlike/README.md`

Content:
- **SimpleTT**: Create, evaluate, sum, compress (from simplett README)
- **ITensorLike TensorTrain**: Orthogonalization, canonical forms, truncation, inner product (from itensorlike README)
- When to use which: simplett for simple numerical work, itensorlike for ITensors.jl-like workflows
- Code examples with assertions

- [ ] **Step 9c: Write tci.md**

Source material: `crates/tensor4all-tensorci/README.md`, `crates/tensor4all-quanticstci/README.md`

Content:
- **Low-level TCI** (`crossinterpolate2`): Define function, set options, run, get TT (from tensorci README)
- **High-level Quantics TCI** (`QuanticsTensorCI2`): Discrete grids, continuous grids with `DiscretizedGrid`, integration (from quanticstci README)
- Important conventions: 1-indexed grid indices, power-of-2 grid sizes
- Code examples with assertions

- [ ] **Step 9d: Write quantics.md**

Source material: `crates/tensor4all-quanticstransform/README.md` (234 lines — the largest README, most content moves here)

Content:
- Operator overview table (flip, shift, phase rotation, cumulative sum, Fourier, binary op, affine)
- Creating and applying operators to tensor trains
- Index mapping and multi-variable encoding
- Big-endian bit ordering, boundary conditions
- Code examples with assertions

- [ ] **Step 9e: Write tree-tn.md**

Source material: `crates/tensor4all-treetn/README.md`

Content:
- Creating a TreeTN from tensors
- Canonicalization and truncation
- Norm, dense conversion, addition
- Sweep counting (nfullsweeps vs nhalfsweeps)
- Code examples with assertions

- [ ] **Step 9f: Verify mdbook builds**

```bash
mdbook build docs/book
```

Expected: SUCCESS

- [ ] **Step 9g: Commit**

```bash
git add docs/book/src/guides/
git commit -m "docs(book): write guide pages (tensor-basics, tensor-train, tci, quantics, tree-tn)"
```

---

## Task 10: mdBook — Conventions

**Files:**
- Modify: `docs/book/src/conventions.md`

- [ ] **Step 1: Write conventions page**

Content migrated from current README sections:

- **Dense layout**: Column-major (matches Julia, ITensors.jl, tenferro-rs). NumPy interop: use `order="F"`.
- **Indexing**: 0-indexed sites in Rust (unlike ITensors.jl which is 1-indexed). Exception: `quanticstci` grid indices are 1-indexed (Julia convention).
- **Truncation tolerance**: `rtol` (relative tolerance). ITensors.jl uses `cutoff` where `rtol = sqrt(cutoff)`.
- **ITensors.jl type correspondence** (table from current README lines 39-49, without QSpace column):

| ITensors.jl | tensor4all-rs |
|-------------|---------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `ITensor` | `IdxTensor` |
| `Dense` | `Storage::StructuredF64/C64` |
| `Diag` | `Storage::StructuredF64/C64` (diagonal) |
| `A * B` | `a.contract(&b)` |

- **Scalar types**: `f64` and `Complex64` supported via generic APIs

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/conventions.md
git commit -m "docs(book): write conventions page"
```

---

## Task 11: mdBook — Julia Bindings

**Files:**
- Modify: `docs/book/src/julia-bindings.md`

- [ ] **Step 1: Write Julia bindings page**

Content:
- Julia bindings are maintained in a separate repo: [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl)
- Installation: `Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")`
- Link to Tensor4all.jl README for detailed docs
- Brief mention of C API layer (`tensor4all-capi`) as the bridge

- [ ] **Step 2: Commit**

```bash
git add docs/book/src/julia-bindings.md
git commit -m "docs(book): write Julia bindings page"
```

---

## Task 12: Rewrite Top-Level README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README.md**

Replace the entire README with ~80 lines:

```markdown
# tensor4all-rs

[![CI](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml/badge.svg)](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml)

A Rust implementation of tensor networks: TCI, Quantics Tensor Train, and Tree Tensor Networks.

## Features

- **ITensors.jl-like dynamic tensors**: Flexible `Index` system with dynamic-rank `Tensor`
- **Tensor Cross Interpolation**: TCI2 algorithm for efficient high-dimensional function approximation
- **Quantics Tensor Train**: Binary encoding of continuous variables with transformation operators
- **Tree Tensor Networks**: Arbitrary topology with canonicalization, truncation, and contraction
- **C API**: Full functionality exposed for language bindings (Julia)

## Quick Start

Add to your `Cargo.toml`:

\```toml
[dependencies]
tensor4all-simplett = "0.1"
\```

\```rust
use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, TensorTrain};

let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
let value = tt.evaluate(&[0, 1, 2]).unwrap();
assert!((value - 1.0).abs() < 1e-12);

let total = tt.sum();
assert!((total - 24.0).abs() < 1e-12);

let options = CompressionOptions {
    tolerance: 1e-10,
    max_bond_dim: 20,
    ..Default::default()
};
let compressed = tt.compressed(&options).unwrap();
assert!((compressed.sum() - 24.0).abs() < 1e-10);
\```

## Crate Overview

| Crate | Description |
|-------|-------------|
| [tensor4all-core](crates/tensor4all-core/) | Core types: Index, Tensor, contraction, SVD, QR |
| [tensor4all-simplett](crates/tensor4all-simplett/) | Simple TT/MPS with compression |
| [tensor4all-itensorlike](crates/tensor4all-itensorlike/) | ITensors.jl-like TensorTrain API |
| [tensor4all-treetn](crates/tensor4all-treetn/) | Tree tensor networks with arbitrary topology |
| [tensor4all-tensorci](crates/tensor4all-tensorci/) | Tensor Cross Interpolation (TCI2) |
| [tensor4all-quanticstci](crates/tensor4all-quanticstci/) | High-level Quantics TCI interface |
| [tensor4all-quanticstransform](crates/tensor4all-quanticstransform/) | Quantics transformation operators |
| [tensor4all-treetci](crates/tensor4all-treetci/) | Tree-structured cross interpolation |
| [tensor4all-partitionedtt](crates/tensor4all-partitionedtt/) | Partitioned Tensor Train |
| [tensor4all-hdf5](crates/tensor4all-hdf5/) | ITensors.jl-compatible HDF5 serialization |
| [tensor4all-capi](crates/tensor4all-capi/) | C FFI for language bindings |

## Documentation

- **[User Guide](https://tensor4all.github.io/tensor4all-rs/)** — tutorials, architecture, conventions
- **[API Reference (rustdoc)](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_core/)** — generated API documentation
- **[Julia Bindings](https://github.com/tensor4all/Tensor4all.jl)** — Tensor4all.jl wrapper
- **[Design Documents](docs/design/index.md)** — architecture and design decisions

## Acknowledgments

Inspired by [ITensors.jl](https://github.com/ITensor/ITensors.jl). We acknowledge fruitful discussions with M. Fishman and E. M. Stoudenmire at CCQ, Flatiron Institute.

**Citation:** If you use this code in research, please cite:

> We used tensor4all-rs (https://github.com/tensor4all/tensor4all-rs), inspired by ITensors.jl.
>
> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", arXiv:2007.14822 (2020)

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
```

- [ ] **Step 2: Verify no broken references**

```bash
grep -n "python\|Python" README.md
```

Expected: no matches

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: slim README to entry point with Book links"
```

---

## Task 13: Unify Crate READMEs (Existing 7 — excluding capi)

**Files:**
- Modify: `crates/tensor4all-core/README.md`
- Modify: `crates/tensor4all-treetn/README.md`
- Modify: `crates/tensor4all-itensorlike/README.md`
- Modify: `crates/tensor4all-simplett/README.md`
- Modify: `crates/tensor4all-tensorci/README.md`
- Modify: `crates/tensor4all-quanticstci/README.md`
- Modify: `crates/tensor4all-quanticstransform/README.md`

For each crate, replace the existing README with the unified template format:

```markdown
# tensor4all-{name}

{1-2 line description}

## Key Types

- `TypeA` — description
- `TypeB` — description

## Example

\```rust
// Minimal working example with assertions
\```

## Documentation

- [User Guide: {guide name}](https://tensor4all.github.io/tensor4all-rs/{path}.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_{name}/)
```

- [ ] **Step 1: Rewrite each README**

Read each existing README, extract the key types and one short example with assertions. Discard detailed explanations (now in Book). Keep the example minimal (~10-15 lines of code).

Key types per crate:
- **core**: `Index`, `IdxTensor`, `Storage`, `contract()`, `svd()`
- **treetn**: `TreeTN`, `from_tensors()`, `canonicalize()`, `truncate()`
- **itensorlike**: `TensorTrain`, `orthogonalize()`, `truncate()`, `inner()`
- **simplett**: `TensorTrain`, `evaluate()`, `sum()`, `compressed()`
- **tensorci**: `crossinterpolate2()`, `TCI2Options`
- **quanticstci**: `QuanticsTensorCI2`, `DiscretizedGrid`
- **quanticstransform**: `LinearOperator`, `shift()`, `flip()`, `fourier()`

Link targets:
- core → guides/tensor-basics.html
- treetn → guides/tree-tn.html
- itensorlike → guides/tensor-train.html
- simplett → guides/tensor-train.html
- tensorci → guides/tci.html
- quanticstci → guides/tci.html
- quanticstransform → guides/quantics.html

- [ ] **Step 2: Verify no broken builds**

```bash
cargo doc --workspace --no-deps 2>&1 | head -20
```

- [ ] **Step 3: Commit**

```bash
git add crates/*/README.md
git commit -m "docs: unify existing crate READMEs to minimal template"
```

---

## Task 14: Create Missing Crate READMEs (5 crates)

**Files:**
- Create: `crates/tensor4all-tensorbackend/README.md`
- Create: `crates/tensor4all-tcicore/README.md`
- Create: `crates/tensor4all-hdf5/README.md`
- Create: `crates/tensor4all-partitionedtt/README.md`
- Modify: `crates/tensor4all-treetci/README.md`

- [ ] **Step 1: Create tensorbackend README**

```markdown
# tensor4all-tensorbackend

> This is an internal crate. Most users should use `tensor4all-core` instead.

Scalar types (`f64`, `Complex64`), storage backends, and tensor algebra primitives
backed by tenferro-rs.

## Key Types

- `AnyScalar` — dynamic scalar type (f64 or Complex64)
- `Storage` — dense/diagonal tensor storage
- `StructuredStorage` — axis-class-aware storage snapshots

## Documentation

- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_tensorbackend/)
```

- [ ] **Step 2: Create tcicore README**

```markdown
# tensor4all-tcicore

> This is an internal crate. Most users should use `tensor4all-tensorci` or `tensor4all-quanticstci` instead.

Low-level TCI infrastructure: matrix cross interpolation, LUCI/rrLU algorithms,
cached function evaluation, and index set management.

## Key Types

- `MatrixLUCI` — matrix LU-based cross interpolation
- `CachedFunction` — thread-safe cached function evaluation
- `IndexSet` — bidirectional index sets for pivot management

## Documentation

- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_tcicore/)
```

- [ ] **Step 3: Create hdf5 README**

```markdown
# tensor4all-hdf5

HDF5 serialization for tensor4all-rs, compatible with ITensors.jl / ITensorMPS.jl file formats.

## Key Types

- `save_itensor()` / `load_itensor()` — read/write `IdxTensor` as ITensors.jl `ITensor`
- `save_mps()` / `load_mps()` — read/write `TensorTrain` as ITensorMPS.jl `MPS`

## Feature Flags

- `link` (default) — compile-time HDF5 linking
- `runtime-loading` — dlopen for FFI environments

## Documentation

- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_hdf5/)
```

- [ ] **Step 4: Create partitionedtt README**

```markdown
# tensor4all-partitionedtt

Partitioned Tensor Train for representing functions over non-overlapping subdomains
with projectors.

## Key Types

- `PartitionedTT` — collection of non-overlapping subdomain tensor trains
- `SubDomainTT` — tensor train restricted to a specific subdomain
- `Projector` — maps tensor indices to fixed values defining subdomains

## Documentation

- [User Guide: Tensor Train](https://tensor4all.github.io/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_partitionedtt/)
```

- [ ] **Step 5: Rewrite treetci README**

```markdown
# tensor4all-treetci

Tree Tensor Cross Interpolation — a Rust port of
[TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl) by Ryo Watanabe.

Computes tensor cross interpolation on tree-structured graphs, producing TreeTN output.

## Key Types

- `crossinterpolate2()` — high-level entry point for tree TCI
- `TreeTCI2` — algorithm state
- `TreeTciGraph` — graph structure definition

## Documentation

- [User Guide: Tree Tensor Networks](https://tensor4all.github.io/tensor4all-rs/guides/tree-tn.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_treetci/)
```

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/README.md \
       crates/tensor4all-tcicore/README.md \
       crates/tensor4all-hdf5/README.md \
       crates/tensor4all-partitionedtt/README.md \
       crates/tensor4all-treetci/README.md
git commit -m "docs: add missing crate READMEs with unified template"
```

---

## Task 15: Rustdoc Examples — High Priority Crates

**Files:**
- Modify: public items in `crates/tensor4all-core/src/`
- Modify: public items in `crates/tensor4all-simplett/src/`
- Modify: public items in `crates/tensor4all-tensorci/src/`
- Modify: public items in `crates/tensor4all-quanticstci/src/`

For each crate, identify public types/traits/functions that lack `/// # Examples` and add examples with assertions. The worker must:

1. Run `cargo doc --workspace --no-deps` to find current public API surface
2. Read `docs/api/tensor4all_{name}.md` for API listing
3. For each public item missing examples, add `/// # Examples` with assertions

- [ ] **Step 1: Add doc examples to tensor4all-core**

Key items: `Index::new()`, `IdxTensor` creation, `contract()`, `svd()`, `qr()`

Pattern for each:
```rust
/// # Examples
///
/// ```
/// use tensor4all_core::{Index, IdxTensor};
///
/// let i = Index::new(3);
/// assert_eq!(i.dim(), 3);
/// ```
```

All examples must compile, run, and assert.

- [ ] **Step 2: Run doctests for core**

```bash
cargo test --release -p tensor4all-core --doc
```

Expected: all doctests pass

- [ ] **Step 3: Add doc examples to tensor4all-simplett**

Key items: `TensorTrain::constant()`, `evaluate()`, `sum()`, `compressed()`

- [ ] **Step 4: Run doctests for simplett**

```bash
cargo test --release -p tensor4all-simplett --doc
```

- [ ] **Step 5: Add doc examples to tensor4all-tensorci**

Key items: `crossinterpolate2()`, `TCI2Options`

- [ ] **Step 6: Run doctests for tensorci**

```bash
cargo test --release -p tensor4all-tensorci --doc
```

- [ ] **Step 7: Add doc examples to tensor4all-quanticstci**

Key items: `QuanticsTensorCI2`, `DiscretizedGrid`

- [ ] **Step 8: Run doctests for quanticstci**

```bash
cargo test --release -p tensor4all-quanticstci --doc
```

- [ ] **Step 9: Commit**

```bash
git add crates/tensor4all-core/ crates/tensor4all-simplett/ \
       crates/tensor4all-tensorci/ crates/tensor4all-quanticstci/
git commit -m "docs: add rustdoc examples with assertions to high-priority crates"
```

---

## Task 16: Rustdoc Examples — Medium Priority Crates

**Files:**
- Modify: public items in `crates/tensor4all-treetn/src/`
- Modify: public items in `crates/tensor4all-quanticstransform/src/`
- Modify: public items in `crates/tensor4all-itensorlike/src/`

Same approach as Task 15.

- [ ] **Step 1: Add doc examples to tensor4all-treetn**

Key items: `TreeTN::from_tensors()`, `canonicalize()`, `truncate()`, `norm()`, `to_dense()`

- [ ] **Step 2: Run doctests**

```bash
cargo test --release -p tensor4all-treetn --doc
```

- [ ] **Step 3: Add doc examples to tensor4all-quanticstransform**

Key items: `LinearOperator`, `shift()`, `flip()`, `fourier()`, `apply()`

- [ ] **Step 4: Run doctests**

```bash
cargo test --release -p tensor4all-quanticstransform --doc
```

- [ ] **Step 5: Add doc examples to tensor4all-itensorlike**

Key items: `TensorTrain::new()`, `orthogonalize()`, `truncate()`, `inner()`

- [ ] **Step 6: Run doctests**

```bash
cargo test --release -p tensor4all-itensorlike --doc
```

- [ ] **Step 7: Commit**

```bash
git add crates/tensor4all-treetn/ crates/tensor4all-quanticstransform/ \
       crates/tensor4all-itensorlike/
git commit -m "docs: add rustdoc examples with assertions to medium-priority crates"
```

---

## Task 17: Rustdoc Examples — Low Priority Crates

**Files:**
- Modify: public items in `crates/tensor4all-tcicore/src/`
- Modify: public items in `crates/tensor4all-tensorbackend/src/`
- Modify: public items in `crates/tensor4all-hdf5/src/`
- Modify: public items in `crates/tensor4all-partitionedtt/src/`
- Modify: public items in `crates/tensor4all-treetci/src/`

Same approach. Internal crates (tcicore, tensorbackend) get minimal examples. HDF5 examples may need `ignore` attribute if HDF5 lib not available in doctests.

- [ ] **Step 1: Add doc examples to low-priority crates**
- [ ] **Step 2: Run doctests for each**

```bash
cargo test --release -p tensor4all-tcicore --doc
cargo test --release -p tensor4all-tensorbackend --doc
cargo test --release -p tensor4all-hdf5 --doc
cargo test --release -p tensor4all-partitionedtt --doc
cargo test --release -p tensor4all-treetci --doc
```

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-tcicore/ crates/tensor4all-tensorbackend/ \
       crates/tensor4all-hdf5/ crates/tensor4all-partitionedtt/ \
       crates/tensor4all-treetci/
git commit -m "docs: add rustdoc examples with assertions to low-priority crates"
```

---

## Task 18: Final Verification

- [ ] **Step 1: Full build and test**

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
cargo doc --workspace --no-deps
mdbook build docs/book
```

All must pass.

- [ ] **Step 2: Review mdBook output**

Open `docs/book/book/index.html` in a browser. Verify:
- All navigation links work
- Code examples render correctly
- No placeholder/stub content remains

- [ ] **Step 3: Verify GitHub Pages deployment structure**

```bash
# Simulate what CI does
mkdir -p _site
cp -r docs/book/book/* _site/
mkdir -p _site/rustdoc
cp -r target/doc/* _site/rustdoc/
```

Verify links from Book to rustdoc work (relative paths).

- [ ] **Step 4: Final commit if any formatting changes**

```bash
git add -A
git commit -m "docs: final formatting and verification"
```
