> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Documentation Improvement Design

**Date:** 2026-04-08
**Status:** Approved
**Scope:** Non-C-API documentation for tensor4all-rs

## Goal

Make tensor4all-rs accessible to all levels of users — from TCI/QTT experts
new to this library, to Rust developers new to tensor networks, to Julia users
exploring the Rust backend — through a layered documentation structure with
clear navigation.

## Document Hierarchy

```
README.md (entry point, ~80 lines)
  │
  ├── docs/book/ (GitHub Pages, user documentation)
  │     Layered learning path for all audiences
  │
  ├── crate READMEs (minimal, ~30-50 lines each)
  │     Overview + key types + short example + link to Book
  │
  ├── rustdoc (/// # Examples with assertions)
  │     API reference with runnable, verified code examples
  │
  └── docs/design/index.md (contributor-facing)
        Index of architecture/design documents
```

**Principle:** Information lives in exactly one place. Everything else links to it.

## 1. README.md Slimming

Reduce from 318 lines to ~80 lines. Keep:

- Project description (1-2 sentences)
- Features list (bullet points)
- Quick Start (one simplett example)
- Crate overview table (existing, keep capi)
- Documentation links (Book, Julia bindings, design docs)
- Citation / Acknowledgments (brief)
- License

Remove:

- Design Philosophy details (move to Book: architecture.md)
- Dense Layout Semantics (move to Book: conventions.md)
- Backend Status (internal)
- Type Correspondence table (move to Book: conventions.md, drop QSpace column)
- Truncation Tolerance (move to Book: conventions.md)
- Solve-Bug Entrypoints (AGENTS.md only)
- TCI usage example (move to Book: guides/tci.md)
- Python bindings section (delete entirely)
- Near-Term Work / TODO / Known Issues (use GitHub Issues)
- Development / Pre-commit Checks (already in AGENTS.md)

## 2. mdBook Structure

Published via GitHub Pages at `https://tensor4all.github.io/tensor4all-rs/`.

```
docs/book/src/
├── SUMMARY.md
├── README.md                # What is tensor4all-rs? Overview + reader-level guide
├── getting-started.md       # cargo add, first TT creation/compression example
├── architecture.md          # Crate dependency diagram, layers, which crate to use
├── concepts.md              # TCI, QTT, TreeTN concepts (for readers without math background)
├── guides/
│   ├── tensor-basics.md     # Index, Tensor, contraction
│   ├── tensor-train.md      # TT/MPS operations (simplett + itensorlike)
│   ├── tci.md               # TCI usage (tensorci + quanticstci)
│   ├── quantics.md          # Quantics transform operators
│   └── tree-tn.md           # TreeTN
├── conventions.md           # Column-major, 0-indexed, tolerance, ITensors.jl correspondence
└── julia-bindings.md        # Tensor4all.jl installation + link
```

SUMMARY.md:

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

### GitHub Pages Deployment

Add a GitHub Actions workflow that runs `mdbook build docs/book` and deploys
to GitHub Pages on push to main.

## 3. Crate README Unified Format

All 13 crate READMEs follow this template:

```markdown
# tensor4all-{name}

{1-2 line description}

## Key Types
- `TypeA` -- description
- `TypeB` -- description

## Example

\`\`\`rust
// Minimal working example with assertions
\`\`\`

## Documentation

- [User Guide: {relevant guide}](https://tensor4all.github.io/tensor4all-rs/{path})
- [API Reference](https://docs.rs/tensor4all-{name})
```

For internal crates (tensorbackend, tcicore), prepend:

> This is an internal crate. Most users should use `tensor4all-{higher-level}` instead.

### Changes by Crate

| Crate | Action |
|-------|--------|
| tensor4all-core | Slim down existing README |
| tensor4all-treetn | Slim down existing README |
| tensor4all-itensorlike | Slim down existing README, move details to Book |
| tensor4all-simplett | Slim down existing README |
| tensor4all-tensorci | Slim down existing README |
| tensor4all-quanticstci | Slim down existing README |
| tensor4all-quanticstransform | Slim down existing README, move operator table to Book |
| tensor4all-capi | Keep as-is (out of scope) |
| tensor4all-tensorbackend | **New** (internal crate notice) |
| tensor4all-tcicore | **New** (internal crate notice) |
| tensor4all-hdf5 | **New** |
| tensor4all-partitionedtt | **New** |
| tensor4all-treetci | **Rewrite** (currently minimal) |

## 4. AGENTS.md Changes

### New Section: Documentation Requirements

```markdown
## Documentation Requirements

Every public type, trait, and function **must** include usage examples in its
doc comments (`/// # Examples`). Examples must include assertions to verify
correctness (e.g., `assert!`, `assert_eq!`, `approx::assert_abs_diff_eq!`).
Use `ignore` attribute on examples that cannot run in doctests.
```

### Pre-PR Checks Update

Add `cargo doc --workspace --no-deps` to the existing checklist:

```bash
cargo fmt --all                        # Format all code
cargo clippy --workspace               # Check for common issues
cargo nextest run --release --workspace # Run all tests
cargo doc --workspace --no-deps        # Build rustdoc
```

## 5. Rustdoc Examples (/// # Examples)

Add assertion-bearing examples to major public API items. Priority:

| Priority | Crate | Key items |
|----------|-------|-----------|
| High | tensor4all-core | `Index`, `IdxTensor`, `contract`, `svd`, `qr` |
| High | tensor4all-simplett | `TensorTrain`, `evaluate`, `sum`, `compressed` |
| High | tensor4all-tensorci | `crossinterpolate2`, `TCI2Options` |
| High | tensor4all-quanticstci | `QuanticsTensorCI2`, `DiscretizedGrid` |
| Medium | tensor4all-treetn | `TreeTN`, `from_tensors`, `canonicalize`, `truncate` |
| Medium | tensor4all-quanticstransform | `LinearOperator`, `shift`, `flip`, `fourier` |
| Medium | tensor4all-itensorlike | `TensorTrain`, `orthogonalize`, `truncate` |
| Low | tensor4all-tcicore | Internal crate |
| Low | tensor4all-tensorbackend | Internal crate |
| Low | tensor4all-hdf5 | `save_mps` / `load_mps` |
| Low | tensor4all-partitionedtt | `PartitionedTT`, `Projector` |
| Low | tensor4all-treetci | `crossinterpolate2` (tree version) |

All examples must include assertions verifying computed results.

## 6. Design Document Index

Create `docs/design/index.md`:

```markdown
# Design Documents

## Architecture & Backend

| Document | Description |
|----------|-------------|
| [t4a_unified_tensor_backend.md](...) | Unified tensor backend design |
| [torch_backend.md](...) | PyTorch backend design exploration |
| [tenferro_ad_scalar_operator_extension_note.md](...) | Tenferro AD scalar operator extension |

## Automatic Differentiation

| Document | Description |
|----------|-------------|
| [three_mode_ad_design.md](...) | Three-mode AD design |

## Julia Compatibility

| Document | Description |
|----------|-------------|
| [quanticstransform_julia_comparison.md](...) | Quantics transform Julia comparison |
```

## 7. Python Bindings Removal

Delete unmaintained Python code and documentation:

- `python/tensor4all/` directory
- `docs/examples/python/` directory (if exists)
- Python references in README (handled by Section 1)
- Python-related CI steps (`scripts/run_python_tests.sh`, workflow steps)

## Out of Scope

- C API documentation (`tensor4all-capi` README stays as-is)
- New feature development
- Existing design document content changes
