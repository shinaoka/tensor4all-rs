# Architecture & Crate Guide

This page describes how tensor4all-rs is organised, what each crate does, how to
choose the right crate for your use case, and the two-stack design that keeps
the public API predictable.

## Two stacks, no facade

The workspace is organised as **two independent stacks**. Each stack owns its
tensor-train representation and its algorithm crates. There is **no facade
crate** that wraps everything: `tensor4all-capi` is a C FFI layer, not a Rust
facade, and each crate can be used on its own.

```text
                        tensor4all-core (Index, Tensor, contraction, SVD/QR)
                                      |
              +-----------------------+------------------------+
              |                                                |
   NETWORK STACK (tree-based)                     SIMPLETT STACK (positional)
              |                                                |
   tensor4all-treetn  (TreeTN)                    tensor4all-simplett (SimpleTensorTrain)
              |                                                |
   tensor4all-itensorlike (TensorTrain)           tensor4all-tensorci (TCI1/TCI2)
              |                                                |
   tensor4all-partitionedtt                        tensor4all-quanticstci
              |                                                |
   tensor4all-treetci                              tensor4all-quanticstransform

   tensor4all-capi  (C FFI for language bindings; depends on both stacks)
   tensor4all-hdf5  (MPS serialization, ITensors.jl-compatible)
```

### The sanctioned crossing: `treetn::simplett_bridge`

The only sanctioned place where the two stacks convert into each other is
`treetn::simplett_bridge`:

- `tensor_train_to_treetn` / `tensor_train_to_treetn_with_names` /
  `tensor_train_to_treetn_with_names_and_site_indices`: simplett
  `SimpleTensorTrain` -> network `TreeTN`;
- `treetn_to_tensor_train`: network `TreeTN` (linear chain) -> simplett
  `SimpleTensorTrain`;
- `fix_and_remove_site_from_treetn_chain`,
  `insert_onehot_site_in_treetn_chain`,
  `weighted_remove_site_from_treetn_chain`: chain-topology helpers.

Crates that need to cross stacks must route through this bridge. New ad hoc
bridges (hand-rolled conversion logic inside another crate) are rejected.
`tensor4all-partitionedtt` consumes simplett output (TCI2 results) only via
`tensor_train_to_treetn`.

One deliberate exception exists: `tensor4all-quanticstransform` builds
TreeTN-based `LinearOperator`s from simplett **MPO** data (two site indices
per node) via `tensortrain_to_linear_operator`. That is operator construction,
not a general stack conversion: the bridge stays MPS-shaped (one site index
per node), and the transform layer keeps the MPO-specific index bookkeeping.
No *new* ad hoc conversions may be added; MPO conversion belongs in the
transform layer, and MPS conversion belongs in the bridge.

### Which stack does a new feature crate target?

- **Application-level features** (operators, evolution, DMRG, TDVP, algorithm
  composition on a known topology) target the **network stack** by default
  (`tensor4all-treetn`, or `tensor4all-itensorlike` when ITensors.jl-style
  semantics are wanted).
- **Numerical-core features** (cross interpolation, factorizations, quantics
  kernels) target the **simplett stack** by default
  (`tensor4all-simplett`, `tensor4all-tensorci`; the former
  `tensor4all-tcicore` was dissolved into `tensor4all-core`, #639).
- A feature that must serve both stacks is implemented in the target stack
  and exposed through `treetn::simplett_bridge`; it is not duplicated in both.

## Vocabulary conventions

Beyond the naming-policy suffixes (`_mut`, `_into`, `_batched`), the following
operation names are unified across crates:

- **Inner product**: `inner_product` everywhere. (`dot` was retired in
  `tensor4all-simplett`.)
- **Densification**: `full_tensor` materializes a tensor train/MPO into a flat
  column-major buffer plus shape; `to_dense` materializes a single tensor into
  a dense `IdxTensor`. The distinction is intentional (whole-TT vs
  one-tensor).
- **Canonicalization**: `treetn` uses `canonicalize` / `canonicalize_mut`;
  `itensorlike` uses `orthogonalize` (ITensors.jl-compatible vocabulary) for
  the same center-site normalization. Each stack keeps its standard term; do
  not introduce a third synonym.
- **Options**: bond caps use `max_bond_dim: Option<usize>`; truncation
  tolerances use `SvdTruncationPolicy` (`rtol`/`cutoff`); sweep counts use
  `nfullsweeps`/`nsweeps` (TreeTN) or `nhalfsweeps` (itensorlike) as documented
  in the skills guide.

## `TensorTrain` naming resolution

Two different types are named `TensorTrain`-family, which is the historical
naming trap that PR 3 resolves:

| Type | Crate | Representation | Use when |
|------|-------|----------------|----------|
| `SimpleTensorTrain<T>` | `tensor4all-simplett` | positional cores (`Tensor3<T>`), no named indices | lightweight create/evaluate/compress |
| `TensorTrain` | `tensor4all-itensorlike` | `TreeTN<IdxTensor, usize>` wrapper with orthogonality tracking | ITensors.jl-style interface |

Rules:

- `tensor4all-simplett::SimpleTensorTrain` is the **positional** chain type.
  Prefer it for numerical kernels.
- `tensor4all-itensorlike::TensorTrain` is the **tree-based** chain type with
  named indices, canonical forms, and orthogonality tracking.
- There are no compatibility aliases: `TensorTrain` alone always means the
  itensorlike type inside that crate, and simplett code must write
  `SimpleTensorTrain`.

## Layer descriptions

### Foundation (internal)

| Crate | Description |
|-------|-------------|
| **tensorbackend** | *Internal.* Compact `f64`/`Complex64` storage, four-dtype eager tensor bridges, and tenferro-backed primitives. Users do not need to depend on this crate directly. |
| **core** | Foundation for everything else. Provides the `Index` system, dynamic-rank `Tensor`, contraction, and SVD/QR/LU factorizations. |

### Network stack

| Crate | Description |
|-------|-------------|
| **treetn** | Tree tensor networks with arbitrary graph topology. Supports canonicalization, truncation, contraction, DMRG/TDVP, and hosts the sanctioned `simplett_bridge`. |
| **itensorlike** | ITensors.jl-inspired `TensorTrain` (tree-based) with orthogonality tracking and multiple canonical forms. |
| **partitionedtt** | Partitioned tensor trains for subdomain decomposition. Builds on itensorlike and crosses to simplett via `simplett_bridge`. |
| **treetci** | Tree TCI: cross interpolation on tree-structured tensor networks. |

### Simplett stack

| Crate | Description |
|-------|-------------|
| **simplett** | Lightweight positional tensor train (`SimpleTensorTrain`) for numerical computation. |
| **tensorci** | Tensor Cross Interpolation. Contains TCI2 (primary algorithm) and TCI1 (legacy). |
| **quanticstci** | High-level Quantics TCI. Interpolates functions on discrete or continuous grids in the quantics format. |

### Quantics & transforms

| Crate | Description |
|-------|-------------|
| **quanticstransform** | Quantics transformation operators: shift, flip, Fourier, affine, and more. Consumes simplett-stack data; constructs TreeTN-based `LinearOperator`s (see the bridge exception above). |
| **interpolativeqtt** | Interpolative QTT construction on a coarse grid (simplett stack). See the [interpolative-QTT tutorial](tutorials/quantics-basics/interpolative-qtt.md). |

### Applications

| Crate | Description |
|-------|-------------|
| **aci** | Alternating Cross Interpolation (ACI) for elementwise tensor-train operations. |

### I/O & bindings

### I/O & bindings

| Crate | Description |
|-------|-------------|
| **hdf5** | HDF5 serialization compatible with ITensors.jl/ITensorMPS.jl file formats. |
| **capi** | C FFI for language bindings (Julia, Python, etc.). Out of scope for this guide; see [Julia Bindings](julia-bindings.md). |

## Which crate should I use?

| Goal | Recommended crate |
|------|-------------------|
| TCI on a black-box function (high level) | `tensor4all-quanticstci` |
| TCI with fine-grained control | `tensor4all-tensorci` |
| Tree TCI | `tensor4all-treetci` |
| Simple positional tensor train (create, evaluate, compress) | `tensor4all-simplett` (`SimpleTensorTrain`) |
| Tensor train with ITensors.jl-style interface | `tensor4all-itensorlike` (`TensorTrain`) |
| Tree tensor networks | `tensor4all-treetn` |
| Subdomain decomposition via partitioned TT | `tensor4all-partitionedtt` |
| Quantics transform operators | `tensor4all-quanticstransform` |
| HDF5 I/O compatible with Julia | `tensor4all-hdf5` |
| Interpolative QTT on a coarse grid | `tensor4all-interpolativeqtt` |
| Elementwise TT ops via ACI | `tensor4all-aci` |

## Error remedies

Public error types use `thiserror`, preserve the source error, carry
structured fields, and name an actionable remedy in the rustdoc when one is
documented. Recurring cases avoid opaque string payloads. `cargo clippy` is
configured with `-D clippy::missing_errors_doc -D clippy::missing_panics_doc`
so every fallible or panicking public function documents its failure modes.

## Internal crates

`tensor4all-tensorbackend` is an implementation detail. The former
`tensor4all-tcicore` crate was dissolved into `tensor4all-core` (#639): the
matrix CI / LUCI / rrLU algorithms, `CachedFunction`, and `MultiIndex` now
live in core. These are still not part of the application-facing public API
surface and their interfaces may change without notice.
