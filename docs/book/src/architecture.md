# Architecture & Crate Guide

This page describes how tensor4all-rs is organised, what each crate does, and how
to choose the right crate for your use case.

## Crate Dependency Diagram

```text
tensor4all-tensorbackend    (scalar types, storage)
        |
tensor4all-core             (Index, Tensor, contraction, SVD/QR)
        |
   +---------+-----------+-----------+
   |         |           |           |
 treetn  itensorlike  simplett    tcicore
   |                     |           |
   |              partitionedtt   tensorci
   |                                 |
 treetci                       quanticstci
                                     |
                              quanticstransform
```

`tensor4all-hdf5` depends on `tensor4all-core` and `tensor4all-itensorlike` (for
MPS serialization).  `tensor4all-capi` depends on most crates and forms the C FFI
layer used by language bindings.

## Layer Descriptions

### Foundation (internal)

| Crate | Description |
|-------|-------------|
| **tensorbackend** | *Internal.* Compact `f64`/`Complex64` storage, all-four-dtype eager tensor bridges, and tenferro-backed primitives. Users do not need to depend on this crate directly. |
| **core** | Foundation for everything else. Provides the `Index` system, dynamic-rank `Tensor`, contraction, and SVD/QR/LU factorizations. |

### Tensor Train & Tree Tensor Networks

| Crate | Description |
|-------|-------------|
| **treetn** | Tree tensor networks with arbitrary graph topology. Supports canonicalization, truncation, and contraction. |
| **itensorlike** | ITensors.jl-inspired `TensorTrain` with orthogonality tracking and multiple canonical forms. Useful when compatibility with the ITensors.jl mental model is important. |
| **simplett** | Lightweight tensor train for numerical computation. The go-to crate for creating, evaluating, and compressing tensor trains without extra overhead. |
| **partitionedtt** | Partitioned tensor trains for subdomain decomposition. Builds on `simplett`. |
| **treetci** | Tree TCI: cross interpolation on tree-structured tensor networks. |

### Tensor Cross Interpolation

| Crate | Description |
|-------|-------------|
| **tcicore** | *Internal.* Matrix CI, LUCI/rrLU algorithms, and cached function evaluation. Users do not need to depend on this crate directly. |
| **tensorci** | Tensor Cross Interpolation. Contains TCI2 (primary algorithm) and TCI1 (legacy). Use this for low-level TCI control. |
| **quanticstci** | High-level Quantics TCI. Interpolates functions on discrete or continuous grids in the quantics format. |

### Quantics & Transforms

| Crate | Description |
|-------|-------------|
| **quanticstransform** | Quantics transformation operators: shift, flip, Fourier, affine, and more. |

### I/O & Bindings

| Crate | Description |
|-------|-------------|
| **hdf5** | HDF5 serialization compatible with ITensors.jl/ITensorMPS.jl file formats. |
| **capi** | C FFI for language bindings (Julia, Python, etc.). Out of scope for this guide; see [Julia Bindings](julia-bindings.md). |

## Which Crate Should I Use?

| Goal | Recommended crate |
|------|-------------------|
| TCI on a black-box function (high level) | `tensor4all-quanticstci` |
| TCI with fine-grained control | `tensor4all-tensorci` |
| Tree TCI | `tensor4all-treetci` |
| Simple tensor train (create, evaluate, compress) | `tensor4all-simplett` |
| Tensor train with ITensors.jl-style interface | `tensor4all-itensorlike` |
| Tree tensor networks | `tensor4all-treetn` |
| Subdomain decomposition via partitioned TT | `tensor4all-partitionedtt` |
| Quantics transform operators | `tensor4all-quanticstransform` |
| HDF5 I/O compatible with Julia | `tensor4all-hdf5` |

## Internal Crates

`tensor4all-tensorbackend` and `tensor4all-tcicore` are implementation details.
They are not part of the public API surface and you should not depend on them
directly in application code.  Their interfaces may change without notice.
