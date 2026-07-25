# tensor4all-rs

[![CI](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml/badge.svg)](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml)

A Rust implementation of tensor networks: TCI, Quantics Tensor Train, and Tree Tensor Networks.

Related project: [tenferro-rs](https://github.com/tensor4all/tenferro-rs) provides tensor backend components used by tensor4all-rs.

## Features

- **ITensors.jl-like dynamic tensors**: Flexible `Index` system with dynamic-rank `Tensor`
- **Tensor Cross Interpolation**: TCI2 primary algorithm plus legacy TCI1 compatibility for efficient high-dimensional function approximation
- **Quantics Tensor Train**: Binary encoding of continuous variables with transformation operators
- **Tree Tensor Networks**: Arbitrary topology with canonicalization, truncation, and contraction
- **C API**: Minimal Julia-facing FFI surface for indices, structured tensors, TreeTN, and quantics materialization

## Quick Start

The crates are not published to crates.io yet. Use git dependencies from an
external project:

```toml
[dependencies]
tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-simplett" }
```

When working from a local checkout, use a path dependency instead:

```toml
[dependencies]
tensor4all-simplett = { path = "../tensor4all-rs/crates/tensor4all-simplett" }
```

```rust
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
assert!((tt.evaluate(&[0, 1, 2]).unwrap() - 1.0).abs() < 1e-12);
assert!((tt.sum() - 24.0).abs() < 1e-12);
```

The root `README.md` is kept intentionally short and its Rust snippets are
validated in CI. Longer runnable examples live in the
[User Guide](https://tensor4all.org/tensor4all-rs/).

## Crate Overview

| Crate | Description |
|-------|-------------|
| [tensor4all-core](crates/tensor4all-core/) | Core types: Index, Tensor, contraction, SVD, QR |
| [tensor4all-simplett](crates/tensor4all-simplett/) | Simple TT/MPS with compression |
| [tensor4all-aci](crates/tensor4all-aci/) | Alternating Cross Interpolation for elementwise tensor train operations |
| [tensor4all-itensorlike](crates/tensor4all-itensorlike/) | ITensors.jl-like TensorTrain API |
| [tensor4all-treetn](crates/tensor4all-treetn/) | Tree tensor networks with arbitrary topology |
| [tensor4all-tensorci](crates/tensor4all-tensorci/) | Tensor Cross Interpolation (TCI2 primary, TCI1 legacy) |
| [tensor4all-quanticstci](crates/tensor4all-quanticstci/) | High-level Quantics TCI interface |
| [tensor4all-interpolativeqtt](crates/tensor4all-interpolativeqtt/) | Interpolative QTT construction |
| [tensor4all-quanticstransform](crates/tensor4all-quanticstransform/) | Quantics transformation operators |
| [tensor4all-treetci](crates/tensor4all-treetci/) | Tree-structured cross interpolation |
| [tensor4all-partitionedtt](crates/tensor4all-partitionedtt/) | Partitioned tensor trains and adaptive TCI interpolation |
| [tensor4all-hdf5](crates/tensor4all-hdf5/) | ITensors.jl-compatible HDF5 serialization |
| [tensor4all-capi](crates/tensor4all-capi/) | C FFI for language bindings |

## Documentation

- **[User Guide](https://tensor4all.org/tensor4all-rs/)** — tutorials, architecture, conventions
- **[API Reference (rustdoc)](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_core/)** — generated API documentation
- **[Julia Bindings](https://github.com/tensor4all/Tensor4all.jl)** — Tensor4all.jl wrapper
- **[Design Documents](docs/design/index.md)** — architecture and design decisions
- **[Provenance and Citation Policy](docs/PROVENANCE_AND_CITATION_POLICY.md)** — which components build on which external projects, algorithm origins, and how to cite

## Acknowledgments

Inspired by [ITensors.jl](https://github.com/ITensor/ITensors.jl) and
[ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl). We
acknowledge fruitful discussions with M. Fishman and E. M. Stoudenmire at CCQ,
Flatiron Institute.

A per-component table of which external projects each crate builds on, and
the algorithm-origin references, is maintained in the
[Provenance and Citation Policy](docs/PROVENANCE_AND_CITATION_POLICY.md).

## How to Cite

If you use tensor4all-rs in research, please read the
[Provenance and Citation Policy](docs/PROVENANCE_AND_CITATION_POLICY.md)
and cite the original papers of the algorithms your work relies on, together
with the software papers of the upstream libraries those components build on
(for example the TCI paper,
[SciPost Phys. 18, 104 (2025)](https://doi.org/10.21468/SciPostPhys.18.3.104),
for the TCI components, and the ITensor paper,
[SciPost Phys. Codebases 4 (2022)](https://doi.org/10.21468/SciPostPhysCodeb.4),
for the tree tensor network components). Please also check the citation
policies of those upstream projects and apply them recursively. This is the
permanent citation style for this project: a future tensor4all-rs software
paper will add to, not replace, these upstream citations. Until then,
reference tensor4all-rs directly by repository URL and version or commit.

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
