# Getting Started

## Prerequisites

You need the Rust toolchain. If you do not have it installed, follow the instructions at <https://rustup.rs/>.

## Adding tensor4all-rs to Your Project

tensor4all-rs is a collection of crates. The crates are not published to
crates.io yet, so use git dependencies from an external project:

```toml
[dependencies]
# Basic tensor train construction and manipulation
tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-simplett" }

# Indexed tensors and ITensors-like TensorTrain/TT-SVD
tensor4all-core = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-core" }
tensor4all-itensorlike = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-itensorlike" }

# Tensor Cross Interpolation (TCI)
tensor4all-tensorci = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-tensorci" }

# Quantics TCI (combines quantics encoding with TCI)
tensor4all-quanticstci = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-quanticstci" }

# Tree tensor networks
tensor4all-treetn = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-treetn" }
```

You do not need to add all of them — only include the crates relevant to your use case.

When working from a local checkout, use path dependencies instead. Paths are
relative to your project's `Cargo.toml`; adjust `../tensor4all-rs` as needed:

```toml
[dependencies]
tensor4all-simplett = { path = "../tensor4all-rs/crates/tensor4all-simplett" }
tensor4all-core = { path = "../tensor4all-rs/crates/tensor4all-core" }
tensor4all-itensorlike = { path = "../tensor4all-rs/crates/tensor4all-itensorlike" }
tensor4all-tensorci = { path = "../tensor4all-rs/crates/tensor4all-tensorci" }
tensor4all-quanticstci = { path = "../tensor4all-rs/crates/tensor4all-quanticstci" }
tensor4all-treetn = { path = "../tensor4all-rs/crates/tensor4all-treetn" }
```

## First Example: Tensor Trains

The following example uses `tensor4all-simplett` to create a constant tensor train, evaluate it at a specific index, and compress it.

```rust
use tensor4all_simplett::prelude::*;

fn main() {
    // Create a constant tensor train with local dimensions [2, 3, 4].
    // Every entry of the represented tensor equals 1.0.
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

    // Evaluate at a specific multi-index.
    let value = tt.evaluate(&[0, 1, 2]).unwrap();
    assert!((value - 1.0).abs() < 1e-12);

    // Sum over all indices (2 * 3 * 4 = 24 elements, all 1.0).
    let total = tt.sum();
    assert!((total - 24.0).abs() < 1e-12);

    // Compress with a truncation tolerance.
    let options = CompressionOptions {
        tolerance: 1e-10,
        max_bond_dim: Some(20),
        ..Default::default()
    };
    let compressed = tt.compressed(&options).unwrap();
    assert!((compressed.sum() - 24.0).abs() < 1e-10);

    println!("sum = {}", compressed.sum());
}
```

Run it with:

```bash
cargo run
```

## Next Steps

- [Concepts](concepts.md) — learn about tensor trains, bond dimensions, and TCI before diving deeper.
- [Dense Tensor to Tensor Train with TT-SVD](guides/dense-tt-svd.md) — decompose an existing full indexed tensor with sequential SVD.
- [Guides](guides/tensor-basics.md) — step-by-step walkthroughs for tensor basics, TCI, quantics transforms, and tree tensor networks.
