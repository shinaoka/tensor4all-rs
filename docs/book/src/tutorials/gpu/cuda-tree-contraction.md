# CUDA TreeTN Contraction

This experimental path contracts a small dense `TreeTN<IdxTensor>` on one NVIDIA GPU. Transfers are explicit: tensor4all-rs never uploads, downloads, or falls back to the CPU inside the contraction.

## Prerequisites

- A stable Rust toolchain; see [Getting Started](../../getting-started.md).
- An NVIDIA GPU and CUDA driver/toolkit compatible with the CUDA versions used by the pinned tenferro/cubecl dependencies. No wider version or compute-capability support matrix is currently promised.
- The intended device must be visible as CUDA ordinal 0.
- Build with the non-default `tenferro-cuda` feature. The commands also name the default `tenferro-cpu-faer` feature explicitly because the quickstart computes a CPU reference alongside the CUDA contraction.

Clone tensor4all-rs and run the checked quickstart:

```bash
git clone https://github.com/tensor4all/tensor4all-rs.git
cd tensor4all-rs
cargo run --release -p tensor4all-treetn --example cuda_quickstart \
  --features tenferro-cuda,tenferro-cpu-faer
```

The complete checked source is embedded below. It is rendered as text rather than an mdBook-tested Rust block because executing it requires CUDA hardware; the feature-gated example itself is compile-checked and run on CUDA.

```text
{{#include ../../../../../crates/tensor4all-treetn/examples/cuda_quickstart.rs}}
```

Source: [`cuda_quickstart.rs`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-treetn/examples/cuda_quickstart.rs).

It performs this flow:

1. Build a two-node host `TreeTN` and compute a CPU reference.
2. Create one caller-owned `CudaExecutionContext` for visible ordinal 0.
3. Upload every node with `TreeTN::upload_cuda`.
4. Contract all internal bonds with `contract_to_tensor_cuda`.
5. Verify that the result is still resident in the same CUDA context.
6. Download explicitly and assert a maximum CPU/GPU residual of at most `1e-10`.

A successful run prints the GPU name and residual, for example:

```text
device="NVIDIA A100 80GB PCIe" residual_max_abs=0.000e0
```

## Using it from another project

The crates are not published to crates.io yet. Enable CUDA on both crates imported by the quickstart:

```toml
[dependencies]
tensor4all-core = { git = "https://github.com/tensor4all/tensor4all-rs", features = ["tenferro-cuda", "tenferro-cpu-faer"] }
tensor4all-treetn = { git = "https://github.com/tensor4all/tensor4all-rs", features = ["tenferro-cuda", "tenferro-cpu-faer"] }
```

Create a binary project, put the dependency entries above under `[dependencies]` in `Cargo.toml`, and copy the embedded program to `src/main.rs`:

```bash
cd ..
cargo new cuda-tree-quickstart
cd cuda-tree-quickstart
# Add the dependency entries above to Cargo.toml.
cp ../tensor4all-rs/crates/tensor4all-treetn/examples/cuda_quickstart.rs src/main.rs
cargo run --release
```

Keep one `CudaExecutionContext` for upload, contraction, synchronization, and download. Mixing host and CUDA nodes, CUDA contexts, or node dtypes returns a typed error before contraction; it does not trigger a hidden transfer or CPU fallback.

## Current limits

This is a dense full-network contraction, so output memory scales with the product of external-index dimensions. It currently supports only:

- dense, untracked `IdxTensor` nodes;
- one dtype and one CUDA context across the tree;
- visible CUDA ordinal 0;
- full contraction to one dense tensor.

CUDA SVD, QR, truncation, TreeTN-to-TreeTN contraction, zip-up/fitting, TCI/ACI, automatic device selection, and multi-GPU execution are not yet supported.

For timing, use the separate `cuda_tree_contraction` example, which reports context setup, upload, warm-up, steady-state GPU contraction plus synchronization, download, and CPU contraction independently:

```bash
cargo run --release -p tensor4all-treetn --example cuda_tree_contraction \
  --features tenferro-cuda,tenferro-cpu-faer
```
