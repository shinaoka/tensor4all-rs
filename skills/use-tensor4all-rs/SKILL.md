---
name: use-tensor4all-rs
description: Use the tensor4all-rs Rust tensor-network library (TCI, quantics tensor trains, tree tensor networks, DMRG/TDVP/GSE time evolution, adaptive patch interpolation). Use when working with `tensor4all-*` crates (core/simplett/tensorci/quanticstci/treetn/partitionedtt/...), their `IdxTensor`/`SimpleTensorTrain`/`TreeTN`/`QuanticsTensorCI2`/`PartitionedTT` APIs, or the `dmrg`/`tdvp`/`gse_tdvp`/`adaptiveinterpolate` entry points. Also when porting ITensors.jl, QuanticsTCI.jl, or TCI patterns to Rust, or debugging column-major, `rtol`, or 0-vs-1 indexing mismatches in tensor4all-rs code.
license: MIT
---

# Use tensor4all-rs

tensor4all-rs is a Rust workspace of tensor-network crates — TCI, quantics tensor trains (QTT), and tree tensor networks (TreeTN) — inspired by ITensors.jl / ITensorNetworks.jl. Two rules run through everything: **column-major** dense layout, and **conventions** that differ per crate (indexing base, tolerance name). Both bite silently — check them every time.

The library is a **stack**: a layered set of crates where each layer builds on the one below. Land on the right layer before writing code, then honor the conventions that span every layer.

## 1. Set up the project

The library is **not on crates.io**; declare a git or path dependency, and build in release.

```toml
[dependencies]
# Current main; Cargo.lock records the resolved commit:
tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-simplett" }
# Exact source revision or release:
# tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "<sha>", package = "tensor4all-simplett" }
# tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", tag = "<release-tag>", package = "tensor4all-simplett" }
# Local checkout — path relative to your Cargo.toml:
# tensor4all-simplett = { path = "../tensor4all-rs/crates/tensor4all-simplett" }
```

This skill tracks `main`; a release tag may not contain every API described below. Add each crate that your code imports directly. Do not add transitive crates that you never name.

- **Import the crate's `prelude`** — every user-facing crate (`tensor4all-core`, `tensor4all-simplett`, `tensor4all-treetn`, `tensor4all-itensorlike`, `tensor4all-tensorci`, `tensor4all-quanticstci`, `tensor4all-aci`, `tensor4all-interpolativeqtt`) ships a `prelude` re-exporting its public traits plus the types needed to call them. Start with `use tensor4all_<crate>::prelude::*;` instead of guessing trait imports; a missing trait import is the top first-try failure (`error[E0599]`).

- **Backend defaults to pure-Rust `faer` — no system BLAS.** Compute crates enable `tenferro-cpu-faer` by default, so a plain dependency compiles standalone. To link a system BLAS (OpenBLAS / MKL / Apple Accelerate) instead, set `default-features = false` and enable `tenferro-system-blas` on each directly imported crate where it is exposed.
- **Build with `--release`.** Tensor linalg in debug is orders of magnitude slower; TCI and DMRG are unusable without optimization. For benchmarks set `opt-level = 3` (and `lto`, `codegen-units = 1`).
- **IdxTensor scalars** are `f32`, `f64`, `Complex32`, and `Complex64` (`num-complex` 0.4). Compact `Storage` snapshots support only `f64`/`Complex64`; 32-bit tensors retain eager authoritative payloads without promotion. Recipes that build random tensors assume you add `rand` + `rand_chacha` (matching the library's `0.9`) as dev-dependencies.
- **`tensor4all-tensorbackend` is internal** — never depend on it or instantiate `CpuBackend` directly; get scalars, storage, and linalg through the public crates. `tensor4all-tcicore` is the home of `CachedFunction` and `MultiIndex`: depend on it for `CachedFunction`, or take `MultiIndex` re-exported from `tensor4all-partitionedtt`.

Done when `cargo build --release` succeeds and a constant TT evaluates:

```rust
use tensor4all_simplett::prelude::*;
let tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 1.0);
assert!((tt.evaluate(&[0, 0])? - 1.0).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## 2. Land on the right crate

Match the goal to the crate. This is the leading decision — the rest follows.

| Goal | Crate |
|------|-------|
| TCI on a black-box function, high level | `tensor4all-quanticstci` |
| TCI with fine-grained, low-level control | `tensor4all-tensorci` |
| Tree-structured cross interpolation | `tensor4all-treetci` |
| Elementwise ops on tensor trains (ACI) | `tensor4all-aci` |
| Simple TT: create / evaluate / compress | `tensor4all-simplett` |
| TT with ITensors.jl-style named indices, orthogonality center | `tensor4all-itensorlike` |
| Tree tensor networks, arbitrary topology; DMRG/TDVP/GSE sweeps | `tensor4all-treetn` |
| Quantics transform operators (shift, flip, Fourier, affine) | `tensor4all-quanticstransform` |
| Interpolative QTT construction | `tensor4all-interpolativeqtt` |
| Adaptive TCI over subdomain patches; patching add/contract/truncate | `tensor4all-partitionedtt` |
| Core Index / Tensor / contraction / SVD-QR-LU | `tensor4all-core` |
| HDF5 I/O compatible with ITensors.jl | `tensor4all-hdf5` |
| C FFI for language bindings | `tensor4all-capi` |

For each crate's key types and entry points, read [`references/crates.md`](references/crates.md).

## 3. Read the authoritative API, not the source

tensor4all-rs keeps source and rustdoc as the API source of truth, and generates a plain-text API dump you should read first:

- **`docs/api/*.md` in a repository checkout** — generated by `cargo run -p api-dump --release -- . -o docs/api`. Read the relevant generated file when present; never hand-edit it.
- **rustdoc** — `cargo doc -p <crate> --open`, or online at `https://tensor4all.org/tensor4all-rs/rustdoc/<crate>/`.
- **mdBook user guide** — `docs/book/src/` (run `mdbook serve docs/book`). Online: `https://tensor4all.org/tensor4all-rs/`. The guide snippets are runnable and CI-checked.

Only read source when the API doc is insufficient. For concrete task patterns (build a TT, run TCI, apply a quantics transform, contract a TreeTN), pull the recipe from [`references/recipes.md`](references/recipes.md) rather than reconstructing from memory.

## 4. Honor the cross-cutting conventions

These apply across crates and fail silently when ignored. Check them against every data-handling path.

**Dense layout is column-major (Fortran order).** `IdxTensor::from_dense(indices, data)` and all flat buffers, `reshape`, the C API, and HDF5 use column-major: the **first listed index varies fastest**. Matches Julia / ITensors.jl. When feeding NumPy data, use `order="F"`.

**Indexing is 0-indexed everywhere.** Rust sites and grid indices are **0-indexed**, unlike ITensors.jl / QuanticsTCI.jl (1-indexed); QuanticsTCI.jl scripts must subtract 1 from grid indices at the call boundary. This includes `tensor4all-quanticstci` (its callback receives 0-indexed grid indices as `&[usize]`), the low-level `tensor4all-tensorci::crossinterpolate2` (`0..local_dim`), and `tensor4all-partitionedtt::adaptiveinterpolate` (full-domain 0-indexed pivots).

**SVD truncation uses `rtol`, not `cutoff`.** ITensors.jl uses `cutoff`; convert with `rtol = sqrt(cutoff)` (so `cutoff=1e-10` → `rtol=1e-5`). Algorithm-specific option structs such as `CompressionOptions` and `TCI2Options` instead name their threshold `tolerance`; check the exact type. Singular values are indexed `s[[0, i]]`, not `s[[i, i]]`.

**Sweep counts differ by API.** TreeTN `ApplyOptions` uses `nfullsweeps` (each edge visited twice, forward+back); TreeTN DMRG/TDVP options call the same unit `nsweeps`. `tensor4all-itensorlike` uses `nhalfsweeps`, with `nfullsweeps = nhalfsweeps / 2`; its `ContractOptions`/`LinsolveOptions` provide `with_nsweeps(n)` = `with_nhalfsweeps(2 * n)`.

**Quantics bits are big-endian.** Site 0 = most-significant bit, site R-1 = least-significant. `quantics_fourier_operator` output is **bit-reversed** frequency order. `quanticscrossinterpolate_discrete` requires all grid dimensions equal and powers of 2.

**Prefer generic APIs over scalar-specific names.** `f32`, `f64`, `Complex32`, and `Complex64` flow through the same generic entry points. `*_f64` / `*_c64` names exist only at the C-API / FFI boundary.

**No hidden dense materialization in production paths.** `to_dense()`, `contract_to_tensor()`, and full-network `evaluate`-every-element loops are for tests and small examples only — they scale as the product of index dimensions. For long TT / TreeTN comparisons use a direct-sum difference plus a tensor-network norm, or sampled `evaluate()` checks. `ApplyOptions::naive()` is local exact apply (bond dims may grow as products), not full dense.

**Index identity is full-index equality, not `id()`.** Two indices with the same id but different prime level, tags, or direction are distinct. Key maps and sets by the full `Index` value. Select concrete legs/sites by passing the full `Index`, never an id. `AdaptiveInterpolateOptions::patch_order` and `PatchingOptions::patch_order` are validated as exact full-`Index` permutations of the site indices — matching by id alone is rejected.

## 4.5 Common pitfalls

- **`TensorTrain` / `MPS` / `MPO` share one representation.** In `tensor4all-itensorlike`, `TensorTrain` (tree-based), `MPS`, and `MPO` are the same underlying type; MPS-like (1 site index per node) vs MPO-like (2 site indices per node) is a runtime property (`is_mps_like`/`is_mpo_like`), not a type. `tensor4all-simplett`'s positional chain type is a *different* type named `SimpleTensorTrain`. Do not mix the two `TensorTrainError` types (`tensor4all_simplett::TensorTrainError` vs `tensor4all_itensorlike::TensorTrainError`) — they have different variants; qualify the path.
- **MPO-MPO contraction has two layers.** `tensor4all-simplett` (positional `SimpleTensorTrain`, `ContractionOptions`) and `tensor4all-treetn` (TreeTN, `apply_linear_operator`) both contract MPOs, but they are different API layers with different topology models. The TreeTN route is the canonical/general path used by the higher-level APIs; use it unless you specifically want the lightweight positional path.
- **Contraction method changes cost and reliability.** MPO contraction supports different methods (naive / zip-up / fit, selected via `ContractionOptions::method` / `ApplyOptions`). The default is a deliberate tradeoff; benchmark the method for your sizes. `naive` means local exact tensor-network contraction (not full dense materialization) unless the docs explicitly say dense/reference.
- **ACI tolerance scaling.** The option is named `scale_tolerance` (not `rescale_tolerance`). Enable it when the target function's scale varies across the domain: it makes the convergence tolerance scale-relative. It is not an accuracy certificate — hold out samples and check per-sweep diagnostics, and be aware of the early-convergence caveat tracked in tensor4all-rs issue #572.
- **0-indexing everywhere.** Sites and grid indices are 0-indexed in Rust (unlike ITensors.jl's 1-indexing). QuanticsTCI.jl scripts must subtract 1 from grid indices at the call boundary.
- **Bond cap / tolerance vocabulary.** The bond-dimension cap is `max_bond_dim: Option<usize>` (`None` = unlimited) everywhere — ITensors.jl `maxdim` maps to `max_bond_dim: Some(d)`, and `cutoff` maps to `rtol` with `rtol = sqrt(cutoff)`. See `docs/book/src/conventions.md`.

## 5. If you are inside the tensor4all-rs repo itself

This skill is for *using* the library. If the working tree is the tensor4all-rs checkout (you see `AGENTS.md` and `REPOSITORY_RULES.md`), the repo's own rules take over — this skill does not override them:

- Read `AGENTS.md` then `REPOSITORY_RULES.md` before non-trivial work (`AGENTS.md` first points to the shared `tensor4all-agent-rules` repo). Early development: no backward compatibility; remove deprecated code.
- Follow the repo's Rust rules — `thiserror` typed errors in library APIs, `anyhow` only for internal plumbing/binaries/examples/tests, no `unwrap()`/`expect()` in library code. For general Rust idiom, lean on the `rust-skills` skill; this repo hard-enforces the error-handling and no-unwrap rules even where a generic style guide would leave it to judgement.
- Run `cargo fmt --all` and `cargo clippy` before committing. Doc examples must run with real assertions (no `ignore`/`no_run`); verify with `cargo test --doc --release --workspace` and `./scripts/test-mdbook.sh`.
- Dense linalg, SVD/QR/einsum go through `tensor4all-tensorbackend` wrappers (don't write hand-rolled matrix kernels, don't form explicit inverses — use solves). Graph traversals use `petgraph` (via `node_name_network` / `site_index_network`).

## Sources of truth

- Library APIs: source + rustdoc (online rustdoc at `https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_core/`).
- Generated API dump: `docs/api/` (from `api-dump`; do not edit).
- User guide: `docs/book/src/` (mdBook); online `https://tensor4all.org/tensor4all-rs/`.
- In a repository checkout, the developer entry point is `AGENTS.md`; durable repo rules are in `REPOSITORY_RULES.md`; shared agent rules live in the `tensor4all-agent-rules` repo.
- Julia bindings: https://github.com/tensor4all/Tensor4all.jl (wraps `tensor4all-capi`).
- Repository paths: design docs `docs/design/index.md`; plans `docs/plans/`; C API design `docs/CAPI_DESIGN.md`; provenance/citation `docs/PROVENANCE_AND_CITATION_POLICY.md`.
