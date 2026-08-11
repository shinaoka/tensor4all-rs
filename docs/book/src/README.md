# tensor4all-rs

tensor4all-rs is a Rust implementation of tensor network algorithms, focused on:

- **Tensor Cross Interpolation (TCI)** — adaptive low-rank tensor approximation with TCI2 primary and TCI1 legacy algorithms
- **Quantics Tensor Train (QTT)** — representation of functions in exponentially fine grids
- **Tree Tensor Networks (TreeTN)** — tensor networks with arbitrary tree topology

The library is structured as a workspace of independent crates under `crates/`, designed for modular use and AI-agentic development workflows. Language bindings for Julia are provided through the C API layer.

Source code and issue tracker: [github.com/tensor4all/tensor4all-rs](https://github.com/tensor4all/tensor4all-rs)

The repository root `README.md` stays intentionally concise. Longer runnable
examples live in this guide and the guide examples are exercised in CI.

---

## Where to start

| I want to... | Go to |
|---|---|
| Understand tensor networks from scratch | [Concepts](concepts.md) |
| Install and run my first example | [Getting Started](getting-started.md) |
| Understand the crate structure | [Architecture & Crate Guide](architecture.md) |
| Come from ITensors.jl and map types | [Conventions](conventions.md) |
| Browse the full API reference | [rustdoc API reference](rustdoc/tensor4all_core/) |
| Use tensor4all-rs from Julia | [Julia Bindings](julia-bindings.md) |

---

## Feature highlights

- **Dynamic Index/Tensor system** inspired by ITensors.jl: indices carry semantic identity, tensor contraction aligns axes by index rather than position
- **Tensor Cross Interpolation (TCI)**: approximates high-dimensional tensors from a small number of evaluations using cross-approximation; TCI2 is the primary algorithm and TCI1 is available for legacy parity
- **Quantics Tensor Train (QTT)**: represents smooth functions on exponentially fine grids; includes transformation operators (affine, shift, sum)
- **Tree Tensor Networks**: arbitrary-topology TTN, not limited to chains (MPS/MPO); supports standard MPS/MPO as special cases with runtime topology checks
- **C API** (`tensor4all-capi`): FFI surface for language bindings, currently used by [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl). This repository is early development with no backward-compatibility guarantee; the C ABI should not be treated as stable.

---

## Not sure where you fit?

If you are new to the library, read [Getting Started](getting-started.md) for a short working example, then consult [Concepts](concepts.md) for background on the data structures used throughout.
