> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# TensorCI Migration Design

## Goal

Complete the remaining `tensor4all-tensorci` migration work from
`TensorCrossInterpolation.jl` while preserving the Rust crate split and the
existing TreetCI/Torch-style batch convention.

Tracked issues:

- tensor4all/tensor4all-rs#514: public `TensorCI1` and `crossinterpolate1`
- tensor4all/tensor4all-rs#515: `optimize_with_finder` for TCI2
- tensor4all/tensor4all-rs#516: TensorCI conversion constructors
- tensor4all/tensor4all-rs#517: Gauss-Kronrod integration table parity

## Non-Goals

- Do not port `contract` from `TensorCrossInterpolation.jl`.
- Do not reintroduce Julia's `BatchEvaluator` / `Val(1)` / `Val(2)` batch
  interface. Rust keeps the existing TreetCI/Torch-style batch callback
  convention.
- Do not add the deprecated Julia `crossinterpolate` alias.
- Do not expose Julia-internal helper names such as `TtimesPinv`,
  `PinvtimesT`, `getPi`, or `getcross`.

## TensorCI1

Port `TensorCI1<T>` as a public legacy API in `tensor4all-tensorci`.

Public surface:

- `TensorCI1<T>`
- `crossinterpolate1`
- `TCI1Options`
- Accessors for length, local dimensions, rank/link dimensions, maximum sample
  value, and pivot errors
- Evaluation support, with tensor-train conversion as the preferred path for
  repeated evaluation
- Conversion to `TensorTrain<T>`

Implementation notes:

- `MatrixCI` should be an implementation helper, not a public API. Start with a
  `tensorci`-local helper unless another crate needs it later.
- Use existing `tensor4all-tcicore` matrix/LU abstractions where possible.
- Avoid explicit pivot-matrix inverses. Implement Julia's `AtimesBinv` and
  `AinvtimesB` behavior with solve-based formulations.
- Add real and complex tests ported from `test_tensorci1.jl`, including
  additional/global pivot behavior.
- Port the major Julia TCI1 test scenarios, converted to Rust's zero-based
  indices: trivial constant MPS initialization/update behavior, real and complex
  Lorentz functions, duplicate global pivots, `crossinterpolate1` convergence,
  additional pivots, and TensorTrain/evaluate parity on a small dense grid.
- Add a pre-PR speed gate comparing Rust TCI1 against
  `../TensorCrossInterpolation.jl` TCI1 on the same benchmark cases. The gate
  must report Rust seconds, Julia seconds, and the Rust/Julia ratio, and the PR
  should not be marked ready without explicit approval if Rust is more than 2x
  slower on the median wall-clock time for the shared TCI1 benchmark.

## TCI2 Optimization

Add an optimizer entry point that lets callers provide a custom global pivot
finder while keeping `TCI2Options` as a plain configuration struct.

Preferred public shape:

```rust
pub fn optimize_with_finder<T, F, B, G>(
    tci: TensorCI2<T>,
    f: F,
    batched_f: Option<B>,
    options: TCI2Options,
    finder: G,
) -> Result<(TensorCI2<T>, Vec<usize>, Vec<f64>)>
where
    G: GlobalPivotFinder;
```

`crossinterpolate2` should remain the simple default wrapper and internally use
`DefaultGlobalPivotFinder`.

Implementation notes:

- Keep the current generic `GlobalPivotFinder` trait unless object-safety is
  needed later.
- Consider Julia's `checkconvglobalpivot` behavior explicitly. The current Rust
  convergence rule requires no global pivots in the recent history; either keep
  that as the default or expose an option with clear docs.
- Do not add logging complexity unless it is needed; `verbosity` is enough for
  the initial API.

## Conversion Constructors

Add conversion paths that do not require dense full-tensor materialization.

First priority:

- `TensorCI2::from_tensor_train(tt, options) -> Result<Self>` where `tt:
  TensorTrain<T>` is consumed.
- `TensorCI2::from_index_sets(local_dims, i_set, j_set, f) -> Result<Self>` if
  validation can be kept strict and clear.

After `TensorCI1` exists:

- `TensorCI2::from_tci1(&TensorCI1<T>) -> Result<Self>`
- `TensorCI1::from_tci2(&TensorCI2<T>, f) -> Result<Self>`

Validation should reject inconsistent dimensions, out-of-range indices,
malformed pivot sets, and empty/zero cases before any expensive reconstruction.

## Integration

Rust integration should continue to use embedded fixed Gauss-Kronrod tables,
not a runtime rule generator and not a new quadrature dependency.

Implementation notes:

- Keep existing `GK15` and `GK31` support.
- Add additional verified tables as needed, such as `GK41`, `GK51`, and `GK61`.
- Unsupported `gk_order` values should produce a typed `TCIError` with the
  supported order list.
- Document the intended difference from Julia: Julia uses `QuadGK.kronrod`;
  Rust supports a deterministic set of embedded, verified rules.
- Test real and complex polynomial integrals, arbitrary bounds, unsupported
  orders, mismatched bounds, and one-dimensional behavior.

## Verification

For each implementation issue:

- Run `cargo fmt --all`.
- Run `cargo test --release -p tensor4all-tensorci`.
- Run broader workspace checks when public APIs, shared helpers, or docs change.
- Add rustdoc examples for new public APIs, with assertions.
- For the TCI1 issue, run the Julia comparison speed gate against the sibling
  `../TensorCrossInterpolation.jl` checkout before PR creation and include the
  measured Rust/Julia timing ratio in the PR notes.
