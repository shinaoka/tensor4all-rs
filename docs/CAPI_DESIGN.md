# C API Design Guidelines

This document describes the current `tensor4all-capi` surface and the design
rules that new C-facing entry points must follow. The crate intentionally keeps
a small Julia-oriented ABI rather than mirroring the full Rust API.

## Current Scope

The exported surface is limited to:

- `t4a_index`: immutable tensor indices
- `t4a_tensor`: dense real/complex tensors plus contraction
- `t4a_treetn`: general tree tensor networks used as state/operator handles
- `t4a_qtt_layout`: canonical binary QTT layout descriptors
- `t4a_qtransform_*_materialize`: quantics operators materialized directly as
  `t4a_treetn`
- `t4a_last_error_message`: thread-local error retrieval

Removed subsystems such as SimpleTT, TreeTCI, QuanticsTCI, quantics-grid
objects, and HDF5 are intentionally outside this ABI.

## Design Rules

### Opaque Handles

All exported objects are opaque pointer wrappers around Rust-owned values:

- `t4a_index`
- `t4a_tensor`
- `t4a_treetn`
- `t4a_qtt_layout`

Callers never inspect internal layout directly. Construction happens through
`*_new(..., out)` functions, and ownership is released with `*_release`.

### Ownership and Mutability

| Type | Mutability | Clone semantics | Notes |
|------|------------|-----------------|-------|
| `t4a_index` | Immutable | Deep clone | Small metadata handle |
| `t4a_tensor` | Immutable | Deep clone | Dense tensor value |
| `t4a_treetn` | Mutable | Deep clone | Supports orthogonalization/truncation |
| `t4a_qtt_layout` | Immutable | Deep clone | Layout descriptor only |

Every opaque type provides:

- `t4a_<type>_release`
- `t4a_<type>_clone`
- `t4a_<type>_is_assigned`

Prefer explicit `t4a_<type>_release`, `t4a_<type>_clone`, and
`t4a_<type>_is_assigned` definitions that delegate to the shared helpers in
`lib.rs`.

### Error Handling

Functions that can fail return the generated `enum t4a_status_code` status
enum. The public header defines it as `typedef enum t4a_status_code { ... }
t4a_status_code;` so C consumers get a named status type rather than a bare
`int`:

- `T4A_SUCCESS`
- `T4A_NULL_POINTER`
- `T4A_INVALID_ARGUMENT`
- `T4A_TAG_OVERFLOW`
- `T4A_TAG_TOO_LONG`
- `T4A_BUFFER_TOO_SMALL`
- `T4A_INTERNAL_ERROR`
- `T4A_NOT_IMPLEMENTED`

Do not add new `int` status returns, generic `StatusCode` typedefs, or
function-specific status integer types. Keep new status variants in the shared
`t4a_status_code` enum unless there is a binding-level reason to introduce a
new typed result channel.

Error details are stored in thread-local storage. The failing FFI call and
`t4a_last_error_message` retrieval must run on the same OS thread; diagnostics
are not process-global and do not cross thread boundaries. Preserve the
original diagnostic across the FFI boundary and do not discard `Err(e)` or
panic payloads. New entry points should use the helpers in `lib.rs`:

- `run_catching` for constructor-style `out` functions
- `unwrap_catch` / `unwrap_catch_ptr` for `catch_unwind` wrappers
- `capi_error` / `err_status` when mapping Rust failures to status codes

Bindings retrieve diagnostics via:

```c
size_t needed = 0;
t4a_last_error_message(NULL, 0, &needed);
```

followed by a second call with a sufficiently large UTF-8 buffer.

### Panic Safety

Rust panics must never cross the FFI boundary. Every exported function must
either:

- use `run_catching`, or
- wrap its body in `catch_unwind` and forward the message via `unwrap_catch`

The panic message itself is part of the user-facing diagnostic surface.

### Query-Then-Fill Buffers

Variable-length outputs use a two-call pattern:

1. Call with `buf = NULL` to query required length.
2. Allocate that length on the caller side.
3. Call again to fill the buffer.

This pattern is used for:

- `t4a_index_tags`
- `t4a_tensor_dims`
- `t4a_tensor_indices`
- `t4a_treetn_neighbors`
- `t4a_treetn_site_indices`
- `t4a_last_error_message`

When the caller provides an undersized buffer, return
`T4A_BUFFER_TOO_SMALL` and write the required length to `out_len`.

### Data Layout

Dense tensor payloads use column-major order to match Julia and Fortran
conventions.

- Real buffers are plain `double[]`
- Complex buffers are interleaved doubles:
  `[re0, im0, re1, im1, ...]`

This applies to both tensor constructors and copy-out functions:

- `t4a_tensor_new_dense_f64`
- `t4a_tensor_new_dense_c64`
- `t4a_tensor_copy_dense_f64`
- `t4a_tensor_copy_dense_c64`

### TreeTN Semantics

`t4a_treetn` is the general structural handle used by bindings:

- construct from site tensors with `t4a_treetn_new`
- inspect topology via `num_vertices`, `neighbors`, `site_indices`, `linkind`
- mutate tensors in place with `t4a_treetn_set_tensor`
- canonicalize or truncate with `t4a_treetn_orthogonalize` and
  `t4a_treetn_truncate`
- evaluate amplitudes / matrix elements with handle-based
  `t4a_treetn_evaluate`
- materialize to a dense tensor with `t4a_treetn_to_dense`

For operator-like nodes, `t4a_treetn_site_indices` returns external indices in the
tensor's own external-index order. This is required so bindings can interpret
materialized transform operators consistently.

### QTT Layout and Materialization

Quantics transforms do not expose an intermediate linear-operator handle in the
C ABI. Instead, the binding provides a canonical layout and materializes the
operator directly as a `t4a_treetn`.

Supported layout kinds: `Interleaved` and `Fused`.

Current materializers:

- `t4a_qtransform_shift_materialize`
- `t4a_qtransform_flip_materialize`
- `t4a_qtransform_phase_rotation_materialize`
- `t4a_qtransform_cumsum_materialize`
- `t4a_qtransform_fourier_materialize`
- `t4a_qtransform_affine_materialize`

Current intentional limitation: affine materialization requires `Fused`
layout. The pullback operator `f(y) = g(A * y + b)` is not exposed as a
separate C API entry point; bindings derive it by calling
`t4a_qtransform_affine_materialize` and transposing the resulting
`LinearOperator` at the binding layer (the pullback is exactly the transpose
of the forward operator).

These constraints should stay explicit in both the function documentation and
the error message returned to bindings.

## Naming Conventions

- Functions use the `t4a_` prefix
- Status codes use the `T4A_` prefix
- Opaque types and enums use `snake_case`, including `t4a_status_code`
- Constructor-style functions write to an `out` pointer and return
  `enum t4a_status_code`

Prefer type-generic names inside Rust. Scalar-specialized names are acceptable
at the FFI boundary when the ABI needs concrete layouts.

## Header Generation

The public C header is generated with `cbindgen`:

```bash
mkdir -p crates/tensor4all-capi/include
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Regenerate the header whenever exported types, enums, constants, or function
signatures change.

## Module Reference

| Module | Export family | Purpose |
|--------|---------------|---------|
| `index.rs` | `t4a_index_*` | Index constructors and metadata access |
| `tensor.rs` | `t4a_tensor_*` | Dense tensor construction, export, contraction |
| `treetn.rs` | `t4a_treetn_*` | Tree tensor network inspection and core ops |
| `quanticstransform.rs` | `t4a_qtt_layout_*`, `t4a_qtransform_*` | Canonical QTT layouts and transform materialization |
| `types.rs` | exported enums and opaque handles | ABI-facing type definitions |
| `lib.rs` | `t4a_status_code`, `t4a_last_error_message` | shared error/panic handling |

## Binding Notes

For Julia bindings specifically:

- pass column-major dense buffers directly
- represent complex arrays as interleaved `Float64`
- treat all opaque pointers as owned handles with explicit release
- always read `t4a_last_error_message` after a non-success status

The C surface is deliberately smaller than the Rust surface. Higher-level Julia
ergonomics should be implemented in `Tensor4all.jl`, not by expanding the ABI
without a clear need.
