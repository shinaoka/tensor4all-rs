# tensor4all-tensorbackend

> This is an internal crate. Most users should use `tensor4all-core` instead.

Compact storage backends for `f64`/`Complex64`, eager tensor algebra primitives
for all four core scalar types (`f32`, `f64`, `Complex32`, `Complex64`)
backed by tenferro-rs.

## Key Types

- `AnyScalar` — dynamic scalar type covering `f32`, `f64`, `Complex32`, and `Complex64`
- `Matrix` — shared column-major dense matrix boundary for tensor4all crates
- `Storage` — compact `f64`/`Complex64` dense, diagonal, and structured payload snapshots
- `StructuredStorage` — axis-class-aware compact payload storage
- `svd_backend`, `qr_backend`, `solve_matrix`, `full_piv_lu_matrix` —
  tenferro-backed dense linear algebra entry points

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_tensorbackend/)
