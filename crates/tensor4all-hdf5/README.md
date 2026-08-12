# tensor4all-hdf5

HDF5 serialization for tensor4all-rs, compatible with ITensors.jl / ITensorMPS.jl file formats.

## Key Types

- `save_itensor()` / `load_itensor()` — read/write `IdxTensor` as ITensors.jl `ITensor`
- `save_mps()` / `load_mps()` — read/write `TensorTrain` as ITensorMPS.jl `MPS`
- `save_treetn()` / `append_treetn()` / `load_treetn()` — read/write a general
  `TreeTN<IdxTensor, V>` (`String` or `usize` node names) using the
  tensor4all-rs `TreeTN` v1 schema: one `node_i/` ITensor subgroup per node
  with a `node_name` attribute; topology is recovered on load from shared
  index identity via `TreeTN::from_tensors`

## TreeTN schema (v1)

```text
<group>/
  @type = "TreeTN"
  @version = 1
  node_count: Int64
  node_1/ ... node_N/
    @node_name: VarLenUnicode
    (ITensor — same per-node schema as save_itensor)
```

Edges are not stored explicitly: bond connections are recovered on load from
shared `Index` identity, exactly as the tree was assembled originally.

## Feature Flags

- `link` (default) — compile-time HDF5 linking
- `runtime-loading` — dlopen for FFI environments

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_hdf5/)
