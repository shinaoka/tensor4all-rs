# HDF5 Serialization

The `tensor4all-hdf5` crate reads and writes tensor4all-rs data structures in
HDF5 files. Three storage schemas are supported:

| Type | Schema | Notes |
|------|--------|-------|
| [`IdxTensor`](rustdoc/tensor4all_core/struct.IdxTensor.html) | `ITensor` | ITensors.jl compatible |
| [`TensorTrain`](rustdoc/tensor4all_itensorlike/struct.TensorTrain.html) | `MPS` | ITensorMPS.jl compatible |
| [`TreeTN`](rustdoc/tensor4all_treetn/struct.TreeTN.html) | `TreeTN` (v1) | tensor4all-rs schema; no ITensorNetworks.jl equivalent exists |

## TreeTN storage

A general tree tensor network is stored as one ITensor subgroup per node,
plus a `node_count` dataset:

```text
<name>/
  @type = "TreeTN"
  @version = 1
  node_count: Int64
  node_1/ ... node_N/
    @node_name: VarLenUnicode
    (ITensor — same per-node schema as save_itensor)
```

Two design decisions keep the schema minimal:

- **Edges are not stored explicitly.** Bond connections are recovered on load
  from shared `Index` identity via `TreeTN::from_tensors`, exactly as the
  tree was assembled originally. The per-node ITensor schema already
  preserves full index identity (id + prime level + tags), and every bond
  index appears in exactly the two nodes it connects, so reconstruction is
  exact.
- **Node names travel with each node** as a `node_name` attribute. Unlike the
  topology, node names (`TreeTN`'s `V` type) are *not* recoverable from the
  tensors, so they are stored explicitly. Any node name type that round-trips
  through strings works — `String` and `usize` are the common cases.

## Example

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_hdf5::{load_treetn, save_treetn};
use tensor4all_treetn::TreeTN;

// A 3-site chain: t0 -- t1 -- t2
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_dyn(4);
let b12 = DynIndex::new_dyn(4);

let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0; 8])?;
let t1 = IdxTensor::from_dense(vec![b01, s1, b12.clone()], vec![2.0; 32])?;
let t2 = IdxTensor::from_dense(vec![b12, s2], vec![3.0; 8])?;

let tn = TreeTN::<IdxTensor, String>::from_tensors(
    vec![t0, t1, t2],
    vec!["left".to_string(), "center".to_string(), "right".to_string()],
)?;

let dir = tempfile::tempdir()?;
let path = dir.path().join("treetn.h5");
let path = path.to_str().unwrap();

save_treetn(path, "tn", &tn)?;
let loaded = load_treetn::<String>(path, "tn")?;

// Structure and node names survive the round trip.
assert_eq!(loaded.node_count(), 3);
assert_eq!(loaded.edge_count(), 2);
assert_eq!(loaded.node_names(), vec!["left".to_string(), "center".to_string(), "right".to_string()]);
# Ok(())
# }
```

## When to use which schema

- **Chains (MPS/MPO)**: use `save_mps` / `load_mps` for ITensorMPS.jl
  compatibility, or `save_treetn` when you want the orthogonality-center-free
  TreeTN representation.
- **General trees**: `save_treetn` / `load_treetn` is the only option — no
  upstream ITensorNetworks.jl format exists.
