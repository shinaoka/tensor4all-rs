# Tree Tensor Networks

The `tensor4all-treetn` crate provides a generic tree tensor network (`TreeTN`) that supports
arbitrary tree topologies — not just linear chains. This guide covers construction, canonicalization,
common operations, and the sweep-counting convention used by iterative algorithms.

## Creating a TreeTN

Use `TreeTN::from_tensors` to build a network from individual tensors. The topology is inferred
automatically: two tensors are connected by an edge when they share a **bond index** (a `DynIndex`
that appears in both tensors). Physical (site) indices appear in exactly one tensor.

The example below builds a 3-site MPS chain `t0 -- t1 -- t2`:

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
use tensor4all_treetn::TreeTN;

// Site indices (appear in one tensor each)
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);

// Bond indices (shared between adjacent tensors)
let b01 = DynIndex::new_dyn(3);
let b12 = DynIndex::new_dyn(3);

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0; 6]).unwrap();
let t1 = TensorDynLen::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 18]).unwrap();
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0; 6]).unwrap();

let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(
    vec![t0, t1, t2],
    vec![0, 1, 2],   // vertex labels
).unwrap();
assert_eq!(ttn.node_count(), 3);
assert_eq!(ttn.edge_count(), 2);
```

Each vertex is labelled by the user-supplied key (here `usize`). Any type that is `Eq + Hash` works.
The tensor at vertex `v` is retrieved with `ttn[v]`.

## Non-Chain Topologies

`TreeTN` is not limited to linear chains. Any tree structure works, including Y-shapes, stars,
and arbitrary branching topologies. Below is a **Y-shaped** tree with a central hub connected
to three leaves:

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
use tensor4all_treetn::TreeTN;

// Site indices for the four nodes
let s_hub = DynIndex::new_dyn(2);
let s_a   = DynIndex::new_dyn(2);
let s_b   = DynIndex::new_dyn(2);
let s_c   = DynIndex::new_dyn(2);

// Bond indices connecting hub to each leaf
let b_ha = DynIndex::new_dyn(3);
let b_hb = DynIndex::new_dyn(3);
let b_hc = DynIndex::new_dyn(3);

// Hub tensor has 1 site index + 3 bond indices (2 * 3 * 3 * 3 = 54 elements)
let t_hub = TensorDynLen::from_dense(
    vec![s_hub, b_ha.clone(), b_hb.clone(), b_hc.clone()],
    vec![1.0_f64; 54],
).unwrap();

// Leaf tensors each have 1 site index + 1 bond index
let t_a = TensorDynLen::from_dense(vec![b_ha, s_a], vec![1.0; 6]).unwrap();
let t_b = TensorDynLen::from_dense(vec![b_hb, s_b], vec![1.0; 6]).unwrap();
let t_c = TensorDynLen::from_dense(vec![b_hc, s_c], vec![1.0; 6]).unwrap();

let ttn = TreeTN::<_, String>::from_tensors(
    vec![t_hub, t_a, t_b, t_c],
    vec!["hub".into(), "A".into(), "B".into(), "C".into()],
).unwrap();

// Y-shape: 4 nodes, 3 edges
assert_eq!(ttn.node_count(), 4);
assert_eq!(ttn.edge_count(), 3);
```

## Canonicalization

Canonicalization orthogonalizes the network toward a chosen **root vertex**, turning all tensors
except the root into isometries. This puts the full norm information into the root tensor and is a
prerequisite for efficient norm computation and truncation.

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
use tensor4all_treetn::{TreeTN, CanonicalizationOptions, TruncationOptions};

let s0 = DynIndex::new_dyn(2);
let bond = DynIndex::new_dyn(3);
let s1 = DynIndex::new_dyn(2);

let t0 = TensorDynLen::from_dense(
    vec![s0, bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let t1 = TensorDynLen::from_dense(
    vec![bond, s1], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();

let ttn = TreeTN::<_, i32>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();

// Canonicalize toward vertex 0
let ttn = ttn.canonicalize([0], CanonicalizationOptions::default()).unwrap();
assert!(ttn.is_canonicalized());

// Truncate bond dimensions after canonicalization
let ttn = ttn.truncate([0], TruncationOptions::default().with_max_rank(2)).unwrap();
assert_eq!(ttn.node_count(), 2);
```

`TruncationOptions` supports both a maximum rank (`with_max_rank`) and a relative tolerance
(`with_rtol`). Truncation discards small singular values on each bond, reducing memory and
contraction cost at the expense of a controlled approximation error.

## Operations

### Norm Computation

`norm()` returns the Frobenius norm of the tensor represented by the network. It canonicalizes
internally so the result is always exact (up to floating-point precision).

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
use tensor4all_treetn::TreeTN;

let s = DynIndex::new_dyn(2);
let t = TensorDynLen::from_dense(vec![s], vec![3.0_f64, 4.0]).unwrap();
let mut ttn = TreeTN::<_, i32>::from_tensors(vec![t], vec![0]).unwrap();

let norm = ttn.norm().unwrap();
// ||[3, 4]|| = 5
assert!((norm - 5.0).abs() < 1e-10);
```

### Dense Conversion

`to_dense()` contracts all bond indices and returns a single `TensorDynLen` whose indices are the
physical (site) indices of the network. For large networks this can be expensive — use it mainly
for testing or small examples.

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex, TensorLike};
use tensor4all_treetn::TreeTN;

let s0 = DynIndex::new_dyn(2);
let bond = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);

let t0 = TensorDynLen::from_dense(
    vec![s0, bond.clone()], vec![1.0_f64, 0.0, 0.0, 1.0],
).unwrap();
let t1 = TensorDynLen::from_dense(
    vec![bond, s1], vec![1.0_f64, 0.0, 0.0, 1.0],
).unwrap();

let ttn = TreeTN::<_, i32>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
let dense = ttn.to_dense().unwrap();
assert_eq!(dense.num_external_indices(), 2);
```

### Addition

Two `TreeTN`s with the **same topology and matching site indices** can be added with `add`. The
result has the same tree structure, but each bond dimension is the sum of the two input bond
dimensions (direct-sum construction).

```rust
use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
use tensor4all_treetn::TreeTN;

let s = DynIndex::new_dyn(2);
let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap();
let ttn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();

// sum represents ttn + ttn
let sum = ttn.add(&ttn).unwrap();
let dense = sum.to_dense().unwrap();
let expected = TensorDynLen::from_dense(vec![s], vec![2.0, 4.0]).unwrap();
assert!(dense.distance(&expected).unwrap() < 1e-12);
```

After addition the bond dimensions grow, so it is common to follow up with `truncate` to keep them
manageable.

### Contraction

A `TreeTN` can be fully contracted to a scalar or dense tensor via `to_dense()`. For
network-to-network contractions (e.g., computing inner products or applying operators), use the
higher-level contraction APIs provided by `tensor4all-treetn`. Refer to the crate documentation for
variational (fit) contraction, which avoids the exponential cost of naive full contraction.

## SimpleTT vs TreeTN

`tensor4all-simplett` provides a simpler `SimpleTensorTrain` type optimized for linear chains.
Choose based on your needs:

| Feature | `SimpleTensorTrain` (simplett) | `TreeTN` (treetn) |
|---------|--------------------------|-------------------|
| Topology | Linear chain only | Any tree |
| Storage | `Vec<Tensor3>` (3-leg tensors) | Named graph of arbitrary-rank tensors |
| Performance | Lower overhead for chains | General but slightly more overhead |
| Use case | MPS, simple 1D | Branching geometries, general TTN |

**Rule of thumb**: If your tensor network is a linear chain and you want maximum performance,
use `SimpleTensorTrain`. If you need branching structure, named nodes, or plan to compose multiple
operators on a tree, use `TreeTN`. You can convert between them using
`tensor_train_to_treetn` from the `simplett_bridge` module.

## Sweep Counting

Iterative algorithms (DMRG-like local updates, fitting) sweep through the network edges multiple
times. `tensor4all-treetn` uses the **`nfullsweeps`** convention:

| Term | Meaning |
|------|---------|
| **Half sweep** | Visit each edge once in a single direction (used in `tensor4all-itensorlike`) |
| **Full sweep** | Visit each edge twice — forward **and** backward (Euler tour) |

The relationship is `nfullsweeps = nhalfsweeps / 2`.

Concretely:

- `nfullsweeps = 1` — each edge is updated twice (once per direction).
- `nfullsweeps = 2` — each edge is updated four times total.

When interoperating with code that uses `nhalfsweeps`, divide by two before passing the value to
`tensor4all-treetn` APIs.
