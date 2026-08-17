# tensor4all-rs recipes

Doctest-style task patterns distilled from the mdBook guides (`docs/book/src/`). Each is the
minimal correct shape — confirm option defaults and scalar support against rustdoc before
production use.

## Dependency

Declare every `tensor4all-*` crate imported by your code as a git/path dependency and build
in release — see [`SKILL.md`](../SKILL.md) §1 for the authoritative setup (revision pinning,
backend features, `--release`, dev-deps). Each recipe below assumes that is done.

## Build, evaluate, compress a TT (simplett)

```rust
use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, SimpleTensorTrain};

let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
assert!((tt.evaluate(&[0, 1, 2]).unwrap() - 1.0).abs() < 1e-12);
assert!((tt.sum() - 24.0).abs() < 1e-12); // 2*3*4 entries

// Adding two constants inflates bond dim; compression recovers rank 1.
let big = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0)
    .add(&SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 2.0))?;
assert_eq!(big.rank(), 2);
let opts = CompressionOptions { tolerance: 1e-10, max_bond_dim: 20, ..Default::default() };
let c = big.compressed(&opts)?;
assert_eq!(c.rank(), 1);
assert!((c.evaluate(&[0, 1, 2])? - 3.0).abs() < 1e-10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Core: indices, tensors, contraction, factorize

```rust
use rand::SeedableRng;
use tensor4all_core::{Index, IndexLike, IdxTensor, contract, factorize, FactorizeOptions, SvdTruncationPolicy};

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);

let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6])?;
let b = IdxTensor::from_dense(vec![j, k.clone()], vec![1.0_f64; 12])?;
let c = contract(&[&a, &b])?;          // j summed away -> [i, k]
assert_eq!(c.dims(), vec![2, 4]);

// SVD split along i | k, truncating below rtol.
let t = IdxTensor::random::<f64, _>(&mut rand_chacha::ChaCha8Rng::seed_from_u64(1), vec![i.clone(), k.clone()])?;
let opts = FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1e-10));
let r = factorize(&t, &[i], &opts)?;   // r.left=[i,bond], r.right=[bond,k]
# Ok::<(), Box<dyn std::error::Error>>(())
```

## TCI on a black-box function (low-level, 0-indexed)

```rust
use tensor4all_simplett::prelude::*;
use tensor4all_tensorci::prelude::*;

let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f, None, vec![4, 4], vec![vec![3, 3]],            // pivot where |f| is large
    TCI2Options { tolerance: 1e-10, seed: Some(42), ..Default::default() },
)?;
assert_eq!(result.termination, tensor4all_tensorci::TCI2Termination::Converged);
assert!(*result.errors.last().unwrap() < 1e-10);
let tt = result.tci.to_tensor_train()?;
assert!((tt.evaluate(&[2, 3])? - 6.0).abs() < 1e-10); // f(2,3)=6
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Quantics TCI on a discrete grid (0-indexed, equal powers of 2)

```rust
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;      // 0-indexed
let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
    &vec![16, 16], f, None,
    QtciOptions::default().with_tolerance(1e-10),
)?;
assert!(*errors.last().unwrap() < 1e-10);
assert!((qtci.evaluate(&[4, 9])? - 13.0).abs() < 1e-8);
// sum of (i+j) for i,j in 0..16 = 2 * 16 * (16*15/2) = 3840
assert!((qtci.sum()? - 3840.0).abs() < 1e-6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Quantics TCI on a continuous interval + integral

```rust
use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions};

let grid = DiscretizedGrid::builder(&[4])              // 2^4 = 16 points
    .with_lower_bound(&[0.0]).with_upper_bound(&[1.0]).build()?;
let (qtci, _ranks, errors) = quanticscrossinterpolate::<f64, _>(
    &grid, |x: &[f64]| x[0] * x[0], None, QtciOptions::default(),
)?;
assert!(*errors.last().unwrap() < 1e-8);
assert!((qtci.evaluate(&[0])? - 0.0).abs() < 1e-10);   // grid point 0 -> x=0
let integral = qtci.integral()?;                         // left Riemann sum, O(h)
assert!((integral - 1.0 / 3.0).abs() < 5e-2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a manual integral on a uniform half-open `[x_min, x_max)`:
`integral = tt.sum() * (x_max - x_min) / 2^R`.

## Cache expensive evaluations

```rust
use tensor4all_core::CachedFunction;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

let r = 8;
let local_dims = vec![2; r];
let step = 1.0 / (1usize << r) as f64;
let cf = CachedFunction::new(
    |idx: &[usize]| {
        let q = idx.iter().fold(0usize, |acc, &b| (acc << 1) | b);
        (-(3.0 * q as f64 * step)).exp()
    },
    &local_dims,
)?;
let cached_f = |idx: &Vec<usize>| cf.eval(idx).unwrap();
let _result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    cached_f, None, local_dims, vec![vec![0; r]],
    TCI2Options { tolerance: 1e-12, seed: Some(42), ..Default::default() },
)?;
assert!(cf.num_cache_hits() > 0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Quantics transform: build + apply an operator

```rust
use tensor4all_core::SvdTruncationPolicy;
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions, shift_operator, BoundaryCondition};
use tensor4all_treetn::ApplyOptions;

let r = 8;
let shift = shift_operator(r, 10, BoundaryCondition::Periodic)?;     // each -> LinearOperator
let qft = quantics_fourier_operator(r, FourierOptions::forward())?;
assert_eq!(qft.mpo().node_count(), r);

// Apply to a TreeTN state. zipUp is the default; naive grows bonds, fit compresses.
let opts = ApplyOptions::zipup()
    .with_max_bond_dim(64)
    .with_svd_policy(SvdTruncationPolicy::new(1e-10));
// let result = apply_linear_operator(&qft, &state, opts)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Partial apply (subset of sites, e.g. Fourier only the x-variable of an interleaved 2D
encoding): rename operator nodes to the x-sites, `set_input_space_from_state` /
`set_output_space_from_state`, then `apply_linear_operator`. The Steiner tree inserts
identity tensors at the skipped sites automatically.

## Tree TN: build, canonicalize, truncate

```rust
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{TreeTN, TruncationOptions};

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let bond = DynIndex::new_dyn(3);
let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0_f64; 6])?;
let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0_f64; 6])?;
let mut ttn = TreeTN::<_, i32>::from_tensors(vec![t0, t1], vec![0, 1])?;
assert_eq!(ttn.edge_count(), 1);

let norm_before = ttn.norm()?;
let mut ttn = ttn.canonicalize([0], Default::default())?; // root 0 holds the norm
assert!(ttn.is_canonicalized());
let mut ttn = ttn.truncate([0], TruncationOptions::default().with_max_bond_dim(2))?;
assert_eq!(ttn.node_count(), 2);
assert!((ttn.norm()? - norm_before).abs() / norm_before < 1e-10); // canonicalization preserves the value
# Ok::<(), Box<dyn std::error::Error>>(())
```

## TreeTN DMRG / TDVP (treetn optimization sweeps)

DMRG (ground state) and TDVP (real/imaginary time evolution) sweep a `TreeTN`
state against a `LinearOperator` MPO, canonicalized at a root `center` node.
Build the mapping with `LinearOperator::from_mpo_and_state(mpo, &state)`, or
pass a bare MPO `TreeTN` to the `*_with_treetn_operator` wrappers. v1: one input
and one output site mapping per node.

```rust
use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{tdvp_with_treetn_operator, TdvpOptions, TreeTN};

// One-site TDVP under a 2x2 identity MPO: |0> evolves under exp(-0.1i * I).
let site = DynIndex::new_dyn(2);
let state_tensor = IdxTensor::from_dense(vec![site.clone()], vec![1.0, 0.0])?;
let state = TreeTN::<IdxTensor, usize>::from_tensors(vec![state_tensor], vec![0])?;

let op_in = DynIndex::new_dyn(2);
let op_out = DynIndex::new_dyn(2);
let op_tensor = IdxTensor::from_dense(vec![op_out, op_in], vec![1.0, 0.0, 0.0, 1.0])?;
let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![op_tensor], vec![0])?;

let result = tdvp_with_treetn_operator(
    &mpo, state, &0,
    TdvpOptions::default()
        .with_nsite(1)                                  // fixed-rank one-site
        .with_exponent_step(Complex64::new(0.0, -0.1)),  // exp(-0.1i * H)
)?;
assert_eq!(result.sweeps_completed, 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

- DMRG: swap to `dmrg_with_treetn_operator(&mpo, state, &0, DmrgOptions::default().with_nsweeps(4).with_max_bond_dim(32).with_energy_tol(1e-10))`; read `result.energy` (Rayleigh quotient), `result.converged`, `result.max_residual_norm`.
- Real time step `dt` → `exponent_step = Complex64::new(0.0, -dt)`; imaginary time `tau` → `-tau`. `order` ∈ {1,2,4}. `nsite 2` grows bonds (set `max_bond_dim`/`svd_policy`); `nsite 1` is fixed-rank.
- To grow rank on the fly use `gse_tdvp` with `GseTdvpOptions { gse: GseOptions::default().with_krylov_dim(3), tdvp: TdvpOptions::default() }`; it expands bond bases with `H·ψ, H²·ψ, …` between sweeps.

## ITensorLike TT: orthogonalize, truncate, inner

```rust
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions};

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let t1 = IdxTensor::from_dense(vec![b01, s1], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let mut tt = SimpleTensorTrain::new(vec![t0, t1])?;

let norm_before = tt.norm();
tt.orthogonalize(1)?;
assert!(tt.is_ortho());
tt.truncate(&TruncateOptions::svd()
    .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
    .with_max_bond_dim(2))?;
let inner = tt.inner(&tt)?;                  // <tt|tt> = norm^2
assert!((inner.real() - norm_before * norm_before).abs() < 1e-10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Compress existing data without materializing (TCI)

Define `f(idx)` computing each element on demand; let TCI find low-rank structure.
`local_dims = vec![128; 3]` for a 128³ grid — never allocate the full array.

```rust
use std::f64::consts::PI;
use tensor4all_simplett::prelude::*;
use tensor4all_tensorci::prelude::*;

let f = |idx: &Vec<usize>| {
    let x = 2.0 * PI * idx[0] as f64 / 128.0;
    let y = 2.0 * PI * idx[1] as f64 / 128.0;
    let z = 2.0 * PI * idx[2] as f64 / 128.0;
    x.cos() + y.cos() + z.cos()           // exact TT rank 2
};
let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f, None, vec![128, 128, 128], vec![vec![0, 0, 0]],
    TCI2Options { tolerance: 1e-12, max_bond_dim: 64, ..Default::default() },
)?;
assert_eq!(result.termination, tensor4all_tensorci::TCI2Termination::Converged);
assert!(*result.errors.last().unwrap() < 1e-10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Adaptive TCI over subdomain patches (partitionedtt, 0-indexed)

`adaptiveinterpolate` runs TCI2 per patch and subdivides patches that miss the
tolerance. Pivots are full-domain and **0-indexed** (tensorci territory, not
quanticstci). The result is a `PartitionedTT`; `to_tensor_train()` recombines it.

```rust
use tensor4all_core::contract;
use tensor4all_partitionedtt::{
    adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex,
};

let sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
// full-domain, 0-indexed evaluator
let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
    f,
    None,                                  // batched_f: None when batching unavailable
    sites,
    vec![vec![1, 1]],                      // initial pivots where |f| is large
    AdaptiveInterpolateOptions::default(), // tci_options tol 1e-8, n_initial_pivots 5
)?;
assert_eq!(result.len(), 1);               // one patch covered the whole domain

let tt = result.to_tensor_train()?;
let dense = contract(&[tt.tensor(0)?, tt.tensor(1)?])?;
assert_eq!(dense.to_vec::<f64>()?, vec![1.0, 2.0, 2.0, 4.0]); // column-major
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a function low-rank only after fixing an index, raise `n_initial_pivots`
(10–20 for multi-feature/high-dim), set `recycle_pivots(true)` to seed children
from a rejected parent, and supply known-nonzero pivots for sparse functions
(patches whose candidates all sample below `1e-30` are stored as zero).

## Elementwise op on TTs (ACI)

```rust
use tensor4all_aci::{elementwise, AciOptions};
use tensor4all_simplett::AbstractTensorTrain;

let a = tensor4all_simplett::SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
let b = tensor4all_simplett::SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
let result = elementwise(|xs: &[f64]| xs[0] * xs[1], &[a, b], &AciOptions::default())?;
assert!((result.tensor_train.evaluate(&[1, 2])? - 8.0).abs() < 1e-10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## HDF5 I/O (ITensors.jl-compatible)

```rust
use tensor4all_hdf5::{load_mps, save_mps};
// save_mps("state.h5", "MPS", &tt)?;      // or save_itensor for a single ITensor
// let loaded = load_mps("state.h5", "MPS")?;
```

## Convergence debugging checklist

- `errors.last()` plateaus above `tolerance`: raise `max_bond_dim` (needs higher rank), raise `max_iter` (more sweeps), or pick better initial pivots where `|f|` is large (`opt_first_pivot`).
- Quantics discrete: all `sizes` equal and powers of 2; raise `n_random_init_pivot` (10–20) for multi-feature / high-dim.
- Off-by-one at a boundary: all tensor4all-rs indices are 0-indexed — QuanticsTCI.jl scripts must subtract 1 from grid indices.
- Tolerance mismatch with Julia: `rtol = sqrt(cutoff)`.
- Running in a debug build (no `--release`): tensor linalg is orders of magnitude slower — always run TCI/DMRG in release.
