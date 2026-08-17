# tensor4all-rs crate reference

Per-crate key types and entry points. Pull the crate you landed on in step 2 of `SKILL.md`.
Signatures are abbreviated; confirm exact shapes in `docs/api/` or rustdoc before relying on them.
All `tensor4all_*` crate names map to `tensor4all-<name>` packages.

## tensor4all-core — foundation

Indices, dynamic-rank tensors, contraction, factorization. Most other crates re-export from here.

- `Index` (alias `DynIndex = Index<DynId, TagSet>`) — identifies a tensor axis by unique identity, optional tags, optional prime level.
  - `Index::new_dyn(dim)` — site index, auto id, no tags.
  - `Index::new_dyn_with_tag(dim, tag) -> Result` — named index.
  - `DynIndex::new_bond(dim) -> Result` — bond index; fallible (metadata validated).
  - `.prime()` / `.noprime()` — raise / clear prime level (ket vs bra).
  - `.dim()`, `.plev()` via the `IndexLike` trait (in `tensor4all_core::prelude`).
  - Two independently created indices are always distinct even with equal dim.
- `IdxTensor` — dynamic-rank tensor supporting `f32`, `f64`, `Complex32`, and `Complex64`; `f64`/`Complex64` may use compact dense/diagonal/structured snapshots while 32-bit tensors retain eager authoritative payloads. Axes are matched by index identity, with no fixed ordering.
  - `IdxTensor::from_dense(indices, data)` — column-major `data`; first index varies fastest.
  - `IdxTensor::zeros::<f64>(indices)`, `::random::<f64, _>(&mut rng, indices)`.
  - `.to_vec::<f32>()` / `.to_vec::<f64>()` / `.to_vec::<Complex32>()` / `.to_vec::<Complex64>()`, `.sum()`, `.dims()`.
- `contract(&[&a, &b, ...])` — sum over all shared indices; inputs must be connected (else error). `outer_product(&a, &b)` for disconnected products.
- `factorize(&t, &[left_indices], &opts) -> FactorizeResult { left, right, rank, singular_values }`.
  - `FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(rtol))` / `.with_max_bond_dim(n)`.
  - `FactorizeOptions::qr().with_qr_rtol(tol)` / `.with_max_bond_dim(n)`.

## tensor4all-simplett — lightweight TT/MPS

Plain generic tensor train (`f32`, `f64`, `Complex32`, or `Complex64` where the selected operations support the dtype); no named indices needed. The go-to for numerics.

- `SimpleTensorTrain::<f64>` / `::<Complex64>` — chain of 3-leg cores.
  - `SimpleTensorTrain::constant(&[dims], val)`, `::zeros(&[dims])`.
  - `SimpleTensorTrain::new(tensors: Vec<Tensor3>)`.
  - `trait AbstractTensorTrain`: `.evaluate(&[idx]) -> Result`, `.sum()`, `.norm()`, `.rank()`, `.len()`, `.site_dims()`, `.link_dims()`, `.site_tensor(i)`, `.add(&b)`, `.compressed(&CompressionOptions)`.
- `CompressionOptions { tolerance, max_bond_dim, .. }` — `tolerance` is the relative truncation threshold. Default ~1e-12 near-lossless; 1e-8..1e-6 for science.
- `SiteTensorTrain` (center-canonical), `VidalTensorTrain` (Vidal form with singular values).
- `Tensor3Ops` (`.left_dim()/.site_dim()/.right_dim()/.set3()`), `types::tensor3_zeros(l, s, r)`.
- Bridge: `tensor_train_to_treetn(&mps) -> (TreeTN, site_indices)` (re-exported from `tensor4all-treetn`).

## tensor4all-itensorlike — ITensors.jl-style TT

Named `DynIndex` objects; orthogonality-center tracking; multiple canonical forms. Use when you need named indices or ITensors.jl compatibility.

- `SimpleTensorTrain::new(vec![t0, t1, ...])` from `IdxTensor` cores.
- `.orthogonalize(site)`, `.orthogonalize_with(site, CanonicalForm::...)` (LU / CI forms).
- `.truncate(&TruncateOptions::svd().with_svd_policy(SvdTruncationPolicy::new(rtol)).with_max_bond_dim(n))`.
- `.inner(&other)` — `<self|other>`, conjugates left operand. `.norm()`, `.is_ortho()`, `.ortho_center()`, `.max_bond_dim()`.
- `TruncateOptions`, `CanonicalForm`. `ContractOptions` / `LinsolveOptions` with `with_nsweeps(n)` (= `with_nhalfsweeps(2*n)`).
- Build indices with `DynIndex::new_dyn(dim)` (site) and `DynIndex::new_bond(dim)?` (bond). `from_dense` data is column-major.

## tensor4all-tensorci — low-level TCI

Direct cross interpolation on integer indices. TCI2 is the primary algorithm; TCI1 is legacy.

- `crossinterpolate2::<T, F, B>(f, batched_f: Option<B>, local_dims, initial_pivots, TCI2Options) -> Result<TCI2OptimizationResult<T>>`. The result is a **struct** — access `result.tci` (`TensorCI2<T>`), `result.ranks`, `result.errors`, `result.termination`. (Not a tuple.)
  - `f: &MultiIndex -> T` — **0-indexed** multi-index (`MultiIndex = Vec<usize>`).
  - `batched_f: Option<B>`, `B: Fn(&[MultiIndex]) -> Vec<T>` — same function, batched; pass `None` when batching is unavailable.
  - `initial_pivots` — pick where `|f|` is large.
- `TCI2Options { tolerance, max_bond_dim, max_iter, seed, normalize_error, .. }`.
  - `tolerance` default 1e-8; 1e-6 explore, 1e-12 high accuracy.
  - `max_bond_dim` cap to prevent runaway on expensive functions.
  - `seed: Some(42)` for reproducibility.
- `opt_first_pivot::<T, F>(&f, &local_dims, &start, n_iters) -> MultiIndex` — local search for a large-`|f|` pivot.
- `TensorCI2::to_tensor_train() -> Result<SimpleTensorTrain>`. Convergence: check `result.termination == TCI2Termination::Converged` (bond-error, global-pivot history, and rank stability all met). `MaxBondDimension` / `MaxIterations` are stops, **not** convergence — treat them as "not done".

## tensor4all-core — caching + multi-index plumbing (TCI substrate, ex-tensor4all-tcicore)

Home of `CachedFunction` and `MultiIndex`. Prefer the higher-level crates; reach for this one only for those two types. `MultiIndex` is re-exported from `tensor4all-partitionedtt`, so you need a direct dep only for `CachedFunction`.

- `CachedFunction::new(f, &local_dims)` — memoize expensive evaluations across sweeps. `.eval(&idx)`, `.cache_size()`, `.num_cache_hits()`. Worth it only when `f` is expensive; cheap functions lose to overhead.
- `MultiIndex` = `Vec<usize>`, the coordinate type threaded through `tensorci` and `partitionedtt` callbacks.

Do **not** depend on `tensor4all-tensorbackend` — it is internal (storage + linalg wrappers); use the public crates and never instantiate `CpuBackend` directly.

## tensor4all-quanticstci — high-level quantics TCI

Quantics encoding (binary bits across sites) often yields far lower bond dims. Port of QuanticsTCI.jl.

- Entry points (return `(QuanticsTensorCI2, ranks, errors)`):
  - `quanticscrossinterpolate_discrete::<T, F>(&sizes, f, None, opts)` — integer grid; `f` receives **0-indexed** grid indices as `&[usize]` (`0..grid_size`). `sizes` must be equal powers of 2.
  - `quanticscrossinterpolate(&grid, f, None, opts)` — continuous domain via `DiscretizedGrid`.
  - `quanticscrossinterpolate_from_arrays` — explicit coordinate arrays.
  - `quanticscrossinterpolate_batched` — vector/tensor-valued `f`.
- `DiscretizedGrid::builder(&[r_bits]).with_lower_bound(&[x0]).with_upper_bound(&[x1]).include_endpoint(bool).build()`.
- `QtciOptions::default().with_tolerance(t).with_max_bond_dim(m).with_nrandominitpivot(k).with_unfoldingscheme(UnfoldingScheme::Interleaved)`.
  - `n_random_init_pivot` default 5; raise to 10–20 for multi-feature / high-dim.
- `QuanticsTensorCI2`: `.evaluate(&[idx])`, `.sum()`, `.integral()` (left Riemann sum; O(h)), `.tci()`.
- Interleaved multivariate encoding: site `n` encodes one bit per variable, local dim `2^num_vars`.

## tensor4all-quanticstransform — quantics operators

Every constructor returns a `LinearOperator` (from `tensor4all-treetn`). All use **big-endian** bits; `r` = bits per variable (`2^r` grid points).

- `flip_operator(r, BoundaryCondition)`, `shift_operator(r, offset, BoundaryCondition)`, `phase_rotation_operator(r, theta)`, `cumsum_operator(r)`, `quantics_fourier_operator(r, FourierOptions)`, `affine_operator(r, &AffineParams, &bc)`.
- `_multivar` variants use interleaved encoding (site local dim `2^num_vars`).
- `BoundaryCondition::{Periodic, Open}`. `FourierOptions::{forward, inverse, default}` (`normalize` default true = isometry). QFT output is **bit-reversed**.
- `AffineParams::new(a: Vec<Rational64>, b: Vec<Rational64>, n_out, n_in)`.
- `LinearOperator`: `.mpo()`, `.node_count()`, `.rename_nodes(&[(from, to)])`, `.set_input_space_from_state(&state)`, `.set_output_space_from_state(&state)`, `.get_input_mapping(&i)`, `.get_output_mapping(&i)`.
- Apply via `tensor4all_treetn::apply_linear_operator(&op, &state, ApplyOptions)`. Partial apply (subset of sites) builds a Steiner tree automatically — no manual identity tensors.
- Error conditions: `r == 0`; `r == 1` for cumsum/triangle/fourier; `r >= 64` for shift (overflow); NaN/Inf `theta`.

## tensor4all-treetn — tree tensor networks

Generic `TreeTN` over arbitrary tree topology (TT/MPS is the path-graph special case).

- `TreeTN::<IdxTensor, V>::from_tensors(tensors, labels) -> Result` — topology inferred from shared bond indices; site indices appear once, bond indices appear twice. `V: Eq + Hash` labels vertices.
- `.node_count()`, `.edge_count()`, `ttn[v]` access, `.norm()`, `.to_dense()` (test/small only), `.add(&b)` (same topology + matching site indices; bonds grow as direct sum), `.replaceind(&old, &new)`.
- `.canonicalize([root], CanonicalizationOptions)`, `.truncate([root], TruncationOptions::default().with_max_bond_dim(n).with_rtol(t))`.
- `apply_linear_operator(&op, &state, ApplyOptions)`:
  - `ApplyOptions::naive()` — local exact, no truncation; bond dims grow as products. Small/debug only.
  - `ApplyOptions::zipup().with_max_bond_dim(n).with_svd_policy(policy)` — default; single sweep, controllable.
  - `ApplyOptions::fit().with_max_bond_dim(n).with_nfullsweeps(k)` — iterative; best compression.
- Sweep convention: `ApplyOptions::fit().with_nfullsweeps(k)` uses `nfullsweeps` (full = forward+back); `nfullsweeps = nhalfsweeps / 2`. The DMRG/TDVP option structs instead name the same full-sweep count `nsweeps` (no `full` prefix) — there is no `with_nfullsweeps` on `DmrgOptions`/`TdvpOptions`.
- `tensor_train_to_treetn(&mps) -> (TreeTN, Vec<site_index>)` bridge from simplett.
- Optimization sweeps (ground state + time evolution). All take a `LinearOperator` (or a bare MPO `TreeTN` via the `*_with_treetn_operator` wrappers), an initial `TreeTN` state, and a `center: &V` root node; the state is canonicalized at `center` first. v1: one input and one output site mapping per node. Build the mapping with `LinearOperator::from_mpo_and_state(mpo, &state)` when site indices are unambiguous, else hand-build `IndexMapping { true_index, internal_index }` input/output maps.
  - `dmrg(operator, init, center, DmrgOptions) -> Result<DmrgResult>` — two-site ground-state DMRG (Lanczos local solve; Rayleigh-quotient energy, no dense materialization). `DmrgOptions::default()` = `nsite 2, nsweeps 5, max_bond_dim None, svd_policy None, energy_tol None`. Builders: `with_nsweeps`, `with_max_bond_dim`, `with_svd_policy`, `with_energy_tol`, `with_lanczos_options`. `DmrgResult { state, energy, sweeps_completed, local_updates, converged, max_residual_norm }`. `nsite` must be 2.
  - `tdvp(operator, init, center, TdvpOptions) -> Result<TdvpResult>` — time-dependent variational principle via Krylov `exp(exponent_step·H)`. `exponent_step` is the Hamiltonian coefficient: real time `dt` → `Complex64::new(0.0, -dt)`; imaginary time `tau` → `-tau` (real negative). `TdvpOptions::default()` = `nsite 2, nsweeps 1, order 2, exponent_step -0.1i, max_bond_dim None, svd_policy None`. `order` ∈ {1,2,4} (Suzuki–Trotter/applyexp, ITensorNetworks.jl parity). `nsite 1` = fixed-rank one-site (rejects truncation options); `nsite 2` allows bond changes. For ITensors `cutoff` parity use `SvdTruncationPolicy::new(cutoff).with_squared_values().with_discarded_tail_sum()`. `TdvpResult { state, sweeps_completed, local_updates, max_error_estimate, max_krylov_iterations }`.
  - `gse_tdvp(operator, init, center, GseTdvpOptions) -> Result<GseTdvpResult>` — Global Subspace Expansion: build Krylov references `H·ψ, H²·ψ, …` and expand bond bases between TDVP sweeps to grow rank where needed. `GseTdvpOptions { gse: GseOptions, tdvp: TdvpOptions }`; `GseOptions::default()` = `krylov_dim 0, density_weight_cutoff 1e-12, hermitian_tol 1e-12, normalize_references true, expand_before_first_sweep true`. Also `global_subspace_expand(operator, init, center, GseOptions)` and `global_subspace_expand_with_references(init, references, center, GseOptions)` for standalone expansion. v1: `IdxTensor` states, one state site index per node.

## tensor4all-aci — elementwise TT operations

Ports AlternatingCrossInterpolation.jl. Approximates an elementwise op on input TTs.

- `elementwise(|xs: &[f64]| ..., &[a, b, ...], &AciOptions::default()) -> AciResult`.
- `elementwise_batched(|batch: ElementwiseBatch, out: &mut [f64]| ..., &[a, b], &AciOptions)` — amortized batched callback; `batch.get(input, point)`.
- `AciOptions` controls tolerance, sweep limits, maximum bond dimension, scaling, initial guess, and random seed; `AciResult { tensor_train, ... }`.

## tensor4all-treetci — tree TCI

Ports TreeTCI.jl. Cross interpolation on tree-structured graphs → TreeTN.

- `crossinterpolate2()` (tree entry), `TreeTCI2`, `TreeTciGraph`.
- `tensor4all_treetci::materialize::to_treetn(tci_state, batch_eval, Some(root))` — materialize as TreeTN with a batched evaluator.

## tensor4all-interpolativeqtt — interpolative QTT

Ports InterpolativeQTT.jl; returns `SimpleTensorTrain<f64>`.

- `interpolate_single_scale(f, x_min, x_max, R, oversampling, &InterpolativeQttOptions::default())`.

## tensor4all-partitionedtreetn — named TreeTN subdomains + adaptive patching

Use this crate for new partitioned work on named TreeTNs. It stores eagerly
masked `TreeTN<IdxTensor, V>` patches, supports branched topologies and multiple
site indices per node, and does not implement adaptive interpolation.

- `Projector` — maps full `DynIndex` identities to zero-based coordinates.
- `SubDomainTreeTN<V>` — eagerly masked TreeTN plus its projector.
- `PartitionedTreeTN<V>` — homogeneous, pairwise-disjoint patches.
- `PatchingOptions { rtol, max_bond_dim, patch_order, split_strategy }`.
- `PatchSplitStrategy::{Sequential, ExactParameterGain}` — exact gain uses
  checked logical local tensor element counts after child truncation.
- `add_with_patching(patches, &center, &options)` — split over-cap patches.
- `truncate_adaptive(&partition, &center, rtol, max_bond_dim)` — assign
  volume-proportional absolute squared-tail budgets and drop patches below them.
- `contract_adaptive(&left, &right, &center, &contract_options, &patching_options)` —
  contract and retruncate against the corrected output norm.

All truncating and contracting operations require an explicit existing node name
as `center`. No production path re-applies eager projectors or materializes a
full network densely.

## tensor4all-partitionedtt — legacy subdomain patches + adaptive TCI

This crate is deprecated during migration. Use `tensor4all-partitionedtreetn`
for new named TreeTN work. It remains buildable and receives correctness and
security fixes only; no removal date is set.

Split a function's domain into non-overlapping projected patches, each its own TT. Use when a function is low-rank only after fixing some site indices. Re-exports `DynIndex`, `IdxTensor`, `MultiIndex`, `SimpleTensorTrain`, `ContractOptions`/`TruncateOptions`, `TCI2Options` — get them here rather than reaching into core internals.

- `Projector` — maps site `DynIndex` → fixed coordinate, defining a subdomain.
  - `Projector::new()`, `Projector::from_pairs([(idx, value), ...])?`.
  - `.get(&idx) -> Option<usize>`, `.is_projected_at(&idx)`, `.insert(idx, value)?`, `.projected_indices()`, `.len()`, `.is_empty()`.
- `SubDomainTT` — an itensorlike `TensorTrain` plus its `Projector`.
  - `SubDomainTT::new(tt, projector)?`, `SubDomainTT::from_tt(tt)` (empty projector).
  - `.data()`, `.projector()`, `.max_bond_dim()`, `.into_data()`, `.all_indices()`.
- `PartitionedTT` — collection of mutually disjoint `SubDomainTT`s (disjointness validated at construction and insertion).
  - `PartitionedTT::from_subdomains(vec)?`, `::from_subdomain(one)?`, `::new()`.
  - `.len()`, `.is_empty()`, `.projectors()`, `.iter()`, `.values()`, `.contains(&projector)`.
  - `.to_tensor_train() -> Result<SimpleTensorTrain>` — recombine all patches into one TT (drops the partition structure).
  - `.contract(&other, &ContractOptions)?` (also free `contract` / `proj_contract`).
- Adaptive patching — bond-cap-driven splitting plus volume-proportional truncation:
  - `add_with_patching(subdomains: Vec<SubDomainTT>, &PatchingOptions) -> Result<PartitionedTT>` — split over-cap patches along `patch_order`, then truncate.
  - `contract_adaptive(&left, &right, &ContractOptions, &PatchingOptions) -> Result<PartitionedTT>`.
  - `truncate_adaptive(&PartitionedTT, rtol, max_bond_dim) -> Result<PartitionedTT>` — total budget `rtol²·‖F‖²` shared by patch volume; patches at/below their budget are dropped.
  - `PatchingOptions { rtol: 1e-12, max_bond_dim: 100, patch_order: Vec<DynIndex>, split_strategy: ExactParameterGain }`.
  - `PatchSplitStrategy::{Sequential, ExactParameterGain (default)}` — `ExactParameterGain` forms + counts child cores to pick the smallest split; `Sequential` takes the first unprojected `patch_order` index.
- Adaptive TCI interpolation — ports TCIAlgorithms.jl `adaptiveinterpolate` / `createpatch` / `_globalpivots`:
  - `adaptiveinterpolate::<T, F, B>(f, batched_f: Option<B>, site_indices: Vec<DynIndex>, initial_pivots: Vec<MultiIndex>, AdaptiveInterpolateOptions) -> Result<PartitionedTT>`.
    - `f: &MultiIndex -> T` receives one **full, 0-based** multi-index (tensorci territory, not quanticstci).
    - `batched_f: Option<B>` with `B: Fn(&[MultiIndex]) -> Vec<T>` — same function, batched; pass `None` when batching is unavailable.
    - `initial_pivots` — full-domain, 0-based; empty allowed.
  - `AdaptiveInterpolateOptions { tci_options: TCI2Options, patch_order: Vec<DynIndex>, n_initial_pivots: 5, recycle_pivots: false }`. Empty `patch_order` ⇒ site order; nonempty must be an exact full-`Index` permutation of `site_indices`.
  - Flow: run TCI2 per patch on the active (unprojected) sites; **accept** only when `TCI2Termination::Converged` **and** final error ≤ `tci_options.tolerance`; otherwise split at the next `patch_order` index into one child per coordinate. Patches with 0/1 active sites are evaluated exactly. Reaching `max_bond_dim`/`max_iter` is a stop, not convergence — such a patch is split.
  - `recycle_pivots = true` seeds children with a rejected parent's diagonal TCI pivots (opt-in; incompatible pivots discarded; every child is still replenished to `n_initial_pivots`).
  - Sampled-zero policy: if every candidate evaluates below `1e-30` the patch is stored as zero — supply known-nonzero pivots for sparse functions.

## tensor4all-hdf5 — ITensors.jl-compatible I/O

- `save_itensor()` / `load_itensor()` — `IdxTensor` as ITensors.jl `ITensor`.
- `save_mps()` / `load_mps()` — `SimpleTensorTrain` as ITensorMPS.jl `MPS`.
- Features: `link` (default, compile-time HDF5), `runtime-loading` (dlopen for FFI).

## tensor4all-capi — C FFI

Status enum `t4a_status_code` with `T4A_` prefix (`T4A_SUCCESS`, `T4A_NULL_POINTER`, `T4A_INTERNAL_ERROR`). Header `crates/tensor4all-capi/include/tensor4all_capi.h` regenerated with `cbindgen`. See `docs/CAPI_DESIGN.md`. Wrapped by Tensor4all.jl. Not for direct Rust use.
