//! Regression tests for adaptive low-rank FIT (variational) contraction.
//!
//! These verify the *algorithmic properties* required of `ContractMethod::Fit`:
//!
//! 1. No `max_bond_dim` is required: ranks grow adaptively from a numerical
//!    tolerance (`svd_policy`) when starting from a small `C₀`.
//! 2. The default low-rank initializer never constructs a χ_A·χ_B product bond
//!    (`low_rank_initializer_tree_tn` keeps every bond at the requested small
//!    dimension; the treetn structural tests pin that directly).
//! 3. Tighter tolerances give equal-or-larger ranks and equal-or-better error.
//!
//! The product constructed here is compressible: `A` is a bond-`χ_a` random
//! MPO and `B` is the identity embedded with bond-`χ_b` (rank-1 content), so
//! `A·B = A` has exact rank `χ_a` while the naïve product bond would be
//! `χ_a·χ_b`. Starting from rank 1 with only a tolerance, fit must grow to
//! `χ_a` on its own.

use rand::rngs::StdRng;
use rand::SeedableRng;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_itensorlike::{ContractOptions, FitInitializer, TensorTrain};

fn random_mpo(
    length: usize,
    input: &[DynIndex],
    output: &[DynIndex],
    links: &[DynIndex],
    rng: &mut StdRng,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    for i in 0..length {
        let mut indices = vec![input[i].clone(), output[i].clone()];
        if i > 0 {
            indices.insert(0, links[i - 1].clone());
        }
        if i < length - 1 {
            indices.push(links[i].clone());
        }
        let tensor = IdxTensor::random::<f64, _>(rng, indices).unwrap();
        tensors.push(tensor);
    }
    TensorTrain::new(tensors).unwrap()
}

/// Identity operator MPO (`shared` -> `output`) whose physical bonds all have
/// dimension `bond_dim` but whose content is rank-1 (only bond coordinate 0).
fn identity_mpo_with_bond(
    length: usize,
    shared: &[DynIndex],
    output: &[DynIndex],
    bond_dim: usize,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    let mut prev: Option<DynIndex> = None;
    for i in 0..length {
        let mut indices: Vec<DynIndex> = Vec::new();
        if let Some(p) = prev.take() {
            indices.push(p);
        }
        indices.push(shared[i].clone());
        indices.push(output[i].clone());
        if i < length - 1 {
            let b = DynIndex::new_dyn(bond_dim);
            indices.push(b.clone());
            prev = Some(b);
        }
        let dim = shared[i].dim();
        let has_l = i > 0;
        let has_r = i < length - 1;
        let mut dims: Vec<usize> = Vec::new();
        if has_l {
            dims.push(bond_dim);
        }
        dims.push(dim);
        dims.push(dim);
        if has_r {
            dims.push(bond_dim);
        }
        let total: usize = dims.iter().product();
        let mut data = vec![0.0_f64; total];
        for j in 0..dim {
            let mut coords: Vec<usize> = Vec::new();
            if has_l {
                coords.push(0);
            }
            coords.push(j);
            coords.push(j);
            if has_r {
                coords.push(0);
            }
            let mut flat = 0usize;
            let mut stride = 1usize;
            for (k, c) in coords.iter().enumerate().rev() {
                flat += c * stride;
                stride *= dims[k];
            }
            data[flat] = 1.0;
        }
        let tensor = IdxTensor::from_dense(indices, data).unwrap();
        tensors.push(tensor);
    }
    TensorTrain::new(tensors).unwrap()
}

fn rel_error(result: &TensorTrain, reference: &TensorTrain) -> f64 {
    let diff = result.axpby(1.0.into(), reference, (-1.0).into()).unwrap();
    diff.norm().unwrap() / reference.norm().unwrap()
}

struct Fixtures {
    chi_a: usize,
    chi_b: usize,
    a: TensorTrain,
    b: TensorTrain,
    /// Exact product computed by untruncated zipup.
    exact: TensorTrain,
}

fn fixtures(length: usize, chi_a: usize, chi_b: usize, phys: usize) -> Fixtures {
    let s_input: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("si={}", i + 1)).unwrap())
        .collect();
    let s_shared: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("sc={}", i + 1)).unwrap())
        .collect();
    let s_output: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("so={}", i + 1)).unwrap())
        .collect();
    let links_a: Vec<DynIndex> = (0..length - 1).map(|_| DynIndex::new_dyn(chi_a)).collect();

    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let a = random_mpo(length, &s_input, &s_shared, &links_a, &mut rng);
    let b = identity_mpo_with_bond(length, &s_shared, &s_output, chi_b);
    let exact = a.contract(&b, &ContractOptions::zipup()).unwrap();
    Fixtures {
        chi_a,
        chi_b,
        a,
        b,
        exact,
    }
}

fn adaptive_fit(
    fx: &Fixtures,
    tol: f64,
    nsweeps: usize,
    bond_dim: usize,
    seed: u64,
) -> TensorTrain {
    fx.a.contract(
        &fx.b,
        &ContractOptions::fit()
            .with_initializer(FitInitializer::LowRankRandom {
                bond_dim,
                seed: Some(seed),
            })
            .with_svd_policy(SvdTruncationPolicy::new(tol))
            .with_nsweeps(nsweeps),
    )
    .unwrap()
}

/// Starting from rank 1 with *no `max_bond_dim`*, fit must grow the bond to
/// the rank the product actually needs (χ_a = 5 here) and approximate it.
#[test]
fn adaptive_fit_grows_rank_without_cap() {
    let fx = fixtures(5, 5, 5, 2);
    assert_eq!(
        fx.exact.max_bond_dim(),
        5,
        "test setup: exact product bond {}, chi_a*chi_b={}",
        fx.exact.max_bond_dim(),
        fx.chi_a * fx.chi_b
    );

    let fit = adaptive_fit(&fx, 1e-4, 4, 1, 7);
    let bd = fit.max_bond_dim();
    let err = rel_error(&fit, &fx.exact);
    eprintln!("rank1->grow: maxbd={bd}, rel_err={err:.6e}");
    assert!(bd > 1, "fit must grow from rank 1, got maxbd={bd}");
    assert!(
        bd <= fx.chi_a,
        "fit should not need more than the true rank"
    );
    assert!(
        err < 1e-3,
        "fit should approximate the product, rel_err={err:.6e}"
    );
}

/// With a tolerance and no max_bond_dim, tighter tolerance gives equal-or-larger
/// rank and equal-or-better error.
#[test]
fn adaptive_fit_tolerance_controls_rank_and_error() {
    let fx = fixtures(5, 5, 5, 2);
    let loose = adaptive_fit(&fx, 1e-1, 4, 1, 7);
    let tight = adaptive_fit(&fx, 1e-6, 4, 1, 7);

    let bd_loose = loose.max_bond_dim();
    let bd_tight = tight.max_bond_dim();
    let err_loose = rel_error(&loose, &fx.exact);
    let err_tight = rel_error(&tight, &fx.exact);
    eprintln!("loose: bd={bd_loose} err={err_loose:.6e}; tight: bd={bd_tight} err={err_tight:.6e}");
    assert!(
        bd_tight >= bd_loose,
        "tighter tolerance must not shrink the rank"
    );
    assert!(
        err_tight <= err_loose * 1.5,
        "tighter tolerance should give better or equal error"
    );
}

/// Fit must succeed (and produce a small state) even when the naïve product
/// bond χ_a·χ_b is large but the true product is compressible, without ever
/// materializing χ_a·χ_b inside the initializer.
#[test]
fn adaptive_fit_avoids_product_bond_intermediate() {
    // χ_a·χ_b = 144; exact compressible rank is χ_a = 12 (B is the identity).
    let fx = fixtures(4, 12, 12, 2);
    let fit = adaptive_fit(&fx, 1e-3, 3, 1, 11);
    let bd = fit.max_bond_dim();
    let err = rel_error(&fit, &fx.exact);
    eprintln!("large product: maxbd={bd}, rel_err={err:.6e}");
    assert!(
        bd <= 12 + 1,
        "fit should stay near the compressible rank, got {bd}"
    );
    assert!(err < 1e-2, "fit should converge, rel_err={err:.6e}");
}

/// Correctness sweep over chain lengths, physical dims and input bonds against
/// the exact (zipup) product, using the adaptive low-rank path.
#[test]
fn adaptive_fit_matches_exact_across_configs() {
    for &(length, phys, chi_a, chi_b) in &[
        (2usize, 2usize, 3usize, 3usize),
        (3, 2, 3, 3),
        (4, 3, 2, 2),
        (5, 2, 2, 2),
    ] {
        let fx = fixtures(length, chi_a, chi_b, phys);
        // Sweeps from rank-1 with a modest tolerance must reproduce the exact
        // product to within the tolerance's error budget.
        let fit = adaptive_fit(&fx, 1e-8, 4, 1, 99);
        let err = rel_error(&fit, &fx.exact);
        eprintln!(
            "(n={length},d={phys},χ_a={chi_a},χ_b={chi_b}): maxbd={}, rel_err={err:.6e}",
            fit.max_bond_dim()
        );
        assert!(
            err < 1e-6,
            "adaptive fit should match the exact product, rel_err={err:.6e}"
        );
    }
}

/// A larger `bond_dim` initializer should converge faster (fewer sweeps) for
/// the same tolerance, and still respect the cap when one is given.
#[test]
fn adaptive_fit_respects_explicit_cap() {
    let fx = fixtures(4, 5, 5, 2);
    // Cap at 2: ranks must never exceed it.
    let fit =
        fx.a.contract(
            &fx.b,
            &ContractOptions::fit()
                .with_initializer(FitInitializer::LowRankRandom {
                    bond_dim: 1,
                    seed: Some(3),
                })
                .with_svd_policy(SvdTruncationPolicy::new(1e-8))
                .with_max_bond_dim(2)
                .with_nsweeps(3),
        )
        .unwrap();
    let bd = fit.max_bond_dim();
    assert!(bd <= 2, "max_bond_dim=2 must be respected, got {bd}");
    // The cap necessarily limits accuracy for an exact-rank-5 product.
    let err = rel_error(&fit, &fx.exact);
    assert!(
        err > 1e-6,
        "capped fit should lose accuracy vs exact (err={err:.6e})"
    );
}
