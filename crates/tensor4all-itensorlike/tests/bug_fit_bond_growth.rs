//! Bug: fit() does not allow bond dimension growth during sweeps when max_rank
//! is not specified and rtol truncates the initial zipup guess.
//!
//! Scenario:
//!   1. User calls fit() with rtol but without max_rank
//!   2. The zipup initialization uses this rtol, truncating to small bonds
//!   3. During fit sweeps, bond dimensions are capped at the zipup value
//!      (because max_rank is None, the code falls back to existing bond dim)
//!   4. Fit cannot grow bonds beyond the zipup initialization → accuracy loss
//!
//! In Julia's ITensorMPS.jl, maxdim defaults to typemax(Int), so bonds grow
//! freely during sweeps. This is essential when zipup under-estimates the
//! required bond dimension.

use rand::rngs::StdRng;
use rand::SeedableRng;

use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

/// Create a random MPO with specified bond dimension.
fn create_random_mpo(
    length: usize,
    input_indices: &[DynIndex],
    output_indices: &[DynIndex],
    link_indices: &[DynIndex],
    rng: &mut StdRng,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    for i in 0..length {
        let mut indices = vec![input_indices[i].clone(), output_indices[i].clone()];
        if i > 0 {
            indices.insert(0, link_indices[i - 1].clone());
        }
        if i < length - 1 {
            indices.push(link_indices[i].clone());
        }
        let tensor = TensorDynLen::random::<f64, _>(rng, indices).unwrap();
        tensors.push(tensor);
    }
    TensorTrain::new(tensors).unwrap()
}

/// Demonstrates the bond capping bug with rtol-based truncation.
///
/// When fit() is called with rtol (no max_rank):
/// - Zipup initialization truncates bonds using rtol
/// - Fit sweeps cap bonds at zipup's bond dims (because max_rank=None → fallback to existing)
/// - This prevents fit from improving accuracy by growing bonds
///
/// Expected (Julia-like behavior): fit should be able to grow bonds during sweeps
/// when max_rank is not specified, using only rtol for truncation control.
#[test]
fn test_fit_bond_growth_with_rtol() {
    let length = 4;
    let phys_dim = 2;
    let bond_dim = 4;

    let s_input: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("s={}", i + 1)).unwrap())
        .collect();
    let s_shared: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("sc={}", i + 1)).unwrap())
        .collect();
    let s_output: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("so={}", i + 1)).unwrap())
        .collect();
    let links_a: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let links_b: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(123);

    let mpo_a = create_random_mpo(length, &s_input, &s_shared, &links_a, &mut rng1);
    let mpo_b = create_random_mpo(length, &s_shared, &s_output, &links_b, &mut rng2);

    eprintln!(
        "A max bond: {}, B max bond: {}",
        mpo_a.max_bond_dim(),
        mpo_b.max_bond_dim()
    );

    // Reference: zipup without truncation (exact)
    let result_exact = mpo_a.contract(&mpo_b, &ContractOptions::zipup()).unwrap();
    let exact_norm = result_exact.norm().unwrap();
    let exact_bd = result_exact.max_bond_dim();
    eprintln!("exact max bond: {exact_bd}, ||exact|| = {exact_norm:.6e}");

    // Use a large rtol that forces significant truncation in zipup
    let rtol = 0.3;

    // Zipup with rtol: truncates significantly
    let result_zipup = mpo_a
        .contract(
            &mpo_b,
            &ContractOptions::zipup()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
        )
        .unwrap();
    let zipup_bd = result_zipup.max_bond_dim();
    let zipup_err = result_zipup
        .axpby(1.0.into(), &result_exact, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / exact_norm;
    eprintln!("zipup(rtol={rtol}) max bond: {zipup_bd}, rel_err = {zipup_err:.6e}");
    assert!(
        zipup_bd < exact_bd,
        "Test setup: zipup should truncate (got {zipup_bd} >= exact {exact_bd})"
    );

    // Fit with same rtol, NO max_rank:
    // Bug: zipup init uses rtol → truncates to small bonds.
    //       Fit sweeps cap at existing bonds (because max_rank=None).
    //       Bonds cannot grow → accuracy limited by zipup.
    // Expected: fit should be free to grow bonds during sweeps and achieve
    //           better accuracy than zipup alone (limited only by rtol).
    let result_fit = mpo_a
        .contract(
            &mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
        )
        .unwrap();
    let fit_bd = result_fit.max_bond_dim();
    let fit_err = result_fit
        .axpby(1.0.into(), &result_exact, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / exact_norm;
    eprintln!("fit(rtol={rtol}) max bond: {fit_bd}, rel_err = {fit_err:.6e}");

    // Fit with more sweeps
    let result_fit4 = mpo_a
        .contract(
            &mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
                .with_nsweeps(4),
        )
        .unwrap();
    let fit4_bd = result_fit4.max_bond_dim();
    let fit4_err = result_fit4
        .axpby(1.0.into(), &result_exact, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / exact_norm;
    eprintln!("fit(rtol={rtol},4sw) max bond: {fit4_bd}, rel_err = {fit4_err:.6e}");

    // Control: fit with small rtol (no truncation), no max_rank
    let result_fit_small_rtol = mpo_a
        .contract(
            &mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-12)),
        )
        .unwrap();
    let fit_small_rtol_bd = result_fit_small_rtol.max_bond_dim();
    let fit_small_rtol_err = result_fit_small_rtol
        .axpby(1.0.into(), &result_exact, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / exact_norm;
    eprintln!("fit(rtol=1e-12) max bond: {fit_small_rtol_bd}, rel_err = {fit_small_rtol_err:.6e}");

    // Summary
    eprintln!("\nBond dimension summary:");
    eprintln!("  exact:          {exact_bd}");
    eprintln!("  zipup(rtol):    {zipup_bd}");
    eprintln!("  fit(rtol,1sw):  {fit_bd}");
    eprintln!("  fit(rtol,4sw):  {fit4_bd}");
    eprintln!("  fit(rtol=1e-12): {fit_small_rtol_bd}");

    eprintln!("\nRelative error summary:");
    eprintln!("  zipup(rtol):    {zipup_err:.6e}");
    eprintln!("  fit(rtol,1sw):  {fit_err:.6e}");
    eprintln!("  fit(rtol,4sw):  {fit4_err:.6e}");
    eprintln!("  fit(rtol=1e-12): {fit_small_rtol_err:.6e}");

    // Key assertions:
    // 1. Fit should not be worse than zipup with same rtol
    assert!(
        fit_err <= zipup_err * 1.5,
        "fit(rtol) should not be much worse than zipup(rtol): \
         fit={fit_err:.6e} vs zipup={zipup_err:.6e}"
    );

    // 2. Fit with small rtol should maintain near-exact accuracy
    assert!(
        fit_small_rtol_err < 1e-6,
        "fit(rtol=1e-12) should be near-exact: rel_err={fit_small_rtol_err:.6e}"
    );

    // 3. If fit bonds are capped at zipup level, flag it
    if fit_bd <= zipup_bd {
        eprintln!(
            "\nBUG CONFIRMED: fit bond dims ({fit_bd}) did not grow beyond \
             zipup init ({zipup_bd}). Bond capping prevents improvement."
        );
    }
    if fit4_bd <= zipup_bd {
        eprintln!(
            "BUG CONFIRMED: fit(4sw) bond dims ({fit4_bd}) did not grow beyond \
             zipup init ({zipup_bd}). More sweeps do not help."
        );
    }
}
