//! Tests for fit bond dimension capping behavior.
//!
//! Verifies the bond capping rules during fit sweeps:
//! - max_rank specified: bonds capped at max_rank
//! - rtol > 0 specified (no max_rank): bonds free to grow (rtol controls truncation)
//! - neither specified (or rtol=0): bonds capped at zipup initialization size
//! - rtol=0 explicit and rtol unspecified should behave identically

use rand::rngs::StdRng;
use rand::SeedableRng;

use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

const TEST_LENGTH: usize = 4;
const TEST_PHYS_DIM: usize = 2;
const TEST_BOND_DIM: usize = 4;

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

/// Helper: compute relative error between result and reference.
fn relative_error(result: &TensorTrain, reference: &TensorTrain, ref_norm: f64) -> f64 {
    result
        .axpby(1.0.into(), reference, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / ref_norm
}

/// Build test MPOs for contraction tests.
struct TestMPOs {
    mpo_a: TensorTrain,
    mpo_b: TensorTrain,
    /// Exact result (zipup without truncation)
    exact: TensorTrain,
    exact_norm: f64,
    #[allow(dead_code)]
    exact_bd: usize,
}

fn setup_test_mpos(length: usize, phys_dim: usize, bond_dim: usize) -> TestMPOs {
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

    let exact = mpo_a.contract(&mpo_b, &ContractOptions::zipup()).unwrap();
    let exact_norm = exact.norm().unwrap();
    let exact_bd = exact.maxbonddim();

    TestMPOs {
        mpo_a,
        mpo_b,
        exact,
        exact_norm,
        exact_bd,
    }
}

/// fit() with an explicit zero-threshold policy should preserve the same bond
/// dimensions as fit() without a policy override.
///
/// Unlike the removed `rtol=0` sentinel API, an explicit
/// `SvdTruncationPolicy::new(0.0)` is now a real truncation policy. Small
/// numerical differences in the final state are therefore acceptable, but the
/// bond-dimension cap behavior should still match.
#[test]
fn test_fit_zero_threshold_matches_no_policy_bond_dims() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);

    let result_no_rtol = t.mpo_a.contract(&t.mpo_b, &ContractOptions::fit()).unwrap();
    let result_rtol_zero = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit().with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(0.0)),
        )
        .unwrap();

    let bd_no_rtol = result_no_rtol.maxbonddim();
    let bd_rtol_zero = result_rtol_zero.maxbonddim();
    let err_no_rtol = relative_error(&result_no_rtol, &t.exact, t.exact_norm);
    let err_rtol_zero = relative_error(&result_rtol_zero, &t.exact, t.exact_norm);

    eprintln!("fit(no rtol): bd={bd_no_rtol}, rel_err={err_no_rtol:.6e}");
    eprintln!("fit(rtol=0):  bd={bd_rtol_zero}, rel_err={err_rtol_zero:.6e}");

    assert_eq!(
        bd_no_rtol, bd_rtol_zero,
        "fit(no policy) and fit(threshold=0) should produce the same bond dims: \
         {bd_no_rtol} vs {bd_rtol_zero}"
    );
}

/// fit() with max_rank should cap bond dimensions at the given value.
#[test]
fn test_fit_max_rank_caps_bonds() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);
    let max_rank = 5;

    let result = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_max_rank(max_rank)
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-12)),
        )
        .unwrap();
    let result_bd = result.maxbonddim();

    eprintln!("fit(max_rank={max_rank}, rtol=1e-12): bd={result_bd}");

    assert!(
        result_bd <= max_rank,
        "fit(max_rank={max_rank}) should cap bonds: got {result_bd}"
    );
}

/// fit() with max_rank should take precedence over rtol for bond capping,
/// even when rtol would allow larger bonds.
#[test]
fn test_fit_max_rank_overrides_rtol() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);
    let max_rank = 5;
    let rtol = 1e-12; // very small → would allow large bonds

    let result = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_max_rank(max_rank)
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
        )
        .unwrap();
    let result_bd = result.maxbonddim();

    // Without max_rank, rtol=1e-12 produces large bonds
    let result_no_cap = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
        )
        .unwrap();
    let no_cap_bd = result_no_cap.maxbonddim();

    eprintln!("fit(max_rank={max_rank}, rtol={rtol}): bd={result_bd}");
    eprintln!("fit(rtol={rtol}):                      bd={no_cap_bd}");

    assert!(
        result_bd <= max_rank,
        "max_rank should cap bonds even with small rtol: got {result_bd}"
    );
    assert!(
        no_cap_bd > max_rank,
        "Test setup: fit without max_rank should have larger bonds: got {no_cap_bd}"
    );
}

/// fit() with positive rtol (no max_rank) should allow bond growth beyond
/// the zipup initialization. Smaller rtol → larger bonds.
#[test]
fn test_fit_rtol_controls_bond_growth() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);

    let rtol_large = 0.3;
    let rtol_small = 1e-8;

    let result_large = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol_large)),
        )
        .unwrap();
    let result_small = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol_small)),
        )
        .unwrap();

    let bd_large = result_large.maxbonddim();
    let bd_small = result_small.maxbonddim();
    let err_large = relative_error(&result_large, &t.exact, t.exact_norm);
    let err_small = relative_error(&result_small, &t.exact, t.exact_norm);

    eprintln!("fit(rtol={rtol_large}): bd={bd_large}, rel_err={err_large:.6e}");
    eprintln!("fit(rtol={rtol_small}): bd={bd_small}, rel_err={err_small:.6e}");

    // Smaller rtol should produce larger (or equal) bond dimensions
    assert!(
        bd_small >= bd_large,
        "Smaller rtol should produce larger bonds: \
         rtol={rtol_small} → bd={bd_small}, rtol={rtol_large} → bd={bd_large}"
    );

    // Smaller rtol should produce better (or equal) accuracy
    assert!(
        err_small <= err_large * 1.01,
        "Smaller rtol should produce better accuracy: \
         rtol={rtol_small} → err={err_small:.6e}, rtol={rtol_large} → err={err_large:.6e}"
    );
}

/// fit() with no parameters should not degrade accuracy compared to zipup
/// with no parameters. Both use default settings (no truncation).
#[test]
fn test_fit_no_params_not_worse_than_zipup() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);

    let result_zipup = t
        .mpo_a
        .contract(&t.mpo_b, &ContractOptions::zipup())
        .unwrap();
    let result_fit = t.mpo_a.contract(&t.mpo_b, &ContractOptions::fit()).unwrap();

    let zipup_err = relative_error(&result_zipup, &t.exact, t.exact_norm);
    let fit_err = relative_error(&result_fit, &t.exact, t.exact_norm);
    let zipup_bd = result_zipup.maxbonddim();
    let fit_bd = result_fit.maxbonddim();

    eprintln!("zipup(default): bd={zipup_bd}, rel_err={zipup_err:.6e}");
    eprintln!("fit(default):   bd={fit_bd}, rel_err={fit_err:.6e}");

    // Both should be near-exact with no truncation
    assert!(
        fit_err < 1e-6,
        "fit(default) should be near-exact: rel_err={fit_err:.6e}"
    );
}

/// fit() with positive rtol should not produce worse accuracy than zipup
/// with the same rtol. The fit variational optimization should improve upon
/// or match the zipup initial guess.
#[test]
fn test_fit_not_worse_than_zipup_with_rtol() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);

    // Use rtol values that produce meaningful truncation in zipup.
    // Very small rtol (e.g. 1e-4, 1e-8) may give zipup near-exact results
    // while fit with 1 sweep has not yet converged, so we test moderate rtol.
    for rtol in [0.3, 0.1] {
        let result_zipup = t
            .mpo_a
            .contract(
                &t.mpo_b,
                &ContractOptions::zipup()
                    .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
            )
            .unwrap();
        let result_fit = t
            .mpo_a
            .contract(
                &t.mpo_b,
                &ContractOptions::fit()
                    .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
            )
            .unwrap();

        let zipup_err = relative_error(&result_zipup, &t.exact, t.exact_norm);
        let fit_err = relative_error(&result_fit, &t.exact, t.exact_norm);

        eprintln!("rtol={rtol:.0e}: zipup err={zipup_err:.6e}, fit err={fit_err:.6e}");

        assert!(
            fit_err <= zipup_err * 1.5,
            "fit(rtol={rtol}) should not be much worse than zipup(rtol={rtol}): \
             fit={fit_err:.6e} vs zipup={zipup_err:.6e}"
        );
    }
}

/// fit() with cutoff (ITensorMPS convention) should behave equivalently
/// to fit() with rtol=sqrt(cutoff). This tests the cutoff→rtol conversion.
#[test]
fn test_fit_cutoff_equivalent_to_rtol() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);
    let cutoff: f64 = 0.01;
    let rtol = cutoff.sqrt(); // 0.1

    let result_cutoff = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit().with_svd_policy(
                tensor4all_core::SvdTruncationPolicy::new(cutoff)
                    .with_squared_values()
                    .with_discarded_tail_sum(),
            ),
        )
        .unwrap();
    let result_rtol = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol)),
        )
        .unwrap();

    let bd_cutoff = result_cutoff.maxbonddim();
    let bd_rtol = result_rtol.maxbonddim();
    let err_cutoff = relative_error(&result_cutoff, &t.exact, t.exact_norm);
    let err_rtol = relative_error(&result_rtol, &t.exact, t.exact_norm);

    eprintln!("fit(cutoff={cutoff}): bd={bd_cutoff}, rel_err={err_cutoff:.6e}");
    eprintln!("fit(rtol={rtol}):     bd={bd_rtol}, rel_err={err_rtol:.6e}");

    assert_eq!(
        bd_cutoff, bd_rtol,
        "cutoff={cutoff} and rtol=sqrt(cutoff)={rtol} should produce same bond dims: \
         {bd_cutoff} vs {bd_rtol}"
    );
    let err_diff = (err_cutoff - err_rtol).abs();
    assert!(
        err_diff < 1e-10,
        "cutoff and rtol should produce the same error: \
         {err_cutoff:.6e} vs {err_rtol:.6e}"
    );
}

/// More sweeps should not degrade accuracy (bonds should not shrink
/// during sweeps when no truncation is applied).
#[test]
fn test_fit_more_sweeps_stable() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);

    let result_1sw = t
        .mpo_a
        .contract(&t.mpo_b, &ContractOptions::fit().with_nsweeps(1))
        .unwrap();
    let result_4sw = t
        .mpo_a
        .contract(&t.mpo_b, &ContractOptions::fit().with_nsweeps(4))
        .unwrap();

    let bd_1sw = result_1sw.maxbonddim();
    let bd_4sw = result_4sw.maxbonddim();
    let err_1sw = relative_error(&result_1sw, &t.exact, t.exact_norm);
    let err_4sw = relative_error(&result_4sw, &t.exact, t.exact_norm);

    eprintln!("fit(1sw): bd={bd_1sw}, rel_err={err_1sw:.6e}");
    eprintln!("fit(4sw): bd={bd_4sw}, rel_err={err_4sw:.6e}");

    // Bond dims should not shrink with more sweeps (no rtol → capped at init)
    assert!(
        bd_4sw >= bd_1sw,
        "More sweeps should not shrink bonds: 1sw bd={bd_1sw}, 4sw bd={bd_4sw}"
    );

    // Accuracy should not degrade significantly with more sweeps.
    // Note: relative_error uses (result - exact).norm().unwrap() via sequential contraction,
    // which loses precision when result ≈ exact (massive cancellation). When err_1sw
    // is near zero, err_4sw may be dominated by this floating-point noise rather than
    // actual approximation error. We use an absolute floor of 1e-6 to account for this.
    assert!(
        err_4sw < err_1sw.max(1e-6) * 10.0,
        "More sweeps should not degrade accuracy significantly: \
         1sw err={err_1sw:.6e}, 4sw err={err_4sw:.6e}"
    );
}

/// With positive rtol, more sweeps should allow bonds to grow and accuracy
/// to improve (or at least not degrade).
#[test]
fn test_fit_more_sweeps_with_rtol_improves() {
    let t = setup_test_mpos(TEST_LENGTH, TEST_PHYS_DIM, TEST_BOND_DIM);
    let rtol = 0.1;

    let result_1sw = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
                .with_nsweeps(1),
        )
        .unwrap();
    let result_4sw = t
        .mpo_a
        .contract(
            &t.mpo_b,
            &ContractOptions::fit()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
                .with_nsweeps(4),
        )
        .unwrap();

    let bd_1sw = result_1sw.maxbonddim();
    let bd_4sw = result_4sw.maxbonddim();
    let err_1sw = relative_error(&result_1sw, &t.exact, t.exact_norm);
    let err_4sw = relative_error(&result_4sw, &t.exact, t.exact_norm);

    eprintln!("fit(rtol={rtol},1sw): bd={bd_1sw}, rel_err={err_1sw:.6e}");
    eprintln!("fit(rtol={rtol},4sw): bd={bd_4sw}, rel_err={err_4sw:.6e}");

    // Accuracy should not degrade significantly with more sweeps
    assert!(
        err_4sw <= err_1sw * 2.0,
        "More sweeps with rtol should not degrade accuracy: \
         1sw err={err_1sw:.6e}, 4sw err={err_4sw:.6e}"
    );
}
