use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike, TensorIndex};
use tensor4all_treetn::{tdvp, IndexMapping, LinearOperator, TdvpOptions, TreeTN};

#[test]
fn profiling_env_preserves_single_site_identity_evolution() {
    std::env::set_var("T4A_PROFILE_TDVP", "1");

    let site = DynIndex::new_dyn(2);
    let state_tensor = IdxTensor::from_dense(
        vec![site.clone()],
        vec![Complex64::new(0.75, 0.5), Complex64::new(-1.25, 0.25)],
    )
    .unwrap();
    let state =
        TreeTN::<IdxTensor, &'static str>::from_tensors(vec![state_tensor], vec!["site0"]).unwrap();
    let before = state.contract_to_tensor().unwrap();

    let input = DynIndex::new_dyn(2);
    let output = DynIndex::new_dyn(2);
    let op_tensor = IdxTensor::from_dense(
        vec![output.clone(), input.clone()],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let mpo =
        TreeTN::<IdxTensor, &'static str>::from_tensors(vec![op_tensor], vec!["site0"]).unwrap();
    let operator = LinearOperator::new(
        mpo,
        HashMap::from([(
            "site0",
            IndexMapping {
                true_index: site.clone(),
                internal_index: input,
            },
        )]),
        HashMap::from([(
            "site0",
            IndexMapping {
                true_index: site,
                internal_index: output,
            },
        )]),
    );

    let exponent = Complex64::new(0.0, -0.125);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_order(1)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 1);

    let after = result
        .state
        .contract_to_tensor()
        .unwrap()
        .permuteinds(&before.external_indices())
        .unwrap();
    let before_data = before.to_vec::<Complex64>().unwrap();
    let after_data = after.to_vec::<Complex64>().unwrap();
    let phase = exponent.exp();
    let max_error = before_data
        .iter()
        .zip(after_data)
        .map(|(before_value, after_value)| (after_value - phase * *before_value).norm())
        .fold(0.0, f64::max);
    assert!(
        max_error < 1.0e-10,
        "profiling changed TDVP identity evolution: max_error={max_error:.3e}"
    );
}
