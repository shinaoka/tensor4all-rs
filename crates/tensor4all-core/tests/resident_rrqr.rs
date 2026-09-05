use num_complex::Complex64;
use std::sync::Arc;
use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    DynIndex, ExecutionContext, IdxTensor, IndexLike, TensorContractionLike,
    TensorFactorizationLike,
};
use tensor4all_tensorbackend::CpuExecutionContext;

#[test]
fn resident_probe_factorization_uses_upstream_rrqr_and_rank_only_readback() {
    let source = include_str!("../src/defaults/idx_tensor.rs");
    let start = source
        .find("    fn resident_probe_batch_qr(")
        .expect("resident RRQR function");
    let end = source[start..]
        .find("    fn resident_src_error_estimate(")
        .map(|offset| start + offset)
        .expect("next resident SRC function");
    let section = &source[start..end];
    assert!(section.contains(".rank_revealing_qr("));
    assert!(!section.contains(".qr()"));
    assert_eq!(section.matches("read_resident_rank(").count(), 1);
    assert!(!section.contains("read_decision_data("));
    assert!(!section.contains("column_permutation.to_tensor"));
}

fn explicit_cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

#[test]
fn explicit_cpu_rrqr_drops_interspersed_dependent_columns_and_restores_order() {
    let context = explicit_cpu_context();
    let row = DynIndex::new_dyn(4);
    let batch = DynIndex::new_dyn(5);
    // Columns 0, 1, and 3 are independent; column 2 is interspersed
    // dependence and column 4 is their linear combination.
    let values = vec![
        1.0_f64, 0.0, 0.0, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        2.0, 0.0, 0.0, 0.0, // column 2
        0.0, 0.0, 1.0, 0.0, // column 3
        1.0, 1.0, 1.0, 0.0, // column 4
    ];
    for scale in [1.0e-100, 1.0, 1.0e100] {
        let scaled = values.iter().map(|value| value * scale).collect::<Vec<_>>();
        let sketch =
            IdxTensor::from_dense_in(&context, vec![row.clone(), batch.clone()], scaled.clone())
                .unwrap();

        let factor = IdxTensor::factorize_probe_batch_incremental_in(
            None,
            &sketch,
            &batch,
            std::slice::from_ref(&row),
            &context,
        )
        .unwrap();

        assert_eq!(factor.rank, 3, "scale {scale:e}");
        assert_eq!(factor.left.indices()[1].dim(), 3);
        assert_eq!(factor.right.indices()[1].dim(), 5);
        let reconstruction = factor.left.contract_pair(&factor.right).unwrap();
        let got = reconstruction.to_vec::<f64>().unwrap();
        let relative_residual = got
            .iter()
            .zip(&scaled)
            .map(|(actual, expected)| (actual - expected).abs() / scale)
            .fold(0.0_f64, f64::max);
        assert!(
            relative_residual < 1.0e-12,
            "RRQR relative residual {relative_residual} at scale {scale:e}"
        );
    }
}

#[test]
fn explicit_cpu_complex_rrqr_restores_order_and_supports_square_estimator() {
    let context = explicit_cpu_context();
    let row = DynIndex::new_dyn(3);
    let batch = DynIndex::new_dyn(3);
    // Column norms force a non-identity pivot while the original factor remains
    // full rank and therefore eligible for the SRC square-factor estimator.
    let values = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(4.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.25, 0.5),
        Complex64::new(2.0, 0.0),
    ];
    let sketch =
        IdxTensor::from_dense_in(&context, vec![row.clone(), batch.clone()], values.clone())
            .unwrap();
    let factor = IdxTensor::factorize_probe_batch_incremental_in(
        None,
        &sketch,
        &batch,
        std::slice::from_ref(&row),
        &context,
    )
    .unwrap();

    assert_eq!(factor.rank, 3);
    let reconstruction = factor.left.contract_pair(&factor.right).unwrap();
    let got = reconstruction.to_vec::<Complex64>().unwrap();
    let residual = got
        .iter()
        .zip(&values)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(residual < 1.0e-12, "complex RRQR residual {residual}");
    let estimate = factor.right.src_error_estimate_in(&context).unwrap();
    assert!(estimate.error.is_finite() && estimate.error > 0.0);
    assert!(estimate.norm.is_finite() && estimate.norm > 0.0);
}
