//! CPU tests for the PR4a context seam (issue #720).
//!
//! Explicit-CPU construction/validation/factorization routing, exercised
//! generically through the `TensorLike` traits where SRC will use them.

use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    factorize_full_rank_in, factorize_in, Canonical, DynIndex, ExecutionContext, FactorizeAlg,
    FactorizeOptions, IdxTensor, TensorConstructionLike, TensorFactorizationLike,
};
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetn::TreeTN;

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

#[test]
fn generic_construction_and_validation_use_the_supplied_context() {
    let context = cpu_context();
    let index = DynIndex::new_dyn(2);

    let dense = <IdxTensor as TensorConstructionLike>::from_dense_in(
        &context,
        vec![index.clone()],
        vec![1.0_f64, 2.0],
    )
    .unwrap();
    let ones =
        <IdxTensor as TensorConstructionLike>::ones_in(&context, std::slice::from_ref(&index))
            .unwrap();
    assert_eq!(ones.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);

    dense.validate_context(&context).unwrap();
    let foreign = cpu_context();
    assert!(dense.validate_context(&foreign).is_err());

    let tree = TreeTN::from_tensors(vec![dense], vec![0]).unwrap();
    tree.validate_context(&context).unwrap();
    assert!(tree.validate_context(&foreign).is_err());

    // A host-built tree does not belong to an explicit context.
    let host = IdxTensor::from_dense(vec![index], vec![1.0_f64, 2.0]).unwrap();
    let host_tree = TreeTN::from_tensors(vec![host], vec![0]).unwrap();
    assert!(host_tree.validate_context(&context).is_err());
}

#[test]
fn factorize_in_matches_host_results_on_explicit_cpu() {
    let context = cpu_context();
    let i = DynIndex::new_dyn(6);
    let j = DynIndex::new_dyn(4);
    let data: Vec<f64> = (0..24)
        .map(|k| {
            let (r, c) = (k % 6, k / 6);
            if r == c {
                4.0 + r as f64
            } else {
                0.1 * (r + c) as f64
            }
        })
        .collect();
    let tensor = <IdxTensor as TensorConstructionLike>::from_dense_in(
        &context,
        vec![i.clone(), j.clone()],
        data,
    )
    .unwrap();

    // Truncated QR via generic options.
    let scoped = factorize_in(
        &tensor,
        std::slice::from_ref(&i),
        &FactorizeOptions::qr(),
        &context,
    )
    .unwrap();
    let host = tensor4all_core::factorize(
        &tensor4all_core::IdxTensor::from_dense(
            vec![i.clone(), j.clone()],
            (0..24)
                .map(|k| {
                    let (r, c) = (k % 6, k / 6);
                    if r == c {
                        4.0 + r as f64
                    } else {
                        0.1 * (r + c) as f64
                    }
                })
                .collect(),
        )
        .unwrap(),
        std::slice::from_ref(&i),
        &FactorizeOptions::qr(),
    )
    .unwrap();
    assert_eq!(scoped.rank, host.rank);
    scoped.left.validate_context(&context).unwrap();
    scoped.right.validate_context(&context).unwrap();

    // Full-rank SVD stays in context.
    let full = factorize_full_rank_in(
        &tensor,
        std::slice::from_ref(&i),
        FactorizeAlg::SVD,
        Canonical::Left,
        &context,
    )
    .unwrap();
    full.left.validate_context(&context).unwrap();
    full.right.validate_context(&context).unwrap();

    // LU/CI have no scoped path.
    assert!(factorize_in(
        &tensor,
        std::slice::from_ref(&i),
        &FactorizeOptions::lu(),
        &context
    )
    .is_err());

    // Estimator + incremental batch route through the generic trait surface.
    let r = scoped.right;
    let estimate = r.src_error_estimate_in(&context).unwrap();
    let expected = r.src_error_estimate().unwrap();
    assert!((estimate.error - expected.error).abs() < 1e-12);

    let batch = DynIndex::new_dyn(2);
    let block = <IdxTensor as TensorConstructionLike>::from_dense_in(
        &context,
        vec![i.clone(), batch.clone()],
        vec![1.0_f64; 12],
    )
    .unwrap();
    let grown = IdxTensor::factorize_probe_batch_incremental_in(
        None,
        &block,
        &batch,
        std::slice::from_ref(&i),
        &context,
    )
    .unwrap();
    assert_eq!(grown.rank, 2);
    grown.left.validate_context(&context).unwrap();
}

#[test]
fn trait_defaults_reject_without_silent_fallback() {
    // DefaultOnlyTensor-style check via IdxTensor host input + foreign context:
    // validation must fail, never silently proceed.
    let context = cpu_context();
    let host = IdxTensor::from_dense(vec![DynIndex::new_dyn(1)], vec![1.0_f64]).unwrap();
    assert!(host.validate_context(&context).is_err());
    assert!(factorize_in(
        &host,
        &[DynIndex::new_dyn(1)],
        &FactorizeOptions::qr(),
        &context
    )
    .is_err());
}
