use super::*;
use crate::defaults::tensordynlen::TensorDynLen;
use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::krylov::{gmres, GmresOptions};
use crate::tensor_like::TensorContractionLike;

/// Helper to create a 1D tensor (vector) with given data and shared index.
fn make_vector_with_index(data: Vec<f64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

// ========================================================================
// Test 0: BlockTensor invariants
// ========================================================================

#[test]
fn test_try_new_valid() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);

    let block = BlockTensor::try_new(vec![b1, b2], (2, 1));
    assert!(block.is_ok());
    let block = block.unwrap();
    assert_eq!(block.shape(), (2, 1));
    assert_eq!(block.num_blocks(), 2);
}

#[test]
fn test_try_new_invalid_shape() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);

    // 2 blocks but shape says 3x1 = 3 blocks
    let result = BlockTensor::try_new(vec![b1, b2], (3, 1));
    assert!(result.is_err());
}

#[test]
fn test_axpby_shape_mismatch() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let b3 = make_vector_with_index(vec![5.0, 6.0], &idx);

    let block_2x1 = BlockTensor::new(vec![b1.clone(), b2], (2, 1)).unwrap();
    let block_1x1 = BlockTensor::new(vec![b3], (1, 1)).unwrap();

    let result = block_2x1.axpby(
        AnyScalar::new_real(1.0),
        &block_1x1,
        AnyScalar::new_real(1.0),
    );
    assert!(result.is_err());
}

#[test]
fn test_inner_product_shape_mismatch() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let b3 = make_vector_with_index(vec![5.0, 6.0], &idx);

    let block_2x1 = BlockTensor::new(vec![b1.clone(), b2], (2, 1)).unwrap();
    let block_1x1 = BlockTensor::new(vec![b3], (1, 1)).unwrap();

    let result = block_2x1.inner_product(&block_1x1);
    assert!(result.is_err());
}

#[test]
fn test_fuse_indices_delegates_to_blocks_and_preserves_shape() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let fused = DynIndex::new_dyn(4);
    let block_a =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let block_b =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let block = BlockTensor::new(vec![block_a, block_b], (1, 2)).unwrap();

    let fused_block =
        <BlockTensor<TensorDynLen> as crate::tensor_like::TensorContractionLike>::fuse_indices(
            &block,
            &[i, j],
            fused,
            crate::tensor_like::LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    assert_eq!(fused_block.shape(), (1, 2));
    assert_eq!(fused_block.num_blocks(), 2);
    assert_eq!(fused_block.get(0, 0).unwrap().dims(), vec![4]);
    assert_eq!(fused_block.get(0, 1).unwrap().dims(), vec![4]);
    assert_eq!(
        fused_block.get(0, 0).unwrap().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        fused_block.get(0, 1).unwrap().to_vec::<f64>().unwrap(),
        vec![5.0, 6.0, 7.0, 8.0]
    );
}

// ========================================================================
// Test 1: Identity operator GMRES
// ========================================================================

#[test]
fn test_gmres_identity_block() {
    // Solve Ax = b where A = I (identity)
    // 2x1 block structure, each block is a 2-element vector
    let idx = DynIndex::new_dyn(2);

    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let b = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1)).unwrap();

    // Identity operator: A x = x
    let apply_a = |x: &BlockTensor<TensorDynLen>| -> anyhow::Result<BlockTensor<TensorDynLen>> {
        Ok(x.clone())
    };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge for identity");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10, "Solution should equal b");
}

// ========================================================================
// Test 2: Diagonal block matrix GMRES
// ========================================================================

#[test]
fn test_gmres_diagonal_block() {
    // A = [[D1, 0], [0, D2]] where D1 = diag(2, 3), D2 = diag(4, 5)
    // b = [[2, 3]^T, [4, 5]^T] -> x = [[1, 1]^T, [1, 1]^T]
    let idx = DynIndex::new_dyn(2);

    let b1 = make_vector_with_index(vec![2.0, 3.0], &idx);
    let b2 = make_vector_with_index(vec![4.0, 5.0], &idx);
    let b = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1)).unwrap();

    let expected1 = make_vector_with_index(vec![1.0, 1.0], &idx);
    let expected2 = make_vector_with_index(vec![1.0, 1.0], &idx);
    let expected = BlockTensor::new(vec![expected1, expected2], (2, 1)).unwrap();

    // Diagonal block operator
    let diag1 = [2.0, 3.0];
    let diag2 = [4.0, 5.0];

    let apply_a =
        move |x: &BlockTensor<TensorDynLen>| -> anyhow::Result<BlockTensor<TensorDynLen>> {
            let x1 = x.get(0, 0).unwrap();
            let x2 = x.get(1, 0).unwrap();

            // Apply D1 to x1
            let x1_data = x1.to_vec::<f64>()?;
            let y1_data: Vec<f64> = x1_data
                .iter()
                .zip(diag1.iter())
                .map(|(&xi, &di)| xi * di)
                .collect();
            let y1 = TensorDynLen::from_dense(x1.indices.clone(), y1_data).unwrap();

            // Apply D2 to x2
            let x2_data = x2.to_vec::<f64>()?;
            let y2_data: Vec<f64> = x2_data
                .iter()
                .zip(diag2.iter())
                .map(|(&xi, &di)| xi * di)
                .collect();
            let y2 = TensorDynLen::from_dense(x2.indices.clone(), y2_data).unwrap();

            BlockTensor::new(vec![y1, y2], (2, 1)).map_err(anyhow::Error::from)
        };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

// ========================================================================
// Test 3: Upper triangular block matrix GMRES
// ========================================================================

#[test]
fn test_gmres_upper_triangular_block() {
    // A = [[I, B], [0, I]] where B = I (identity)
    // Ax = [x1 + x2, x2]^T
    // b = [[2, 3]^T, [1, 1]^T]
    // x2 = [1, 1]^T, x1 = [2, 3]^T - [1, 1]^T = [1, 2]^T
    let idx = DynIndex::new_dyn(2);

    let b1 = make_vector_with_index(vec![2.0, 3.0], &idx);
    let b2 = make_vector_with_index(vec![1.0, 1.0], &idx);
    let b = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1)).unwrap();

    let expected1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let expected2 = make_vector_with_index(vec![1.0, 1.0], &idx);
    let expected = BlockTensor::new(vec![expected1, expected2], (2, 1)).unwrap();

    // Upper triangular block operator: A = [[I, I], [0, I]]
    let apply_a = |x: &BlockTensor<TensorDynLen>| -> anyhow::Result<BlockTensor<TensorDynLen>> {
        let x1 = x.get(0, 0).unwrap();
        let x2 = x.get(1, 0).unwrap();

        // y1 = x1 + x2
        let y1 = x1.axpby(AnyScalar::new_real(1.0), x2, AnyScalar::new_real(1.0))?;
        // y2 = x2
        let y2 = x2.clone();

        BlockTensor::new(vec![y1, y2], (2, 1)).map_err(anyhow::Error::from)
    };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

// ========================================================================
// Test: Basic vector space operations
// ========================================================================

#[test]
fn test_norm_squared() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    // norm_squared = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
    let norm_sq = block.norm_squared().unwrap();
    assert!((norm_sq - 30.0).abs() < 1e-10);
}

#[test]
fn test_isapprox_uses_trait_default_with_error_propagation() {
    let idx = DynIndex::new_dyn(2);
    let a = BlockTensor::new(vec![make_vector_with_index(vec![1.0, 2.0], &idx)], (1, 1)).unwrap();
    let b = BlockTensor::new(
        vec![make_vector_with_index(vec![1.0 + 1.0e-13, 2.0], &idx)],
        (1, 1),
    )
    .unwrap();
    assert!(TensorVectorSpace::isapprox(&a, &b, 1.0e-12, 0.0).unwrap());

    // A NaN block makes the default norm fail; the trait default propagates it.
    let nan = BlockTensor::new(vec![TensorDynLen::scalar(f64::NAN).unwrap()], (1, 1)).unwrap();
    assert!(TensorVectorSpace::isapprox(&nan, &a, 1.0e-12, 1.0e-12).is_err());

    // A shape mismatch makes the default sub fail before any norm is taken.
    let wide = BlockTensor::new(vec![TensorDynLen::scalar(1.0).unwrap(); 2], (2, 1)).unwrap();
    assert!(TensorVectorSpace::isapprox(&a, &wide, 1.0e-12, 1.0e-12).is_err());
}

#[test]
fn test_scale() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let scaled = block.scale(AnyScalar::new_real(2.0)).unwrap();

    // Check scaled values
    let expected1 = make_vector_with_index(vec![2.0, 4.0], &idx);
    let expected2 = make_vector_with_index(vec![6.0, 8.0], &idx);
    let expected = BlockTensor::new(vec![expected1, expected2], (2, 1)).unwrap();

    let diff = scaled
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10);
}

#[test]
fn test_inner_product() {
    let idx = DynIndex::new_dyn(2);
    let a1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let a2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let a = BlockTensor::new(vec![a1, a2], (2, 1)).unwrap();

    let b1 = make_vector_with_index(vec![5.0, 6.0], &idx);
    let b2 = make_vector_with_index(vec![7.0, 8.0], &idx);
    let b = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    // inner_product = (1*5 + 2*6) + (3*7 + 4*8) = 17 + 53 = 70
    let ip = a.inner_product(&b).unwrap();
    assert!((ip.real() - 70.0).abs() < 1e-10);
}

#[test]
fn test_conj() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    // For real tensors, conj should be identity
    let conjugated = block.conj();
    let diff = conjugated
        .axpby(AnyScalar::new_real(1.0), &block, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10);
}

// ========================================================================
// validate_indices tests
// ========================================================================

#[test]
fn test_validate_indices_column_shared() {
    // Column vector with shared index → should pass
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();
    assert!(block.validate_indices().is_ok());
}

#[test]
fn test_validate_indices_column_independent() {
    // Column vector with independent indices → should also pass
    // (different rows can have independent indices; the operator determines relationships)
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx1);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx2);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();
    assert!(block.validate_indices().is_ok());
}

#[test]
fn test_validate_indices_matrix_shared() {
    // 2x2 matrix: same-row blocks share one index, same-column blocks share another
    let row0_idx = DynIndex::new_dyn(2);
    let row1_idx = DynIndex::new_dyn(2);
    let col0_idx = DynIndex::new_dyn(3);
    let col1_idx = DynIndex::new_dyn(3);

    // Block (0,0): [col0_idx, row0_idx]
    let b00 =
        TensorDynLen::from_dense(vec![col0_idx.clone(), row0_idx.clone()], vec![0.0; 6]).unwrap();
    // Block (0,1): [col1_idx, row0_idx] — same row → shares row0_idx
    let b01 =
        TensorDynLen::from_dense(vec![col1_idx.clone(), row0_idx.clone()], vec![0.0; 6]).unwrap();
    // Block (1,0): [col0_idx, row1_idx] — same column → shares col0_idx
    let b10 =
        TensorDynLen::from_dense(vec![col0_idx.clone(), row1_idx.clone()], vec![0.0; 6]).unwrap();
    // Block (1,1): [col1_idx, row1_idx]
    let b11 =
        TensorDynLen::from_dense(vec![col1_idx.clone(), row1_idx.clone()], vec![0.0; 6]).unwrap();

    let block = BlockTensor::new(vec![b00, b01, b10, b11], (2, 2)).unwrap();
    assert!(block.validate_indices().is_ok());
}

#[test]
fn test_validate_indices_matrix_no_row_sharing() {
    // 2x2 matrix: all indices independent → should fail (no common indices in same row)
    let b00 = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![0.0; 2]).unwrap();
    let b01 = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![0.0; 2]).unwrap();
    let b10 = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![0.0; 2]).unwrap();
    let b11 = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![0.0; 2]).unwrap();

    let block = BlockTensor::new(vec![b00, b01, b10, b11], (2, 2)).unwrap();
    assert!(block.validate_indices().is_err());
}

#[test]
fn test_validate_indices_matrix_same_id_prime_is_not_shared_index() {
    let row = DynIndex::new_dyn(2);
    let row_prime = row.prime();
    let b00 = TensorDynLen::from_dense(vec![row], vec![0.0; 2]).unwrap();
    let b01 = TensorDynLen::from_dense(vec![row_prime], vec![0.0; 2]).unwrap();

    let block = BlockTensor::new(vec![b00, b01], (1, 2)).unwrap();

    assert!(block.validate_indices().is_err());
}

#[test]
fn test_external_indices_deduplication() {
    // Column vector with shared index → external_indices should return 1 unique index
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();
    let ext = block.external_indices();
    assert_eq!(ext.len(), 1, "Shared index should appear once");
    assert_eq!(ext[0], idx);
}

#[test]
fn test_external_indices_preserves_same_id_prime_pair() {
    let idx = DynIndex::new_dyn(2);
    let idx_prime = idx.prime();
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx_prime);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let ext = block.external_indices();

    assert_eq!(ext, vec![idx, idx_prime]);
}

#[test]
fn test_external_indices_independent() {
    // Column vector with independent indices → external_indices returns 2 unique indices
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx1);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx2);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();
    let ext = block.external_indices();
    assert_eq!(ext.len(), 2, "Independent indices should both appear");
}

// ========================================================================
// Accessor tests
// ========================================================================

#[test]
fn test_get_mut() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let mut block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    // Mutate block (1, 0) via get_mut
    let blk = block.get_mut(1, 0).unwrap();
    *blk = make_vector_with_index(vec![10.0, 20.0], &idx);

    let data = block.get(1, 0).unwrap().to_vec::<f64>().unwrap();
    assert!((data[0] - 10.0).abs() < 1e-10);
    assert!((data[1] - 20.0).abs() < 1e-10);
}

#[test]
fn test_blocks_and_blocks_mut() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let mut block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    // blocks() returns immutable slice
    assert_eq!(block.blocks().len(), 2);

    // blocks_mut() returns mutable slice
    block.blocks_mut()[0] = make_vector_with_index(vec![10.0, 20.0], &idx);
    let data = block.get(0, 0).unwrap().to_vec::<f64>().unwrap();
    assert!((data[0] - 10.0).abs() < 1e-10);
}

#[test]
fn test_into_blocks() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let blocks = block.into_blocks();
    assert_eq!(blocks.len(), 2);
    let data = blocks[1].to_vec::<f64>().unwrap();
    assert!((data[0] - 3.0).abs() < 1e-10);
}

// ========================================================================
// TensorIndex: replaceind / replaceinds
// ========================================================================

#[test]
fn test_replaceind() {
    let idx = DynIndex::new_dyn(2);
    let new_idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let replaced = block.replaceind(&idx, &new_idx).unwrap();

    // All blocks should now use new_idx
    for blk in replaced.blocks() {
        let ext = blk.external_indices();
        assert_eq!(ext[0], new_idx);
    }
    // Data should be preserved
    let data = replaced.get(0, 0).unwrap().to_vec::<f64>().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_replaceinds() {
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    let new_idx1 = DynIndex::new_dyn(2);
    let new_idx2 = DynIndex::new_dyn(3);

    let b1 = TensorDynLen::from_dense(vec![idx1.clone(), idx2.clone()], vec![0.0; 6]).unwrap();
    let b2 = TensorDynLen::from_dense(vec![idx1.clone(), idx2.clone()], vec![1.0; 6]).unwrap();
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let replaced = block
        .replaceinds(
            &[idx1.clone(), idx2.clone()],
            &[new_idx1.clone(), new_idx2.clone()],
        )
        .unwrap();

    // Check that indices were replaced in both blocks
    for blk in replaced.blocks() {
        let ext = blk.external_indices();
        let ids: Vec<_> = ext.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(new_idx1.id()));
        assert!(ids.contains(new_idx2.id()));
    }
    assert_eq!(replaced.shape(), (2, 1));
}

// ========================================================================
// maxabs
// ========================================================================

#[test]
fn test_maxabs() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, -5.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block = BlockTensor::new(vec![b1, b2], (2, 1)).unwrap();

    let m = block.maxabs().unwrap();
    assert!((m - 5.0).abs() < 1e-10);
}

// ========================================================================
// Unsupported operations
// ========================================================================

#[test]
fn test_factorize_unsupported() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let block = BlockTensor::new(vec![b1], (1, 1)).unwrap();

    let result = block.factorize(&[idx], &FactorizeOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_direct_sum_unsupported() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block1 = BlockTensor::new(vec![b1], (1, 1)).unwrap();
    let block2 = BlockTensor::new(vec![b2], (1, 1)).unwrap();

    let result = block1.direct_sum(&block2, &[(idx.clone(), idx.clone())]);
    assert!(result.is_err());
}

#[test]
fn test_outer_product_unsupported() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
    let block1 = BlockTensor::new(vec![b1], (1, 1)).unwrap();
    let block2 = BlockTensor::new(vec![b2], (1, 1)).unwrap();

    let result = block1.outer_product(&block2);
    assert!(result.is_err());
}

#[test]
fn test_permuteinds_unsupported() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let block = BlockTensor::new(vec![b1], (1, 1)).unwrap();

    let result = block.permuteinds(&[idx]);
    assert!(result.is_err());
}

#[test]
fn test_contract_unsupported() {
    let idx = DynIndex::new_dyn(2);
    let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
    let block = BlockTensor::new(vec![b1], (1, 1)).unwrap();

    let result = BlockTensor::<TensorDynLen>::contract(&[&block]);
    assert!(result.is_err());
}

#[test]
fn test_diagonal_unsupported() {
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(2);

    let result = BlockTensor::<TensorDynLen>::diagonal(&idx1, &idx2);
    assert!(result.is_err());
}

#[test]
fn test_scalar_one_unsupported() {
    let result = BlockTensor::<TensorDynLen>::scalar_one();
    assert!(result.is_err());
}

#[test]
fn test_ones_unsupported() {
    let idx = DynIndex::new_dyn(2);

    let result = BlockTensor::<TensorDynLen>::ones(&[idx]);
    assert!(result.is_err());
}

#[test]
fn test_onehot_unsupported() {
    let idx = DynIndex::new_dyn(2);

    let result = BlockTensor::<TensorDynLen>::onehot(&[(idx, 0)]);
    assert!(result.is_err());
}
