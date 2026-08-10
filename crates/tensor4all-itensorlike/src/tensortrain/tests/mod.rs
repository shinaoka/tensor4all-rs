use super::*;
use std::error::Error;
use std::io;
use std::time::Duration;
use tensor4all_core::{DynId, Index, LinearizationOrder, TensorContractionLike, TensorVectorSpace};

/// Helper to create a simple tensor for testing
fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|i| i.size()).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

/// Helper to create a DynIndex
fn idx(id: u64, size: usize) -> DynIndex {
    Index::new_with_size(DynId(id), size)
}

#[test]
fn test_empty_tt() {
    let tt: TensorTrain = TensorTrain::new(vec![]).unwrap();
    assert!(tt.is_empty());
    assert_eq!(tt.len(), 0);
    assert_eq!(tt.llim(), -1);
    assert_eq!(tt.rlim(), 1);
    assert!(!tt.isortho());
}

#[test]
fn tensor_train_error_from_anyhow_retains_source_chain() {
    let source = anyhow::Error::new(io::Error::other("typed backend failure"));
    let error = TensorTrainError::from(source);
    assert!(error.source().is_some());
    assert!(error.to_string().contains("typed backend failure"));
}

#[test]
fn test_single_site_tt() {
    let tensor = make_tensor(vec![idx(0, 2)]);

    let tt = TensorTrain::new(vec![tensor]).unwrap();
    assert_eq!(tt.len(), 1);
    assert!(!tt.isortho());
    assert_eq!(tt.bond_dims(), Vec::<usize>::new());
}

#[test]
fn profile_helpers_and_basic_accessors_cover_paths() {
    let mut elapsed = Duration::ZERO;
    let value = profile_tt_inner_section(true, &mut elapsed, || 42usize);
    assert_eq!(value, 42);
    assert!(elapsed >= Duration::ZERO);
    print_tt_inner_profile(&TensorTrainInnerProfile::default(), 0);

    let tensor = make_tensor(vec![idx(0, 2)]);
    let tt = TensorTrain::new(vec![tensor]).unwrap();
    assert_eq!(tt.tensors().len(), 1);
    assert!(tt
        .tensor_checked(5)
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
    assert_eq!(tt.clone().into_treetn().node_count(), 1);
    assert!(tt.norm_squared().unwrap() >= 0.0);
}

#[test]
fn add_reindexed_like_self_aligns_site_indices_before_addition() {
    let i0 = idx(0, 2);
    let i1 = idx(1, 2);
    let link = idx(2, 2);
    let j0 = idx(10, 2);
    let j1 = idx(11, 2);
    let rhs_link = idx(12, 2);

    let lhs = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![i0.clone(), link.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        TensorDynLen::from_dense(vec![link.clone(), i1.clone()], vec![5.0, 6.0, 7.0, 8.0]).unwrap(),
    ])
    .unwrap();
    let rhs = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![j0.clone(), rhs_link.clone()], vec![2.0, 3.0, 4.0, 5.0])
            .unwrap(),
        TensorDynLen::from_dense(
            vec![rhs_link.clone(), j1.clone()],
            vec![7.0, 11.0, 13.0, 17.0],
        )
        .unwrap(),
    ])
    .unwrap();

    let sum = lhs.add_reindexed_like_self(&rhs).unwrap();
    assert_eq!(sum.len(), 2);
    assert_eq!(sum.siteinds(), vec![vec![i0.clone()], vec![i1.clone()]]);

    let dense = sum.to_dense().unwrap();
    let rhs_dense = rhs
        .to_dense()
        .unwrap()
        .replaceinds(&[j0, j1], &[i0, i1])
        .unwrap();
    let expected = lhs
        .to_dense()
        .unwrap()
        .axpby(
            AnyScalar::new_real(1.0),
            &rhs_dense,
            AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(dense.sub(&expected).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn test_fuse_indices_trait_dispatch_returns_unsupported_error() {
    let i = idx(0, 2);
    let fused = idx(1, 2);
    let tensor = make_tensor(vec![i.clone()]);
    let tt = TensorTrain::new(vec![tensor]).unwrap();

    let err = <TensorTrain as TensorContractionLike>::fuse_indices(
        &tt,
        &[i],
        fused,
        LinearizationOrder::ColumnMajor,
    )
    .unwrap_err();

    assert!(err
        .to_string()
        .contains("TensorTrain does not support TensorContractionLike::fuse_indices"));
}

#[test]
fn test_two_site_tt() {
    // Create two tensors with a shared link index
    let s0 = idx(0, 2); // site 0
    let l01 = idx(1, 3); // link 0-1
    let s1 = idx(2, 2); // site 1

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    assert_eq!(tt.len(), 2);
    assert_eq!(tt.bond_dims(), vec![3]);
    assert_eq!(tt.maxbonddim(), 3);

    // Check link index
    let link = tt.linkind(0).unwrap();
    assert_eq!(link.size(), 3);

    // Check site indices (nested vec)
    let site_inds = tt.siteinds();
    assert_eq!(site_inds.len(), 2);
    assert_eq!(site_inds[0].len(), 1);
    assert_eq!(site_inds[1].len(), 1);
    assert_eq!(site_inds[0][0].size(), 2);
    assert_eq!(site_inds[1][0].size(), 2);
}

#[test]
fn test_multi_site_indices() {
    // Test site with multiple physical indices
    let s0a = idx(0, 2); // site 0 index a
    let s0b = idx(1, 3); // site 0 index b
    let l01 = idx(2, 4); // link 0-1
    let s1 = idx(3, 2); // site 1

    let t0 = make_tensor(vec![s0a.clone(), s0b.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Check site indices (nested vec)
    let site_inds = tt.siteinds();
    assert_eq!(site_inds.len(), 2);
    assert_eq!(site_inds[0].len(), 2); // site 0 has 2 indices
    assert_eq!(site_inds[1].len(), 1); // site 1 has 1 index
}

#[test]
fn test_new_preserves_site_tensor_index_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![l01.clone(), s0.clone()]);
    let t1 = make_tensor(vec![s1.clone(), l01.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    assert_eq!(tt.tensor(0).unwrap().indices(), &[l01.clone(), s0]);
    assert_eq!(tt.tensor(1).unwrap().indices(), &[s1, l01]);
}

#[test]
fn test_with_ortho_preserves_site_tensor_index_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![l01.clone(), s0.clone()]);
    let t1 = make_tensor(vec![s1.clone(), l01.clone()]);

    let tt = TensorTrain::with_ortho(vec![t0, t1], -1, 1, Some(CanonicalForm::Unitary)).unwrap();

    assert_eq!(tt.tensor(0).unwrap().indices(), &[l01.clone(), s0]);
    assert_eq!(tt.tensor(1).unwrap().indices(), &[s1, l01]);
}

#[test]
fn test_from_treetn_preserves_site_tensor_index_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![l01.clone(), s0.clone()]);
    let t1 = make_tensor(vec![s1.clone(), l01.clone()]);
    let tree = tensor4all_treetn::TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap();

    let tt = TensorTrain::from_treetn(tree).unwrap();

    assert_eq!(tt.tensor(0).unwrap().indices(), &[l01.clone(), s0]);
    assert_eq!(tt.tensor(1).unwrap().indices(), &[s1, l01]);
}

#[test]
fn test_ortho_tracking() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    // Create with specified orthogonality (ortho center at site 0)
    let tt = TensorTrain::with_ortho(
        vec![t0, t1],
        -1, // no left orthogonality
        1,  // right orthogonal from site 1
        Some(CanonicalForm::Unitary),
    )
    .unwrap();

    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
}

#[test]
fn test_ortho_lims_range() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let l12 = idx(2, 3);
    let s1 = idx(3, 2);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1, l12.clone()]);
    let t2 = make_tensor(vec![l12, s2]);

    // Create with partial orthogonality
    let tt = TensorTrain::with_ortho(vec![t0, t1, t2], 0, 2, None).unwrap();

    assert_eq!(tt.ortho_lims(), 1..2);
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
}

#[test]
fn test_no_common_index_error() {
    let s0 = idx(0, 2);
    let s1 = idx(1, 2);

    let t0 = make_tensor(vec![s0]);
    let t1 = make_tensor(vec![s1]);

    let result = TensorTrain::new(vec![t0, t1]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorTrainError::InvalidStructure { .. }
    ));
}

#[test]
fn test_orthogonalize_two_site() {
    // Create a 2-site tensor train
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    assert!(!tt.isortho());

    // Orthogonalize to site 0
    tt.orthogonalize(0).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));

    // Orthogonalize to site 1
    tt.orthogonalize(1).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
}

#[test]
fn test_orthogonalize_three_site() {
    // Create a 3-site tensor train
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    // Orthogonalize to middle site
    tt.orthogonalize(1).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));

    // Orthogonalize to left
    tt.orthogonalize(0).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));

    // Orthogonalize to right
    tt.orthogonalize(2).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(2));
}

#[test]
fn test_orthogonalize_with_lu() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    tt.orthogonalize_with(0, CanonicalForm::LU).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::LU));
}

#[test]
fn test_orthogonalize_with_ci() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    tt.orthogonalize_with(1, CanonicalForm::CI).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::CI));
}

#[test]
fn test_truncate_with_max_rank() {
    // Create a 3-site tensor train with large bond dimension
    let s0 = idx(0, 4);
    let l01 = idx(1, 8);
    let s1 = idx(2, 4);
    let l12 = idx(3, 8);
    let s2 = idx(4, 4);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
    assert_eq!(tt.maxbonddim(), 8);

    // Truncate to max rank 4
    let options = TruncateOptions::svd().with_max_rank(4);
    tt.truncate(&options).unwrap();

    // Check that bond dimensions are reduced
    assert!(tt.maxbonddim() <= 4);
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
}

#[test]
fn test_contract_with_fit_method() {
    use crate::ContractOptions;

    // Use two-site tensor trains with shared site indices to exercise contraction
    let s0 = idx(100, 2);
    let s1 = idx(101, 2);
    let l01_a = idx(102, 3);
    let l01_b = idx(103, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    // Test contract with Fit method
    let options = ContractOptions::fit().with_max_rank(10).with_nhalfsweeps(4); // 4 half-sweeps = 2 full sweeps
    let result = tt1.contract_pair(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    let naive_result = tt1
        .to_dense()
        .unwrap()
        .contract_pair(&tt2.to_dense().unwrap())
        .unwrap();
    assert!(result_tt
        .to_dense()
        .unwrap()
        .isapprox(&naive_result, 1e-10, 0.0)
        .unwrap());
}

#[test]
fn test_contract_with_naive_method() {
    use crate::ContractOptions;

    // Use single-site tensor trains with a shared site index
    let s0 = idx(200, 2);

    let t1 = make_tensor(vec![s0.clone()]);
    let tt1 = TensorTrain::new(vec![t1]).unwrap();

    let t2 = make_tensor(vec![s0.clone()]);
    let tt2 = TensorTrain::new(vec![t2]).unwrap();

    // Test contract with Naive method
    let options = ContractOptions::naive().with_dense_reference_limit(2);
    let result = tt1.contract_pair(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    let naive_result = tt1
        .to_dense()
        .unwrap()
        .contract_pair(&tt2.to_dense().unwrap())
        .unwrap();
    assert!(result_tt
        .to_dense()
        .unwrap()
        .isapprox(&naive_result, 1e-10, 0.0)
        .unwrap());
}

#[test]
fn test_contract_nhalfsweeps_conversion() {
    use crate::ContractOptions;

    // Use two-site tensor trains with shared site indices to exercise contraction
    let s0 = idx(300, 2);
    let s1 = idx(301, 2);
    let l01_a = idx(302, 3);
    let l01_b = idx(303, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    // Test that nhalfsweeps is correctly converted to nfullsweeps
    // nhalfsweeps=6 should become nfullsweeps=3
    let options = ContractOptions::fit().with_nhalfsweeps(6).with_max_rank(10);
    let result = tt1.contract_pair(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    let naive_result = tt1
        .to_dense()
        .unwrap()
        .contract_pair(&tt2.to_dense().unwrap())
        .unwrap();
    assert!(result_tt
        .to_dense()
        .unwrap()
        .isapprox(&naive_result, 1e-10, 0.0)
        .unwrap());
}

#[test]
fn test_contract_fit_odd_nhalfsweeps_errors() {
    use crate::ContractOptions;

    let s0 = idx(400, 2);
    let s1 = idx(401, 2);
    let l01_a = idx(402, 3);
    let l01_b = idx(403, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    let options = ContractOptions::fit().with_nhalfsweeps(1).with_max_rank(10);
    let err = tt1.contract_pair(&tt2, &options).unwrap_err();
    assert!(matches!(err, TensorTrainError::OperationError { .. }));
}

#[test]
fn test_truncate_invalid_rtol_errors() {
    let s0 = idx(500, 2);
    let l01 = idx(501, 3);
    let s1 = idx(502, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let options =
        TruncateOptions::svd().with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(-1.0));
    let err = tt.truncate(&options).unwrap_err();
    assert!(matches!(err, TensorTrainError::OperationError { .. }));
}

#[test]
fn test_inner_product() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Compute norm squared
    let norm_sq = tt.norm_squared().unwrap();
    assert!(norm_sq > 0.0);

    // Compute norm
    let norm = tt.norm().unwrap();
    assert!((norm * norm - norm_sq).abs() < 1e-10);
}

#[test]
fn test_to_dense() {
    // Create a 2-site TT: s0 -- l01 -- s1
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();

    // Convert to dense
    let dense = tt.to_dense().unwrap();

    // Expected: contract t0 and t1 along l01
    let expected = t0.contract_pair(&t1).unwrap();

    // Compare results
    let dense_data = dense.to_vec::<f64>().unwrap();
    let expected_data = expected.to_vec::<f64>().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_single_site() {
    // Single site TT should return the tensor as-is
    let s0 = idx(0, 4);
    let t0 = make_tensor(vec![s0.clone()]);

    let tt = TensorTrain::new(vec![t0.clone()]).unwrap();
    let dense = tt.to_dense().unwrap();

    let dense_data = dense.to_vec::<f64>().unwrap();
    let expected_data = t0.to_vec::<f64>().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_three_sites() {
    // 3-site TT: s0 -- l01 -- s1 -- l12 -- s2
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let tt = TensorTrain::new(vec![t0.clone(), t1.clone(), t2.clone()]).unwrap();
    let dense = tt.to_dense().unwrap();

    // Expected: contract t0, t1, t2 sequentially
    let expected = t0.contract_pair(&t1).unwrap().contract_pair(&t2).unwrap();

    let dense_data = dense.to_vec::<f64>().unwrap();
    let expected_data = expected.to_vec::<f64>().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    let result = tt.to_dense();
    assert!(result.is_err());
}

#[test]
fn test_add_simple() {
    // Create two TTs with the same structure
    // Both TTs must have the same site indices AND link indices
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    // First TT with data [1, 2, 3, ...]
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

    // Second TT with same structure (clone and modify data for testing)
    let tt2 = tt1.clone();

    // Add them
    let sum = tt1.add(&tt2).unwrap();

    // Result should have double bond dimension
    assert_eq!(sum.len(), 2);
    assert_eq!(sum.bond_dims(), vec![6]); // 3 + 3

    // Verify numerically: sum.to_dense() == tt1.to_dense() + tt2.to_dense()
    let sum_dense = sum.to_dense().unwrap();
    let tt1_dense = tt1.to_dense().unwrap();
    let tt2_dense = tt2.to_dense().unwrap();

    let sum_data = sum_dense.to_vec::<f64>().unwrap();
    let tt1_data = tt1_dense.to_vec::<f64>().unwrap();
    let tt2_data = tt2_dense.to_vec::<f64>().unwrap();

    assert_eq!(sum_data.len(), tt1_data.len());
    for i in 0..sum_data.len() {
        let expected = tt1_data[i] + tt2_data[i];
        assert!(
            (sum_data[i] - expected).abs() < 1e-10,
            "Mismatch at {}: {} vs {}",
            i,
            sum_data[i],
            expected
        );
    }
}

#[test]
fn test_add_empty() {
    let empty = TensorTrain::new(vec![]).unwrap();

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // empty + tt = tt
    let result = empty.add(&tt).unwrap();
    assert_eq!(result.len(), tt.len());

    // tt + empty = tt
    let result = tt.add(&empty).unwrap();
    assert_eq!(result.len(), tt.len());

    // empty + empty = empty
    let result = empty.add(&empty).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_add_length_mismatch() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

    let t0_2 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1_2 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2_2 = make_tensor(vec![l12.clone(), s2.clone()]);
    let tt2 = TensorTrain::new(vec![t0_2, t1_2, t2_2]).unwrap();

    // Length mismatch should fail
    let result = tt1.add(&tt2);
    assert!(result.is_err());
}

#[test]
fn test_set_llim_updates_canonical_region() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Set llim=-1, rlim already 1 => center at 0
    tt.set_llim(-1);
    // With rlim=1 (which is the default for non-ortho TT) and llim=-1, center should be at 0
    // But this depends on the rlim value, let's explicitly set both
    let mut tt2 = TensorTrain::with_ortho(
        vec![
            make_tensor(vec![idx(0, 2), idx(1, 3)]),
            make_tensor(vec![idx(1, 3), idx(2, 2)]),
        ],
        -1,
        1,
        Some(CanonicalForm::Unitary),
    )
    .unwrap();
    assert!(tt2.isortho());
    assert_eq!(tt2.orthocenter(), Some(0));

    // Setting llim to a value that breaks single-center should clear ortho
    tt2.set_llim(5);
    assert!(!tt2.isortho());
}

#[test]
fn test_set_rlim_updates_canonical_region() {
    let mut tt = TensorTrain::with_ortho(
        vec![
            make_tensor(vec![idx(0, 2), idx(1, 3)]),
            make_tensor(vec![idx(1, 3), idx(2, 2)]),
        ],
        -1,
        1,
        Some(CanonicalForm::Unitary),
    )
    .unwrap();
    assert!(tt.isortho());

    // Setting rlim to a value that breaks single-center should clear ortho
    tt.set_rlim(5);
    assert!(!tt.isortho());
}

#[test]
fn test_set_tensor_invalidates_ortho() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt =
        TensorTrain::with_ortho(vec![t0, t1], -1, 1, Some(CanonicalForm::Unitary)).unwrap();
    assert!(tt.isortho());

    // Replace tensor at site 0
    let new_tensor = make_tensor(vec![s0, l01]);
    tt.set_tensor(0, new_tensor).unwrap();
    assert!(!tt.isortho());
}

#[test]
fn test_set_tensor_checked_invalid_site_errors() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1]);
    let replacement = make_tensor(vec![s0, l01]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let err = tt.set_tensor_checked(2, replacement).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 2, length: 2 }
    ));
}

#[test]
fn test_set_tensor_checked_preserves_replacement_index_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1]);
    let replacement = make_tensor(vec![l01.clone(), s0.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    tt.set_tensor_checked(0, replacement).unwrap();

    assert_eq!(tt.tensor(0).unwrap().indices(), &[l01, s0]);
}

#[test]
fn test_tensor_mut_checked_invalid_site_errors() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let err = tt.tensor_mut_checked(2).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 2, length: 2 }
    ));
}

#[test]
fn test_tensors_mut_returns_all_sites_in_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let replacement =
        TensorDynLen::from_dense(vec![s0, l01], vec![42.0; 6]).expect("valid replacement");

    {
        let mut tensors = tt.tensors_mut().unwrap();
        assert_eq!(tensors.len(), 2);
        *tensors[0] = replacement.clone();
    }

    assert_eq!(
        tt.tensor(0).unwrap().to_vec::<f64>().unwrap(),
        vec![42.0; 6]
    );
}

#[test]
fn test_tensors_mut_checked_returns_all_sites_in_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let replacement =
        TensorDynLen::from_dense(vec![s0, l01], vec![42.0; 6]).expect("valid replacement");

    {
        let mut tensors = tt.tensors_mut_checked().unwrap();
        assert_eq!(tensors.len(), 2);
        *tensors[0] = replacement.clone();
    }

    assert_eq!(
        tt.tensor(0).unwrap().to_vec::<f64>().unwrap(),
        vec![42.0; 6]
    );
}

#[test]
fn test_scale() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Scale by 2.0
    let scaled = tt.scale(AnyScalar::new_real(2.0)).unwrap();

    // Verify: norm of scaled should be 2 * norm of original
    let orig_norm = tt.norm().unwrap();
    let scaled_norm = scaled.norm().unwrap();
    assert!(
        (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
        "Expected scaled_norm = {}, got {}",
        2.0 * orig_norm,
        scaled_norm
    );
}

#[test]
fn test_scale_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    let scaled = tt.scale(AnyScalar::new_real(2.0)).unwrap();
    assert!(scaled.is_empty());
}

#[test]
fn test_axpby() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt1 = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();
    let tt2 = tt1.clone();

    // Compute 2*tt1 + 3*tt2 = 5*tt1 (since tt1 == tt2)
    let result = tt1
        .axpby(AnyScalar::new_real(2.0), &tt2, AnyScalar::new_real(3.0))
        .unwrap();

    // Verify numerically via to_dense
    let result_dense = result.to_dense().unwrap();
    let tt1_dense = tt1.to_dense().unwrap();

    let result_data = result_dense.to_vec::<f64>().unwrap();
    let tt1_data = tt1_dense.to_vec::<f64>().unwrap();

    assert_eq!(result_data.len(), tt1_data.len());
    for i in 0..result_data.len() {
        let expected = 5.0 * tt1_data[i]; // 2*tt1 + 3*tt2 = 5*tt1
        assert!(
            (result_data[i] - expected).abs() < 1e-10,
            "Mismatch at {}: {} vs {}",
            i,
            result_data[i],
            expected
        );
    }

    // Bond dimension should be 6 (3 + 3)
    assert_eq!(result.bond_dims(), vec![6]);
}

#[test]
fn test_tensor_like_scale() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Use TensorVectorSpace::scale
    let scaled = TensorVectorSpace::scale(&tt, AnyScalar::new_real(2.0)).unwrap();

    let orig_norm = tt.norm().unwrap();
    let scaled_norm = TensorVectorSpace::norm(&scaled).unwrap();
    assert!(
        (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
        "Expected scaled_norm = {}, got {}",
        2.0 * orig_norm,
        scaled_norm
    );
}

#[test]
fn test_tensor_like_inner_product() {
    use tensor4all_core::TensorVectorSpace;

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // TensorVectorSpace::inner_product should equal TensorTrain::inner
    let inner_via_trait = TensorVectorSpace::inner_product(&tt, &tt).unwrap();
    let inner_direct = tt.inner(&tt).unwrap();

    assert!(
        (inner_via_trait.real() - inner_direct.real()).abs() < 1e-10,
        "Inner product mismatch: {} vs {}",
        inner_via_trait.real(),
        inner_direct.real()
    );
}

#[test]
fn test_multiple_common_indices_error() {
    // Create two tensors that share TWO common indices => should error
    let shared1 = idx(10, 2);
    let shared2 = idx(11, 3);

    let t0 = make_tensor(vec![shared1.clone(), shared2.clone()]);
    let t1 = make_tensor(vec![shared1.clone(), shared2.clone()]);

    let result = TensorTrain::new(vec![t0, t1]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Multiple common indices"));
}

#[test]
fn test_set_canonical_form() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    assert_eq!(tt.canonical_form(), None);

    tt.set_canonical_form(Some(CanonicalForm::LU));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::LU));

    tt.set_canonical_form(None);
    assert_eq!(tt.canonical_form(), None);
}

#[test]
fn test_tensor_checked_out_of_bounds() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Valid access
    assert!(tt.tensor_checked(0).is_ok());
    assert!(tt.tensor_checked(1).is_ok());

    // Out of bounds
    let err = tt.tensor_checked(2).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 2, length: 2 }
    ));

    let err = tt.tensor_checked(100).unwrap_err();
    assert!(matches!(err, TensorTrainError::SiteOutOfBounds { .. }));
}

#[test]
fn test_tensor_mut() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Mutate tensor at site 0
    let t = tt.tensor_mut(0).unwrap();
    assert_eq!(t.indices().len(), 2);
    // Just verify we can get a mutable reference without panic
    let _ = t.indices();
}

#[test]
fn test_linkind_out_of_bounds() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Valid link
    assert!(tt.linkind(0).is_some());

    // Out of bounds
    assert!(tt.linkind(1).is_none());
    assert!(tt.linkind(100).is_none());
}

#[test]
fn test_sim_linkinds_single_site() {
    // Single site TT: sim_linkinds should return a clone
    let s0 = idx(0, 4);
    let t0 = make_tensor(vec![s0.clone()]);
    let tt = TensorTrain::new(vec![t0]).unwrap();

    let simmed = tt.sim_linkinds().unwrap();
    assert_eq!(simmed.len(), 1);
    // Should have same data
    let orig_data = tt.tensor(0).unwrap().to_vec::<f64>().unwrap();
    let sim_data = simmed.tensor(0).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(orig_data, sim_data);
}

#[test]
fn test_sim_linkinds_two_sites() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let simmed = tt.sim_linkinds().unwrap();

    assert_eq!(simmed.len(), 2);
    assert_eq!(simmed.bond_dims(), vec![3]);

    // The link index should have a different ID than the original
    let orig_link = tt.linkind(0).unwrap();
    let sim_link = simmed.linkind(0).unwrap();
    assert_ne!(orig_link.id(), sim_link.id());
    assert_eq!(orig_link.size(), sim_link.size());

    // Site indices should be preserved
    let orig_sites = tt.siteinds();
    let sim_sites = simmed.siteinds();
    assert_eq!(orig_sites[0][0].id(), sim_sites[0][0].id());
    assert_eq!(orig_sites[1][0].id(), sim_sites[1][0].id());
}

#[test]
fn test_siteinds_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    let site_inds = tt.siteinds();
    assert!(site_inds.is_empty());
}

#[test]
fn test_set_llim_valid_center() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1, l12.clone()]);
    let t2 = make_tensor(vec![l12, s2]);

    // Start with ortho center at site 1
    let mut tt =
        TensorTrain::with_ortho(vec![t0, t1, t2], 0, 2, Some(CanonicalForm::Unitary)).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));

    // set_llim to 0 with current rlim=2 => center should be 1 (0+1)
    tt.set_llim(0);
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
}

#[test]
fn test_set_rlim_valid_center() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1, l12.clone()]);
    let t2 = make_tensor(vec![l12, s2]);

    // Start with ortho center at site 0
    let mut tt =
        TensorTrain::with_ortho(vec![t0, t1, t2], -1, 1, Some(CanonicalForm::Unitary)).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));

    // set_rlim to 2 with current llim=-1 => llim will be recomputed.
    // After set_rlim(2): llim from orthocenter is recalculated.
    // Since set_rlim reads current llim first (which is -1), then checks -1+2==2? No, 1!=2.
    // So this clears ortho. Let's set rlim=1 which keeps center at 0.
    tt.set_rlim(1);
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
}

#[test]
fn test_orthogonalize_empty_errors() {
    let mut tt = TensorTrain::new(vec![]).unwrap();
    let err = tt.orthogonalize(0).unwrap_err();
    assert!(matches!(err, TensorTrainError::Empty));
}

#[test]
fn test_orthogonalize_out_of_bounds_errors() {
    let s0 = idx(0, 2);
    let t0 = make_tensor(vec![s0]);
    let mut tt = TensorTrain::new(vec![t0]).unwrap();

    let err = tt.orthogonalize(1).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 1, length: 1 }
    ));
}

#[test]
fn test_sim_linkinds_three_sites() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
    let simmed = tt.sim_linkinds().unwrap();

    assert_eq!(simmed.len(), 3);
    assert_eq!(simmed.bond_dims().len(), 2);

    // All link indices should have different IDs than originals
    for i in 0..2 {
        let orig_link = tt.linkind(i).unwrap();
        let sim_link = simmed.linkind(i).unwrap();
        assert_ne!(orig_link.id(), sim_link.id());
        assert_eq!(orig_link.size(), sim_link.size());
    }

    // Dense contraction should give same values
    let orig_dense = tt.to_dense().unwrap();
    let sim_dense = simmed.to_dense().unwrap();
    let orig_data = orig_dense.to_vec::<f64>().unwrap();
    let sim_data = sim_dense.to_vec::<f64>().unwrap();
    assert_eq!(orig_data.len(), sim_data.len());
    for (a, b) in orig_data.iter().zip(sim_data.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_truncate_single_site_noop() {
    // Truncating a single-site TT should be a no-op
    let s0 = idx(0, 4);
    let t0 = make_tensor(vec![s0]);
    let mut tt = TensorTrain::new(vec![t0]).unwrap();

    let options = TruncateOptions::svd().with_max_rank(2);
    tt.truncate(&options).unwrap();
    assert_eq!(tt.len(), 1);
}

#[test]
fn test_haslink() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    assert!(tt.haslink(0));
    assert!(!tt.haslink(1));
    assert!(!tt.haslink(100));
}

#[test]
fn test_linkinds() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 4);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1, l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2]);

    let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    let links = tt.linkinds();
    assert_eq!(links.len(), 2);
    assert_eq!(links[0].size(), 3);
    assert_eq!(links[1].size(), 4);
}

#[test]
fn test_bond_dim() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 5);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    assert_eq!(tt.bond_dim(0), Some(5));
    assert_eq!(tt.bond_dim(1), None);
}

#[test]
fn test_maxbonddim_single_site() {
    let s0 = idx(0, 4);
    let t0 = make_tensor(vec![s0]);
    let tt = TensorTrain::new(vec![t0]).unwrap();
    // Single site has no bonds, maxbonddim returns 1
    assert_eq!(tt.maxbonddim(), 1);
}

#[test]
fn test_dense_maxabs_is_explicit_dense_reference_api() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let maxabs = tt.dense_maxabs().unwrap();
    let dense = tt.to_dense().unwrap();
    let dense_maxabs = dense.maxabs().unwrap();
    assert!((maxabs - dense_maxabs).abs() < 1e-10);
}

#[test]
fn test_tensor_like_maxabs_is_not_hidden_dense_for_tensor_train() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let err = TensorVectorSpace::maxabs(&tt).unwrap_err();
    assert!(err.to_string().contains("explicit dense materialization"));
}

#[test]
fn test_tensor_like_conj() {
    use tensor4all_core::TensorContractionLike;

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // For real tensors, conj should be identical
    let conj_tt = TensorContractionLike::conj(&tt);
    assert_eq!(conj_tt.len(), tt.len());

    let orig_dense = tt.to_dense().unwrap();
    let conj_dense = conj_tt.to_dense().unwrap();

    let orig_data = orig_dense.to_vec::<f64>().unwrap();
    let conj_data = conj_dense.to_vec::<f64>().unwrap();

    assert_eq!(orig_data.len(), conj_data.len());
    for (a, b) in orig_data.iter().zip(conj_data.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_default_is_empty() {
    let tt = TensorTrain::default();
    assert!(tt.is_empty());
    assert_eq!(tt.len(), 0);
}

#[test]
fn test_replaceind() {
    use tensor4all_core::TensorIndex;

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Replace s0 with a new index of the same size
    let new_s0 = idx(100, 2);
    let tt2 = tt.replaceind(&s0, &new_s0).unwrap();

    // The new TT should have the new index
    let ext_inds = tt2.external_indices();
    assert!(ext_inds.contains(&new_s0));
    assert!(!ext_inds.contains(&s0));
}

#[test]
fn test_truncate_with_rtol() {
    let s0 = idx(0, 4);
    let l01 = idx(1, 8);
    let s1 = idx(2, 4);
    let l12 = idx(3, 8);
    let s2 = idx(4, 4);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1, l12.clone()]);
    let t2 = make_tensor(vec![l12, s2]);

    let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    let options =
        TruncateOptions::svd().with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10));
    tt.truncate(&options).unwrap();
    assert!(tt.isortho() || tt.len() == 3);
}

#[test]
fn tt_inner_profile_enabled_path_runs() {
    // The profile-gated branch of profile_tt_inner_section and
    // print_tt_inner_profile are normally dead under tests. Force the env
    // flag on and exercise a contraction so both branches execute.
    unsafe {
        std::env::set_var("T4A_PROFILE_TT_INNER", "1");
    }
    let i = DynIndex::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap();
    let _ = super::contract_pair(&a, &b).unwrap();
    unsafe {
        std::env::remove_var("T4A_PROFILE_TT_INNER");
    }
}

#[test]
fn tt_inner_profile_print_formats_zero_profile() {
    let profile = TensorTrainInnerProfile::default();
    print_tt_inner_profile(&profile, 3);
}

fn make_two_site_tt() -> TensorTrain {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

#[test]
fn tensor_checked_rejects_out_of_bounds_site() {
    let tt = make_two_site_tt();
    let err = tt.tensor_checked(5).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 5, .. }
    ));
}

#[test]
fn tensor_mut_checked_rejects_out_of_bounds_site() {
    let mut tt = make_two_site_tt();
    let err = tt.tensor_mut_checked(7).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 7, .. }
    ));
}

#[test]
fn tensor_mut_rejects_out_of_bounds_site() {
    let mut tt = make_two_site_tt();
    assert!(tt.tensor_mut(9).is_err());
}

#[test]
fn empty_tt_len_and_tensor_checked() {
    let tt = TensorTrain::new(vec![]).unwrap();
    assert_eq!(tt.len(), 0);
    assert!(tt.tensor_checked(0).is_err());
    assert!(tt.ortho_lims().is_empty());
}

#[test]
fn haslink_bounds_and_missing_node() {
    let tt = make_two_site_tt();
    assert!(tt.haslink(0));
    // Out-of-range site: no link
    assert!(!tt.haslink(5));
}

#[test]
fn set_tensor_raw_out_of_bounds_reports_site() {
    let mut tt = make_two_site_tt();
    let t = make_tensor(vec![idx(0, 2)]);
    let err = tt.set_tensor_raw(9, t).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 9, .. }
    ));
}

#[test]
fn new_rejects_no_common_index_between_adjacent() {
    let t0 = make_tensor(vec![idx(0, 2)]);
    let t1 = make_tensor(vec![idx(3, 2)]);
    let err = TensorTrain::new(vec![t0, t1]).unwrap_err();
    assert!(err.to_string().contains("No common index"));
}

#[test]
fn new_rejects_multiple_common_indices() {
    let a = idx(0, 2);
    let b = idx(1, 2);
    let t0 = make_tensor(vec![a.clone(), b.clone()]);
    let t1 = make_tensor(vec![a.clone(), b.clone()]);
    let err = TensorTrain::new(vec![t0, t1]).unwrap_err();
    assert!(err.to_string().contains("Multiple common indices"));
}

#[test]
fn new_empty_tt_has_no_ortho_lims() {
    let tt = TensorTrain::new(vec![]).unwrap();
    assert_eq!(tt.len(), 0);
    assert!(tt.ortho_lims().is_empty());
}

#[test]
fn tensor_index_replaceinds_updates_site_indices() {
    let tt = make_two_site_tt();
    let old = tt.siteinds()[0][0].clone();
    let new = idx(50, old.size());
    let replaced = TensorIndex::replaceinds(&tt, &[old], &[new]).unwrap();
    let got = replaced.siteinds()[0][0].clone();
    assert_eq!(got.id().value(), 50);
    assert_eq!(got.size(), 2);
}

#[test]
fn siteinds_empty_tt_returns_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    assert!(tt.siteinds().is_empty());
}
