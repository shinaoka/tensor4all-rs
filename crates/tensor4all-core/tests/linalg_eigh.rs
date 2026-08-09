use num_complex::Complex64;
use tensor4all_core::{AnyScalar, DynIndex, TensorContractionLike, TensorDynLen};

#[test]
fn hermitian_eigendecomposition_solves_complex_residuals() {
    let row = DynIndex::new_dyn(2);
    let col = DynIndex::new_dyn(2);
    let matrix = TensorDynLen::from_dense(
        vec![row.clone(), col.clone()],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();

    let decomp = matrix.hermitian_eigendecomposition(1.0e-12).unwrap();

    assert_eq!(decomp.eigenvalues.len(), 2);
    assert!((decomp.eigenvalues[0] - 1.0).abs() < 1.0e-12);
    assert!((decomp.eigenvalues[1] - 3.0).abs() < 1.0e-12);
    assert_eq!(
        decomp.eigenvectors.indices(),
        &[row.clone(), decomp.eigenvector_index.clone()]
    );

    for (position, &lambda) in decomp.eigenvalues.iter().enumerate() {
        let vector = decomp
            .eigenvectors
            .select_indices(std::slice::from_ref(&decomp.eigenvector_index), &[position])
            .unwrap();
        let vector_as_col = vector.replaceind(&row, &col).unwrap();
        let applied = TensorDynLen::contract(&[&matrix, &vector_as_col]).unwrap();
        let expected = vector.scale(AnyScalar::new_real(lambda)).unwrap();

        assert!(
            applied.isapprox(&expected, 1.0e-10, 0.0).unwrap(),
            "eigenpair {position} residual maxabs={}",
            applied.sub(&expected).unwrap().maxabs().unwrap()
        );
    }
}

#[test]
fn hermitian_eigendecomposition_keeps_eigenvectors_tracked() {
    let row = DynIndex::new_dyn(2);
    let col = DynIndex::new_dyn(2);
    let matrix = TensorDynLen::from_dense(vec![row, col], vec![2.0_f64, 0.25, 0.25, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let decomp = matrix.hermitian_eigendecomposition(1.0e-12).unwrap();

    assert!(decomp.eigenvectors.tracks_grad());
}
