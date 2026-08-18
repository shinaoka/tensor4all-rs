use tensor4all_core::{
    factorize_full_rank, svd, Canonical, FactorizeAlg, IdxTensor, Index, TensorContractionLike,
};

fn assert_f64_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "entry {idx}: expected {expected}, got {actual}"
        );
    }
}

fn finite_diff_svd_singular_value_sum(data: &[f64], index: usize) -> f64 {
    const STEP: f64 = 1.0e-6;

    fn eval(data: &[f64]) -> f64 {
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(2);
        let tensor = IdxTensor::from_dense(vec![i.clone(), j], data.to_vec()).unwrap();
        let (_u, s, _v) = svd::<f64>(&tensor, &[i]).unwrap();
        s.sum().unwrap().real()
    }

    let mut plus = data.to_vec();
    let mut minus = data.to_vec();
    plus[index] += STEP;
    minus[index] -= STEP;
    (eval(&plus) - eval(&minus)) / (2.0 * STEP)
}

#[test]
fn tensor_sum_returns_rank_zero_anyscalar_with_backward() {
    let i = Index::new_dyn(3);
    let x = IdxTensor::from_dense(vec![i], vec![1.0, 2.0, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let loss = x.sum().unwrap();
    assert!(loss.tracks_grad());

    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn tensor_only_preserves_tracking_for_scalar_tensor() {
    let x = IdxTensor::scalar(2.0).unwrap().enable_grad().unwrap();

    let scalar = x.only().unwrap();
    let loss = &scalar * &scalar;
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert!((grad.only().unwrap().real() - 4.0).abs() < 1e-12);
}

#[test]
fn factorize_qr_reconstruction_preserves_gradient_to_input() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let x = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![2.0_f64, 0.5, 1.0, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let result = factorize_full_rank(
        &x,
        std::slice::from_ref(&i),
        FactorizeAlg::QR,
        Canonical::Left,
    )
    .unwrap();
    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    let loss = reconstructed.sum().unwrap();
    assert!(loss.tracks_grad());
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_f64_slice_close(&grad.to_vec::<f64>().unwrap(), &[1.0, 1.0, 1.0, 1.0], 1e-8);
}

fn assert_ci_reconstruction_rejects_tracked_input(canonical: Canonical) {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let x = IdxTensor::from_dense(vec![i.clone(), j], vec![2.0_f64, 0.5, 1.0, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let error =
        factorize_full_rank(&x, std::slice::from_ref(&i), FactorizeAlg::CI, canonical).unwrap_err();
    assert!(matches!(
        error,
        tensor4all_core::FactorizeError::UnsupportedStorage(_)
    ));
    assert!(error.to_string().contains("tracked tensors"));
}

#[test]
fn factorize_ci_left_rejects_tracked_input() {
    assert_ci_reconstruction_rejects_tracked_input(Canonical::Left);
}

#[test]
fn factorize_ci_right_rejects_tracked_input() {
    assert_ci_reconstruction_rejects_tracked_input(Canonical::Right);
}

#[test]
fn svd_singular_value_loss_preserves_gradient_to_input() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let data = vec![2.0_f64, 0.5, 1.0, 3.0];
    let x = IdxTensor::from_dense(vec![i.clone(), j.clone()], data.clone())
        .unwrap()
        .enable_grad()
        .unwrap();

    let (_u, s, _v) = svd::<f64>(&x, std::slice::from_ref(&i)).unwrap();
    let loss = s.sum().unwrap();
    assert!(loss.tracks_grad());
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    let expected: Vec<f64> = (0..data.len())
        .map(|index| finite_diff_svd_singular_value_sum(&data, index))
        .collect();
    assert_f64_slice_close(&grad.to_vec::<f64>().unwrap(), &expected, 1e-5);
}
