use super::*;

#[test]
fn test_mpo_zeros() {
    let mpo = MPO::<f64>::zeros(&[(2, 2), (3, 3), (2, 2)]);
    assert_eq!(mpo.len(), 3);
    assert_eq!(mpo.site_dims(), vec![(2, 2), (3, 3), (2, 2)]);
    assert_eq!(mpo.rank(), 1);
}

#[test]
fn test_mpo_constant() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 5.0);
    assert_eq!(mpo.len(), 2);

    // Sum should be 5.0 * (2*2) * (2*2) = 5.0 * 4 * 4 = 80.0
    let sum = mpo.sum().unwrap();
    assert!((sum - 80.0).abs() < 1e-10);
}

#[test]
fn test_mpo_new_rejects_invalid_structure() {
    let left: Tensor4<f64> = tensor4_zeros(1, 2, 2, 2);
    let right: Tensor4<f64> = tensor4_zeros(3, 2, 2, 1);
    let err = MPO::new(vec![left, right]).unwrap_err();
    assert!(err.to_string().contains("Bond shape mismatch"));

    let bad_left: Tensor4<f64> = tensor4_zeros(2, 2, 2, 1);
    let err = MPO::new(vec![bad_left]).unwrap_err();
    assert!(err.to_string().contains("Invalid boundary"));

    let bad_right: Tensor4<f64> = tensor4_zeros(1, 2, 2, 2);
    let err = MPO::new(vec![bad_right]).unwrap_err();
    assert!(err.to_string().contains("Invalid boundary"));
}

#[test]
fn test_mpo_empty_and_single_site_constructors() {
    let empty = MPO::<f64>::constant(&[], 3.0);
    assert!(empty.is_empty());
    assert_eq!(empty.link_dims(), Vec::<usize>::new());
    assert_eq!(empty.rank(), 1);
    assert_eq!(empty.fulltensor(), (Vec::new(), Vec::new()));
    assert_eq!(empty.sum().unwrap(), 0.0);
    assert!(empty
        .evaluate(&[])
        .unwrap_err()
        .to_string()
        .contains("MPO is empty"));

    let single = MPO::<f64>::constant(&[(2, 3)], 4.0);
    assert_eq!(single.site_dim(0), (2, 3));
    assert_eq!(single.link_dims(), Vec::<usize>::new());
    assert_eq!(single.evaluate(&[1, 2]).unwrap(), 4.0);
    assert_eq!(single.scaled(0.5).evaluate(&[1, 2]).unwrap(), 2.0);

    let identity_empty = MPO::<f64>::identity(&[]).unwrap();
    assert!(identity_empty.is_empty());
}

#[test]
fn test_mpo_identity() {
    let mpo = MPO::<f64>::identity(&[2, 3]).unwrap();
    assert_eq!(mpo.len(), 2);
    assert_eq!(mpo.site_dims(), vec![(2, 2), (3, 3)]);

    // Identity: O[i, j, k, l] = delta(i, j) * delta(k, l)
    // So evaluate([0, 0, 0, 0]) = 1
    assert!((mpo.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    // evaluate([0, 1, 0, 0]) = 0 (i != j)
    assert!((mpo.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
    // evaluate([1, 1, 2, 2]) = 1
    assert!((mpo.evaluate(&[1, 1, 2, 2]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_mpo_evaluate() {
    // Create a simple MPO
    let mut t0: Tensor4<f64> = tensor4_zeros(1, 2, 2, 1);
    t0.set4(0, 0, 0, 0, 1.0);
    t0.set4(0, 0, 1, 0, 2.0);
    t0.set4(0, 1, 0, 0, 3.0);
    t0.set4(0, 1, 1, 0, 4.0);

    let mut t1: Tensor4<f64> = tensor4_zeros(1, 2, 2, 1);
    t1.set4(0, 0, 0, 0, 1.0);
    t1.set4(0, 0, 1, 0, 0.5);
    t1.set4(0, 1, 0, 0, 2.0);
    t1.set4(0, 1, 1, 0, 1.5);

    let mpo = MPO::new(vec![t0, t1]).unwrap();

    // evaluate([0, 0, 0, 0]) = t0[0, 0, 0, 0] * t1[0, 0, 0, 0] = 1 * 1 = 1
    assert!((mpo.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    // evaluate([1, 1, 0, 1]) = t0[0, 1, 1, 0] * t1[0, 0, 1, 0] = 4 * 0.5 = 2
    assert!((mpo.evaluate(&[1, 1, 0, 1]).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_mpo_evaluate_reports_invalid_indices() {
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let err = mpo.evaluate(&[0, 0, 0]).unwrap_err();
    assert!(err.to_string().contains("Expected 4 indices"));

    let err = mpo.evaluate(&[2, 0, 0, 0]).unwrap_err();
    assert!(err.to_string().contains("Index out of bounds"));
    assert!(err.to_string().contains("site 0"));

    let err = mpo.evaluate(&[0, 0, 0, 2]).unwrap_err();
    assert!(err.to_string().contains("Index out of bounds"));
    assert!(err.to_string().contains("site 1"));
}

#[test]
fn test_mpo_fulltensor_zero_dimension_returns_shape() {
    let mpo = MPO::<f64>::zeros(&[(0, 2)]);
    let (data, shape) = mpo.fulltensor();
    assert!(data.is_empty());
    assert_eq!(shape, vec![0, 2]);
}

#[test]
fn test_mpo_scale() {
    let mut mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    mpo.scale(3.0);

    // Sum should be 3.0 * (2*2) * (2*2) = 3.0 * 16 = 48.0
    let sum = mpo.sum().unwrap();
    assert!((sum - 48.0).abs() < 1e-10);
}

#[test]
fn test_mpo_link_dims() {
    let mut t0: Tensor4<f64> = tensor4_zeros(1, 2, 2, 3);
    let mut t1: Tensor4<f64> = tensor4_zeros(3, 2, 2, 2);
    let t2: Tensor4<f64> = tensor4_zeros(2, 2, 2, 1);

    // Fill with some values
    t0.set4(0, 0, 0, 0, 1.0);
    t1.set4(0, 0, 0, 0, 1.0);

    let mpo = MPO::new(vec![t0, t1, t2]).unwrap();
    assert_eq!(mpo.link_dims(), vec![3, 2]);
    assert_eq!(mpo.rank(), 3);
}

#[test]
fn test_mpo_fulltensor() {
    let mpo = MPO::<f64>::constant(&[(2, 2)], 5.0);
    let (data, shape) = mpo.fulltensor();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(data.len(), 4);

    // All elements should be 5.0
    for val in &data {
        assert!((val - 5.0).abs() < 1e-10);
    }
}

#[test]
fn test_mpo_fulltensor_matches_evaluate() {
    let mut tensor: Tensor4<f64> = tensor4_zeros(1, 2, 3, 1);
    tensor.set4(0, 0, 0, 0, 1.0);
    tensor.set4(0, 1, 0, 0, 2.0);
    tensor.set4(0, 0, 1, 0, 3.0);
    tensor.set4(0, 1, 1, 0, 4.0);
    tensor.set4(0, 0, 2, 0, 5.0);
    tensor.set4(0, 1, 2, 0, 6.0);

    let mpo = MPO::new(vec![tensor]).unwrap();
    let (data, shape) = mpo.fulltensor();

    assert_eq!(shape, vec![2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let idx = i + 2 * j;
            let expected = mpo.evaluate(&[i, j]).unwrap();
            assert!(
                (data[idx] - expected).abs() < 1e-10,
                "Mismatch at [{i}, {j}]"
            );
        }
    }
}
