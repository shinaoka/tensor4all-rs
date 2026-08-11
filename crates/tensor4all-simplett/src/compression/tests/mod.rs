use super::*;
use num_complex::Complex64;
use tenferro_tensor::TensorScalar;

// Generic test functions for f64 and Complex64

fn test_compress_constant_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], <T as Scalar>::from_f64(1.0));
    let original_sum = tt.sum();

    let mut tt_compressed = tt.clone();
    tt_compressed
        .compress(&CompressionOptions::default())
        .unwrap();

    let compressed_sum = tt_compressed.sum();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-10);
}

fn test_compress_preserves_values_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a simple tensor train
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, <T as Scalar>::from_f64(1.0));
    t0.set3(0, 0, 1, <T as Scalar>::from_f64(0.5));
    t0.set3(0, 1, 0, <T as Scalar>::from_f64(0.0));
    t0.set3(0, 1, 1, <T as Scalar>::from_f64(1.0));

    let mut t1: Tensor3<T> = tensor3_zeros(2, 3, 2);
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(
                    l,
                    s,
                    r,
                    <T as Scalar>::from_f64(((l + s + r) as f64) * 0.1 + 0.1),
                );
            }
        }
    }

    let mut t2: Tensor3<T> = tensor3_zeros(2, 2, 1);
    t2.set3(0, 0, 0, <T as Scalar>::from_f64(1.0));
    t2.set3(0, 1, 0, <T as Scalar>::from_f64(0.5));
    t2.set3(1, 0, 0, <T as Scalar>::from_f64(0.5));
    t2.set3(1, 1, 0, <T as Scalar>::from_f64(1.0));

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let original_sum = tt.sum();

    let mut tt_compressed = tt.clone();
    tt_compressed
        .compress(&CompressionOptions::default())
        .unwrap();

    let compressed_sum = tt_compressed.sum();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-8);
}

fn test_compress_with_max_bond_dim_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a tensor train with higher bond dimension
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 3);
    for s in 0..2 {
        for r in 0..3 {
            t0.set3(0, s, r, <T as Scalar>::from_f64((s + r + 1) as f64));
        }
    }

    let mut t1: Tensor3<T> = tensor3_zeros(3, 2, 1);
    for l in 0..3 {
        for s in 0..2 {
            t1.set3(l, s, 0, <T as Scalar>::from_f64((l + s + 1) as f64));
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let original_norm = tt.norm();

    let options = CompressionOptions {
        max_bond_dim: Some(2),
        tolerance: 1e-12,
        ..Default::default()
    };

    let mut tt_compressed = tt.clone();
    tt_compressed.compress(&options).unwrap();

    // Norm should be approximately preserved (with some truncation error)
    let compressed_norm = tt_compressed.norm();
    assert!((original_norm - compressed_norm).abs() < original_norm * 0.1);
}

// f64 tests
#[test]
fn test_compress_constant_f64() {
    test_compress_constant_generic::<f64>();
}

#[test]
fn test_compress_preserves_values_f64() {
    test_compress_preserves_values_generic::<f64>();
}

#[test]
fn test_compress_with_max_bond_dim_f64() {
    test_compress_with_max_bond_dim_generic::<f64>();
}

// Complex64 tests
#[test]
fn test_compress_constant_c64() {
    test_compress_constant_generic::<Complex64>();
}

#[test]
fn test_compress_preserves_values_c64() {
    test_compress_preserves_values_generic::<Complex64>();
}

#[test]
fn test_compress_with_max_bond_dim_c64() {
    test_compress_with_max_bond_dim_generic::<Complex64>();
}

fn test_compress_svd_constant_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], <T as Scalar>::from_f64(1.0));
    let original_sum = tt.sum();

    let mut tt_compressed = tt.clone();
    let options = CompressionOptions {
        method: CompressionMethod::SVD,
        ..Default::default()
    };
    tt_compressed.compress(&options).unwrap();

    let compressed_sum = tt_compressed.sum();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-10);
}

fn test_compress_svd_with_truncation_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a rank-3 TT and compress with SVD to max_bond_dim=2
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 3);
    for s in 0..2 {
        for r in 0..3 {
            t0.set3(0, s, r, <T as Scalar>::from_f64((s + r + 1) as f64));
        }
    }
    let mut t1: Tensor3<T> = tensor3_zeros(3, 2, 1);
    for l in 0..3 {
        for s in 0..2 {
            t1.set3(l, s, 0, <T as Scalar>::from_f64((l + s + 1) as f64));
        }
    }
    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let original_norm = tt.norm();

    let options = CompressionOptions {
        method: CompressionMethod::SVD,
        max_bond_dim: Some(2),
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut tt_compressed = tt.clone();
    tt_compressed.compress(&options).unwrap();

    let compressed_norm = tt_compressed.norm();
    assert!((original_norm - compressed_norm).abs() < original_norm * 0.1);
}

#[test]
fn test_compress_svd_constant_f64() {
    test_compress_svd_constant_generic::<f64>();
}

#[test]
fn test_compress_svd_constant_c64() {
    test_compress_svd_constant_generic::<Complex64>();
}

#[test]
fn test_compress_svd_with_truncation_f64() {
    test_compress_svd_with_truncation_generic::<f64>();
}

#[test]
fn test_compress_svd_with_truncation_c64() {
    test_compress_svd_with_truncation_generic::<Complex64>();
}
