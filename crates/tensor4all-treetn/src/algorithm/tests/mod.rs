use super::*;

#[test]
fn test_contraction_algorithm_roundtrip() {
    for alg in [
        ContractionAlgorithm::Naive,
        ContractionAlgorithm::ZipUp,
        ContractionAlgorithm::Fit,
        ContractionAlgorithm::Src,
    ] {
        let i = alg.to_i32();
        let recovered = ContractionAlgorithm::from_i32(i).unwrap();
        assert_eq!(alg, recovered);
    }
}

#[test]
fn test_compression_algorithm_roundtrip() {
    for alg in [
        CompressionAlgorithm::SVD,
        CompressionAlgorithm::LU,
        CompressionAlgorithm::CI,
        CompressionAlgorithm::Variational,
    ] {
        let i = alg.to_i32();
        let recovered = CompressionAlgorithm::from_i32(i).unwrap();
        assert_eq!(alg, recovered);
    }
}

#[test]
fn test_canonical_form_roundtrip() {
    for form in [CanonicalForm::Unitary, CanonicalForm::LU, CanonicalForm::CI] {
        let i = form.to_i32();
        let recovered = CanonicalForm::from_i32(i).unwrap();
        assert_eq!(form, recovered);
    }
}

#[test]
fn test_invalid_values() {
    assert!(ContractionAlgorithm::from_i32(-1).is_none());
    assert!(ContractionAlgorithm::from_i32(100).is_none());
    assert!(CompressionAlgorithm::from_i32(-1).is_none());
    assert!(CompressionAlgorithm::from_i32(100).is_none());
    assert!(CanonicalForm::from_i32(-1).is_none());
    assert!(CanonicalForm::from_i32(100).is_none());
}

#[test]
fn test_default() {
    assert_eq!(ContractionAlgorithm::default(), ContractionAlgorithm::Naive);
    assert_eq!(CompressionAlgorithm::default(), CompressionAlgorithm::SVD);
    assert_eq!(CanonicalForm::default(), CanonicalForm::Unitary);
}

#[test]
fn test_contraction_algorithm_name() {
    assert_eq!(ContractionAlgorithm::Naive.name(), "naive");
    assert_eq!(ContractionAlgorithm::ZipUp.name(), "zipup");
    assert_eq!(ContractionAlgorithm::Fit.name(), "fit");
    assert_eq!(ContractionAlgorithm::Src.name(), "src");
}

#[test]
fn test_canonical_form_name() {
    assert_eq!(CanonicalForm::Unitary.name(), "unitary");
    assert_eq!(CanonicalForm::LU.name(), "lu");
    assert_eq!(CanonicalForm::CI.name(), "ci");
}

#[test]
fn test_compression_algorithm_name() {
    assert_eq!(CompressionAlgorithm::SVD.name(), "svd");
    assert_eq!(CompressionAlgorithm::LU.name(), "lu");
    assert_eq!(CompressionAlgorithm::CI.name(), "ci");
    assert_eq!(CompressionAlgorithm::Variational.name(), "variational");
}
