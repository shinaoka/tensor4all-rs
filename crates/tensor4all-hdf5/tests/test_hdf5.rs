use approx::assert_abs_diff_eq;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::File;
use num_complex::Complex64;
use std::str::FromStr;
use tensor4all_core::index::{DynId, DynIndex, Index, TagSet};
use tensor4all_core::IdxTensor;
use tensor4all_hdf5::{
    append_itensor, append_mps, append_treetn, load_itensor, load_mps, load_treetn, save_itensor,
    save_mps, save_treetn,
};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain};

fn temp_path(name: &str) -> String {
    let dir = std::env::temp_dir();
    dir.join(format!("tensor4all_hdf5_test_{}.h5", name))
        .to_string_lossy()
        .to_string()
}

struct TempHdf5Path(String);

impl TempHdf5Path {
    fn new(name: &str) -> Self {
        let path = temp_path(name);
        std::fs::remove_file(&path).ok();
        Self(path)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl Drop for TempHdf5Path {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

fn itensor_error(name: &str, mutate: impl FnOnce(&hdf5_metno::Group)) -> String {
    let path = temp_path(name);
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    mutate(&file.group("tensor").unwrap());
    drop(file);

    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    std::fs::remove_file(path).ok();
    error
}

fn mps_error(name: &str, mutate: impl FnOnce(&hdf5_metno::Group)) -> String {
    let path = temp_path(name);
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    mutate(&file.group("mps").unwrap());
    drop(file);

    let error = load_mps(&path, "mps").unwrap_err().to_string();
    std::fs::remove_file(path).ok();
    error
}

/// Create a simple 2x3 f64 tensor with known data.
fn make_test_tensor_f64() -> IdxTensor {
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Site,n=1").unwrap()).set_plev(1);
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Link,l=1").unwrap()).set_plev(2);
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    IdxTensor::from_dense(vec![i1, i2], data).unwrap()
}

/// Create a simple 2x3 complex tensor with known data.
fn make_test_tensor_c64() -> IdxTensor {
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Site,n=1").unwrap()).set_plev(1);
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Link,l=1").unwrap()).set_plev(2);
    let data: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.1),
        Complex64::new(2.0, 0.2),
        Complex64::new(3.0, 0.3),
        Complex64::new(4.0, 0.4),
        Complex64::new(5.0, 0.5),
        Complex64::new(6.0, 0.6),
    ];
    IdxTensor::from_dense(vec![i1, i2], data).unwrap()
}

#[test]
fn test_itensor_f64_roundtrip() {
    let path = temp_path("itensor_f64");
    let tensor = make_test_tensor_f64();

    save_itensor(&path, "tensor", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor").unwrap();

    // Check dimensions
    assert_eq!(tensor.dims(), loaded.dims());

    // Check index properties
    let orig_indices = tensor.indices();
    let loaded_indices = loaded.indices();
    assert_eq!(orig_indices.len(), loaded_indices.len());
    for (orig, loaded) in orig_indices.iter().zip(loaded_indices.iter()) {
        assert_eq!(orig.id.0, loaded.id.0);
        assert_eq!(orig.dim, loaded.dim);
        assert_eq!(orig.plev, loaded.plev);
        assert_eq!(orig.tags, loaded.tags);
    }

    // Check data
    let orig_data = tensor.to_vec::<f64>().unwrap();
    let loaded_data = loaded.to_vec::<f64>().unwrap();
    for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_f64_storage_dataset_uses_column_major_linearization() {
    let path = temp_path("itensor_f64_storage_column_major");
    let tensor = make_test_tensor_f64();

    save_itensor(&path, "tensor", &tensor).unwrap();

    let file = File::open(&path).unwrap();
    let storage_group = file.group("tensor").unwrap().group("storage").unwrap();
    let stored: Vec<f64> = storage_group
        .dataset("data")
        .unwrap()
        .as_reader()
        .read_1d()
        .unwrap()
        .to_vec();

    assert_eq!(stored, tensor.to_vec::<f64>().unwrap());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_c64_roundtrip() {
    let path = temp_path("itensor_c64");
    let tensor = make_test_tensor_c64();

    save_itensor(&path, "tensor", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor").unwrap();

    // Check dimensions
    assert_eq!(tensor.dims(), loaded.dims());

    // Check data
    let orig_data = tensor.to_vec::<Complex64>().unwrap();
    let loaded_data = loaded.to_vec::<Complex64>().unwrap();
    for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-15);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_append_itensor_keeps_multiple_named_objects() {
    let path = temp_path("append_itensor_multiple");
    std::fs::remove_file(&path).ok();
    let first = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0]).unwrap();
    let second = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0, 4.0]).unwrap();

    append_itensor(&path, "first", &first).unwrap();
    append_itensor(&path, "second", &second).unwrap();

    assert_eq!(
        load_itensor(&path, "first")
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        load_itensor(&path, "second")
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![3.0, 4.0]
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_c64_storage_dataset_uses_column_major_linearization() {
    let path = temp_path("itensor_c64_storage_column_major");
    let tensor = make_test_tensor_c64();

    save_itensor(&path, "tensor", &tensor).unwrap();

    let file = File::open(&path).unwrap();
    let storage_group = file.group("tensor").unwrap().group("storage").unwrap();
    let stored: Vec<Complex64> = storage_group
        .dataset("data")
        .unwrap()
        .as_reader()
        .read_1d()
        .unwrap()
        .to_vec();

    assert_eq!(stored, tensor.to_vec::<Complex64>().unwrap());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_itensor_3d_roundtrip() {
    let path = temp_path("itensor_3d");
    let i1 = Index::new_dyn_with_tags(2, TagSet::from_str("Link,l=0").unwrap());
    let i2 = Index::new_dyn_with_tags(3, TagSet::from_str("Site,n=1").unwrap());
    let i3 = Index::new_dyn_with_tags(4, TagSet::from_str("Link,l=1").unwrap());
    let n = 2 * 3 * 4;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i1, i2, i3], data.clone()).unwrap();

    save_itensor(&path, "tensor3d", &tensor).unwrap();
    let loaded = load_itensor(&path, "tensor3d").unwrap();

    assert_eq!(tensor.dims(), loaded.dims());
    let loaded_data = loaded.to_vec::<f64>().unwrap();
    for (a, b) in data.iter().zip(loaded_data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn negative_index_dimension_is_rejected() {
    let path = temp_path("negative_index_dim");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("tensor/inds/index_1")
        .unwrap()
        .dataset("dim")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);

    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    assert!(error.contains("dim"));
    assert!(error.contains("-1"));
    std::fs::remove_file(path).ok();
}

#[test]
fn negative_index_set_length_is_rejected() {
    let path = temp_path("negative_index_set_length");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("tensor/inds")
        .unwrap()
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);

    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    assert!(error.contains("length"));
    assert!(error.contains("-1"));
    std::fs::remove_file(path).ok();
}

#[test]
fn index_set_length_exceeding_child_groups_is_rejected() {
    let path = temp_path("index_set_length_exceeds_groups");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("tensor/inds")
        .unwrap()
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&3_i64)
        .unwrap();
    drop(file);

    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    assert!(error.contains("length"));
    assert!(error.contains("3"));
    assert!(error.contains("child groups"));
    std::fs::remove_file(path).ok();
}

#[test]
fn index_set_excess_child_group_is_rejected() {
    let error = itensor_error("index_set_excess_child_group", |tensor| {
        tensor
            .group("inds")
            .unwrap()
            .create_group("index_3")
            .unwrap();
    });
    assert!(error.contains("index_3"));
    assert!(error.contains("declared range"), "{error}");
}

#[test]
fn index_set_misleading_prefix_child_is_rejected() {
    let error = itensor_error("index_set_misleading_prefix", |tensor| {
        tensor
            .group("inds")
            .unwrap()
            .create_group("index_stale")
            .unwrap();
    });
    assert!(error.contains("index_stale"));
}

#[test]
fn index_set_expected_dataset_is_rejected() {
    let error = itensor_error("index_set_expected_dataset", |tensor| {
        let inds = tensor.group("inds").unwrap();
        inds.unlink("index_1").unwrap();
        inds.new_dataset::<i64>()
            .shape(())
            .create("index_1")
            .unwrap();
    });
    assert!(error.contains("index_1"));
}

#[test]
fn index_set_noncanonical_child_names_are_rejected() {
    for (case, replacement) in [("leading_zero", "index_01"), ("ordinal_zero", "index_0")] {
        let name = format!("index_set_noncanonical_{case}");
        let error = itensor_error(&name, |tensor| {
            tensor
                .group("inds")
                .unwrap()
                .relink("index_1", replacement)
                .unwrap();
        });
        assert!(error.contains(replacement), "{case}: {error}");
    }
}

#[test]
fn index_set_expected_soft_link_is_rejected() {
    let path = temp_path("index_set_expected_soft_link");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let tensor = file.group("tensor").unwrap();
    let inds = tensor.group("inds").unwrap();
    tensor.link_hard("inds/index_1", "target_index").unwrap();
    inds.unlink("index_1").unwrap();
    inds.link_soft("/tensor/target_index", "index_1").unwrap();
    drop(file);

    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    assert!(error.contains("index_1"), "{error}");
    std::fs::remove_file(path).ok();
}

#[test]
fn index_set_zero_length_with_children_is_rejected() {
    let error = itensor_error("index_set_zero_length_with_children", |tensor| {
        tensor
            .group("inds")
            .unwrap()
            .dataset("length")
            .unwrap()
            .as_writer()
            .write_scalar(&0_i64)
            .unwrap();
    });
    assert!(error.contains("index_1") || error.contains("length"));
}

#[test]
fn index_set_zero_length_with_many_unrelated_members_is_rejected_without_panic() {
    let path = TempHdf5Path::new("index_set_zero_length_many_unrelated");
    save_itensor(path.as_str(), "tensor", &make_test_tensor_f64()).unwrap();
    {
        let file = File::open_rw(path.as_str()).unwrap();
        let inds = file.group("tensor/inds").unwrap();
        inds.dataset("length")
            .unwrap()
            .as_writer()
            .write_scalar(&0_i64)
            .unwrap();
        for n in 0..64 {
            let name = format!("unrelated_{n}");
            inds.new_dataset::<u8>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .as_writer()
                .write_scalar(&0_u8)
                .unwrap();
        }
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_itensor(path.as_str(), "tensor")
    }));
    assert!(result.is_ok(), "reader panicked: {:?}", result.err());
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("length"), "{error}");
}

/// Create a simple 3-site MPS for testing.
fn make_test_mps() -> TensorTrain {
    // Site 0: (1, d0=2, chi01=3) → indices: [link_left_dummy, site0, link01]
    // Site 1: (chi01=3, d1=2, chi12=4) → indices: [link01, site1, link12]
    // Site 2: (chi12=4, d2=2, 1) → indices: [link12, site2, link_right_dummy]

    let site_tags: Vec<TagSet> = (0..3)
        .map(|n| TagSet::from_str(&format!("Site,n={}", n)).unwrap())
        .collect();

    // Link indices (shared between adjacent sites)
    let link01_id = 100u64;
    let link12_id = 200u64;

    let link_tags_01 = TagSet::from_str("Link,l=1").unwrap();
    let link_tags_12 = TagSet::from_str("Link,l=2").unwrap();

    // Dummy bond indices for boundary (dim=1)
    let left_dummy = DynIndex::new_dyn(1);
    let right_dummy = DynIndex::new_dyn(1);

    // Site indices
    let site0 = Index::new_dyn_with_tags(2, site_tags[0].clone());
    let site1 = Index::new_dyn_with_tags(2, site_tags[1].clone());
    let site2 = Index::new_dyn_with_tags(2, site_tags[2].clone());

    // Link indices (same id for shared bond)
    let link01_left = Index::new_with_tags(DynId(link01_id), 3, link_tags_01.clone());
    let link01_right = link01_left.clone();
    let link12_left = Index::new_with_tags(DynId(link12_id), 4, link_tags_12.clone());
    let link12_right = link12_left.clone();

    // Tensor 0: shape (1, 2, 3) = 6 elements
    let data0: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
    let t0 = IdxTensor::from_dense(vec![left_dummy, site0, link01_left], data0).unwrap();

    // Tensor 1: shape (3, 2, 4) = 24 elements
    let data1: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
    let t1 = IdxTensor::from_dense(vec![link01_right, site1, link12_left], data1).unwrap();

    // Tensor 2: shape (4, 2, 1) = 8 elements
    let data2: Vec<f64> = (0..8).map(|i| i as f64 * 0.05).collect();
    let t2 = IdxTensor::from_dense(vec![link12_right, site2, right_dummy], data2).unwrap();

    TensorTrain::new(vec![t0, t1, t2]).unwrap()
}

#[test]
fn negative_mps_length_is_rejected() {
    let path = temp_path("negative_mps_length");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("mps")
        .unwrap()
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);

    let error = load_mps(&path, "mps").unwrap_err().to_string();
    assert!(error.contains("length"));
    assert!(error.contains("-1"));
    std::fs::remove_file(path).ok();
}

#[test]
fn oversized_mps_limits_are_rejected() {
    let path = temp_path("oversized_mps_limits");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.group("mps").unwrap();
    group
        .dataset("llim")
        .unwrap()
        .as_writer()
        .write_scalar(&i64::MAX)
        .unwrap();
    group
        .dataset("rlim")
        .unwrap()
        .as_writer()
        .write_scalar(&i64::MIN)
        .unwrap();
    drop(file);

    let error = load_mps(&path, "mps").unwrap_err().to_string();
    assert!(error.contains("llim") || error.contains("rlim"));
    assert!(error.contains("9223372036854775807") || error.contains("-9223372036854775808"));
    std::fs::remove_file(path).ok();
}

#[test]
fn mps_length_exceeding_child_groups_is_rejected() {
    let path = temp_path("mps_length_exceeds_groups");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.group("mps").unwrap();
    group
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&4_i64)
        .unwrap();
    group
        .dataset("rlim")
        .unwrap()
        .as_writer()
        .write_scalar(&5_i64)
        .unwrap();
    drop(file);

    let error = load_mps(&path, "mps").unwrap_err().to_string();
    assert!(error.contains("length"));
    assert!(error.contains("4"));
    assert!(
        error.contains("MPS[4]") || error.contains("expected child groups"),
        "{error}"
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn mps_i32_max_limits_are_rejected() {
    for field in ["llim", "rlim"] {
        let name = format!("mps_i32_max_{field}");
        let error = mps_error(&name, |group| {
            group
                .dataset(field)
                .unwrap()
                .as_writer()
                .write_scalar(&(i32::MAX as i64))
                .unwrap();
        });
        assert!(error.contains(field), "{field}: {error}");
        assert!(error.contains("2147483647"), "{field}: {error}");
        assert!(error.contains("length"), "{field}: {error}");
    }
}

#[test]
fn mps_in_range_invalid_orthogonality_limits_are_rejected() {
    for (name, llim, rlim) in [
        ("below_left_boundary", -2_i64, 4_i64),
        ("past_right_boundary", -1, 5),
        ("wrong_gap", 0, 3),
        ("center_past_end", 2, 4),
    ] {
        let name = format!("mps_invalid_ortho_{name}");
        let error = mps_error(&name, |group| {
            group
                .dataset("llim")
                .unwrap()
                .as_writer()
                .write_scalar(&llim)
                .unwrap();
            group
                .dataset("rlim")
                .unwrap()
                .as_writer()
                .write_scalar(&rlim)
                .unwrap();
        });
        assert!(error.contains("llim"), "{llim}, {rlim}: {error}");
        assert!(error.contains("rlim"), "{llim}, {rlim}: {error}");
        assert!(error.contains(&llim.to_string()), "{llim}, {rlim}: {error}");
        assert!(error.contains(&rlim.to_string()), "{llim}, {rlim}: {error}");
        assert!(error.contains("length"), "{llim}, {rlim}: {error}");
    }
}

#[test]
fn valid_mps_orthogonality_boundaries_roundtrip() {
    for (name, llim, rlim) in [("first", -1, 1), ("last", 1, 3)] {
        let path = temp_path(&format!("mps_valid_ortho_{name}"));
        let source = make_test_mps();
        let tensors = source.tensors().into_iter().cloned().collect();
        let mps =
            TensorTrain::with_ortho(tensors, llim, rlim, Some(CanonicalForm::Unitary)).unwrap();

        save_mps(&path, "mps", &mps).unwrap();
        let loaded = load_mps(&path, "mps").unwrap();
        assert_eq!(loaded.llim(), llim);
        assert_eq!(loaded.rlim(), rlim);
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn mps_excess_child_group_is_rejected() {
    let error = mps_error("mps_excess_child_group", |group| {
        group.create_group("MPS[4]").unwrap();
    });
    assert!(error.contains("MPS[4]"));
    assert!(error.contains("declared range"), "{error}");
}

#[test]
fn mps_misleading_prefix_child_is_rejected() {
    let error = mps_error("mps_misleading_prefix", |group| {
        group.create_group("MPS[stale]").unwrap();
    });
    assert!(error.contains("MPS[stale]"));
}

#[test]
fn mps_expected_dataset_is_rejected() {
    let error = mps_error("mps_expected_dataset", |group| {
        group.unlink("MPS[1]").unwrap();
        group
            .new_dataset::<i64>()
            .shape(())
            .create("MPS[1]")
            .unwrap();
    });
    assert!(error.contains("MPS[1]"));
}

#[test]
fn mps_noncanonical_child_names_are_rejected() {
    for (case, replacement) in [
        ("missing_closing_bracket", "MPS[1"),
        ("leading_zero", "MPS[01]"),
        ("ordinal_zero", "MPS[0]"),
    ] {
        let name = format!("mps_noncanonical_{case}");
        let error = mps_error(&name, |group| {
            group.relink("MPS[1]", replacement).unwrap();
        });
        assert!(error.contains(replacement), "{case}: {error}");
    }
}

#[test]
fn mps_expected_soft_link_is_rejected() {
    let path = temp_path("mps_expected_soft_link");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.group("mps").unwrap();
    file.link_hard("mps/MPS[1]", "mps_target").unwrap();
    group.unlink("MPS[1]").unwrap();
    group.link_soft("/mps_target", "MPS[1]").unwrap();
    drop(file);

    let error = load_mps(&path, "mps").unwrap_err().to_string();
    assert!(error.contains("MPS[1]"));
    std::fs::remove_file(path).ok();
}

#[test]
fn mps_zero_length_with_children_is_rejected() {
    let path = TempHdf5Path::new("mps_zero_length_with_children");
    let mps = TensorTrain::new(Vec::new()).unwrap();
    save_mps(path.as_str(), "mps", &mps).unwrap();
    {
        let file = File::open_rw(path.as_str()).unwrap();
        file.group("mps").unwrap().create_group("MPS[1]").unwrap();
    }

    let error = load_mps(path.as_str(), "mps").unwrap_err().to_string();
    assert!(error.contains("MPS[1]"));
}

#[test]
fn mps_zero_length_with_many_unrelated_members_is_rejected_without_panic() {
    let path = TempHdf5Path::new("mps_zero_length_many_unrelated");
    save_mps(path.as_str(), "mps", &TensorTrain::new(Vec::new()).unwrap()).unwrap();
    {
        let file = File::open_rw(path.as_str()).unwrap();
        let group = file.group("mps").unwrap();
        for n in 0..64 {
            let name = format!("unrelated_{n}");
            group
                .new_dataset::<u8>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .as_writer()
                .write_scalar(&0_u8)
                .unwrap();
        }
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_mps(path.as_str(), "mps")
    }));
    assert!(result.is_ok(), "reader panicked: {:?}", result.err());
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("length"), "{error}");
}

#[test]
fn test_mps_roundtrip() {
    let path = temp_path("mps");
    let mps = make_test_mps();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    // Check length
    assert_eq!(mps.len(), loaded.len());

    // Check each tensor
    let orig_tensors = mps.tensors();
    let loaded_tensors = loaded.tensors();
    for (i, (orig, loaded_t)) in orig_tensors.iter().zip(loaded_tensors.iter()).enumerate() {
        assert_eq!(orig.dims(), loaded_t.dims(), "Dims mismatch at site {}", i);

        // Check index IDs are preserved
        let orig_inds = orig.indices();
        let loaded_inds = loaded_t.indices();
        for (oi, li) in orig_inds.iter().zip(loaded_inds.iter()) {
            assert_eq!(oi.id.0, li.id.0, "Index ID mismatch at site {}", i);
            assert_eq!(oi.dim, li.dim, "Index dim mismatch at site {}", i);
            assert_eq!(oi.plev, li.plev, "Index plev mismatch at site {}", i);
        }

        // Check data
        let orig_data = orig.to_vec::<f64>().unwrap();
        let loaded_data = loaded_t.to_vec::<f64>().unwrap();
        for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_mps_load_preserves_site_tensor_index_order() {
    let path = temp_path("mps_preserve_index_order");

    let site0 = Index::new_with_size(DynId(10), 2);
    let link = Index::new_with_size(DynId(11), 3);
    let site1 = Index::new_with_size(DynId(12), 2);
    let left = IdxTensor::from_dense(
        vec![link.clone(), site0.clone()],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    let right = IdxTensor::from_dense(
        vec![site1.clone(), link.clone()],
        vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    )
    .unwrap();
    let mps = TensorTrain::new(vec![left, right]).unwrap();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    assert_eq!(loaded.tensor(0).unwrap().indices(), &[link.clone(), site0]);
    assert_eq!(loaded.tensor(1).unwrap().indices(), &[site1, link]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_append_mps_keeps_multiple_named_objects() {
    let path = temp_path("append_mps_multiple");
    std::fs::remove_file(&path).ok();
    let first = TensorTrain::new(vec![IdxTensor::from_dense(
        vec![DynIndex::new_dyn(2)],
        vec![1.0, 2.0],
    )
    .unwrap()])
    .unwrap();
    let second = TensorTrain::new(vec![IdxTensor::from_dense(
        vec![DynIndex::new_dyn(2)],
        vec![3.0, 4.0],
    )
    .unwrap()])
    .unwrap();

    append_mps(&path, "first", &first).unwrap();
    append_mps(&path, "second", &second).unwrap();

    assert_eq!(
        load_mps(&path, "first").unwrap().site_indices()[0][0].size(),
        2
    );
    assert_eq!(
        load_mps(&path, "second").unwrap().tensors()[0]
            .to_vec::<f64>()
            .unwrap(),
        vec![3.0, 4.0]
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_mps_ortho_roundtrip() {
    let path = temp_path("mps_ortho");
    let mps = make_test_mps();

    // Check that llim/rlim survive roundtrip
    let llim = mps.llim();
    let rlim = mps.rlim();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    assert_eq!(loaded.llim(), llim);
    assert_eq!(loaded.rlim(), rlim);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_mps_canonical_form_roundtrip() {
    let path = temp_path("mps_canonical_form");
    let mut mps = make_test_mps();
    mps.orthogonalize_with(0, CanonicalForm::LU).unwrap();

    save_mps(&path, "mps", &mps).unwrap();
    let loaded = load_mps(&path, "mps").unwrap();

    assert_eq!(loaded.canonical_form(), Some(CanonicalForm::LU));
    assert_eq!(loaded.llim(), mps.llim());
    assert_eq!(loaded.rlim(), mps.rlim());

    std::fs::remove_file(&path).ok();
}

#[test]
fn mps_missing_canonical_form_ignores_many_unrelated_attributes() {
    let path = TempHdf5Path::new("mps_missing_canonical_form_many_attributes");
    save_mps(path.as_str(), "mps", &make_test_mps()).unwrap();
    {
        let file = File::open_rw(path.as_str()).unwrap();
        let group = file.group("mps").unwrap();
        for n in 0..64 {
            let name = format!("unrelated_{n}");
            group
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .as_writer()
                .write_scalar(&n)
                .unwrap();
        }
    }

    let loaded = load_mps(path.as_str(), "mps").unwrap();
    assert_eq!(loaded.canonical_form(), None);
}

#[test]
fn mps_invalid_canonical_form_value_is_rejected_with_context() {
    let error = mps_error("mps_invalid_canonical_form_value", |group| {
        let attr = group
            .new_attr::<i32>()
            .shape(())
            .create("canonical_form")
            .unwrap();
        attr.as_writer().write_scalar(&999_i32).unwrap();
    });
    assert!(error.contains("canonical_form"), "{error}");
    assert!(error.contains("999"), "{error}");
}

#[test]
fn mps_invalid_canonical_form_type_is_rejected_with_context() {
    let error = mps_error("mps_invalid_canonical_form_type", |group| {
        let attr = group
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("canonical_form")
            .unwrap();
        attr.as_writer()
            .write_scalar(&VarLenUnicode::from_str("LU").unwrap())
            .unwrap();
    });
    assert!(error.contains("canonical_form"), "{error}");
    assert!(error.contains("read"), "{error}");
}

#[test]
fn mps_attribute_lookup_does_not_enumerate_attribute_names() {
    for (path, source) in [
        ("src/mps.rs", include_str!("../src/mps.rs")),
        ("src/backend.rs", include_str!("../src/backend.rs")),
    ] {
        assert!(
            !source.contains("attr_names("),
            "{path} must use constant-space attribute lookup"
        );
    }
}

#[test]
fn test_roundtrip_preserves_same_id_distinct_metadata_indices() -> anyhow::Result<()> {
    use tensor4all_core::{DynId, IdxTensor, Index, TagSet};
    use tensor4all_hdf5::{load_itensor, save_itensor};

    let tags_a = TagSet::from_str("Site,A")?;
    let tags_b = TagSet::from_str("Site,B")?;
    let i = Index::new_with_tags(DynId(7), 2, tags_a);
    let j = Index::new_with_tags(DynId(7), 3, tags_b).set_plev(1);
    assert_ne!(i, j);

    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("same_id_metadata.h5");
    let path = path.to_str().unwrap();

    save_itensor(path, "tensor", &tensor)?;
    let loaded = load_itensor(path, "tensor")?;

    assert_eq!(loaded.indices(), tensor.indices());
    assert_eq!(loaded.to_vec::<f64>()?, tensor.to_vec::<f64>()?);
    Ok(())
}

#[test]
fn test_type_mismatch_error() {
    // Write an ITensor, then try to load it as MPS → should get a clear error
    let path = temp_path("type_mismatch");
    let tensor = make_test_tensor_f64();
    save_itensor(&path, "obj", &tensor).unwrap();

    let err = load_mps(&path, "obj").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Expected HDF5 type 'MPS'"),
        "Expected type mismatch error, got: {}",
        msg
    );

    std::fs::remove_file(&path).ok();
}

use tensor4all_treetn::TreeTN;

/// A 3-node tree (root connected to two leaves) with known data.
fn make_test_treetn() -> TreeTN<IdxTensor, String> {
    let s_left = DynIndex::new_dyn(2);
    let s_root = DynIndex::new_dyn(3);
    let s_right = DynIndex::new_dyn(2);
    let b_left = DynIndex::new_dyn(4);
    let b_right = DynIndex::new_dyn(5);

    let left = IdxTensor::from_dense(vec![s_left, b_left.clone()], vec![1.0; 8]).unwrap();
    let root = IdxTensor::from_dense(vec![b_left, s_root, b_right.clone()], vec![2.0; 60]).unwrap();
    let right = IdxTensor::from_dense(vec![b_right, s_right], vec![3.0; 10]).unwrap();

    TreeTN::from_tensors(
        vec![left, root, right],
        vec!["left".to_string(), "root".to_string(), "right".to_string()],
    )
    .unwrap()
}

fn treetn_node_tensors(tn: &TreeTN<IdxTensor, String>) -> Vec<IdxTensor> {
    tn.node_indices()
        .iter()
        .map(|idx| tn.tensor(*idx).unwrap().clone())
        .collect()
}

#[test]
fn test_treetn_roundtrip() {
    let path = TempHdf5Path::new("treetn_roundtrip");
    let tn = make_test_treetn();

    save_treetn(path.as_str(), "tn", &tn).unwrap();
    let loaded = load_treetn::<String>(path.as_str(), "tn").unwrap();

    // Structural equality: same node count, same edges, same names in order.
    assert_eq!(loaded.node_count(), tn.node_count());
    assert_eq!(loaded.edge_count(), tn.edge_count());
    assert_eq!(loaded.node_names(), tn.node_names());

    // Numerical equality: each node tensor is identical (index identity and
    // data), so the reconstructed topology matches the original exactly.
    let orig = treetn_node_tensors(&tn);
    let reloaded = treetn_node_tensors(&loaded);
    assert_eq!(orig.len(), reloaded.len());
    for (a, b) in orig.iter().zip(&reloaded) {
        assert_eq!(a.indices(), b.indices());
        assert!(a.distance(b).unwrap() < 1e-15);
    }
}

#[test]
fn test_treetn_append_multiple() {
    let path = TempHdf5Path::new("treetn_append");
    let tn = make_test_treetn();

    save_treetn(path.as_str(), "a", &tn).unwrap();
    append_treetn(path.as_str(), "b", &tn).unwrap();

    let a = load_treetn::<String>(path.as_str(), "a").unwrap();
    let b = load_treetn::<String>(path.as_str(), "b").unwrap();
    assert_eq!(a.node_count(), 3);
    assert_eq!(b.node_count(), 3);
    assert_eq!(a.node_names(), b.node_names());
}

#[test]
fn test_treetn_usize_node_names_roundtrip() {
    let path = TempHdf5Path::new("treetn_usize");
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(4);
    let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0; 8]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01, s1], vec![1.0; 8]).unwrap();
    let tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();

    save_treetn(path.as_str(), "tn", &tn).unwrap();
    let loaded = load_treetn::<usize>(path.as_str(), "tn").unwrap();
    assert_eq!(loaded.node_names(), vec![0, 1]);
    assert_eq!(loaded.edge_count(), 1);
}

#[test]
fn test_treetn_bad_node_name_parse_error() {
    // Save with String names, then request usize: parsing must fail loudly.
    let path = TempHdf5Path::new("treetn_bad_name");
    let tn = make_test_treetn();
    save_treetn(path.as_str(), "tn", &tn).unwrap();

    let err = load_treetn::<usize>(path.as_str(), "tn").unwrap_err();
    assert!(
        err.to_string().contains("Failed to parse TreeTN node name"),
        "Expected node-name parse error, got: {}",
        err
    );
}

#[test]
fn test_treetn_missing_child_group_error() {
    // Corrupt the file: remove a node group, then loading must fail with a
    // child-count mismatch rather than reading a partial tree.
    let path = temp_path("treetn_missing_child");
    save_treetn(&path, "tn", &make_test_treetn()).unwrap();

    {
        let file = File::open_rw(&path).unwrap();
        let group = file.group("tn").unwrap();
        group.unlink("node_3").unwrap();
    }

    let err = load_treetn::<String>(&path, "tn").unwrap_err();
    assert!(
        err.to_string().contains("declares 3 TreeTN children"),
        "Expected child-count mismatch error, got: {}",
        err
    );

    std::fs::remove_file(&path).ok();
}
