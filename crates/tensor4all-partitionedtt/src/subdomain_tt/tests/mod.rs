use super::*;

fn projector(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Projector {
    Projector::from_pairs(pairs).unwrap()
}

fn subdomain(data: TensorTrain, projector: Projector) -> SubDomainTT {
    SubDomainTT::new(data, projector).unwrap()
}
use std::sync::Arc;
use tensor4all_core::index::Index;
use tensor4all_core::{IdxTensorError, TensorStorageError};
use tensor4all_itensorlike::TensorTrainError;
fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

fn make_tensor(indices: Vec<DynIndex>) -> IdxTensor {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
    IdxTensor::from_dense(indices, data).unwrap()
}

fn make_simple_tt() -> (TensorTrain, Vec<DynIndex>, Vec<DynIndex>) {
    // Create a 2-site tensor train
    let s0 = make_index(2); // site 0
    let l01 = make_index(3); // link 0-1
    let s1 = make_index(2); // site 1

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    (tt, vec![s0, s1], vec![l01])
}

#[test]
fn test_subdomain_tt_creation() {
    let (tt, site_inds, _) = make_simple_tt();
    let projector = projector([(site_inds[0].clone(), 1)]);

    let subdomain = subdomain(tt, projector);

    assert_eq!(subdomain.len(), 2);
    assert!(subdomain.is_projected_at(&site_inds[0]));
    assert!(!subdomain.is_projected_at(&site_inds[1]));
}

#[test]
fn test_subdomain_tt_from_tt() {
    let (tt, _, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);

    assert_eq!(subdomain.len(), 2);
    assert!(subdomain.projector().is_empty());
}

#[test]
fn test_subdomain_tt_project() {
    let (tt, site_inds, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);

    // Project to fix site 0 to value 1
    let projector = projector([(site_inds[0].clone(), 1)]);
    let projected = subdomain.project(&projector).unwrap();

    assert!(projected.is_some());
    let projected = projected.unwrap();
    assert!(projected.is_projected_at(&site_inds[0]));
    assert_eq!(projected.projector().get(&site_inds[0]), Some(1));
}

#[test]
fn test_subdomain_tt_project_value_one_numeric() {
    let (tt, site_inds, _) = make_simple_tt();
    let full = tt.to_dense().unwrap();
    let full_data = full.to_vec::<f64>().unwrap();

    let subdomain = SubDomainTT::from_tt(tt);
    let projector = projector([(site_inds[0].clone(), 1)]);
    let projected = subdomain.project(&projector).unwrap().unwrap();
    let projected_full = projected.data().to_dense().unwrap();
    let projected_data = projected_full.to_vec::<f64>().unwrap();

    assert_eq!(projected_data.len(), full_data.len());
    assert_eq!(projected_data[0], 0.0);
    assert_eq!(projected_data[1], full_data[1]);
    assert_eq!(projected_data[2], 0.0);
    assert_eq!(projected_data[3], full_data[3]);
}

#[test]
fn test_subdomain_tt_project_incompatible() {
    let (tt, site_inds, _) = make_simple_tt();
    let projector1 = projector([(site_inds[0].clone(), 0)]);
    let subdomain = subdomain(tt, projector1);

    // Try to project with incompatible projector (different value at same site)
    let projector2 = projector([(site_inds[0].clone(), 1)]);
    let projected = subdomain.project(&projector2).unwrap();

    assert!(projected.is_none());
}

#[test]
fn test_subdomain_tt_all_indices() {
    let (tt, site_inds, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);

    let all_indices = subdomain.all_indices();
    assert_eq!(all_indices.len(), 2);
    assert!(all_indices.contains(&site_inds[0]));
    assert!(all_indices.contains(&site_inds[1]));
}

#[test]
fn test_subdomain_tt_norm() {
    let (tt, _, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);

    let norm = subdomain.norm().unwrap();
    assert!(norm > 0.0);
}

#[test]
fn subdomain_new_rejects_absent_projector_index() {
    let (tt, site_inds, _) = make_simple_tt();
    let absent = make_index(5);
    let projector = projector([(site_inds[0].clone(), 1), (absent.clone(), 0)]);

    let error = SubDomainTT::new(tt, projector).unwrap_err();

    assert!(matches!(
        error,
        PartitionedTTError::ProjectorIndexNotFound { index } if index == absent
    ));
}

#[test]
fn subdomain_new_rejects_same_id_tag_or_prime_variants() {
    let (tt, site_inds, _) = make_simple_tt();
    let tagged = tensor4all_core::index::Index::new_with_tags(
        site_inds[0].id,
        site_inds[0].dim,
        tensor4all_core::TagSet::from_str("Site").unwrap(),
    );
    let primed = site_inds[0].prime();

    for absent in [tagged, primed] {
        let error = SubDomainTT::new(tt.clone(), projector([(absent.clone(), 0)])).unwrap_err();
        assert!(matches!(
            error,
            PartitionedTTError::ProjectorIndexNotFound { index } if index == absent
        ));
    }
}

#[test]
fn subdomain_new_validates_against_matched_tensor_train_dimension() {
    let (tt, site_inds, _) = make_simple_tt();
    let mismatched_dimension =
        tensor4all_core::index::Index::new_with_tags(site_inds[0].id, 5, site_inds[0].tags.clone());
    let projector = projector([(mismatched_dimension.clone(), 2)]);

    let error = SubDomainTT::new(tt, projector).unwrap_err();

    assert!(matches!(
        error,
        PartitionedTTError::ProjectorCoordinateOutOfBounds {
            index,
            value: 2,
            dim: 2,
        } if index == mismatched_dimension
    ));
}

#[test]
fn project_rejects_out_of_range_coordinate_against_tt_dimension() {
    let (tt, site_inds, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);
    let mismatched_dimension =
        tensor4all_core::index::Index::new_with_tags(site_inds[0].id, 5, site_inds[0].tags.clone());
    let projector = projector([(mismatched_dimension.clone(), 2)]);

    let error = subdomain.project(&projector).unwrap_err();
    assert!(matches!(
        error,
        PartitionedTTError::ProjectorCoordinateOutOfBounds {
            index,
            value: 2,
            dim: 2,
        } if index == mismatched_dimension
    ));
}

#[test]
fn project_rejects_index_absent_from_tensor_train() {
    let (tt, _, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);
    let absent = make_index(2);
    let projector = projector([(absent.clone(), 0)]);

    let error = subdomain.project(&projector).unwrap_err();
    assert!(matches!(
        error,
        PartitionedTTError::ProjectorIndexNotFound { index } if index == absent
    ));
}

#[test]
fn project_preserves_autodiff_metadata_and_backward_values() {
    let site = make_index(2);
    let source = IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let source_alias = source.clone();
    let subdomain = SubDomainTT::from_tt(TensorTrain::new(vec![source]).unwrap());

    let projected = subdomain.project(&projector([(site, 1)])).unwrap().unwrap();
    let projected_tensor = projected.data().tensor(0).unwrap();
    assert!(projected_tensor.tracks_grad());
    assert_eq!(projected_tensor.to_vec::<f64>().unwrap(), vec![0.0, 4.0]);

    projected_tensor.sum().unwrap().backward().unwrap();
    assert_eq!(
        source_alias
            .grad()
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![0.0, 1.0]
    );
}

#[test]
fn project_error_mapping_retains_typed_storage_source() {
    let storage_source = TensorStorageError::Materialization {
        source: Arc::new(std::io::Error::other("forced projection storage failure")),
    };
    let error = SubDomainTT::tensor_operation_error(IdxTensorError::Storage {
        source: storage_source,
    });

    match error {
        PartitionedTTError::TensorStorage { source } => {
            assert!(source
                .to_string()
                .contains("forced projection storage failure"));
            assert!(std::error::Error::source(&source).is_some());
        }
        other => panic!("expected typed storage error, got {other:?}"),
    }
}

#[test]
fn subdomain_contract_preserves_tensor_train_error_variant_and_source() {
    let (nonempty, _, _) = make_simple_tt();
    let empty = SubDomainTT::from_tt(TensorTrain::new(vec![]).unwrap());
    let nonempty = SubDomainTT::from_tt(nonempty);

    let error = empty
        .contract(&nonempty, &ContractOptions::default())
        .unwrap_err();
    match error {
        PartitionedTTError::TensorTrain { source } => {
            assert!(matches!(source, TensorTrainError::InvalidStructure { .. }));
            assert!(std::error::Error::source(&source).is_none());
        }
        other => panic!("expected preserved tensor-train error, got {other:?}"),
    }
}
