use std::collections::HashSet;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, LinearizationOrder};
use tensor4all_treetn::{SiteIndexNetwork, TreeTN};

#[test]
fn replace_site_index_with_indices_preserves_dense_tensor_values() {
    let fused = DynIndex::new_dyn(4);
    let tensor = IdxTensor::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);

    let unfused = tn
        .replace_site_index_with_indices(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    let dense = unfused.contract_to_tensor().unwrap();
    let expected =
        IdxTensor::from_dense(vec![left.clone(), right.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);

    let (site_indices, _) = unfused.all_site_indices().unwrap();
    let site_id_set: HashSet<_> = site_indices.iter().map(|idx| *idx.id()).collect();
    assert_eq!(site_indices.len(), 2);
    assert!(site_id_set.contains(left.id()));
    assert!(site_id_set.contains(right.id()));
}

#[test]
fn split_to_preserves_sparse_tensor_with_zero_leading_qr_row() {
    let out0 = DynIndex::new_dyn(2);
    let out1 = DynIndex::new_dyn(2);
    let in0 = DynIndex::new_dyn(2);
    let in1 = DynIndex::new_dyn(2);

    let mut data = vec![0.0; 16];
    data[6] = 1.0;
    let tensor = IdxTensor::from_dense(
        vec![out0.clone(), out1.clone(), in0.clone(), in1.clone()],
        data,
    )
    .unwrap();
    let tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap();

    let mut target = SiteIndexNetwork::<usize, DynIndex>::new();
    target
        .add_node(0, HashSet::from([out0.clone(), in0.clone()]))
        .unwrap();
    target
        .add_node(1, HashSet::from([out1.clone(), in1.clone()]))
        .unwrap();
    target.add_edge(&0, &1).unwrap();

    let split = tn.split_to(&target, &Default::default()).unwrap();
    let dense = split.contract_to_tensor().unwrap();
    let expected = tn.contract_to_tensor().unwrap();

    assert!(
        dense.distance(&expected).unwrap() < 1.0e-12,
        "exact split_to must not drop rows when QR produces a zero leading row"
    );
}
