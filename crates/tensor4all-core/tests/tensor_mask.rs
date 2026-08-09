use tensor4all_core::{DynIndex, TensorDynLen};

#[test]
fn mask_index_preserves_values_indices_and_reverse_mode_graph() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let source = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let masked = source.mask_index(&i, 1).unwrap();
    assert_eq!(masked.indices(), &[i.clone(), j.clone()]);
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
    assert!(masked.tracks_grad());

    let loss = masked.sum().unwrap();
    loss.backward().unwrap();
    assert_eq!(
        source.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn mask_index_preserves_compact_diagonal_storage() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let source = TensorDynLen::from_diag(vec![i.clone(), j], vec![3.0_f64, 4.0]).unwrap();

    let masked = source.mask_index(&i, 1).unwrap();
    assert!(masked.is_diag());
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 0.0, 0.0, 4.0]);
}
