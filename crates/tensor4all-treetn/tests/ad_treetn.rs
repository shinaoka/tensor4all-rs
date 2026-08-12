//! Tests for reverse-mode automatic differentiation through TreeTN operations.

use tensor4all_core::{contract, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

fn make_three_site_mps_data() -> (Vec<Vec<DynIndex>>, Vec<Vec<f64>>) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);

    let index_sets = vec![
        vec![s0, bond01.clone()],
        vec![bond01, s1, bond12.clone()],
        vec![bond12, s2],
    ];

    let data = vec![
        vec![1.0, 0.5, 0.3, 2.0],
        vec![3.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 2.0],
        vec![1.5, 0.2, 0.4, 1.0],
    ];

    (index_sets, data)
}

#[test]
fn backward_ad_to_dense_propagates_gradients() {
    let (index_sets, data) = make_three_site_mps_data();

    let tensors: Vec<IdxTensor> = index_sets
        .iter()
        .zip(&data)
        .map(|(idx, d)| {
            IdxTensor::from_dense(idx.clone(), d.clone())
                .unwrap()
                .enable_grad()
                .unwrap()
        })
        .collect();

    let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
    let dense = ttn.to_dense().unwrap();

    // Contract dense with ones to get scalar = sum(dense)
    let ones = IdxTensor::from_dense(
        dense.indices().to_vec(),
        vec![1.0; dense.indices().iter().map(|i| i.dim()).product::<usize>()],
    )
    .unwrap();
    let scalar = contract(&[&dense, &ones]).unwrap();

    scalar.backward().unwrap();

    for (i, &ni) in ttn.node_indices().iter().enumerate() {
        let grad = ttn.tensor(ni).unwrap().grad().unwrap();
        assert!(grad.is_some(), "node {i} has no gradient after backward");
    }
}

#[test]
fn backward_ad_gradient_matches_finite_diff() {
    let (index_sets, data) = make_three_site_mps_data();
    let eps = 1e-6;

    let tensors: Vec<IdxTensor> = index_sets
        .iter()
        .zip(&data)
        .map(|(idx, d)| {
            IdxTensor::from_dense(idx.clone(), d.clone())
                .unwrap()
                .enable_grad()
                .unwrap()
        })
        .collect();

    let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
    let dense = ttn.to_dense().unwrap();

    let ones = IdxTensor::from_dense(
        dense.indices().to_vec(),
        vec![1.0; dense.indices().iter().map(|i| i.dim()).product::<usize>()],
    )
    .unwrap();
    let scalar = contract(&[&dense, &ones]).unwrap();

    scalar.backward().unwrap();

    // Verify gradient of first node vs finite difference
    let node_0 = ttn.node_index(&0).unwrap();
    let grad_t0 = ttn
        .tensor(node_0)
        .unwrap()
        .grad()
        .unwrap()
        .expect("t0 gradient missing");
    let grad_t0_vec = grad_t0.to_vec::<f64>().unwrap();

    for elem_idx in 0..data[0].len() {
        let make_perturbed_sum = |delta: f64| -> f64 {
            let tensors: Vec<IdxTensor> = index_sets
                .iter()
                .zip(&data)
                .enumerate()
                .map(|(i, (idx, d))| {
                    let mut d = d.clone();
                    if i == 0 {
                        d[elem_idx] += delta;
                    }
                    IdxTensor::from_dense(idx.clone(), d).unwrap()
                })
                .collect();
            let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
            ttn.to_dense().unwrap().sum().unwrap().real()
        };

        let fd = (make_perturbed_sum(eps) - make_perturbed_sum(-eps)) / (2.0 * eps);
        let ad = grad_t0_vec[elem_idx];
        let err = (ad - fd).abs();
        assert!(
            err < 1e-4,
            "backward grad[0][{elem_idx}] = {ad}, finite diff = {fd}, err = {err}"
        );
    }
}

#[test]
fn backward_accumulates_until_clear_grad_across_treetn_nodes() {
    let (index_sets, data) = make_three_site_mps_data();

    let tensors: Vec<IdxTensor> = index_sets
        .iter()
        .zip(&data)
        .map(|(idx, d)| {
            IdxTensor::from_dense(idx.clone(), d.clone())
                .unwrap()
                .enable_grad()
                .unwrap()
        })
        .collect();

    let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
    let dense = ttn.to_dense().unwrap();
    let ones = IdxTensor::from_dense(
        dense.indices().to_vec(),
        vec![1.0; dense.indices().iter().map(|i| i.dim()).product::<usize>()],
    )
    .unwrap();

    let scalar = contract(&[&dense, &ones]).unwrap();
    scalar.backward().unwrap();

    let first_grads: Vec<Vec<f64>> = ttn
        .node_indices()
        .iter()
        .map(|&ni| {
            ttn.tensor(ni)
                .unwrap()
                .grad()
                .unwrap()
                .expect("missing gradient after first backward")
                .to_vec::<f64>()
                .unwrap()
        })
        .collect();

    let scalar = contract(&[&dense, &ones]).unwrap();
    scalar.backward().unwrap();

    for (node_pos, (&ni, first_grad)) in ttn.node_indices().iter().zip(&first_grads).enumerate() {
        let grad = ttn
            .tensor(ni)
            .unwrap()
            .grad()
            .unwrap()
            .expect("missing gradient after second backward")
            .to_vec::<f64>()
            .unwrap();
        assert_eq!(grad.len(), first_grad.len());
        for (elem_pos, (&got, &expected_once)) in grad.iter().zip(first_grad).enumerate() {
            let expected = 2.0 * expected_once;
            let err = (got - expected).abs();
            assert!(
                err < 1e-10,
                "node {node_pos} grad[{elem_pos}] = {got}, expected {expected}, err = {err}"
            );
        }
    }

    for ni in ttn.node_indices() {
        ttn.tensor(ni).unwrap().clear_grad().unwrap();
        assert!(ttn.tensor(ni).unwrap().grad().unwrap().is_none());
    }
}
