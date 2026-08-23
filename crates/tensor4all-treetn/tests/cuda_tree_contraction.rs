#![cfg(feature = "tenferro-cuda")]

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CudaTreeTNError, TreeTN};

fn dense(indices: Vec<DynIndex>, seed: usize) -> IdxTensor {
    let len = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .unwrap();
    let values = (0..len)
        .map(|offset| ((seed + offset) % 17) as f64 / 17.0 + 0.25)
        .collect();
    IdxTensor::from_dense(indices, values).unwrap()
}

fn two_node_tree() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    TreeTN::from_tensors(
        vec![
            dense(vec![left_site, bond.clone()], 1),
            dense(vec![bond, right_site], 5),
        ],
        vec![0, 1],
    )
    .unwrap()
}

fn branched_tree() -> TreeTN<IdxTensor, usize> {
    let center_site = DynIndex::new_dyn(2);
    let leaf_sites = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    let bonds = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    TreeTN::from_tensors(
        vec![
            dense(
                vec![
                    center_site,
                    bonds[0].clone(),
                    bonds[1].clone(),
                    bonds[2].clone(),
                ],
                1,
            ),
            dense(vec![bonds[0].clone(), leaf_sites[0].clone()], 3),
            dense(vec![bonds[1].clone(), leaf_sites[1].clone()], 7),
            dense(vec![bonds[2].clone(), leaf_sites[2].clone()], 11),
        ],
        vec![0, 1, 2, 3],
    )
    .unwrap()
}

fn scalar_closed_tree() -> TreeTN<IdxTensor, usize> {
    let bonds = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    TreeTN::from_tensors(
        vec![
            dense(vec![bonds[0].clone(), bonds[1].clone()], 1),
            dense(vec![bonds[0].clone()], 5),
            dense(vec![bonds[1].clone()], 9),
        ],
        vec![0, 1, 2],
    )
    .unwrap()
}

fn single_node_tree() -> TreeTN<IdxTensor, usize> {
    let first_site = DynIndex::new_dyn(2);
    let second_site = DynIndex::new_dyn(3);
    let tree = TreeTN::from_tensors(
        vec![dense(vec![first_site.clone(), second_site.clone()], 13)],
        vec![0],
    )
    .unwrap();
    let canonical: Vec<_> = tree.site_space(&0).unwrap().iter().cloned().collect();
    if canonical != tree.tensor(tree.node_indices()[0]).unwrap().indices() {
        return tree;
    }
    TreeTN::from_tensors(vec![dense(vec![second_site, first_site], 13)], vec![0]).unwrap()
}

fn assert_cuda_parity(tree: TreeTN<IdxTensor, usize>, context: &CudaExecutionContext) {
    let cpu = tree.contract_to_tensor().unwrap();
    let source_indices: Vec<_> = tree
        .node_indices()
        .into_iter()
        .map(|node| tree.tensor(node).unwrap().indices().to_vec())
        .collect();

    let resident_tree = tree.upload_cuda(context).unwrap();
    for node in resident_tree.node_indices() {
        resident_tree
            .tensor(node)
            .unwrap()
            .validate_cuda_residency(context)
            .unwrap();
    }
    let restored_tree = resident_tree.download(context).unwrap();
    for (node, indices) in tree.node_indices().into_iter().zip(source_indices.iter()) {
        let restored = restored_tree.tensor(node).unwrap();
        assert_eq!(restored.indices(), indices);
        assert_eq!(
            restored.to_vec::<f64>().unwrap(),
            tree.tensor(node).unwrap().to_vec::<f64>().unwrap()
        );
    }

    let resident = resident_tree.contract_to_tensor_cuda(context).unwrap();
    resident.validate_cuda_residency(context).unwrap();
    assert!(resident.to_vec::<f64>().is_err());

    let downloaded = resident.download(context).unwrap();
    assert_eq!(downloaded.indices(), cpu.indices());
    let residual = downloaded.sub(&cpu).unwrap().maxabs().unwrap();
    assert!(residual <= 1.0e-10, "CUDA/CPU residual: {residual}");

    for (node, indices) in tree.node_indices().into_iter().zip(source_indices) {
        assert_eq!(tree.tensor(node).unwrap().indices(), indices);
        assert!(tree.tensor(node).unwrap().to_vec::<f64>().is_ok());
    }
}

#[test]
fn two_node_cuda_contraction_matches_cpu() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    assert_cuda_parity(two_node_tree(), &context);
}

#[test]
fn branched_cuda_contraction_matches_cpu() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    assert_cuda_parity(branched_tree(), &context);
}

#[test]
fn scalar_closed_cuda_contraction_matches_cpu() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let tree = scalar_closed_tree();
    assert_cuda_parity(tree, &context);
}

#[test]
fn single_node_cuda_contraction_preserves_index_order_and_matches_cpu() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let tree = single_node_tree();
    let node = tree.node_indices()[0];
    let expected_indices = tree.tensor(node).unwrap().indices().to_vec();
    let cpu = tree.contract_to_tensor().unwrap();
    assert_eq!(cpu.indices(), expected_indices);

    let resident_tree = tree.upload_cuda(&context).unwrap();
    let resident = resident_tree.contract_to_tensor_cuda(&context).unwrap();
    assert_eq!(resident.indices(), expected_indices);

    let downloaded = resident.download(&context).unwrap();
    assert_eq!(downloaded.indices(), expected_indices);
    assert_eq!(
        downloaded.to_vec::<f64>().unwrap(),
        cpu.to_vec::<f64>().unwrap()
    );
}

#[test]
fn empty_cuda_tree_contraction_returns_typed_error() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let tree = TreeTN::<IdxTensor, usize>::from_tensors(Vec::new(), Vec::new()).unwrap();

    assert!(matches!(
        tree.contract_to_tensor_cuda(&context),
        Err(CudaTreeTNError::EmptyNetwork)
    ));
}

#[test]
fn mixed_host_and_cuda_nodes_are_rejected_before_contraction() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let host = two_node_tree();
    let nodes = host.node_indices();
    let mixed = TreeTN::from_tensors(
        vec![
            host.tensor(nodes[0])
                .unwrap()
                .upload_cuda(&context)
                .unwrap(),
            host.tensor(nodes[1]).unwrap().clone(),
        ],
        vec![0, 1],
    )
    .unwrap();

    assert!(matches!(
        mixed.contract_to_tensor_cuda(&context),
        Err(CudaTreeTNError::Residency { .. })
    ));
}

#[test]
fn mixed_cuda_dtypes_are_rejected_before_contraction() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let left_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let mixed = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond, right_site], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap()
    .upload_cuda(&context)
    .unwrap();

    assert!(matches!(
        mixed.contract_to_tensor_cuda(&context),
        Err(CudaTreeTNError::MixedDtype { .. })
    ));
}

#[test]
fn foreign_cuda_context_is_rejected_with_a_typed_error() {
    let first = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let second = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let resident = two_node_tree().upload_cuda(&first).unwrap();

    assert!(matches!(
        resident.contract_to_tensor_cuda(&second),
        Err(CudaTreeTNError::Residency { .. })
    ));
}

#[test]
fn source_contract_requires_pairwise_edge_walk() {
    let source = include_str!("../src/cuda.rs");
    let method = source
        .split_once("pub fn contract_to_tensor_cuda")
        .map(|(_, method)| method)
        .expect("CUDA contraction method must exist");

    assert!(method.contains("contract_pair"));
    assert!(!method.contains("contract_to_tensor("));
    assert!(!method.contains("T::contract"));
}
