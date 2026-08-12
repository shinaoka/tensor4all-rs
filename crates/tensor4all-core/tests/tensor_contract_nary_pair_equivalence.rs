use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{
    factorize, outer_product, svd, Canonical, DynIndex, FactorizeOptions, IdxTensor,
    TensorContractionLike,
};

fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>, dims: &[usize]) -> IdxTensor {
    let expected_len: usize = dims.iter().product();
    assert_eq!(data.len(), expected_len);
    IdxTensor::from_dense(indices, data).unwrap()
}

fn col_major_multi_index(mut offset: usize, dims: &[usize]) -> Vec<usize> {
    dims.iter()
        .map(|&dim| {
            let index = offset % dim;
            offset /= dim;
            index
        })
        .collect()
}

fn col_major_offset(indices: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1;
    let mut offset = 0;
    for (&index, &dim) in indices.iter().zip(dims.iter()) {
        offset += index * stride;
        stride *= dim;
    }
    offset
}

fn permute_col_major(data: &[f64], dims: &[usize], perm: &[usize]) -> Vec<f64> {
    let permuted_dims: Vec<usize> = perm.iter().map(|&axis| dims[axis]).collect();
    let mut permuted = vec![0.0; data.len()];

    for (src_offset, value) in data.iter().enumerate() {
        let src_index = col_major_multi_index(src_offset, dims);
        let dst_index: Vec<usize> = perm.iter().map(|&axis| src_index[axis]).collect();
        let dst_offset = col_major_offset(&dst_index, &permuted_dims);
        permuted[dst_offset] = *value;
    }

    permuted
}

#[test]
fn test_contract_nary_pair_matches_binary_contract() {
    let l01 = Index::new_dyn(3);
    let s1 = Index::new_dyn(2);
    let l12 = Index::new_dyn(3);
    let s2 = Index::new_dyn(2);

    // t1[l01, s1, l12]
    let t1 = make_tensor(
        vec![l01.clone(), s1.clone(), l12.clone()],
        (1..=18).map(|x| x as f64).collect(),
        &[3, 2, 3],
    );
    // t2[l12, s2]
    let t2 = make_tensor(
        vec![l12.clone(), s2.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );

    let binary = t1.contract_pair(&t2).unwrap();
    let multi = <IdxTensor as TensorContractionLike>::contract(&[&t1, &t2]).expect("contract");

    assert!(
        multi.isapprox(&binary, 1e-12, 0.0).unwrap(),
        "nary contract and binary contract differ: maxabs diff = {}",
        multi.sub(&binary).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_contract_nary_three_matches_sequential_binary_contract() {
    let i = Index::new_dyn(2);
    let a = Index::new_dyn(3);
    let b = Index::new_dyn(2);
    let c = Index::new_dyn(3);
    let k = Index::new_dyn(2);

    let t0 = make_tensor(
        vec![i.clone(), a.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[2, 3],
    );
    let t1 = make_tensor(
        vec![a.clone(), b.clone(), c.clone()],
        (1..=18).map(|x| x as f64).collect(),
        &[3, 2, 3],
    );
    let t2 = make_tensor(
        vec![c.clone(), k.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );

    let sequential = t0.contract_pair(&t1).unwrap().contract_pair(&t2).unwrap();
    let multi = <IdxTensor as TensorContractionLike>::contract(&[&t0, &t1, &t2]).expect("contract");

    assert!(
        multi.isapprox(&sequential, 1e-12, 0.0).unwrap(),
        "3-tensor nary contract and sequential contract differ: maxabs diff = {}",
        multi.sub(&sequential).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_contract_nary_pair_matches_binary_contract_for_zero_masked_inputs() {
    let s0 = Index::new_dyn(2);
    let l01 = Index::new_dyn(3);
    let s1 = Index::new_dyn(2);

    let t0 = make_tensor(
        vec![s0, l01.clone()],
        vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let t1 = make_tensor(vec![l01, s1], (1..=6).map(|x| x as f64).collect(), &[3, 2]);

    let binary = t0.contract_pair(&t1).unwrap();
    let multi = <IdxTensor as TensorContractionLike>::contract(&[&t0, &t1]).expect("contract");

    assert!(
        multi.isapprox(&binary, 1e-12, 0.0).unwrap(),
        "zero-masked nary contract and binary contract differ: maxabs diff = {}",
        multi.sub(&binary).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_zipup_zero_masked_root_nary_matches_sequential_binary_contract() {
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);
    let s2 = Index::new_dyn(2);
    let l01 = Index::new_dyn(3);
    let l12 = Index::new_dyn(3);

    let a0 = make_tensor(
        vec![s0.clone(), l01.clone()],
        vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let a1 = make_tensor(
        vec![l01.clone(), s1.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );
    let b0 = make_tensor(
        vec![s1.clone(), l12.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[2, 3],
    );
    let b1 = make_tensor(
        vec![l12.clone(), s2.clone()],
        vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0],
        &[3, 2],
    );

    let leaf = outer_product(&a0, &b0).expect("leaf outer product");
    let permuted_leaf = leaf
        .permute_indices(&[s0.clone(), s1.clone(), l01.clone(), l12.clone()])
        .unwrap();
    let expected_permuted = IdxTensor::from_dense(
        vec![s0.clone(), s1.clone(), l01.clone(), l12.clone()],
        permute_col_major(&leaf.to_vec::<f64>().unwrap(), &leaf.dims(), &[0, 2, 1, 3]),
    )
    .unwrap();
    assert!(
        permuted_leaf.isapprox(&expected_permuted, 1e-12, 0.0).unwrap(),
        "native permute for leaf does not match tensor-level column-major expectation: maxabs diff = {}",
        permuted_leaf.sub(&expected_permuted).unwrap().maxabs().unwrap()
    );

    let (u, s, v) = svd::<f64>(&leaf, &[s0.clone(), s1.clone()]).expect("svd");
    let vh = v.conj().permute(&[2, 0, 1]).unwrap();
    let svh = s.contract_pair(&vh).unwrap();
    let svh = svh
        .replaceind(
            &s.indices[1].clone(),
            &v.indices[v.indices.len() - 1].clone(),
        )
        .unwrap();
    let svd_reconstructed = u.contract_pair(&svh).unwrap();
    assert!(
        svd_reconstructed.isapprox(&leaf, 1e-10, 0.0).unwrap(),
        "svd leaf does not reconstruct: maxabs diff = {}",
        svd_reconstructed.sub(&leaf).unwrap().maxabs().unwrap()
    );

    let factorized = factorize(
        &leaf,
        &[s0.clone(), s1.clone()],
        &FactorizeOptions::svd().with_canonical(Canonical::Left),
    )
    .expect("factorize");

    let reconstructed_leaf = factorized.left.contract_pair(&factorized.right).unwrap();
    assert!(
        reconstructed_leaf.isapprox(&leaf, 1e-10, 0.0).unwrap(),
        "factorized leaf does not reconstruct: maxabs diff = {}",
        reconstructed_leaf.sub(&leaf).unwrap().maxabs().unwrap()
    );

    let sequential = factorized
        .right
        .contract_pair(&a1)
        .unwrap()
        .contract_pair(&b1)
        .unwrap();
    let multi = <IdxTensor as TensorContractionLike>::contract(&[&factorized.right, &a1, &b1])
        .expect("root contract");

    assert!(
        multi.isapprox(&sequential, 1e-10, 0.0).unwrap(),
        "zipup root nary contract and sequential binary contract differ: maxabs diff = {}",
        multi.sub(&sequential).unwrap().maxabs().unwrap()
    );
}
