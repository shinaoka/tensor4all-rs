//! Regression test: swap_site_indices corrupts tensor values on certain MPS
//! structures.
//!
//! Root cause: tenferro's `Tensor::reshape()` incorrectly assigns column-major
//! strides when dimensions contain 1, causing element ordering corruption in
//! the QR/SVD native path during canonicalization.

use std::collections::HashMap;
use tensor4all_core::{
    common_inds, DynIndex, FactorizeOptions, IdxTensor, IndexLike, TensorContractionLike,
    TensorFactorizationLike, TensorIndex,
};
use tensor4all_treetn::{SwapOptions, TreeTN};

/// Build a 2-site MPS from dense data via QR factorization.
fn make_2site_mps(s1: &DynIndex, s2: &DynIndex, data: &[f64]) -> Vec<IdxTensor> {
    let dense = IdxTensor::from_dense(vec![s1.clone(), s2.clone()], data.to_vec()).unwrap();
    let fr = dense
        .factorize(std::slice::from_ref(s1), &FactorizeOptions::qr())
        .unwrap();
    vec![fr.left, fr.right]
}

/// Concatenate two MPS by adding a dim-1 bond between them.
fn concatenate(left: &[IdxTensor], right: &[IdxTensor]) -> Vec<IdxTensor> {
    let bond = DynIndex::new_dyn(1);
    let ones = IdxTensor::ones(std::slice::from_ref(&bond)).unwrap();
    let n_left = left.len();
    let mut tensors = Vec::with_capacity(n_left + right.len());
    for (i, t) in left.iter().enumerate() {
        if i == n_left - 1 {
            tensors.push(t.outer_product(&ones).unwrap());
        } else {
            tensors.push(t.clone());
        }
    }
    for (i, t) in right.iter().enumerate() {
        if i == 0 {
            tensors.push(ones.outer_product(t).unwrap());
        } else {
            tensors.push(t.clone());
        }
    }
    tensors
}

/// contract_to_tensor should be invariant under swap_site_indices.
///
/// Using non-trivial data that produces higher bond dimensions.
#[test]
fn test_swap_preserves_values_nontrivial_data() {
    let z1_1 = DynIndex::new_dyn_with_tag(2, "z1=1").unwrap();
    let z1_2 = DynIndex::new_dyn_with_tag(2, "z1=2").unwrap();
    let z2_1 = DynIndex::new_dyn_with_tag(2, "z2=1").unwrap();
    let z2_2 = DynIndex::new_dyn_with_tag(2, "z2=2").unwrap();

    let left = make_2site_mps(&z1_1, &z1_2, &[1.0, 2.0, 3.0, 4.0]);
    let right = make_2site_mps(&z2_1, &z2_2, &[5.0, 6.0, 7.0, 8.0]);
    let tensors = concatenate(&left, &right);
    let node_names: Vec<usize> = (0..tensors.len()).collect();

    let treetn_before: TreeTN<IdxTensor, usize> =
        TreeTN::from_tensors(tensors.clone(), node_names.clone()).unwrap();
    let dense_before = treetn_before.contract_to_tensor().unwrap();
    let norm_before = dense_before.norm().unwrap();

    // Swap sites 1 and 2: [z1=1, z1=2, z2=1, z2=2] → [z1=1, z2=1, z1=2, z2=2]
    let mut treetn: TreeTN<IdxTensor, usize> = TreeTN::from_tensors(tensors, node_names).unwrap();
    let mut target = HashMap::new();
    target.insert(z1_1.clone(), 0usize);
    target.insert(z2_1.clone(), 1usize);
    target.insert(z1_2.clone(), 2usize);
    target.insert(z2_2.clone(), 3usize);
    treetn
        .swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let dense_after = treetn.contract_to_tensor().unwrap();
    let neg = dense_after
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = dense_before.add(&neg).unwrap();
    let rel_err = diff.norm().unwrap() / norm_before;

    assert!(
        rel_err < 1e-10,
        "contract_to_tensor changed after swap: relative error = {rel_err:.3e}"
    );
}

/// Test: contract_to_tensor should be invariant under canonicalization
#[test]
fn test_canonicalize_preserves_contract_to_tensor() {
    let z1_1 = DynIndex::new_dyn_with_tag(2, "z1=1").unwrap();
    let z1_2 = DynIndex::new_dyn_with_tag(2, "z1=2").unwrap();
    let z2_1 = DynIndex::new_dyn_with_tag(2, "z2=1").unwrap();
    let z2_2 = DynIndex::new_dyn_with_tag(2, "z2=2").unwrap();

    let left = make_2site_mps(&z1_1, &z1_2, &[1.0, 2.0, 3.0, 4.0]);
    let right = make_2site_mps(&z2_1, &z2_2, &[5.0, 6.0, 7.0, 8.0]);
    let tensors = concatenate(&left, &right);
    let node_names: Vec<usize> = (0..tensors.len()).collect();

    let treetn_before: TreeTN<IdxTensor, usize> =
        TreeTN::from_tensors(tensors.clone(), node_names.clone()).unwrap();
    let dense_before = treetn_before.contract_to_tensor().unwrap();

    let mut treetn_after: TreeTN<IdxTensor, usize> =
        TreeTN::from_tensors(tensors, node_names).unwrap();
    use tensor4all_treetn::CanonicalizationOptions;
    treetn_after
        .canonicalize_mut(std::iter::once(0usize), CanonicalizationOptions::default())
        .unwrap();
    let dense_after = treetn_after.contract_to_tensor().unwrap();

    let neg = dense_after
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = dense_before.add(&neg).unwrap();
    let rel_err = diff.norm().unwrap() / dense_before.norm().unwrap();
    assert!(
        rel_err < 1e-10,
        "contract_to_tensor changed after canonicalization: relative error = {rel_err:.3e}"
    );
}

/// Contract two tensors and refactorize, check roundtrip is exact.
#[test]
fn test_contract_factorize_roundtrip() {
    let z1_1 = DynIndex::new_dyn_with_tag(2, "z1=1").unwrap();
    let z1_2 = DynIndex::new_dyn_with_tag(2, "z1=2").unwrap();
    let z2_1 = DynIndex::new_dyn_with_tag(2, "z2=1").unwrap();
    let z2_2 = DynIndex::new_dyn_with_tag(2, "z2=2").unwrap();

    let left = make_2site_mps(&z1_1, &z1_2, &[1.0, 2.0, 3.0, 4.0]);
    let right = make_2site_mps(&z2_1, &z2_2, &[5.0, 6.0, 7.0, 8.0]);
    let tensors = concatenate(&left, &right);
    let node_names: Vec<usize> = (0..tensors.len()).collect();

    let mut treetn: TreeTN<IdxTensor, usize> = TreeTN::from_tensors(tensors, node_names).unwrap();
    use tensor4all_treetn::CanonicalizationOptions;
    treetn
        .canonicalize_mut(std::iter::once(0usize), CanonicalizationOptions::default())
        .unwrap();

    let n1 = treetn.node_index(&1).unwrap();
    let n2 = treetn.node_index(&2).unwrap();
    let t1 = treetn.tensor(n1).unwrap().clone();
    let t2 = treetn.tensor(n2).unwrap().clone();

    let contracted = t1.contract_pair(&t2).unwrap();

    // Find left_inds for factorization
    let t1_ids: std::collections::HashSet<_> = t1
        .external_indices()
        .iter()
        .map(|i: &DynIndex| *i.id())
        .collect();
    let t2_ids: std::collections::HashSet<_> = t2
        .external_indices()
        .iter()
        .map(|i: &DynIndex| *i.id())
        .collect();
    let mut left_inds: Vec<DynIndex> = Vec::new();
    for idx in contracted.external_indices() {
        let is_t1_only = t1_ids.contains(idx.id()) && !t2_ids.contains(idx.id());
        let is_z2_1 = idx.id() == z2_1.id();
        if is_t1_only && idx.id() != z1_2.id() {
            left_inds.push(idx.clone());
        }
        if is_z2_1 {
            left_inds.push(idx.clone());
        }
    }

    let factorize_options =
        FactorizeOptions::svd().with_canonical(tensor4all_core::Canonical::Left);
    let result = contracted
        .factorize(&left_inds, &factorize_options)
        .unwrap();

    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    let recon_aligned = reconstructed
        .permute_indices(&contracted.external_indices())
        .unwrap();
    let neg = recon_aligned
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = contracted.add(&neg).unwrap();
    let rel_err = diff.norm().unwrap() / contracted.norm().unwrap();
    assert!(rel_err < 1e-10, "roundtrip error = {rel_err:.3e}");
}

/// Manually simulate sweep_edge steps to verify each step preserves values.
#[test]
fn test_manual_sweep_edge_steps() {
    let z1_1 = DynIndex::new_dyn_with_tag(2, "z1=1").unwrap();
    let z1_2 = DynIndex::new_dyn_with_tag(2, "z1=2").unwrap();
    let z2_1 = DynIndex::new_dyn_with_tag(2, "z2=1").unwrap();
    let z2_2 = DynIndex::new_dyn_with_tag(2, "z2=2").unwrap();

    let left = make_2site_mps(&z1_1, &z1_2, &[1.0, 2.0, 3.0, 4.0]);
    let right = make_2site_mps(&z2_1, &z2_2, &[5.0, 6.0, 7.0, 8.0]);
    let mut t = concatenate(&left, &right);

    // Reference: full contraction of original tensors
    let full_ref = t[3]
        .contract_pair(&t[2])
        .unwrap()
        .contract_pair(&t[1])
        .unwrap()
        .contract_pair(&t[0])
        .unwrap();

    // Helper: contract all 4 tensors and compare to reference
    let check_full = |_label: &str, tensors: &[IdxTensor]| -> f64 {
        let full = tensors[3]
            .contract_pair(&tensors[2])
            .unwrap()
            .contract_pair(&tensors[1])
            .unwrap()
            .contract_pair(&tensors[0])
            .unwrap();
        let aligned = full.permute_indices(&full_ref.external_indices()).unwrap();
        let neg = full_ref
            .scale(tensor4all_core::AnyScalar::new_real(-1.0))
            .unwrap();
        let diff = aligned.add(&neg).unwrap();
        diff.norm().unwrap() / full_ref.norm().unwrap()
    };

    // Helper: find shared index between two tensors
    let find_bond = |a: &IdxTensor, b: &IdxTensor| -> DynIndex {
        let ids: std::collections::HashSet<_> = a
            .external_indices()
            .iter()
            .map(|i: &DynIndex| *i.id())
            .collect();
        b.external_indices()
            .iter()
            .find(|i: &&DynIndex| ids.contains(i.id()))
            .unwrap()
            .clone()
    };

    // Helper: sweep_edge from src to dst
    let sweep_edge = |t: &mut [IdxTensor], src: usize, dst: usize| {
        let bond = find_bond(&t[dst], &t[src]);
        let left_inds: Vec<DynIndex> = t[src]
            .external_indices()
            .iter()
            .filter(|i: &&DynIndex| i.id() != bond.id())
            .cloned()
            .collect();
        let fr = t[src]
            .factorize(&left_inds, &FactorizeOptions::qr())
            .unwrap();
        let new_dst = t[dst].contract_pair(&fr.right).unwrap();
        t[src] = fr.left;
        t[dst] = new_dst;
    };

    assert!(check_full("Before", &t) < 1e-10);

    // Canonicalization towards node 0: sweeps 3→2, 2→1, 1→0
    sweep_edge(&mut t, 3, 2);
    let err = check_full("After 3→2", &t);
    assert!(err < 1e-10, "Corruption at sweep 3→2: {err:.3e}");

    sweep_edge(&mut t, 2, 1);
    let err = check_full("After 2→1", &t);
    assert!(err < 1e-10, "Corruption at sweep 2→1: {err:.3e}");

    sweep_edge(&mut t, 1, 0);
    let err = check_full("After 1→0", &t);
    assert!(err < 1e-10, "Corruption at sweep 1→0: {err:.3e}");
}

/// Minimal reproduction: QR factorize roundtrip of a tensor with dim-1 axis.
///
/// Tensor `[d=1, d=2, d=2]` factorized with `left_inds=[d=2, d=2]` produces
/// a tall matrix `[4, 1]`. The column vector is ambiguously contiguous in both
/// row-major and column-major, causing tenferro's `reshape` to assign wrong
/// strides when converting back to multi-dimensional.
#[test]
fn test_qr_roundtrip_tall_matrix() {
    let i1 = DynIndex::new_dyn_with_tag(1, "bond_small").unwrap();
    let i2 = DynIndex::new_dyn_with_tag(2, "site_a").unwrap();
    let i3 = DynIndex::new_dyn_with_tag(2, "site_b").unwrap();

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = IdxTensor::from_dense(vec![i1.clone(), i2.clone(), i3.clone()], data).unwrap();

    let fr = t
        .factorize(&[i2.clone(), i3.clone()], &FactorizeOptions::qr())
        .unwrap();

    let recon = fr.left.contract_pair(&fr.right).unwrap();
    let recon_aligned = recon.permute_indices(&t.external_indices()).unwrap();
    let neg = t.scale(tensor4all_core::AnyScalar::new_real(-1.0)).unwrap();
    let diff = recon_aligned.add(&neg).unwrap();
    let rel_err = diff.norm().unwrap() / t.norm().unwrap();
    assert!(rel_err < 1e-10, "QR roundtrip error = {rel_err:.3e}");
}

/// Same test but starting from a 4-site MPS built via sequential QR.
#[test]
fn test_swap_preserves_values_plain_mps() {
    let s0 = DynIndex::new_dyn_with_tag(2, "z1=1").unwrap();
    let s1 = DynIndex::new_dyn_with_tag(2, "z1=2").unwrap();
    let s2 = DynIndex::new_dyn_with_tag(2, "z2=1").unwrap();
    let s3 = DynIndex::new_dyn_with_tag(2, "z2=2").unwrap();

    let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let indices = vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()];
    let dense = IdxTensor::from_dense(indices.clone(), data).unwrap();

    let mut tensors: Vec<IdxTensor> = Vec::new();
    let mut remaining = dense;
    for k in 0..3 {
        let mut left_inds = Vec::new();
        if k > 0 {
            let common = common_inds(remaining.indices(), tensors[k - 1].indices());
            left_inds.extend(common);
        }
        left_inds.push(indices[k].clone());
        let fr = remaining
            .factorize(&left_inds, &FactorizeOptions::qr())
            .unwrap();
        tensors.push(fr.left);
        remaining = fr.right;
    }
    tensors.push(remaining);

    let node_names: Vec<usize> = (0..tensors.len()).collect();

    let treetn_before: TreeTN<IdxTensor, usize> =
        TreeTN::from_tensors(tensors.clone(), node_names.clone()).unwrap();
    let dense_before = treetn_before.contract_to_tensor().unwrap();

    let mut treetn: TreeTN<IdxTensor, usize> = TreeTN::from_tensors(tensors, node_names).unwrap();
    let mut target = HashMap::new();
    target.insert(s0.clone(), 0usize);
    target.insert(s2.clone(), 1usize);
    target.insert(s1.clone(), 2usize);
    target.insert(s3.clone(), 3usize);
    treetn
        .swap_site_indices(&target, &SwapOptions::default())
        .unwrap();
    let dense_after = treetn.contract_to_tensor().unwrap();

    let neg = dense_after
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = dense_before.add(&neg).unwrap();
    let rel_err = diff.norm().unwrap() / dense_before.norm().unwrap();

    assert!(
        rel_err < 1e-10,
        "contract_to_tensor changed after swap (plain): relative error = {rel_err:.3e}"
    );
}
