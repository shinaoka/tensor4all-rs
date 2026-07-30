//! Regression tests for `TreeTN::swap_site_indices` dropping singular values
//! that the caller never asked to drop (issue #564).
//!
//! Two independent paths used to fall back to the process-global
//! `DEFAULT_SVD_TRUNCATION_POLICY` (per-value rtol 1e-12):
//!
//! 1. `rtol: None` set no SVD policy on the swap factorization, so the global
//!    default applied instead of an exact decomposition.
//! 2. The canonical-center transport sweep never received `SwapOptions` at all,
//!    so even `rtol: Some(0.0)` could not make the operation exact.
//!
//! The singular value 1e-13 is chosen to sit well below the global default 1e-12
//! and well above f64 roundoff 1e-16, so any fallback to the global policy
//! truncates it and fails these tests.

use std::collections::HashMap;

use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_treetn::{SwapOptions, TreeTN};

/// Orthogonal 4x4 Hadamard/2, column-major flat.
fn hadamard4() -> Vec<f64> {
    let rows = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let mut out = vec![0.0; 16];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            out[r + 4 * c] = v / 2.0;
        }
    }
    out
}

/// 4-site chain "0"-"1"-"2"-"3" whose middle cut 01|23 has singular values
/// (1, 1e-13, 1e-13, 1e-13).
fn chain_with_tiny_singular_values() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let s = [1.0_f64, 1e-13, 1e-13, 1e-13];
    let u = hadamard4();

    // A = U * diag(s) * U^T, split as left = U, right = diag(s) * U^T.
    let mut m = vec![0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            m[row + 4 * col] = s[row] * u[col + 4 * row];
        }
    }

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(4);
    let b23 = DynIndex::new_dyn(2);

    let t0 =
        TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], u).unwrap();
    let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone(), b23.clone()], m).unwrap();
    let t3 =
        TensorDynLen::from_dense(vec![b23.clone(), s3.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut tn = TreeTN::<TensorDynLen, String>::new();
    tn.add_tensor("0".to_string(), t0).unwrap();
    tn.add_tensor("1".to_string(), t1).unwrap();
    tn.add_tensor("2".to_string(), t2).unwrap();
    tn.add_tensor("3".to_string(), t3).unwrap();
    let n0 = tn.node_index(&"0".to_string()).unwrap();
    let n1 = tn.node_index(&"1".to_string()).unwrap();
    let n2 = tn.node_index(&"2".to_string()).unwrap();
    let n3 = tn.node_index(&"3".to_string()).unwrap();
    tn.connect(n0, &b01, n1, &b01).unwrap();
    tn.connect(n1, &b12, n2, &b12).unwrap();
    tn.connect(n2, &b23, n3, &b23).unwrap();
    (tn, s0, s1, s2, s3)
}

/// Largest non-site bond dimension adjacent to `node`.
fn bond_dim_at(tn: &TreeTN<TensorDynLen, String>, node: &str, sites: &[&DynIndex]) -> usize {
    let n = tn.node_index(&node.to_string()).unwrap();
    tn.tensor(n)
        .unwrap()
        .external_indices()
        .iter()
        .filter(|idx| !sites.iter().any(|s| **s == **idx))
        .map(|idx| idx.dim())
        .max()
        .unwrap_or(0)
}

/// A swap step that needs a transport sweep must not truncate: the transport
/// used to run a truncating factorization that ignored `SwapOptions` entirely.
#[test]
fn swap_with_transport_must_not_truncate() {
    let (tn, s0, s1, s2, s3) = chain_with_tiny_singular_values();
    let before = tn.contract_to_tensor().unwrap();
    assert_eq!(
        bond_dim_at(&tn, "1", &[&s1]),
        4,
        "input must have full middle rank"
    );

    // Swap s2 <-> s3 at nodes "2"/"3". Root is "0" (min node name), so the
    // canonical center is transported 0 -> 1 -> 2 across the middle bond first.
    let mut target: HashMap<DynIndex, String> = HashMap::new();
    target.insert(s0, "0".to_string());
    target.insert(s1.clone(), "1".to_string());
    target.insert(s2, "3".to_string());
    target.insert(s3, "2".to_string());

    for (label, opts) in [
        ("rtol: None", SwapOptions::default()),
        (
            "rtol: Some(0.0)",
            SwapOptions {
                max_rank: None,
                rtol: Some(0.0),
            },
        ),
    ] {
        let mut tn = tn.clone();
        tn.swap_site_indices(&target, &opts).unwrap();
        let after = tn.contract_to_tensor().unwrap();
        let err = before.sub(&after).unwrap().maxabs();
        let dim = bond_dim_at(&tn, "1", &[&s1]);
        assert_eq!(dim, 4, "{label}: middle bond rank must be preserved");
        assert!(err < 1e-15, "{label}: swap must be exact, err={err:e}");
    }
}

/// A swap step with no transport at all must not truncate with `rtol: None`:
/// this isolates the "`None` used to mean global default, not exact" cause.
#[test]
fn swap_without_transport_must_not_truncate() {
    // 2-node network A--B with M = H2 * diag(1, 1e-13) * H2^T on the single bond.
    // Root is "A" (min node name), so the swap step has an empty transport path.
    let h = 1.0 / 2.0_f64.sqrt();
    let sig = [1.0_f64, 1e-13];
    let hm = [[h, h], [h, -h]];
    let mut m = vec![0.0; 4];
    for a in 0..2 {
        for b in 0..2 {
            m[a + 2 * b] = (0..2).map(|k| hm[a][k] * sig[k] * hm[b][k]).sum();
        }
    }

    let sa = DynIndex::new_dyn(2);
    let sb = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let ta = TensorDynLen::from_dense(vec![sa.clone(), bond.clone()], m).unwrap();
    let tb =
        TensorDynLen::from_dense(vec![bond.clone(), sb.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut tn = TreeTN::<TensorDynLen, String>::new();
    tn.add_tensor("A".to_string(), ta).unwrap();
    tn.add_tensor("B".to_string(), tb).unwrap();
    let na = tn.node_index(&"A".to_string()).unwrap();
    let nb = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(na, &bond, nb, &bond).unwrap();
    let before = tn.contract_to_tensor().unwrap();
    assert_eq!(bond_dim_at(&tn, "A", &[&sa]), 2);

    let mut target: HashMap<DynIndex, String> = HashMap::new();
    target.insert(sa, "B".to_string());
    target.insert(sb.clone(), "A".to_string());

    let mut tn = tn.clone();
    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();
    let after = tn.contract_to_tensor().unwrap();
    let err = before.sub(&after).unwrap().maxabs();
    let dim = bond_dim_at(&tn, "A", &[&sb]);
    assert_eq!(dim, 2, "rtol: None must not drop the 1e-13 singular value");
    assert!(err < 1e-15, "rtol: None must be exact, err={err:e}");
}
