//! Regression tests for linsolve where x and b behave like MPOs.
//!
//! The current square linsolve infrastructure primarily targets MPS-like states
//! (one site index per node). For MPO-like states, each node carries two site indices.
//! This test checks that the bra/ket convention precheck and ProjectedState `<ref|b>`
//! construction keep the local RHS indices aligned with the local init indices.
//!
//! We intentionally use the simplest possible setup:
//! - 1 node (no bonds)
//! - identity operator (A = I) with internal indices + index mappings
//! - x and b are "MPO-like" in the sense that they have 2 site indices per node:
//!   - one is the "contracted" index that A acts on (mapped via IndexMapping.true_index)
//!   - the other is an "external/batch" index that should remain open
//!
//! This does NOT attempt to validate full MPO×MPO operator multiplication; it only
//! checks index-structure consistency needed to extend linsolve to MPO-like spaces.

use std::collections::{HashMap, HashSet};

use tensor4all_core::{index::DynId, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{
    CanonicalizationOptions, IndexMapping, LinsolveOptions, LocalUpdateStep, LocalUpdater,
    SquareLinsolveUpdater, TreeTN,
};

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize) -> DynIndex {
    loop {
        let idx = DynIndex::new_dyn(dim);
        if used.insert(*idx.id()) {
            return idx;
        }
    }
}

/// Create a 1-node "MPO-like" state tensor with two site indices (external, contracted).
fn one_node_mpo_like_state(external: DynIndex, contracted: DynIndex) -> TreeTN<IdxTensor, String> {
    let mut tn = TreeTN::<IdxTensor, String>::new();
    let nelem = external.dim() * contracted.dim();
    let t = IdxTensor::from_dense(vec![external, contracted], vec![1.0; nelem]).unwrap();
    tn.add_tensor("site0".to_string(), t).unwrap();
    tn
}

/// Create a 1-node identity MPO with internal indices, plus index mappings to a true index.
#[allow(clippy::type_complexity)]
fn one_node_identity_operator_with_mappings(
    phys_dim: usize,
    true_contracted: DynIndex,
    used: &mut HashSet<DynId>,
) -> (
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    // Internal MPO indices (must be distinct IDs)
    let s_in_tmp = unique_dyn_index(used, phys_dim);
    let s_out_tmp = unique_dyn_index(used, phys_dim);

    // Identity matrix on (s_out_tmp, s_in_tmp)
    let mut data = vec![0.0_f64; phys_dim * phys_dim];
    for k in 0..phys_dim {
        data[k * phys_dim + k] = 1.0;
    }
    let t = IdxTensor::from_dense(vec![s_out_tmp.clone(), s_in_tmp.clone()], data).unwrap();

    let mut mpo = TreeTN::<IdxTensor, String>::new();
    mpo.add_tensor("site0".to_string(), t).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: true_contracted.clone(),
            internal_index: s_in_tmp,
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: true_contracted,
            internal_index: s_out_tmp,
        },
    );

    (mpo, input_mapping, output_mapping)
}

#[test]
fn test_linsolve_allows_two_site_indices_per_node_for_rhs_alignment() -> anyhow::Result<()> {
    let phys_dim = 2usize;
    let external_dim = 3usize;

    let mut used = HashSet::<DynId>::new();
    let contracted = unique_dyn_index(&mut used, phys_dim);
    let external = unique_dyn_index(&mut used, external_dim);

    // x and b share the same (external, contracted) index IDs.
    let rhs = one_node_mpo_like_state(external.clone(), contracted.clone());
    let init = rhs.clone();

    let (op, in_map, out_map) =
        one_node_identity_operator_with_mappings(phys_dim, contracted.clone(), &mut used);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(1)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(10)
        .with_gmres_restart_dim(10)
        .with_max_bond_dim(4)
        .with_coefficients(0.0, 1.0);

    let mut x = init.canonicalize(["site0".to_string()], CanonicalizationOptions::default())?;
    let mut updater = SquareLinsolveUpdater::with_index_mappings(op, in_map, out_map, rhs, options);

    let step = LocalUpdateStep {
        nodes: vec!["site0".to_string()],
        new_center: "site0".to_string(),
    };

    // Run a single local update. The main purpose is to ensure we do not fail
    // with index-count mismatch between init and RHS.
    updater.before_step(&step, &x)?;
    let subtree = x.extract_subtree(&step.nodes)?;
    let updated_subtree = updater.update(subtree, &step, &x)?;
    x.replace_subtree(&step.nodes, &updated_subtree)?;
    x.set_canonical_region([step.new_center.clone()])?;
    updater.after_step(&step, &x)?;

    Ok(())
}

/// Init and rhs with different index structures must fail precheck with a clear mismatch message.
#[test]
fn test_linsolve_precheck_fails_when_init_rhs_index_structure_mismatch() {
    let phys_dim = 2usize;
    let external_dim = 3usize;

    let mut used = HashSet::<DynId>::new();
    let contracted_rhs = unique_dyn_index(&mut used, phys_dim);
    let external_rhs = unique_dyn_index(&mut used, external_dim);

    let rhs = one_node_mpo_like_state(external_rhs.clone(), contracted_rhs.clone());

    let contracted_init = unique_dyn_index(&mut used, phys_dim);
    let external_init = unique_dyn_index(&mut used, external_dim);
    let init = one_node_mpo_like_state(external_init, contracted_init);

    let (op, in_map, out_map) =
        one_node_identity_operator_with_mappings(phys_dim, contracted_rhs.clone(), &mut used);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(1)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(10)
        .with_gmres_restart_dim(10)
        .with_max_bond_dim(4)
        .with_coefficients(0.0, 1.0);

    let x = init
        .canonicalize(["site0".to_string()], CanonicalizationOptions::default())
        .unwrap();
    let mut updater = SquareLinsolveUpdater::with_index_mappings(op, in_map, out_map, rhs, options);

    let step = LocalUpdateStep {
        nodes: vec!["site0".to_string()],
        new_center: "site0".to_string(),
    };

    let err = updater.before_step(&step, &x).unwrap_err();
    assert!(
        err.to_string().contains("precheck failed")
            || err.to_string().contains("index structure mismatch")
            || err.to_string().contains("Index structure mismatch"),
        "expected precheck mismatch error, got: {}",
        err
    );
}
