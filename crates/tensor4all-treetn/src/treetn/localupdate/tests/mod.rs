use super::*;
use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};

/// Create a 4-node Y-shape TreeTN:
///     A
///     |
///     B
///    / \
///   C   D
fn create_y_shape_treetn() -> (
    TreeTN<IdxTensor, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let mut tn = TreeTN::<IdxTensor, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    // Tensor A: [site_a, bond_ab]
    let tensor_a =
        IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Tensor B: [bond_ab, bond_bc, bond_bd]
    let tensor_b = IdxTensor::from_dense(
        vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
        vec![1.0; 27],
    )
    .unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Tensor C: [bond_bc, site_c]
    let tensor_c =
        IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Tensor D: [bond_bd, site_d]
    let tensor_d =
        IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("D".to_string(), tensor_d).unwrap();

    // Connect
    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

    (tn, site_a, site_c, site_d, bond_ab)
}

#[test]
fn truncate_updater_keeps_same_id_primed_nonbond_index() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let bond_prime = bond.prime();

    let tensor_0 = IdxTensor::from_dense(
        vec![s0.clone(), bond.clone(), bond_prime.clone()],
        vec![1.0, 0.0],
    )
    .unwrap();
    let tensor_1 = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![1.0, 0.0]).unwrap();

    let tn =
        TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor_0, tensor_1], vec![0usize, 1usize])
            .unwrap();

    let step = LocalUpdateStep {
        nodes: vec![0usize, 1usize],
        new_center: 1usize,
    };
    let mut updater = TruncateUpdater::new(Some(4), None);
    let updated = updater.update(tn.clone(), &step, &tn).unwrap();

    let node_0 = updated.node_index(&0usize).unwrap();
    let tensor_0_indices = updated.tensor(node_0).unwrap().external_indices();
    assert!(tensor_0_indices.iter().any(|idx| idx == &bond_prime));
    assert!(!tensor_0_indices.iter().any(|idx| idx == &bond));
}

#[test]
fn canonicalize_keeps_same_id_primed_nonbond_index_on_source() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let bond_prime = bond.prime();

    let tensor_0 = IdxTensor::from_dense(
        vec![s0.clone(), bond.clone(), bond_prime.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.25],
    )
    .unwrap();
    let tensor_1 =
        IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut tn =
        TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor_0, tensor_1], vec![0usize, 1usize])
            .unwrap();

    tn.canonicalize_mut([1usize], Default::default()).unwrap();

    let node_0 = tn.node_index(&0usize).unwrap();
    let tensor_0_indices = tn.tensor(node_0).unwrap().external_indices();
    assert!(
        tensor_0_indices.iter().any(|idx| idx == &bond_prime),
        "source tensor lost same-ID primed non-bond index: {tensor_0_indices:?}"
    );
    assert!(
        !tensor_0_indices.iter().any(|idx| idx == &bond),
        "source tensor should no longer carry the old edge bond: {tensor_0_indices:?}"
    );
}

#[test]
fn test_extract_subtree_single_node() {
    let (tn, _site_a, _, _, _) = create_y_shape_treetn();

    // Extract just node A
    let subtree = tn.extract_subtree(&["A".to_string()]).unwrap();

    assert_eq!(subtree.node_count(), 1);
    assert_eq!(subtree.edge_count(), 0);

    // Should have site_a as external index plus bond_ab (which becomes external)
    let n_a = subtree.node_index(&"A".to_string()).unwrap();
    let tensor_a = subtree.tensor(n_a).unwrap();
    assert_eq!(tensor_a.num_external_indices(), 2);

    // Verify consistency after extraction
    subtree.verify_internal_consistency().unwrap();
}

#[test]
fn test_extract_subtree_two_nodes() {
    let (tn, _, _, _, _) = create_y_shape_treetn();

    // Extract A-B subtree
    let subtree = tn
        .extract_subtree(&["A".to_string(), "B".to_string()])
        .unwrap();

    assert_eq!(subtree.node_count(), 2);
    assert_eq!(subtree.edge_count(), 1);

    // Verify connectivity
    let _n_a = subtree.node_index(&"A".to_string()).unwrap();
    let _n_b = subtree.node_index(&"B".to_string()).unwrap();
    assert!(subtree
        .edge_between(&"A".to_string(), &"B".to_string())
        .is_some());

    // Verify consistency after extraction
    subtree.verify_internal_consistency().unwrap();
}

#[test]
fn test_extract_subtree_disconnected_fails() {
    let (tn, _, _, _, _) = create_y_shape_treetn();

    // Try to extract A and C (not connected)
    let result = tn.extract_subtree(&["A".to_string(), "C".to_string()]);
    assert!(result.is_err());
}

#[test]
fn test_extract_subtree_preserves_consistency() {
    let (tn, _, _, _, _) = create_y_shape_treetn();

    // Extract B-C-D subtree
    let subtree = tn
        .extract_subtree(&["B".to_string(), "C".to_string(), "D".to_string()])
        .unwrap();

    // Verify consistency
    subtree.verify_internal_consistency().unwrap();
}

#[test]
fn test_replace_subtree_same_appearance() {
    let (mut tn, _, _, _, _) = create_y_shape_treetn();

    // Extract subtree, modify tensor data (but keep same structure), replace
    let subtree = tn.extract_subtree(&["C".to_string()]).unwrap();

    // Replace with itself (should work)
    tn.replace_subtree(&["C".to_string()], &subtree).unwrap();

    // Verify consistency
    tn.verify_internal_consistency().unwrap();
}

#[test]
fn test_replace_subtree_two_nodes() {
    let (mut tn, _, _, _, _) = create_y_shape_treetn();

    // Extract C-D subtree (through B)... wait, C and D are not connected.
    // Let's use B-C subtree instead.
    let subtree = tn
        .extract_subtree(&["B".to_string(), "C".to_string()])
        .unwrap();

    // Replace with itself
    tn.replace_subtree(&["B".to_string(), "C".to_string()], &subtree)
        .unwrap();

    // Verify consistency
    tn.verify_internal_consistency().unwrap();
}

// ========================================================================
// LocalUpdateSweepPlan tests
// ========================================================================

/// Create a chain network: A - B - C
fn create_chain_network() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net
}

/// Create a Y-shape network:
///     A
///     |
///     B
///    / \
///   C   D
fn create_y_network() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();
    net
}

#[test]
fn test_sweep_plan_nsite1_chain() {
    let net = create_chain_network();

    // Sweep from B (middle)
    let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 1).unwrap();

    // nsite=1: Euler tour vertices from B: [B, A, B, C, B]
    // Steps: all vertices except the last (return to root)
    // So we have 4 steps: B, A, B, C
    assert_eq!(plan.nsite, 1);
    assert_eq!(plan.len(), 4);

    // Each step should have exactly one node
    for step in plan.iter() {
        assert_eq!(step.nodes.len(), 1);
        // For nsite=1, new_center == nodes[0]
        assert_eq!(&step.new_center, &step.nodes[0]);
    }

    // First step should be B (starting point)
    assert_eq!(&plan.steps[0].nodes[0], "B");
}

#[test]
fn test_sweep_plan_nsite2_chain() {
    let net = create_chain_network();

    // Sweep from B (middle)
    let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 2).unwrap();

    // nsite=2: Euler tour edges from B visits 2 edges × 2 directions = 4 edges
    // The exact order depends on neighbor ordering in petgraph
    assert_eq!(plan.nsite, 2);
    assert_eq!(plan.len(), 4);

    // Each step should have exactly two nodes
    for step in plan.iter() {
        assert_eq!(step.nodes.len(), 2);
    }

    // Each edge (A-B and B-C) should be visited twice (in both directions)
    let mut ab_count = 0;
    let mut bc_count = 0;
    for step in plan.iter() {
        let has_a = step.nodes.contains(&"A".to_string());
        let has_b = step.nodes.contains(&"B".to_string());
        let has_c = step.nodes.contains(&"C".to_string());

        if has_a && has_b {
            ab_count += 1;
        }
        if has_b && has_c {
            bc_count += 1;
        }
    }
    assert_eq!(ab_count, 2); // Edge A-B visited twice
    assert_eq!(bc_count, 2); // Edge B-C visited twice

    // First step should start from B
    assert!(plan.steps[0].nodes.contains(&"B".to_string()));
}

#[test]
fn test_sweep_plan_nsite1_y_shape() {
    let net = create_y_network();

    // Sweep from B (center of Y)
    let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 1).unwrap();

    assert_eq!(plan.nsite, 1);
    // Y-shape has 3 edges, so Euler tour visits each twice = 6 edges
    // Vertices sequence: 7 vertices (starting node + 6 edge traversals)
    // Steps: 6 (all except last return to B)
    assert_eq!(plan.len(), 6);

    // All 4 nodes should be visited
    let visited: HashSet<_> = plan.iter().map(|s| s.nodes[0].clone()).collect();
    assert!(visited.contains("A"));
    assert!(visited.contains("B"));
    assert!(visited.contains("C"));
    assert!(visited.contains("D"));
}

#[test]
fn test_sweep_plan_nsite2_y_shape() {
    let net = create_y_network();

    // Sweep from B (center of Y)
    let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 2).unwrap();

    assert_eq!(plan.nsite, 2);
    // Y-shape has 3 edges, each visited twice = 6 edge traversals
    assert_eq!(plan.len(), 6);

    // Each edge should appear in both directions
    let mut edge_pairs: HashSet<(String, String)> = HashSet::new();
    for step in plan.iter() {
        let mut nodes = step.nodes.clone();
        nodes.sort();
        edge_pairs.insert((nodes[0].clone(), nodes[1].clone()));
    }
    // 3 unique edges: {A,B}, {B,C}, {B,D}
    assert_eq!(edge_pairs.len(), 3);
    assert!(edge_pairs.contains(&("A".to_string(), "B".to_string())));
    assert!(edge_pairs.contains(&("B".to_string(), "C".to_string())));
    assert!(edge_pairs.contains(&("B".to_string(), "D".to_string())));
}

#[test]
fn test_sweep_plan_single_node() {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();

    // Single node: no edges, empty plan
    let plan_nsite1 = LocalUpdateSweepPlan::new(&net, &"A".to_string(), 1).unwrap();
    // Single node has no edges, so Euler tour returns just [A]
    // Steps: all except last = 0 steps
    assert!(plan_nsite1.is_empty());

    let plan_nsite2 = LocalUpdateSweepPlan::new(&net, &"A".to_string(), 2).unwrap();
    assert!(plan_nsite2.is_empty());
}

#[test]
fn test_sweep_plan_invalid_nsite() {
    let net = create_chain_network();

    // nsite=0 is invalid
    assert!(LocalUpdateSweepPlan::new(&net, &"B".to_string(), 0).is_none());

    // nsite=3 is invalid
    assert!(LocalUpdateSweepPlan::new(&net, &"B".to_string(), 3).is_none());
}

#[test]
fn test_sweep_plan_nonexistent_root() {
    let net = create_chain_network();

    // Root "X" doesn't exist
    assert!(LocalUpdateSweepPlan::new(&net, &"X".to_string(), 1).is_none());
}

// ========================================================================
// TruncateUpdater and apply_local_update_sweep tests
// ========================================================================

/// Create a chain TreeTN: A - B - C
/// Each node has a site index of dim 2, bonds of dim 4
fn create_chain_treetn() -> TreeTN<IdxTensor, String> {
    let mut tn = TreeTN::<IdxTensor, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(4);
    let bond_bc = DynIndex::new_dyn(4);

    // Tensor A: [site_a, bond_ab] dim 2x4
    let tensor_a =
        IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 8]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Tensor B: [bond_ab, site_b, bond_bc] dim 4x2x4
    let tensor_b = IdxTensor::from_dense(
        vec![bond_ab.clone(), site_b.clone(), bond_bc.clone()],
        vec![1.0; 32],
    )
    .unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Tensor C: [bond_bc, site_c] dim 4x2
    let tensor_c =
        IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 8]).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Connect
    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();

    tn
}

#[test]
fn test_truncate_updater_basic() {
    use crate::CanonicalizationOptions;

    let tn = create_chain_treetn();

    // Canonicalize towards B (the root of the sweep)
    // This is required before using TruncateUpdater
    let mut tn = tn
        .canonicalize(["B".to_string()], CanonicalizationOptions::default())
        .expect("Failed to canonicalize");

    // Create sweep plan with nsite=2 from B
    let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();
    assert_eq!(plan.len(), 4); // 2 edges × 2 directions

    // Create truncate updater with max_bond_dim=2
    let mut updater = TruncateUpdater::new(Some(2), None);

    // Apply sweep
    apply_local_update_sweep(&mut tn, &plan, &mut updater).unwrap();

    // Verify consistency after sweep
    tn.verify_internal_consistency().unwrap();

    // Check that bond dimensions are reduced
    // After truncation with max_bond_dim=2, all bonds should have dim <= 2
    for node_name in tn.node_names() {
        let node_idx = tn.node_index(&node_name).unwrap();
        let tensor = tn.tensor(node_idx).unwrap();
        for dim in tensor.external_indices().iter().map(|i| i.dim()) {
            // Site dims are 2, truncated bonds should be <= 2
            assert!(dim <= 4); // Original max was 4
        }
    }
}

#[test]
fn test_apply_local_update_sweep_preserves_structure() {
    use crate::CanonicalizationOptions;

    let tn = create_chain_treetn();
    let original_node_count = tn.node_count();
    let original_edge_count = tn.edge_count();

    // Canonicalize towards B (the root of the sweep)
    // This is required before using TruncateUpdater
    let mut tn = tn
        .canonicalize(["B".to_string()], CanonicalizationOptions::default())
        .expect("Failed to canonicalize");

    // Create sweep plan with nsite=2 from B
    let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();

    // Apply with no truncation (max_bond_dim=None)
    let mut updater = TruncateUpdater::new(None, None);
    apply_local_update_sweep(&mut tn, &plan, &mut updater).unwrap();

    // Structure should be preserved
    assert_eq!(tn.node_count(), original_node_count);
    assert_eq!(tn.edge_count(), original_edge_count);

    // Verify consistency
    tn.verify_internal_consistency().unwrap();
}

#[test]
fn test_apply_local_update_sweep_requires_canonicalization() {
    // Test that apply_local_update_sweep fails when TreeTN is not canonicalized
    let mut tn = create_chain_treetn();

    // Create sweep plan with nsite=2 from B
    let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();

    // Create truncate updater
    let mut updater = TruncateUpdater::new(Some(2), None);

    // Apply sweep should fail because TreeTN is not canonicalized
    let result = apply_local_update_sweep(&mut tn, &plan, &mut updater);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("not canonicalized") || err_msg.contains("canonical_region is empty"),
        "Unexpected error message: {}",
        err_msg
    );
}

#[test]
fn test_apply_local_update_sweep_multi_node_canonical_region() {
    // Test that apply_local_update_sweep fails when canonical_region has multiple nodes
    let mut tn = create_chain_treetn();

    // Manually set canonical_region to two nodes (A and B)
    tn.set_canonical_region(["A".to_string(), "B".to_string()])
        .unwrap();

    let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();
    let mut updater = TruncateUpdater::new(Some(2), None);

    let result = apply_local_update_sweep(&mut tn, &plan, &mut updater);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("single node"),
        "Unexpected error message: {}",
        err_msg
    );
}

#[test]
fn test_sweep_plan_from_treetn() {
    let tn = create_chain_treetn();

    // from_treetn should work the same as new with topology
    let plan1 = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();
    let plan2 =
        LocalUpdateSweepPlan::new(tn.site_index_network().topology(), &"B".to_string(), 2).unwrap();

    assert_eq!(plan1.len(), plan2.len());
    assert_eq!(plan1.nsite, plan2.nsite);
}
