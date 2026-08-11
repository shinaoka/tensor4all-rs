//! Minimal linsolve repros with identity operator (A = I).
//!
//! This example is intentionally kept minimal and contains exactly two cases:
//! - **ok**: identity operator succeeds (single 1-site local update at canonical center)
//! - **fail**: identity operator still fails with
//!   `Dimension mismatch after alignment: self has dims [2, 2, 20, 8], other has [2, 20, 2, 8]`
//!   (2-site local update sweep, first few steps)
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example minimal_linsolve_identity_cases --release -- ok
//!   RUST_BACKTRACE=1 cargo run -p tensor4all-treetn --example minimal_linsolve_identity_cases --release -- fail

use std::collections::{HashMap, HashSet};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tensor4all_core::{index::DynId, DynIndex, IndexLike, TensorContractionLike, TensorDynLen};
use tensor4all_treetn::{
    apply_local_update_sweep, CanonicalizationOptions, IndexMapping, LinsolveOptions,
    LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater, SquareLinsolveUpdater, TreeTN,
};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize) -> DynIndex {
    loop {
        let idx = DynIndex::new_dyn(dim);
        if used.insert(*idx.id()) {
            return idx;
        }
    }
}

fn create_mps_chain_with_sites_all_ones(
    n: usize,
    bond_dim: usize,
    sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    anyhow::ensure!(sites.len() == n, "sites.len() must equal n");

    let mut mps = TreeTN::<TensorDynLen, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim))
        .collect();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        // Match the index ordering used in tests (`tests/linsolve.rs`):
        // - first:  [s0, b01]
        // - middle: [b_{i-1}, s_i, b_i]
        // - last:   [b_{n-2}, s_{n-1}]
        let indices = if n == 1 {
            vec![sites[i].clone()]
        } else if i == 0 {
            vec![sites[i].clone(), bonds[i].clone()]
        } else if i + 1 == n {
            vec![bonds[i - 1].clone(), sites[i].clone()]
        } else {
            vec![bonds[i - 1].clone(), sites[i].clone(), bonds[i].clone()]
        };

        let nelem: usize = indices.iter().map(|idx| idx.dim()).product();
        let t = TensorDynLen::from_dense(indices, vec![1.0_f64; nelem]).unwrap();
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        mps.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok(mps)
}

fn create_random_mps_chain_with_sites(
    rng: &mut ChaCha8Rng,
    n: usize,
    bond_dim: usize,
    sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    anyhow::ensure!(sites.len() == n, "sites.len() must equal n");

    let mut mps = TreeTN::<TensorDynLen, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim))
        .collect();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let indices = if n == 1 {
            vec![sites[i].clone()]
        } else if i == 0 {
            vec![sites[i].clone(), bonds[i].clone()]
        } else if i + 1 == n {
            vec![bonds[i - 1].clone(), sites[i].clone()]
        } else {
            vec![bonds[i - 1].clone(), sites[i].clone(), bonds[i].clone()]
        };
        let t = TensorDynLen::random::<f64, _>(rng, indices)?;
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        mps.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok(mps)
}

/// Create an N-site identity MPO with internal indices (bond dim = 1).
type IdentityMpoWithMappings = (
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
);

fn create_identity_mpo_chain_with_internal_indices(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<IdentityMpoWithMappings> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // MPO bonds: dim 1
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();

    // Internal indices (MPO-only)
    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);

        // Index ordering matches tests/linsolve.rs conventions:
        // - first:  [s_out_tmp, s_in_tmp, bond_right]
        // - middle: [bond_left, s_out_tmp, s_in_tmp, bond_right]
        // - last:   [bond_left, s_out_tmp, s_in_tmp]
        let indices = if n == 1 {
            vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()]
        } else if i == 0 {
            vec![s_out_tmp[i].clone(), s_in_tmp[i].clone(), bonds[i].clone()]
        } else if i + 1 == n {
            vec![
                bonds[i - 1].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ]
        } else {
            vec![
                bonds[i - 1].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
                bonds[i].clone(),
            ]
        };

        // Base identity matrix on (s_out_tmp, s_in_tmp).
        let mut data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            data[k * phys_dim + k] = 1.0;
        }
        let base = TensorDynLen::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], data)
            .unwrap();
        let t = if indices.len() == 2 {
            base
        } else {
            let bond_indices: Vec<_> = indices
                .iter()
                .filter(|idx| idx.dim() == 1)
                .cloned()
                .collect();
            let ones = TensorDynLen::from_dense(bond_indices, vec![1.0_f64; 1]).unwrap();
            TensorDynLen::outer_product(&base, &ones)?
        };

        let node = mpo.add_tensor(node_name.clone(), t).unwrap();
        nodes.push(node);

        input_mapping.insert(
            node_name.clone(),
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            node_name,
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    for i in 0..n.saturating_sub(1) {
        mpo.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok((mpo, input_mapping, output_mapping))
}

fn case_ok_identity_single_1site_step() -> anyhow::Result<()> {
    let n = 10usize;
    let phys_dim = 2usize;
    let state_bond_dim = 20usize;

    let _seed = 1234u64;
    let mut used_ids = HashSet::<DynId>::new();
    let sites: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    // Deterministic state to avoid incidental reorderings.
    let rhs = create_mps_chain_with_sites_all_ones(n, state_bond_dim, &sites, &mut used_ids)?;
    let init = rhs.clone();

    let (operator, input_mapping, output_mapping) =
        create_identity_mpo_chain_with_internal_indices(n, phys_dim, &sites, &mut used_ids)?;

    let center = make_node_name(n / 2);

    // Keep coefficients simple (avoid the a0*x + ... path inside the linop).
    let options = LinsolveOptions::default()
        .with_nfullsweeps(1)
        .with_max_bond_dim(state_bond_dim)
        .with_gmres_tol(1e-6)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(0.0, 1.0);

    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;
    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator,
        input_mapping,
        output_mapping,
        rhs,
        options,
    );

    // Single 1-site local update at the canonical center (no sweep).
    let step = LocalUpdateStep {
        nodes: vec![center.clone()],
        new_center: center,
    };
    updater.before_step(&step, &x)?;
    let subtree = x.extract_subtree(&step.nodes)?;
    let updated_subtree = updater.update(subtree, &step, &x)?;
    x.replace_subtree(&step.nodes, &updated_subtree)?;
    x.set_canonical_region([step.new_center.clone()])?;
    updater.after_step(&step, &x)?;

    println!("OK case succeeded (identity MPO, single 1-site update).");
    Ok(())
}

fn case_fail_identity_2site_sweep() -> anyhow::Result<()> {
    let n = 10usize;
    let phys_dim = 2usize;
    let state_bond_dim = 20usize;

    // 2-site sweep (first few steps only)
    let nsite = 2usize;
    let max_steps = 4usize;
    let root = make_node_name(n / 2);

    // Benchmark-like coefficients (this is where we reproduce the known failure)
    let options = LinsolveOptions::default()
        .with_nfullsweeps(1)
        .with_max_bond_dim(state_bond_dim)
        .with_gmres_tol(1e-6)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(1.0, 1.0);

    let seed = 1234u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut used_ids = HashSet::<DynId>::new();

    let sites: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    let rhs =
        create_random_mps_chain_with_sites(&mut rng, n, state_bond_dim, &sites, &mut used_ids)?;
    let init =
        create_random_mps_chain_with_sites(&mut rng, n, state_bond_dim, &sites, &mut used_ids)?;

    let (operator, input_mapping, output_mapping) =
        create_identity_mpo_chain_with_internal_indices(n, phys_dim, &sites, &mut used_ids)?;

    let mut x = init.canonicalize([root.clone()], CanonicalizationOptions::default())?;

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &root, nsite)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
    anyhow::ensure!(!plan.steps.is_empty(), "2-site plan unexpectedly empty");
    let steps: Vec<_> = plan.steps.iter().take(max_steps).cloned().collect();
    let subplan = LocalUpdateSweepPlan { steps, nsite };

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator,
        input_mapping,
        output_mapping,
        rhs,
        options,
    );

    // Expected to fail (repro):
    // Dimension mismatch after alignment: self has dims [2, 2, 20, 8], other has [2, 20, 2, 8]
    apply_local_update_sweep(&mut x, &subplan, &mut updater)?;

    println!("Unexpected: fail case succeeded.");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "ok".to_string());

    match mode.as_str() {
        "ok" => case_ok_identity_single_1site_step(),
        "fail" => case_fail_identity_2site_sweep(),
        other => {
            anyhow::bail!(
                "Unknown mode '{other}'. Use:\n  ... -- ok\n  ... -- fail\n(Default is ok)"
            )
        }
    }
}
