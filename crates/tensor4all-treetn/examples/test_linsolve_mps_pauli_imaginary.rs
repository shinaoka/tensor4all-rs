//! Test: linsolve with an MPO operator A = i * Pauli-X acting on an MPS state x.
//!
//! Setup:
//! - Build an MPS `x_true` (random complex MPS).
//! - Build an MPO `A = i*X` (pure imaginary Pauli-X).
//! - Compute `b = A * x_true`.
//! - Solve `A * x = b` with `x_init = x_true`.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mps_pauli_imaginary --release

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tensor4all_core::{
    index::DynId, AnyScalar, DynIndex, IdxTensor, IndexLike, TensorContractionLike, TensorIndex,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

type TnWithInternalIndices = (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>);

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

/// Create an N-site Pauli-X MPO operator (operator-only; no external index).
/// Returns (mpo, s_in_tmp, s_out_tmp) where s_in_tmp and s_out_tmp are internal indices.
fn create_n_site_pauli_x_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TnWithInternalIndices> {
    anyhow::ensure!(n_sites >= 1, "Need at least 1 site");
    anyhow::ensure!(phys_dim == 2, "Pauli-X requires phys_dim=2");

    let mut mpo = TreeTN::<IdxTensor, String>::new();

    let s_in_tmp: Vec<DynIndex> = (0..n_sites)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();

    let bonds: Vec<DynIndex> = (0..n_sites.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();

    // X = [[0, 1], [1, 0]]
    let pauli_x = [0.0_f64, 1.0, 1.0, 0.0];

    let mut nodes = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let name = make_node_name(i);
        let data = pauli_x.to_vec();

        let tensor = if n_sites == 1 {
            IdxTensor::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], data).unwrap()
        } else if i == 0 {
            IdxTensor::from_dense(
                vec![s_out_tmp[i].clone(), s_in_tmp[i].clone(), bonds[i].clone()],
                data,
            )
            .unwrap()
        } else if i + 1 == n_sites {
            IdxTensor::from_dense(
                vec![
                    bonds[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            IdxTensor::from_dense(
                vec![
                    bonds[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bonds[i].clone(),
                ],
                data,
            )
            .unwrap()
        };

        let node = mpo.add_tensor(name, tensor).unwrap();
        nodes.push(node);
    }

    for (i, bond) in bonds.iter().enumerate() {
        mpo.connect(nodes[i], bond, nodes[i + 1], bond).unwrap();
    }

    Ok((mpo, s_in_tmp, s_out_tmp))
}

fn create_index_mappings(
    state_sites: &[DynIndex],
    s_in_tmp: &[DynIndex],
    s_out_tmp: &[DynIndex],
) -> (
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    let n = state_sites.len();
    assert_eq!(s_in_tmp.len(), n);
    assert_eq!(s_out_tmp.len(), n);

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for i in 0..n {
        let site = make_node_name(i);
        input_mapping.insert(
            site.clone(),
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            site,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    (input_mapping, output_mapping)
}

/// Scale a TreeTN by a scalar factor (scale only one tensor to avoid scalar^n scaling).
fn scale_treetn(
    treetn: &TreeTN<IdxTensor, String>,
    scalar: AnyScalar,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    let mut scaled = TreeTN::<IdxTensor, String>::new();
    let node_names: Vec<String> = treetn.node_names().into_iter().collect();
    let last_idx = node_names.len().saturating_sub(1);

    for (i, node_name) in node_names.iter().enumerate() {
        let node_idx = treetn.node_index(node_name).unwrap();
        let tensor = treetn.tensor(node_idx).unwrap();
        let scaled_tensor = if i == last_idx {
            tensor.scale(scalar.clone())?
        } else {
            tensor.clone()
        };
        scaled.add_tensor(node_name.clone(), scaled_tensor)?;
    }

    for (node_a, node_b) in treetn.site_index_network().edges() {
        let edge = treetn.edge_between(&node_a, &node_b).unwrap();
        let bond = treetn.bond_index(edge).unwrap();
        let node_a_idx = scaled.node_index(&node_a).unwrap();
        let node_b_idx = scaled.node_index(&node_b).unwrap();
        scaled.connect(node_a_idx, bond, node_b_idx, bond)?;
    }

    Ok(scaled)
}

fn create_random_mps_chain_with_sites_c64(
    rng: &mut ChaCha8Rng,
    n: usize,
    bond_dim: usize,
    sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    anyhow::ensure!(sites.len() == n, "sites.len() must equal n");

    let mut mps = TreeTN::<IdxTensor, String>::new();
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

        let t = IdxTensor::random::<Complex64, _>(rng, indices)?;
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        mps.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok(mps)
}

fn create_mps_chain_with_sites_all_ones_c64(
    n: usize,
    bond_dim: usize,
    sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    anyhow::ensure!(sites.len() == n, "sites.len() must equal n");

    let mut mps = TreeTN::<IdxTensor, String>::new();
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

        let nelem: usize = indices.iter().map(|idx| idx.dim()).product();
        let data = vec![Complex64::new(1.0, 0.0); nelem];
        let t = IdxTensor::from_dense(indices, data).unwrap();
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        mps.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok(mps)
}

fn compute_residual(
    op: &TreeTN<IdxTensor, String>,
    im: &HashMap<String, IndexMapping<DynIndex>>,
    om: &HashMap<String, IndexMapping<DynIndex>>,
    x: &TreeTN<IdxTensor, String>,
    rhs: &TreeTN<IdxTensor, String>,
) -> anyhow::Result<(f64, f64)> {
    let linop = LinearOperator::new(op.clone(), im.clone(), om.clone());
    let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;

    let ax_full = ax.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;

    // Align ax to b's external index ordering.
    let ref_order: Vec<DynIndex> = b_full.external_indices();
    let order_for = |tensor: &IdxTensor| -> anyhow::Result<Vec<DynIndex>> {
        let inds: Vec<DynIndex> = tensor.external_indices();
        let by_id: HashMap<DynId, DynIndex> = inds
            .into_iter()
            .map(|idx: DynIndex| (*idx.id(), idx))
            .collect();
        let mut out = Vec::with_capacity(ref_order.len());
        for r in ref_order.iter() {
            out.push(
                by_id
                    .get(r.id())
                    .ok_or_else(|| anyhow::anyhow!("missing index in alignment"))?
                    .clone(),
            );
        }
        Ok(out)
    };

    let ax_order = order_for(&ax_full)?;
    let ax_aligned = ax_full.permuteinds(&ax_order)?;

    let ax_vec = ax_aligned.to_vec::<Complex64>()?;
    let b_vec = b_full.to_vec::<Complex64>()?;
    anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");

    let mut diff2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for (ax_i, b_i) in ax_vec.iter().zip(b_vec.iter()) {
        let r = ax_i - *b_i;
        diff2 += r.norm_sqr();
        b2 += b_i.norm_sqr();
    }

    let abs_res = diff2.sqrt();
    let rel_res = if b2 > 0.0 { (diff2 / b2).sqrt() } else { 0.0 };

    Ok((abs_res, rel_res))
}

fn compute_state_error(
    x: &TreeTN<IdxTensor, String>,
    x_true: &TreeTN<IdxTensor, String>,
) -> anyhow::Result<(f64, f64)> {
    let x_full = x.contract_to_tensor()?;
    let x_true_full = x_true.contract_to_tensor()?;

    let ref_order: Vec<DynIndex> = x_true_full.external_indices();
    let order_for = |tensor: &IdxTensor| -> anyhow::Result<Vec<DynIndex>> {
        let inds: Vec<DynIndex> = tensor.external_indices();
        let by_id: HashMap<DynId, DynIndex> = inds
            .into_iter()
            .map(|idx: DynIndex| (*idx.id(), idx))
            .collect();
        let mut out = Vec::with_capacity(ref_order.len());
        for r in ref_order.iter() {
            out.push(
                by_id
                    .get(r.id())
                    .ok_or_else(|| anyhow::anyhow!("missing index in alignment"))?
                    .clone(),
            );
        }
        Ok(out)
    };

    let x_order = order_for(&x_full)?;
    let x_aligned = x_full.permuteinds(&x_order)?;

    let x_vec = x_aligned.to_vec::<Complex64>()?;
    let x_true_vec = x_true_full.to_vec::<Complex64>()?;
    anyhow::ensure!(x_vec.len() == x_true_vec.len(), "vector length mismatch");

    let mut diff2 = 0.0_f64;
    let mut x_true2 = 0.0_f64;
    for (x_i, x_true_i) in x_vec.iter().zip(x_true_vec.iter()) {
        let r = *x_i - *x_true_i;
        diff2 += r.norm_sqr();
        x_true2 += x_true_i.norm_sqr();
    }

    let abs_err = diff2.sqrt();
    let rel_err = if x_true2 > 0.0 {
        (diff2 / x_true2).sqrt()
    } else {
        0.0
    };

    Ok((abs_err, rel_err))
}

fn main() -> anyhow::Result<()> {
    let n = 2usize;
    let phys_dim = 2usize;
    let bond_dim = 6usize;

    println!("=== Test: linsolve with MPS and pure imaginary Pauli-X MPO (i*X) ===");
    println!("n = {n}, phys_dim = {phys_dim}, bond_dim = {bond_dim}");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let sites: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    let center = make_node_name(n / 2);

    // Operator: X, then scale by i.
    let (operator_x, s_in_tmp, s_out_tmp) =
        create_n_site_pauli_x_mpo_with_internal_indices(n, phys_dim, &mut used_ids)?;
    let (a_input_mapping, a_output_mapping) = create_index_mappings(&sites, &s_in_tmp, &s_out_tmp);

    let operator_i_x = scale_treetn(&operator_x, AnyScalar::from(Complex64::new(0.0, 1.0)))?;
    let linop_i_x = LinearOperator::new(
        operator_i_x.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
    );

    // x_true: random complex MPS (canonicalized before forming b to keep apply gauge-consistent).
    let mut rng = ChaCha8Rng::seed_from_u64(99999);
    let x_true =
        create_random_mps_chain_with_sites_c64(&mut rng, n, bond_dim, &sites, &mut used_ids)?;
    let x_true = x_true.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    // b = A * x_true
    let b = apply_linear_operator(&linop_i_x, &x_true, ApplyOptions::default())?;

    let options = LinsolveOptions::default()
        .with_nfullsweeps(10)
        .with_max_bond_dim(bond_dim)
        .with_gmres_tol(1e-10)
        .with_gmres_max_restarts(30)
        .with_gmres_restart_dim(30)
        .with_coefficients(0.0, 1.0);

    let n_sweeps = 10usize;

    println!("--- Solving (i*X)*x = b with init=x_true ---");
    let mut x = x_true.clone();

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator_i_x.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
        b.clone(),
        options.clone(),
    );

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs, init_rel) =
        compute_residual(&operator_i_x, &a_input_mapping, &a_output_mapping, &x, &b)?;
    let (init_abs_state, init_rel_state) = compute_state_error(&x, &x_true)?;

    println!(
        "    Initial residual: |Ax-b| = {:.6e}, |Ax-b|/|b| = {:.6e}",
        init_abs, init_rel
    );
    println!(
        "    Initial state err: |x-x_true| = {:.6e}, rel = {:.6e}",
        init_abs_state, init_rel_state
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        if sweep == 1 || sweep == n_sweeps {
            let (_abs, rel) =
                compute_residual(&operator_i_x, &a_input_mapping, &a_output_mapping, &x, &b)?;
            let (_abs_state, rel_state) = compute_state_error(&x, &x_true)?;
            println!(
                "    After {sweep} sweeps: |Ax-b|/|b| = {:.6e}, |x-x_true|/|x_true| = {:.6e}",
                rel, rel_state
            );
        }
    }

    // --- Now run the same test with x_true = all-ones MPS (complex) ---
    println!();
    println!("--- Now: x_true = all-ones MPS (complex) ---");
    let mut used_ids2 = used_ids.clone();
    let x_true_all = create_mps_chain_with_sites_all_ones_c64(n, bond_dim, &sites, &mut used_ids2)?;
    let x_true_all =
        x_true_all.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let b_all = apply_linear_operator(&linop_i_x, &x_true_all, ApplyOptions::default())?;

    let mut x_all = x_true_all.clone();

    let mut updater_all = SquareLinsolveUpdater::with_index_mappings(
        operator_i_x.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
        b_all.clone(),
        options.clone(),
    );

    let plan_all = LocalUpdateSweepPlan::from_treetn(&x_all, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan for all-ones"))?;

    let (init_abs_all, init_rel_all) = compute_residual(
        &operator_i_x,
        &a_input_mapping,
        &a_output_mapping,
        &x_all,
        &b_all,
    )?;
    let (init_abs_state_all, init_rel_state_all) = compute_state_error(&x_all, &x_true_all)?;

    println!(
        "    Initial residual (all-ones): |Ax-b| = {:.6e}, |Ax-b|/|b| = {:.6e}",
        init_abs_all, init_rel_all
    );
    println!(
        "    Initial state err (all-ones): |x-x_true| = {:.6e}, rel = {:.6e}",
        init_abs_state_all, init_rel_state_all
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_all, &plan_all, &mut updater_all)?;
        if sweep == 1 || sweep == n_sweeps {
            let (_abs, rel) = compute_residual(
                &operator_i_x,
                &a_input_mapping,
                &a_output_mapping,
                &x_all,
                &b_all,
            )?;
            let (_abs_state, rel_state) = compute_state_error(&x_all, &x_true_all)?;
            println!(
                "    After {sweep} sweeps (all-ones): |Ax-b|/|b| = {:.6e}, |x-x_true|/|x_true| = {:.6e}",
                rel, rel_state
            );
        }
    }

    Ok(())
}
