//! Compare MPS vs MPO linsolve times with Pauli-X operator.
//!
//! Test 1: x = all-ones MPS, b = A*x, solve A*x = b (x = b start)
//! Test 2: x = all-ones MPO, b = A*x, solve A*x = b (x = b start)
//!
//! Same parameters for both tests; only MPS vs MPO structure differs.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example compare_mps_mpo_pauli --release

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use tensor4all_core::{
    index::DynId, DynIndex, IdxTensor, IndexLike, TensorContractionLike, TensorIndex,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalForm,
    CanonicalizationOptions, IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan,
    SquareLinsolveUpdater, TreeTN, TruncationOptions,
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

    // Pauli X = [[0, 1], [1, 0]]
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

/// Create an all-ones MPS chain.
fn create_all_ones_mps(
    n: usize,
    _phys_dim: usize,
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
        let data = vec![1.0_f64; nelem];
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

/// Create an all-ones MPO state (not operator, but a state with MPO structure).
/// Returns (mpo, input_mapping, output_mapping).
#[allow(clippy::type_complexity)]
fn create_all_ones_mpo_state(
    n: usize,
    phys_dim: usize,
    bond_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(
        true_site_indices.len() == n,
        "true_site_indices.len() must equal n"
    );

    let mut mpo = TreeTN::<IdxTensor, String>::new();

    // MPO bonds
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim))
        .collect();

    // Internal indices (MPO-only: s_in, s_out)
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

        // Build full index list for this site tensor
        let indices: Vec<DynIndex> = if n == 1 {
            // Single site: [external, s_out, s_in]
            vec![
                true_site_indices[i].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ]
        } else if i == 0 {
            // First site: [external, s_out, s_in, bond_right]
            vec![
                true_site_indices[i].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
                bonds[i].clone(),
            ]
        } else if i + 1 == n {
            // Last site: [bond_left, external, s_out, s_in]
            vec![
                bonds[i - 1].clone(),
                true_site_indices[i].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ]
        } else {
            // Middle sites: [bond_left, external, s_out, s_in, bond_right]
            vec![
                bonds[i - 1].clone(),
                true_site_indices[i].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
                bonds[i].clone(),
            ]
        };

        let nelem: usize = indices.iter().map(|idx| idx.dim()).product();
        let data = vec![1.0_f64; nelem];
        let t = IdxTensor::from_dense(indices, data).unwrap();

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

fn compute_residual(
    op: &TreeTN<IdxTensor, String>,
    im: &HashMap<String, IndexMapping<DynIndex>>,
    om: &HashMap<String, IndexMapping<DynIndex>>,
    a0: f64,
    a1: f64,
    x: &TreeTN<IdxTensor, String>,
    rhs: &TreeTN<IdxTensor, String>,
) -> anyhow::Result<f64> {
    let linop = LinearOperator::new(op.clone(), im.clone(), om.clone());
    let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;
    let ax_full = ax.contract_to_tensor()?;
    let x_full = x.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;
    let ref_order = b_full.external_indices();

    let order_for = |tensor: &IdxTensor| -> anyhow::Result<Vec<DynIndex>> {
        let inds = tensor.external_indices();
        let by_id: HashMap<DynId, DynIndex> = inds.into_iter().map(|i| (*i.id(), i)).collect();
        let mut out = Vec::with_capacity(ref_order.len());
        for r in &ref_order {
            let id = *r.id();
            let idx = by_id
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("residual: index {:?} not found in tensor", id))?
                .clone();
            out.push(idx);
        }
        Ok(out)
    };

    let order_x = order_for(&x_full)?;
    let order_ax = order_for(&ax_full)?;
    let x_aligned = x_full.permuteinds(&order_x)?;
    let ax_aligned = ax_full.permuteinds(&order_ax)?;

    let b_vec = b_full.to_vec::<f64>()?;
    let x_vec = x_aligned.to_vec::<f64>()?;
    let ax_vec = ax_aligned.to_vec::<f64>()?;
    anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");
    anyhow::ensure!(x_vec.len() == b_vec.len(), "vector length mismatch");

    let mut r2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for ((ax_i, x_i), b_i) in ax_vec.iter().zip(x_vec.iter()).zip(b_vec.iter()) {
        let opx_i = a0 * x_i + a1 * ax_i;
        let ri = opx_i - b_i;
        r2 += ri * ri;
        b2 += b_i * b_i;
    }
    Ok(if b2 > 0.0 {
        (r2 / b2).sqrt()
    } else {
        r2.sqrt()
    })
}

fn main() -> anyhow::Result<()> {
    let n = 10usize;
    let phys_dim = 2usize;
    let bond_dim = 20usize;
    let max_bond_dim = 20usize;
    let n_sweeps = 5usize;
    let cutoff = 1e-8_f64;
    let rtol = cutoff.sqrt();
    let a0 = 0.0_f64;
    let a1 = 1.0_f64;
    let gmres_tol = 1e-6_f64;
    let gmres_max_restarts = 20usize;
    let gmres_restart_dim = 30usize;

    println!("=== Compare MPS vs MPO linsolve times with Pauli-X operator ===");
    println!("N = {n}");
    println!("phys_dim = {phys_dim}");
    println!("bond_dim = {bond_dim}");
    println!("max_bond_dim = {max_bond_dim}");
    println!("n_sweeps = {n_sweeps}");
    println!("cutoff = {cutoff}");
    println!("rtol = sqrt(cutoff) = {rtol}");
    println!("coefficients: a0 = {a0}, a1 = {a1}");
    println!("GMRES: tol = {gmres_tol}, maxiter = {gmres_max_restarts}, gmres_restart_dim = {gmres_restart_dim}");
    println!();

    let truncation = TruncationOptions::default()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
        .with_max_bond_dim(max_bond_dim);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(n_sweeps)
        .with_truncation(truncation)
        .with_gmres_tol(gmres_tol)
        .with_gmres_max_restarts(gmres_max_restarts)
        .with_gmres_restart_dim(gmres_restart_dim)
        .with_coefficients(a0, a1);

    // ========== Test 1: MPS ==========
    println!("=== Test 1: MPS ===");
    let mut used_ids_mps = HashSet::<DynId>::new();
    let sites_mps: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids_mps, phys_dim))
        .collect();

    // Create Pauli-X operator
    let (operator_a_mps, s_in_tmp_mps, s_out_tmp_mps) =
        create_n_site_pauli_x_mpo_with_internal_indices(n, phys_dim, &mut used_ids_mps)?;
    let (a_input_mapping_mps, a_output_mapping_mps) =
        create_index_mappings(&sites_mps, &s_in_tmp_mps, &s_out_tmp_mps);

    // Create all-ones MPS x_true
    let x_true_mps = create_all_ones_mps(n, phys_dim, bond_dim, &sites_mps, &mut used_ids_mps)?;

    // Compute b = A * x_true
    let linop_a_mps = LinearOperator::new(
        operator_a_mps.clone(),
        a_input_mapping_mps.clone(),
        a_output_mapping_mps.clone(),
    );
    let b_mps = apply_linear_operator(&linop_a_mps, &x_true_mps, ApplyOptions::default())?;

    // Warmup
    println!("Warmup run (excluded)...");
    let center_mps = make_node_name(n / 2);
    {
        let init = b_mps.clone();
        let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        let mut x = init.canonicalize([center_mps.clone()], canon_opts)?;
        let plan = LocalUpdateSweepPlan::from_treetn(&x, &center_mps, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
        let mut updater = SquareLinsolveUpdater::with_index_mappings(
            operator_a_mps.clone(),
            a_input_mapping_mps.clone(),
            a_output_mapping_mps.clone(),
            b_mps.clone(),
            options.clone(),
        );
        for _ in 1..=n_sweeps {
            apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        }
    }

    // Timed run
    let start_mps = Instant::now();
    let init = b_mps.clone();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x_mps = init.canonicalize([center_mps.clone()], canon_opts)?;
    let plan_mps = LocalUpdateSweepPlan::from_treetn(&x_mps, &center_mps, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
    let mut updater_mps = SquareLinsolveUpdater::with_index_mappings(
        operator_a_mps.clone(),
        a_input_mapping_mps.clone(),
        a_output_mapping_mps.clone(),
        b_mps.clone(),
        options.clone(),
    );
    for _ in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_mps, &plan_mps, &mut updater_mps)?;
    }
    let elapsed_mps = start_mps.elapsed();

    let res_mps = compute_residual(
        &operator_a_mps,
        &a_input_mapping_mps,
        &a_output_mapping_mps,
        a0,
        a1,
        &x_mps,
        &b_mps,
    )?;

    println!(
        "MPS solve time: {:.3} ms",
        elapsed_mps.as_secs_f64() * 1000.0
    );
    println!("MPS final residual: {:.6e}", res_mps);
    println!();

    // ========== Test 2: MPO ==========
    println!("=== Test 2: MPO ===");
    let mut used_ids_mpo = HashSet::<DynId>::new();
    let sites_mpo: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids_mpo, phys_dim))
        .collect();

    // Create Pauli-X operator
    let (operator_a_mpo, s_in_tmp_mpo, s_out_tmp_mpo) =
        create_n_site_pauli_x_mpo_with_internal_indices(n, phys_dim, &mut used_ids_mpo)?;
    let (a_input_mapping_mpo, a_output_mapping_mpo) =
        create_index_mappings(&sites_mpo, &s_in_tmp_mpo, &s_out_tmp_mpo);

    // Create all-ones MPO state x_true
    let (x_true_mpo, _x_input_mapping_mpo, _x_output_mapping_mpo) =
        create_all_ones_mpo_state(n, phys_dim, bond_dim, &sites_mpo, &mut used_ids_mpo)?;

    // Compute b = A * x_true (A acts on MPO state)
    let linop_a_mpo = LinearOperator::new(
        operator_a_mpo.clone(),
        a_input_mapping_mpo.clone(),
        a_output_mapping_mpo.clone(),
    );
    let b_mpo = apply_linear_operator(&linop_a_mpo, &x_true_mpo, ApplyOptions::default())?;

    // Warmup
    println!("Warmup run (excluded)...");
    let center_mpo = make_node_name(n / 2);
    {
        let init = b_mpo.clone();
        let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        let mut x = init.canonicalize([center_mpo.clone()], canon_opts)?;
        let plan = LocalUpdateSweepPlan::from_treetn(&x, &center_mpo, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
        let mut updater = SquareLinsolveUpdater::with_index_mappings(
            operator_a_mpo.clone(),
            a_input_mapping_mpo.clone(),
            a_output_mapping_mpo.clone(),
            b_mpo.clone(),
            options.clone(),
        );
        for _ in 1..=n_sweeps {
            apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        }
    }

    // Timed run
    let start_mpo = Instant::now();
    let init = b_mpo.clone();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x_mpo_sol = init.canonicalize([center_mpo.clone()], canon_opts)?;
    let plan_mpo = LocalUpdateSweepPlan::from_treetn(&x_mpo_sol, &center_mpo, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
    let mut updater_mpo = SquareLinsolveUpdater::with_index_mappings(
        operator_a_mpo.clone(),
        a_input_mapping_mpo.clone(),
        a_output_mapping_mpo.clone(),
        b_mpo.clone(),
        options.clone(),
    );
    for _ in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_mpo_sol, &plan_mpo, &mut updater_mpo)?;
    }
    let elapsed_mpo = start_mpo.elapsed();

    let res_mpo = compute_residual(
        &operator_a_mpo,
        &a_input_mapping_mpo,
        &a_output_mapping_mpo,
        a0,
        a1,
        &x_mpo_sol,
        &b_mpo,
    )?;

    println!(
        "MPO solve time: {:.3} ms",
        elapsed_mpo.as_secs_f64() * 1000.0
    );
    println!("MPO final residual: {:.6e}", res_mpo);
    println!();

    // ========== Summary ==========
    println!("=== Summary ===");
    println!(
        "MPS: {:.3} ms (residual: {:.3e})",
        elapsed_mps.as_secs_f64() * 1000.0,
        res_mps
    );
    println!(
        "MPO: {:.3} ms (residual: {:.3e})",
        elapsed_mpo.as_secs_f64() * 1000.0,
        res_mpo
    );
    let overhead = (elapsed_mpo.as_secs_f64() / elapsed_mps.as_secs_f64() - 1.0) * 100.0;
    println!(
        "MPO overhead: {:.1}% slower than MPS",
        if overhead > 0.0 { overhead } else { 0.0 }
    );

    Ok(())
}
