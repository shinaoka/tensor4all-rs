//! Test: linsolve with MPO where A is a Pauli-X operator and x is a random MPO.
//!
//! Test setup: A = Pauli-X operator, x = random MPO (state), b = A*x.
//! Then solve A*x = b (init=rhs, i.e., init=b).
//! If the solver converges correctly, the solution x should be close to the original random MPO.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mpo_pauli_operator --release

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    index::DynId, AnyScalar, DynIndex, IndexLike, TensorContractionLike, TensorDynLen, TensorIndex,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

type TnWithInternalIndices = (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>);

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

fn mpo_node_indices(
    n: usize,
    i: usize,
    bonds: &[DynIndex],
    s_out: &[DynIndex],
    s_in: &[DynIndex],
) -> Vec<DynIndex> {
    if n == 1 {
        vec![s_out[i].clone(), s_in[i].clone()]
    } else if i == 0 {
        vec![s_out[i].clone(), s_in[i].clone(), bonds[i].clone()]
    } else if i + 1 == n {
        vec![bonds[i - 1].clone(), s_out[i].clone(), s_in[i].clone()]
    } else {
        vec![
            bonds[i - 1].clone(),
            s_out[i].clone(),
            s_in[i].clone(),
            bonds[i].clone(),
        ]
    }
}

fn bond_indices(indices: &[DynIndex]) -> Vec<DynIndex> {
    indices
        .iter()
        .filter(|idx| idx.dim() == 1)
        .cloned()
        .collect()
}

/// Create an N-site Pauli-X MPO operator (for operator A).
/// Returns (mpo, s_in_tmp, s_out_tmp) where s_in_tmp and s_out_tmp are internal indices.
/// This is based on test_linsolve_general_coefficients_n3.rs.
fn create_n_site_pauli_x_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<TnWithInternalIndices> {
    anyhow::ensure!(n_sites >= 1, "Need at least 1 site");
    anyhow::ensure!(phys_dim == 2, "Pauli-X requires phys_dim=2");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // Internal indices (independent IDs)
    let s_in_tmp: Vec<DynIndex> = (0..n_sites)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();

    // Bond indices (dim 1 for Pauli-X operator)
    let bond_indices: Vec<DynIndex> = (0..n_sites.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();

    // Pauli X matrix: [[0, 1], [1, 0]]
    // As a tensor [out, in]: X[0,0]=0, X[0,1]=1, X[1,0]=1, X[1,1]=0
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    let mut nodes = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let name = make_node_name(i);
        let mut data = vec![0.0; phys_dim * phys_dim];
        for out_idx in 0..phys_dim {
            for in_idx in 0..phys_dim {
                data[out_idx * phys_dim + in_idx] = pauli_x[out_idx * phys_dim + in_idx];
            }
        }

        let tensor = if n_sites == 1 {
            TensorDynLen::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], data).unwrap()
        } else if i == 0 {
            TensorDynLen::from_dense(
                vec![
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else if i == n_sites - 1 {
            TensorDynLen::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            TensorDynLen::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        };
        let node = mpo.add_tensor(name, tensor).unwrap();
        nodes.push(node);
    }

    // Connect adjacent sites
    for (i, bond) in bond_indices.iter().enumerate() {
        mpo.connect(nodes[i], bond, nodes[i + 1], bond).unwrap();
    }

    Ok((mpo, s_in_tmp, s_out_tmp))
}

/// Create index mappings from MPO operator internal indices to state site indices.
/// Returns (input_mapping, output_mapping).
fn create_index_mappings(
    state_site_indices: &[DynIndex],
    s_in_tmp: &[DynIndex],
    s_out_tmp: &[DynIndex],
) -> (
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    let n_sites = state_site_indices.len();
    assert_eq!(s_in_tmp.len(), n_sites);
    assert_eq!(s_out_tmp.len(), n_sites);

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for i in 0..n_sites {
        let site = make_node_name(i);
        input_mapping.insert(
            site.clone(),
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            site,
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    (input_mapping, output_mapping)
}

/// Create an N-site identity MPO operator (for operator I).
/// Returns (mpo, input_mapping, output_mapping).
#[allow(clippy::type_complexity)]
fn create_identity_mpo_operator(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();
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
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);

        let mut base_data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            base_data[k * phys_dim + k] = 1.0;
        }
        let base =
            TensorDynLen::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], base_data)
                .unwrap();

        let t = if indices.len() == 2 {
            base
        } else {
            let bond_indices = bond_indices(&indices);
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

/// Create a random MPO state (for x) with external indices.
/// Returns (mpo, input_mapping, output_mapping).
/// This creates a state MPO with external indices (true_site_indices) that remain open.
#[allow(clippy::type_complexity)]
fn create_random_mpo_state(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    seed: u64,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
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
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);

        // For state MPOs, we need to include the external index (true_site_indices[i])
        // in the tensor so it appears in site_space.
        // The tensor structure is: [external, s_out_tmp, s_in_tmp, ...bonds]
        // where external = true_site_indices[i] is the index that remains open.
        //
        // Create random tensor with shape [external, s_out, s_in]
        let random_tensor = TensorDynLen::random::<f64, _>(
            &mut rng,
            vec![
                true_site_indices[i].clone(), // external index
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ],
        )?;

        let t = if indices.len() == 2 {
            // n == 1 case: [external, s_out, s_in]
            random_tensor
        } else {
            // Add bond indices via outer product
            let bond_indices = bond_indices(&indices);
            let ones = TensorDynLen::from_dense(bond_indices, vec![1.0_f64; 1]).unwrap();
            TensorDynLen::outer_product(&random_tensor, &ones)?
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

fn print_bond_dims(treetn: &TreeTN<TensorDynLen, String>, label: &str) {
    let edges: Vec<_> = treetn.site_index_network().edges().collect();
    if edges.is_empty() {
        println!("{label}: no bonds");
        return;
    }
    let mut dims = Vec::new();
    for (node_a, node_b) in edges {
        if let Some(edge) = treetn.edge_between(&node_a, &node_b) {
            if let Some(bond) = treetn.bond_index(edge) {
                dims.push(bond.dim);
            }
        }
    }
    println!("{label}: bond_dims = {:?}", dims);
}

/// Scale a TreeTN by a scalar factor (scale only one tensor to avoid scalar^n scaling).
fn scale_treetn(
    treetn: &TreeTN<TensorDynLen, String>,
    scalar: AnyScalar,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    let mut scaled = TreeTN::<TensorDynLen, String>::new();
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

fn compute_residual(
    op: &TreeTN<TensorDynLen, String>,
    im: &HashMap<String, IndexMapping<DynIndex>>,
    om: &HashMap<String, IndexMapping<DynIndex>>,
    a0: f64,
    a1: f64,
    x: &TreeTN<TensorDynLen, String>,
    rhs: &TreeTN<TensorDynLen, String>,
) -> anyhow::Result<(f64, f64)> {
    let linop = LinearOperator::new(op.clone(), im.clone(), om.clone());
    let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;
    let ax_full = ax.contract_to_tensor()?;
    let x_full = x.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;
    let ref_order = b_full.external_indices();

    let order_for = |tensor: &TensorDynLen| -> anyhow::Result<Vec<DynIndex>> {
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
    if b_full.is_complex() {
        let b_vec = b_full.to_vec::<Complex64>()?;
        let x_vec = x_aligned.to_vec::<Complex64>()?;
        let ax_vec = ax_aligned.to_vec::<Complex64>()?;
        anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");
        anyhow::ensure!(x_vec.len() == b_vec.len(), "vector length mismatch");

        let mut r2 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for ((ax_i, x_i), b_i) in ax_vec.iter().zip(x_vec.iter()).zip(b_vec.iter()) {
            let opx_i = x_i * a0 + ax_i * a1;
            let ri = opx_i - b_i;
            r2 += ri.norm_sqr();
            b2 += b_i.norm_sqr();
        }
        let abs_res = r2.sqrt();
        let rel_res = if b2 > 0.0 { (r2 / b2).sqrt() } else { abs_res };
        Ok((abs_res, rel_res))
    } else {
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
        let abs_res = r2.sqrt();
        let rel_res = if b2 > 0.0 { (r2 / b2).sqrt() } else { abs_res };
        Ok((abs_res, rel_res))
    }
}

/// Compute the difference between x and x_true.
/// Returns (absolute error, relative error) where relative error is |x - x_true| / |x_true|.
fn compute_state_error(
    x: &TreeTN<TensorDynLen, String>,
    x_true: &TreeTN<TensorDynLen, String>,
) -> anyhow::Result<(f64, f64)> {
    let x_full = x.contract_to_tensor()?;
    let x_true_full = x_true.contract_to_tensor()?;
    let ref_order = x_true_full.external_indices();

    let order_for = |tensor: &TensorDynLen| -> anyhow::Result<Vec<DynIndex>> {
        let inds = tensor.external_indices();
        let by_id: HashMap<DynId, DynIndex> = inds.into_iter().map(|i| (*i.id(), i)).collect();
        let mut out = Vec::with_capacity(ref_order.len());
        for r in &ref_order {
            let id = *r.id();
            let idx = by_id
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("state error: index {:?} not found in tensor", id))?
                .clone();
            out.push(idx);
        }
        Ok(out)
    };

    let order_x = order_for(&x_full)?;
    let x_aligned = x_full.permuteinds(&order_x)?;

    if x_true_full.is_complex() {
        let x_true_vec = x_true_full.to_vec::<Complex64>()?;
        let x_vec = x_aligned.to_vec::<Complex64>()?;
        anyhow::ensure!(x_vec.len() == x_true_vec.len(), "vector length mismatch");

        let mut diff2 = 0.0_f64;
        let mut x_true2 = 0.0_f64;
        for (x_i, x_true_i) in x_vec.iter().zip(x_true_vec.iter()) {
            let diff_i = x_i - x_true_i;
            diff2 += diff_i.norm_sqr();
            x_true2 += x_true_i.norm_sqr();
        }
        let abs_err = diff2.sqrt();
        let rel_err = if x_true2 > 0.0 {
            (diff2 / x_true2).sqrt()
        } else {
            abs_err
        };
        Ok((abs_err, rel_err))
    } else {
        let x_true_vec = x_true_full.to_vec::<f64>()?;
        let x_vec = x_aligned.to_vec::<f64>()?;
        anyhow::ensure!(x_vec.len() == x_true_vec.len(), "vector length mismatch");

        let mut diff2 = 0.0_f64;
        let mut x_true2 = 0.0_f64;
        for (x_i, x_true_i) in x_vec.iter().zip(x_true_vec.iter()) {
            let diff_i = x_i - x_true_i;
            diff2 += diff_i * diff_i;
            x_true2 += x_true_i * x_true_i;
        }
        let abs_err = diff2.sqrt();
        let rel_err = if x_true2 > 0.0 {
            (diff2 / x_true2).sqrt()
        } else {
            abs_err
        };
        Ok((abs_err, rel_err))
    }
}

fn main() -> anyhow::Result<()> {
    let n = 3usize;
    let phys_dim = 2usize;

    println!("=== Test: linsolve with Pauli-X operator A and random MPO state x ===");
    println!("N = {n}, phys_dim = {phys_dim}");
    println!("Test setup: A = Pauli-X operator, x = random MPO (state), b = A*x.");
    println!("Then solve A*x = b with init=rhs (i.e., init=b).");
    println!("Expected: solution x should converge to the original random MPO.");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let site_indices: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    // Create Pauli-X operator A
    let (operator_a, s_in_tmp, s_out_tmp) =
        create_n_site_pauli_x_mpo_with_internal_indices(n, phys_dim, &mut used_ids)?;
    let (a_input_mapping, a_output_mapping) =
        create_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);
    print_bond_dims(&operator_a, "A (Pauli-X operator) bond dimensions");
    println!();

    // Create random MPO state x (the true solution)
    let (x_true, _x_input_mapping, _x_output_mapping) =
        create_random_mpo_state(n, phys_dim, &site_indices, &mut used_ids, 12345)?;
    print_bond_dims(&x_true, "x_true (random MPO state) bond dimensions");
    println!();

    let center = make_node_name(n / 2);
    let n_sweeps = 10usize;

    // Test case 1: a0=0, a1=1 => A * x = b
    println!("=== Test case 1: a0=0, a1=1 (equation: A*x = b) ===");
    println!();

    // Compute b = A * x_true
    let linop_a = LinearOperator::new(
        operator_a.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
    );
    let b_tree_1 = apply_linear_operator(&linop_a, &x_true, ApplyOptions::default())?;
    print_bond_dims(&b_tree_1, "b = A*x_true bond dimensions");
    println!();

    // Set up linsolve options
    let options_1 = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_bond_dim(4)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(0.0, 1.0); // a0=0, a1=1 => A * x = b

    println!("--- Solving A*x = b with init=rhs (i.e., init=b) ---");
    // Initialize x with b (rhs): x^(0) = b
    println!("    Initial value: x^(0) = b (rhs)");
    let init_1 = b_tree_1.clone();
    let mut x_1 = init_1.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_1 = SquareLinsolveUpdater::with_index_mappings(
        operator_a.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
        b_tree_1.clone(),
        options_1.clone(),
    );
    let plan_1 = LocalUpdateSweepPlan::from_treetn(&x_1, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_1, init_rel_1) = compute_residual(
        &operator_a,
        &a_input_mapping,
        &a_output_mapping,
        0.0,
        1.0,
        &x_1,
        &b_tree_1,
    )?;
    println!(
        "    Initial residual: |Ax - b| = {:.6e}, |Ax - b| / |b| = {:.6e}",
        init_abs_1, init_rel_1
    );

    let (init_abs_state_1, init_rel_state_1) = compute_state_error(&x_1, &x_true)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_1, init_rel_state_1
    );

    for _sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_1, &plan_1, &mut updater_1)?;
    }

    let (final_abs_1, final_rel_1) = compute_residual(
        &operator_a,
        &a_input_mapping,
        &a_output_mapping,
        0.0,
        1.0,
        &x_1,
        &b_tree_1,
    )?;
    println!(
        "    After {} sweeps: |Ax - b| = {:.6e}, |Ax - b| / |b| = {:.6e}",
        n_sweeps, final_abs_1, final_rel_1
    );

    let (final_abs_state_1, final_rel_state_1) = compute_state_error(&x_1, &x_true)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_1, final_rel_state_1
    );

    print_bond_dims(&x_1, "    x bond dimensions (final)");
    println!();

    // Test case 2: a0=2, a1=1 => (2I + A) * x = b
    println!("=== Test case 2: a0=2, a1=1 (equation: (2I + A)*x = b) ===");
    println!();

    // Create identity operator I
    let (operator_i, i_input_mapping, i_output_mapping) =
        create_identity_mpo_operator(n, phys_dim, &site_indices, &mut used_ids)?;
    print_bond_dims(&operator_i, "I (identity operator) bond dimensions");
    println!();

    // Compute b = (2I + A) * x_true = 2*I*x_true + A*x_true
    let linop_i = LinearOperator::new(
        operator_i.clone(),
        i_input_mapping.clone(),
        i_output_mapping.clone(),
    );
    let i_x = apply_linear_operator(&linop_i, &x_true, ApplyOptions::default())?;
    let a_x = apply_linear_operator(&linop_a, &x_true, ApplyOptions::default())?;

    // Scale I*x by 2 using scale_treetn
    use tensor4all_core::AnyScalar;
    let i_x_scaled = scale_treetn(&i_x, AnyScalar::new_real(2.0))?;

    // Add i_x_scaled and a_x using TreeTN::add (direct-sum construction)
    let b_tree_2 = i_x_scaled.add(&a_x)?;

    print_bond_dims(&b_tree_2, "b = (2I + A)*x_true bond dimensions");
    println!();

    // Set up linsolve options for a0=2, a1=1
    let options_2 = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_bond_dim(4)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(2.0, 1.0); // a0=2, a1=1 => (2I + A) * x = b

    println!("--- Solving (2I + A)*x = b with init=rhs (i.e., init=b) ---");
    println!("    Initial value: x^(0) = b (rhs)");
    let init_2 = b_tree_2.clone();
    let mut x_2 = init_2.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_2 = SquareLinsolveUpdater::with_index_mappings(
        operator_a.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
        b_tree_2.clone(),
        options_2.clone(),
    );
    let plan_2 = LocalUpdateSweepPlan::from_treetn(&x_2, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_2, init_rel_2) = compute_residual(
        &operator_a,
        &a_input_mapping,
        &a_output_mapping,
        2.0,
        1.0,
        &x_2,
        &b_tree_2,
    )?;
    println!(
        "    Initial residual: |(2I + A)x - b| = {:.6e}, |(2I + A)x - b| / |b| = {:.6e}",
        init_abs_2, init_rel_2
    );

    let (init_abs_state_2, init_rel_state_2) = compute_state_error(&x_2, &x_true)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_2, init_rel_state_2
    );

    for _sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_2, &plan_2, &mut updater_2)?;
    }

    let (final_abs_2, final_rel_2) = compute_residual(
        &operator_a,
        &a_input_mapping,
        &a_output_mapping,
        2.0,
        1.0,
        &x_2,
        &b_tree_2,
    )?;
    println!(
        "    After {} sweeps: |(2I + A)x - b| = {:.6e}, |(2I + A)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_2, final_rel_2
    );

    let (final_abs_state_2, final_rel_state_2) = compute_state_error(&x_2, &x_true)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_2, final_rel_state_2
    );

    print_bond_dims(&x_2, "    x bond dimensions (final)");
    println!();

    println!("=== All tests completed successfully ===");
    Ok(())
}
