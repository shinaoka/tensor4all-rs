//! Test: linsolve with MPO for x and b (all identity operators).
//!
//! Test setup: A = I, x = I, b = I (MPOs). Equation: I * x = b.
//!
//! **Case 1 (real):** b = I. init=rhs or init=random. Solve I*x = b.
//! **Case 2 (pure imaginary):** b = i*I. init=rhs or init=random. Solve I*x = b.
//! **Case 3:** A = i*I, b = -I, solution x = i*I. init=rhs or init=random. Solve (i*I)*x = -I.
//! Each test prints a short description, then initial/final |Ax - b| / |b|.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mpo_identity --release

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

/// Create an N-site identity MPO with internal indices (bond dim = 1).
/// Returns (mpo, input_mapping, output_mapping).
#[allow(clippy::type_complexity)]
fn create_identity_mpo_with_mappings(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<IdxTensor, String>::new();

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

        // Index ordering matches tests/linsolve.rs conventions.
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);

        // For x and b MPOs, we need to include the external index (true_site_indices[i])
        // in the tensor so it appears in site_space.
        // The tensor structure is: [external, s_out_tmp, s_in_tmp, ...bonds]
        // where external = true_site_indices[i] is the index that remains open.
        //
        // We create an identity on (external, s_out_tmp, s_in_tmp) where:
        // - Identity on (s_out_tmp, s_in_tmp): δ(s_out, s_in)
        // - Identity on external: each value of external passes through unchanged
        // So: I[ext, s_out, s_in] = δ(s_out, s_in) for each ext value
        let mut base_data = vec![0.0_f64; phys_dim * phys_dim * phys_dim];
        for ext_val in 0..phys_dim {
            for k in 0..phys_dim {
                // Identity on (s_out, s_in): δ(s_out, s_in)
                // Index order: [external, s_out, s_in]
                let idx = ext_val * phys_dim * phys_dim + k * phys_dim + k;
                base_data[idx] = 1.0;
            }
        }
        let base = IdxTensor::from_dense(
            vec![
                true_site_indices[i].clone(), // external index (appears in site_space)
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ],
            base_data,
        )
        .unwrap();

        let t = if indices.len() == 2 {
            // n == 1 case: [external, s_out, s_in]
            base
        } else {
            // Add bond indices via outer product
            let bond_indices = bond_indices(&indices);
            let ones = IdxTensor::from_dense(bond_indices, vec![1.0_f64; 1]).unwrap();
            IdxTensor::outer_product(&base, &ones)?
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

/// Create a random MPO with the same structure (same index IDs) as `template`.
/// Replaces each node's tensor with random **real** data. Used for init != rhs (real case).
fn create_random_mpo_matching_structure(
    template: &TreeTN<IdxTensor, String>,
    seed: u64,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mpo = template.clone();
    for node_name in template.node_names() {
        let node_idx = mpo.node_index(&node_name).unwrap();
        let t = mpo.tensor(node_idx).unwrap();
        let indices = t.external_indices();
        let new_t = IdxTensor::random::<f64, _>(&mut rng, indices)?;
        mpo.replace_tensor(node_idx, new_t)?;
    }
    Ok(mpo)
}

/// Like `create_random_mpo_matching_structure` but uses **complex** random tensors.
/// Use for init=random when RHS is complex (e.g. b = i*I) so storage matches DenseC64.
fn create_random_mpo_matching_structure_c64(
    template: &TreeTN<IdxTensor, String>,
    seed: u64,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mpo = template.clone();
    for node_name in template.node_names() {
        let node_idx = mpo.node_index(&node_name).unwrap();
        let t = mpo.tensor(node_idx).unwrap();
        let indices = t.external_indices();
        let new_t = IdxTensor::random::<Complex64, _>(&mut rng, indices)?;
        mpo.replace_tensor(node_idx, new_t)?;
    }
    Ok(mpo)
}

/// Create an N-site identity MPO for the **operator** A only.
/// Tensors have [s_out, s_in] (+ bonds) with no separate "external" index.
/// Mappings use true_site_indices for state connection. This avoids duplicate
/// true_site in ProjectedOperator::apply output (op has no true_site).
#[allow(clippy::type_complexity)]
fn create_identity_mpo_operator_only(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<IdxTensor, String>::new();
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
            IdxTensor::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], base_data)
                .unwrap();

        let t = if indices.len() == 2 {
            base
        } else {
            let bond_indices = bond_indices(&indices);
            let ones = IdxTensor::from_dense(bond_indices, vec![1.0_f64; 1]).unwrap();
            IdxTensor::outer_product(&base, &ones)?
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

fn print_bond_dims(treetn: &TreeTN<IdxTensor, String>, label: &str) {
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

fn compute_residual(
    op: &TreeTN<IdxTensor, String>,
    im: &HashMap<String, IndexMapping<DynIndex>>,
    om: &HashMap<String, IndexMapping<DynIndex>>,
    a0: f64,
    a1: f64,
    x: &TreeTN<IdxTensor, String>,
    rhs: &TreeTN<IdxTensor, String>,
) -> anyhow::Result<(f64, f64)> {
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
            let ri: Complex64 = opx_i - *b_i;
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

fn main() -> anyhow::Result<()> {
    let n = 3usize;
    let phys_dim = 2usize;

    println!("=== Test: linsolve with MPO for x and b (all identity operators) ===");
    println!("N = {n}, phys_dim = {phys_dim}");
    println!("Equation: A * x = b with A = I (identity MPO).");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let b_site_indices: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    let (b_mpo, _b_input_mapping, _b_output_mapping) =
        create_identity_mpo_with_mappings(n, phys_dim, &b_site_indices, &mut used_ids)?;
    print_bond_dims(&b_mpo, "b (RHS) bond dimensions");
    println!();

    let options = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_bond_dim(4)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(1.0, 0.0); // a0=1, a1=0 => I * x = b

    let (operator_a, a_input_mapping, a_output_mapping) =
        create_identity_mpo_operator_only(n, phys_dim, &b_site_indices, &mut used_ids)?;

    let center = make_node_name(n / 2);
    let n_sweeps = 5usize;
    let init_modes = ["rhs", "random"];

    println!("--- Case 1: Real RHS (b = I), init=rhs / init=random ---");
    for init_mode in init_modes {
        println!("  Test: b = I (identity MPO), init={init_mode}. Solve I*x = b.");

        let init = match init_mode {
            "rhs" => b_mpo.clone(),
            "random" => create_random_mpo_matching_structure(&b_mpo, 42)?,
            _ => anyhow::bail!("unknown init_mode {init_mode}"),
        };
        let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

        let mut updater = SquareLinsolveUpdater::with_index_mappings(
            operator_a.clone(),
            a_input_mapping.clone(),
            a_output_mapping.clone(),
            b_mpo.clone(),
            options.clone(),
        );
        let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

        let (_init_abs, init_rel) = compute_residual(
            &operator_a,
            &a_input_mapping,
            &a_output_mapping,
            1.0,
            0.0,
            &x,
            &b_mpo,
        )?;
        println!("    Initial |Ax - b| / |b| = {:.6e}", init_rel);

        for _sweep in 1..=n_sweeps {
            apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        }

        let (_final_abs, final_rel) = compute_residual(
            &operator_a,
            &a_input_mapping,
            &a_output_mapping,
            1.0,
            0.0,
            &x,
            &b_mpo,
        )?;
        println!(
            "    After {} sweeps: |Ax - b| / |b| = {:.6e}",
            n_sweeps, final_rel
        );
        print_bond_dims(&x, "    x bond dimensions (final)");
        println!();
    }

    println!();
    println!("--- Case 2: Pure imaginary RHS (b = i*I), init=rhs / init=random ---");
    let imag_scalar = AnyScalar::from(Complex64::new(0.0, 1.0));
    let b_imag = scale_treetn(&b_mpo, imag_scalar.clone())?;

    for init_mode in ["rhs", "random"] {
        println!("  Test: b = i*I (pure imaginary), init={init_mode}. Solve I*x = b.");

        let init = match init_mode {
            "rhs" => b_imag.clone(),
            "random" => create_random_mpo_matching_structure_c64(&b_imag, 43)?,
            _ => anyhow::bail!("unknown init_mode {init_mode}"),
        };
        let mut x_imag = init
            .clone()
            .canonicalize([center.clone()], CanonicalizationOptions::default())?;
        let mut updater_imag = SquareLinsolveUpdater::with_index_mappings(
            operator_a.clone(),
            a_input_mapping.clone(),
            a_output_mapping.clone(),
            b_imag.clone(),
            options.clone(),
        );
        let plan_imag = LocalUpdateSweepPlan::from_treetn(&x_imag, &center, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

        let (_init_abs, init_rel) = compute_residual(
            &operator_a,
            &a_input_mapping,
            &a_output_mapping,
            1.0,
            0.0,
            &x_imag,
            &b_imag,
        )?;
        println!("    Initial |Ax - b| / |b| = {:.6e}", init_rel);
        for _sweep in 1..=n_sweeps {
            apply_local_update_sweep(&mut x_imag, &plan_imag, &mut updater_imag)?;
        }
        let (_final_abs, final_rel) = compute_residual(
            &operator_a,
            &a_input_mapping,
            &a_output_mapping,
            1.0,
            0.0,
            &x_imag,
            &b_imag,
        )?;
        println!(
            "    After {} sweeps: |Ax - b| / |b| = {:.6e}",
            n_sweeps, final_rel
        );
        print_bond_dims(&x_imag, "    x bond dimensions (final)");
        println!();
    }

    println!();
    println!("--- Case 3: A = i*I, b = -I (solution x = i*I), init=rhs / init=random ---");
    let neg_one = AnyScalar::from(Complex64::new(-1.0, 0.0));
    let a_case3 = scale_treetn(&operator_a, imag_scalar)?;
    let b_case3 = scale_treetn(&b_mpo, neg_one)?;
    let options_case3 = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_bond_dim(4)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(20)
        .with_gmres_restart_dim(30)
        .with_coefficients(0.0, 1.0); // a0=0, a1=1 => (i*I)*x = b

    for init_mode in ["rhs", "random"] {
        println!("  Test: A = i*I, b = -I, init={init_mode}. Solve (i*I)*x = -I.");

        let init = match init_mode {
            "rhs" => b_case3.clone(),
            "random" => create_random_mpo_matching_structure_c64(&b_case3, 44)?,
            _ => anyhow::bail!("unknown init_mode {init_mode}"),
        };
        let mut x_case3 = init
            .clone()
            .canonicalize([center.clone()], CanonicalizationOptions::default())?;
        let mut updater_case3 = SquareLinsolveUpdater::with_index_mappings(
            a_case3.clone(),
            a_input_mapping.clone(),
            a_output_mapping.clone(),
            b_case3.clone(),
            options_case3.clone(),
        );
        let plan_case3 = LocalUpdateSweepPlan::from_treetn(&x_case3, &center, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

        let (_init_abs, init_rel) = compute_residual(
            &a_case3,
            &a_input_mapping,
            &a_output_mapping,
            0.0,
            1.0,
            &x_case3,
            &b_case3,
        )?;
        println!("    Initial |Ax - b| / |b| = {:.6e}", init_rel);
        for _sweep in 1..=n_sweeps {
            apply_local_update_sweep(&mut x_case3, &plan_case3, &mut updater_case3)?;
        }
        let (_final_abs, final_rel) = compute_residual(
            &a_case3,
            &a_input_mapping,
            &a_output_mapping,
            0.0,
            1.0,
            &x_case3,
            &b_case3,
        )?;
        println!(
            "    After {} sweeps: |Ax - b| / |b| = {:.6e}",
            n_sweeps, final_rel
        );
        print_bond_dims(&x_case3, "    x bond dimensions (final)");
        println!();
    }

    println!("=== Test completed successfully ===");
    Ok(())
}
