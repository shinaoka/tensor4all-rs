//! Test: linsolve with MPO where A is a pure imaginary Pauli-X operator (i*X) and x is a complex random MPO.
//!
//! Test setup: A = i*X (pure imaginary * Pauli-X operator), x = complex random MPO (state), b = A*x.
//! Then solve (i*X)*x = b with init=x_true (exact solution).
//!
//! This test case demonstrates numerical instability issues when:
//! - A is a complex operator (i*X)
//! - x is a complex random MPO
//! - Even when starting from the exact solution, numerical errors can accumulate
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mpo_pauli_imaginary --release

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::index::DynId;
use tensor4all_core::{
    AnyScalar, DynIndex, IndexLike, TensorContractionLike, TensorDynLen, TensorIndex,
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

/// Create a random complex MPO state (for x) with external indices.
/// Returns (mpo, input_mapping, output_mapping).
/// This creates a complex state MPO with external indices (true_site_indices) that remain open.
#[allow(clippy::type_complexity)]
fn create_random_mpo_state_c64(
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

        // Create random complex tensor with shape [external, s_out, s_in]
        let random_tensor = TensorDynLen::random::<Complex64, _>(
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

/// Create an identity MPO state (for x) with external indices (complex version).
/// Returns (mpo, input_mapping, output_mapping).
/// This creates a complex identity state MPO with external indices (true_site_indices) that remain open.
/// The identity structure: I[ext, s_out, s_in] = δ(s_out, s_in) for each ext value.
#[allow(clippy::type_complexity)]
fn create_identity_mpo_state_c64(
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

        // Identity on (external, s_out, s_in): I[ext, s_out, s_in] = δ(s_out, s_in) for each ext value
        // Use complex data type
        let mut base_data = vec![Complex64::new(0.0, 0.0); phys_dim * phys_dim * phys_dim];
        for ext_val in 0..phys_dim {
            for k in 0..phys_dim {
                // Index order: [external, s_out, s_in]
                let idx = ext_val * phys_dim * phys_dim + k * phys_dim + k;
                base_data[idx] = Complex64::new(1.0, 0.0);
            }
        }
        let base = TensorDynLen::from_dense(
            vec![
                true_site_indices[i].clone(), // external index
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
            let ones =
                TensorDynLen::from_dense(bond_indices, vec![Complex64::new(1.0, 0.0); 1]).unwrap();
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

/// Create an all-ones MPO state (for x) with external indices (complex version).
/// Returns (mpo, input_mapping, output_mapping).
///
/// Each local tensor has shape [external, s_out, s_in] and is filled with 1+0i.
/// This provides a simple, deterministic x_true with the same index structure as
/// `create_random_mpo_state_c64` / `create_identity_mpo_state_c64`.
#[allow(clippy::type_complexity)]
fn create_all_ones_mpo_state_c64(
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

        // All-ones on (external, s_out, s_in)
        let base_data = vec![Complex64::new(1.0, 0.0); phys_dim * phys_dim * phys_dim];
        let base = TensorDynLen::from_dense(
            vec![
                true_site_indices[i].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ],
            base_data,
        )
        .unwrap();

        let t = if indices.len() == 2 {
            base
        } else {
            let bond_indices = bond_indices(&indices);
            let ones =
                TensorDynLen::from_dense(bond_indices, vec![Complex64::new(1.0, 0.0); 1]).unwrap();
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

fn print_operator_dense_matrix(
    op: &TreeTN<TensorDynLen, String>,
    s_out: &[DynIndex],
    s_in: &[DynIndex],
    label: &str,
) -> anyhow::Result<()> {
    let t = op.contract_to_tensor()?;
    let external = t.external_indices();
    let by_id: HashMap<DynId, DynIndex> = external.into_iter().map(|i| (*i.id(), i)).collect();

    let mut desired_order = Vec::with_capacity(s_out.len() + s_in.len());
    for idx in s_out.iter().chain(s_in.iter()) {
        desired_order.push(
            by_id
                .get(idx.id())
                .ok_or_else(|| anyhow::anyhow!("matrix print: index {:?} not found", idx.id()))?
                .clone(),
        );
    }

    let t = t.permuteinds(&desired_order)?;

    let dim_out: usize = s_out.iter().map(|i| i.dim).product();
    let dim_in: usize = s_in.iter().map(|i| i.dim).product();
    let expected_len = dim_out
        .checked_mul(dim_in)
        .ok_or_else(|| anyhow::anyhow!("matrix print: dimension overflow"))?;

    println!("{label} (dense matrix {dim_out}x{dim_in}):");
    if t.is_complex() {
        let data = t.to_vec::<Complex64>()?;
        anyhow::ensure!(
            data.len() == expected_len,
            "matrix print: length mismatch (got {}, expected {})",
            data.len(),
            expected_len
        );
        for r in 0..dim_out {
            print!("[");
            for c in 0..dim_in {
                let v = data[r * dim_in + c];
                if c + 1 == dim_in {
                    print!("{:+.6}{:+.6}i", v.re, v.im);
                } else {
                    print!("{:+.6}{:+.6}i, ", v.re, v.im);
                }
            }
            println!("]");
        }
    } else {
        let data = t.to_vec::<f64>()?;
        anyhow::ensure!(
            data.len() == expected_len,
            "matrix print: length mismatch (got {}, expected {})",
            data.len(),
            expected_len
        );
        for r in 0..dim_out {
            print!("[");
            for c in 0..dim_in {
                let v = data[r * dim_in + c];
                if c + 1 == dim_in {
                    print!("{:+.6}", v);
                } else {
                    print!("{:+.6}, ", v);
                }
            }
            println!("]");
        }
    }

    Ok(())
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

// Ensure the function is called in the main logic
fn main() -> anyhow::Result<()> {
    let n = 2usize;
    let phys_dim = 2usize;

    println!("=== Test: linsolve with pure imaginary Pauli-X operator (i*X) ===");
    println!("N = {n}, phys_dim = {phys_dim}");
    println!();
    println!("Test case 1: x = complex random MPO");
    println!("Test case 2: x = I (identity operator as state MPO)");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let site_indices: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    // Create Pauli-X operator and scale by pure imaginary i
    let (operator_x, s_in_tmp_x, s_out_tmp_x) =
        create_n_site_pauli_x_mpo_with_internal_indices(n, phys_dim, &mut used_ids)?;
    let (a_input_mapping_x, a_output_mapping_x) =
        create_index_mappings(&site_indices, &s_in_tmp_x, &s_out_tmp_x);

    // Scale Pauli-X by i (pure imaginary)
    let imag_scalar = AnyScalar::from(Complex64::new(0.0, 1.0));
    let operator_i_x = scale_treetn(&operator_x, imag_scalar.clone())?;
    print_bond_dims(
        &operator_i_x,
        "A = i*X (pure imaginary * Pauli-X operator) bond dimensions",
    );
    println!();

    // Print dense matrix representation of A = i*X
    print_operator_dense_matrix(&operator_i_x, &s_out_tmp_x, &s_in_tmp_x, "A = i*X")?;
    println!();

    // Test case 1: x = complex random MPO
    println!("=== Test case 1: A = i*X, x = complex random MPO ===");
    println!("Test setup: A = i*X (pure imaginary * Pauli-X operator), x = complex random MPO (state), b = A*x.");
    println!("Then solve (i*X)*x = b with init=x_true (exact solution).");
    println!(
        "This test demonstrates numerical instability: even when starting from the exact solution,"
    );
    println!("numerical errors can accumulate and cause the solution to diverge.");
    println!();

    // Create complex random MPO state x (the true solution)
    let (x_true_c64, _x_input_mapping_c64, _x_output_mapping_c64) =
        create_random_mpo_state_c64(n, phys_dim, &site_indices, &mut used_ids, 99999)?;
    print_bond_dims(
        &x_true_c64,
        "x_true (complex random MPO state) bond dimensions",
    );

    // Display the vector representation of x_true
    let x_true_tensor = x_true_c64.contract_to_tensor()?;
    let x_true_vec = x_true_tensor.to_vec::<Complex64>()?;

    // Display x_true as matrix (n=2 sites with phys_dim=2 each gives 4x4 matrix)
    println!(
        "x_true as matrix ({phys_dim}^{n} x {phys_dim}^{n} = {} x {}):",
        phys_dim.pow(n as u32),
        phys_dim.pow(n as u32)
    );
    let dim_per_axis = phys_dim.pow(n as u32);
    for r in 0..dim_per_axis {
        print!("[");
        for c in 0..dim_per_axis {
            let v = x_true_vec[r * dim_per_axis + c];
            if c + 1 == dim_per_axis {
                print!("{:+.4}{:+.4}i", v.re, v.im);
            } else {
                print!("{:+.4}{:+.4}i, ", v.re, v.im);
            }
        }
        println!("]");
    }
    println!();

    // Compute b = (i*X) * x_true_c64
    let linop_i_x = LinearOperator::new(
        operator_i_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
    );
    let b_tree = apply_linear_operator(&linop_i_x, &x_true_c64, ApplyOptions::default())?;
    print_bond_dims(&b_tree, "b = (i*X)*x_true bond dimensions");
    println!();

    // Set up linsolve options for a0=0, a1=1
    let options = LinsolveOptions::default()
        .with_nfullsweeps(10)
        .with_max_bond_dim(50)
        .with_gmres_tol(1e-10)
        .with_gmres_max_restarts(30)
        .with_gmres_restart_dim(30)
        .with_coefficients(0.0, 1.0); // a0=0, a1=1 => (i*X) * x = b

    let center = make_node_name(n / 2);
    let n_sweeps = 20usize;

    println!("--- Solving (i*X)*x = b with init=x_true (exact solution) ---");
    println!("    Initial value: x^(0) = x_true (exact solution)");
    let init = x_true_c64.clone();
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator_i_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree.clone(),
        options.clone(),
    );
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs, init_rel) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x,
        &b_tree,
    )?;
    println!(
        "    Initial residual: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        init_abs, init_rel
    );

    let (init_abs_state, init_rel_state) = compute_state_error(&x, &x_true_c64)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state, init_rel_state
    );

    // Use more sweeps for complex case
    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        // Print intermediate results every 5 sweeps
        if sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_i_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x,
                &b_tree,
            )?;
            let (_inter_abs_state, inter_rel_state) = compute_state_error(&x, &x_true_c64)?;
            println!("    After {sweep} sweeps: |(i*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
        // Print residual after the first sweep
        if sweep == 1 {
            let (first_abs, first_rel) = compute_residual(
                &operator_i_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x,
                &b_tree,
            )?;
            println!(
                "    After 1 sweep: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
                first_abs, first_rel
            );
        }
    }

    let (final_abs, final_rel) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x,
        &b_tree,
    )?;
    println!(
        "    After {} sweeps: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs, final_rel
    );

    let (final_abs_state, final_rel_state) = compute_state_error(&x, &x_true_c64)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state, final_rel_state
    );

    print_bond_dims(&x, "    x bond dimensions (final)");
    println!();

    // Test case 2: A = i*X, x = complex random MPO (same x_true_c64)
    println!("=== Test case 2: A = i*X, x = complex random MPO (same as test case 1) ===");
    println!("Test setup: A = i*X (pure imaginary * Pauli-X operator), x = complex random MPO (state), b = A*x.");
    println!(
        "This test uses the same x_true_c64 to verify consistency across different algorithms."
    );
    println!();

    // Compute b = (i*X) * x_true_c64 (same x_true as test case 1)
    let b_tree_2 = apply_linear_operator(&linop_i_x, &x_true_c64, ApplyOptions::default())?;
    print_bond_dims(&b_tree_2, "b = (i*X)*x_true bond dimensions");
    println!();

    println!("--- Solving (i*X)*x = b with init=x_true (exact solution) ---");
    let mut x_2 = x_true_c64.clone();
    x_2 = x_2.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_2 = SquareLinsolveUpdater::with_index_mappings(
        operator_i_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree_2.clone(),
        options.clone(),
    );
    let plan_2 = LocalUpdateSweepPlan::from_treetn(&x_2, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_2, init_rel_2) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2,
        &b_tree_2,
    )?;
    println!(
        "    Initial residual: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        init_abs_2, init_rel_2
    );

    let (init_abs_state_2, init_rel_state_2) = compute_state_error(&x_2, &x_true_c64)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_2, init_rel_state_2
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_2, &plan_2, &mut updater_2)?;
        if sweep == 1 || sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_i_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x_2,
                &b_tree_2,
            )?;
            let (_inter_abs_state, inter_rel_state) = compute_state_error(&x_2, &x_true_c64)?;
            println!("    After {sweep} sweeps: |(i*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
    }

    let (final_abs_2, final_rel_2) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2,
        &b_tree_2,
    )?;
    println!(
        "    After {} sweeps: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_2, final_rel_2
    );

    let (final_abs_state_2, final_rel_state_2) = compute_state_error(&x_2, &x_true_c64)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_2, final_rel_state_2
    );

    print_bond_dims(&x_2, "    x bond dimensions (final)");
    println!();

    // Test case 3: A = i*X, x = I (identity operator as state MPO)
    println!("=== Test case 3: A = i*X, x = I (identity operator as state MPO) ===");
    println!("Test setup: A = i*X (pure imaginary * Pauli-X operator), x = I (identity MPO state), b = (i*X)*I = i*X.");
    println!("Then solve (i*X)*x = b with init=I (identity operator).");
    println!("This test checks if the solver converges when x is the identity operator.");
    println!();

    // Create identity MPO state x (the true solution) - complex version
    let (x_true_identity, _x_input_mapping_identity, _x_output_mapping_identity) =
        create_identity_mpo_state_c64(n, phys_dim, &site_indices, &mut used_ids)?;
    print_bond_dims(
        &x_true_identity,
        "x_true (identity MPO state) bond dimensions",
    );
    println!();

    // Compute b = (i*X) * x_true_identity = (i*X) * I = i*X
    let b_tree_identity =
        apply_linear_operator(&linop_i_x, &x_true_identity, ApplyOptions::default())?;
    print_bond_dims(&b_tree_identity, "b = (i*X)*I = i*X bond dimensions");
    println!();

    println!("--- Solving (i*X)*x = b with init=I (identity operator) ---");
    println!("    Initial value: x^(0) = I (identity operator)");
    let init_identity = x_true_identity.clone();
    let mut x_identity =
        init_identity.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_identity = SquareLinsolveUpdater::with_index_mappings(
        operator_i_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree_identity.clone(),
        options.clone(),
    );
    let plan_identity = LocalUpdateSweepPlan::from_treetn(&x_identity, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_identity, init_rel_identity) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_identity,
        &b_tree_identity,
    )?;
    println!(
        "    Initial residual: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        init_abs_identity, init_rel_identity
    );

    let (init_abs_state_identity, init_rel_state_identity) =
        compute_state_error(&x_identity, &x_true_identity)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_identity, init_rel_state_identity
    );

    // Use more sweeps for complex case
    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_identity, &plan_identity, &mut updater_identity)?;
        // Print intermediate results every 5 sweeps
        if sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_i_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x_identity,
                &b_tree_identity,
            )?;
            let (_inter_abs_state, inter_rel_state) =
                compute_state_error(&x_identity, &x_true_identity)?;
            println!("    After {sweep} sweeps: |(i*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
    }

    let (final_abs_identity, final_rel_identity) = compute_residual(
        &operator_i_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_identity,
        &b_tree_identity,
    )?;
    println!(
        "    After {} sweeps: |(i*X)x - b| = {:.6e}, |(i*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_identity, final_rel_identity
    );

    let (final_abs_state_identity, final_rel_state_identity) =
        compute_state_error(&x_identity, &x_true_identity)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_identity, final_rel_state_identity
    );

    print_bond_dims(&x_identity, "    x bond dimensions (final)");
    println!();

    // Test case: Scale Pauli-X by 2 (real scalar)
    let real_scalar = AnyScalar::new_real(2.0);
    let operator_2_x = scale_treetn(&operator_x, real_scalar.clone())?;
    print_bond_dims(
        &operator_2_x,
        "A = 2*X (real scalar * Pauli-X operator) bond dimensions",
    );
    println!();

    // Print dense matrix representation of A = 2*X
    print_operator_dense_matrix(&operator_2_x, &s_out_tmp_x, &s_in_tmp_x, "A = 2*X")?;
    println!();

    // Additional test setup for A = 2*X
    println!("=== Test case: A = 2*X, x = complex random MPO ===");
    println!("Test setup: A = 2*X (real scalar * Pauli-X operator), x = complex random MPO (state), b = A*x.");
    println!("Then solve (2*X)*x = b with init=x_true (exact solution).\n");

    // Compute b = (2*X) * x_true_c64
    let linop_2_x = LinearOperator::new(
        operator_2_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
    );
    let b_tree_2_x = apply_linear_operator(&linop_2_x, &x_true_c64, ApplyOptions::default())?;
    print_bond_dims(&b_tree_2_x, "b = (2*X)*x_true bond dimensions");
    println!();

    // Solve (2*X)*x = b with init=x_true
    println!("--- Solving (2*X)*x = b with init=x_true (exact solution) ---");
    let mut x_2_x = x_true_c64.clone();
    // Canonicalize x_2_x before applying local update sweeps
    x_2_x = x_2_x.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_2_x = SquareLinsolveUpdater::with_index_mappings(
        operator_2_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree_2_x.clone(),
        options.clone(),
    );
    let plan_2_x = LocalUpdateSweepPlan::from_treetn(&x_2_x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_2_x, init_rel_2_x) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x,
        &b_tree_2_x,
    )?;
    println!(
        "    Initial residual: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        init_abs_2_x, init_rel_2_x
    );

    let (init_abs_state_2_x, init_rel_state_2_x) = compute_state_error(&x_2_x, &x_true_c64)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_2_x, init_rel_state_2_x
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_2_x, &plan_2_x, &mut updater_2_x)?;
        if sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_2_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x_2_x,
                &b_tree_2_x,
            )?;
            let (_inter_abs_state, inter_rel_state) = compute_state_error(&x_2_x, &x_true_c64)?;
            println!("    After {sweep} sweeps: |(2*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
    }

    let (final_abs_2_x, final_rel_2_x) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x,
        &b_tree_2_x,
    )?;
    println!(
        "    After {} sweeps: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_2_x, final_rel_2_x
    );

    let (final_abs_state_2_x, final_rel_state_2_x) = compute_state_error(&x_2_x, &x_true_c64)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_2_x, final_rel_state_2_x
    );

    print_bond_dims(&x_2_x, "    x bond dimensions (final)");
    println!();

    // Test case: A = 2*X, x_true = all-twos MPO state (2.0 * all-ones)
    println!("=== Test case: A = 2*X, x_true = All-twos MPO state ===");
    println!("Test setup: A = 2*X (real scalar * Pauli-X operator), x_true = all-twos MPO state (2.0 * all-ones), b = A*x_true.");
    println!("Then solve (2*X)*x = b with init=x_true (exact solution).\n");

    let (x_true_ones_state, _, _) =
        create_all_ones_mpo_state_c64(n, phys_dim, &site_indices, &mut used_ids)?;
    let x_true_twos_state = scale_treetn(&x_true_ones_state, AnyScalar::new_real(2.0))?;
    print_bond_dims(
        &x_true_twos_state,
        "x_true (all-twos MPO state) bond dimensions",
    );
    println!();

    let b_tree_2_x_twos =
        apply_linear_operator(&linop_2_x, &x_true_twos_state, ApplyOptions::default())?;
    print_bond_dims(&b_tree_2_x_twos, "b = (2*X)*x_true_twos bond dimensions");
    println!();

    println!("--- Solving (2*X)*x = b with init=x_true (all-twos MPO state) ---");
    let mut x_2_x_twos = x_true_twos_state.clone();
    x_2_x_twos = x_2_x_twos.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_2_x_twos = SquareLinsolveUpdater::with_index_mappings(
        operator_2_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree_2_x_twos.clone(),
        options.clone(),
    );
    let plan_2_x_twos = LocalUpdateSweepPlan::from_treetn(&x_2_x_twos, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_2_x_twos, init_rel_2_x_twos) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x_twos,
        &b_tree_2_x_twos,
    )?;
    println!(
        "    Initial residual: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        init_abs_2_x_twos, init_rel_2_x_twos
    );

    let (init_abs_state_2_x_twos, init_rel_state_2_x_twos) =
        compute_state_error(&x_2_x_twos, &x_true_twos_state)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_2_x_twos, init_rel_state_2_x_twos
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_2_x_twos, &plan_2_x_twos, &mut updater_2_x_twos)?;
        if sweep == 1 || sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_2_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x_2_x_twos,
                &b_tree_2_x_twos,
            )?;
            let (_inter_abs_state, inter_rel_state) =
                compute_state_error(&x_2_x_twos, &x_true_twos_state)?;
            println!("    After {sweep} sweeps: |(2*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
    }

    let (final_abs_2_x_twos, final_rel_2_x_twos) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x_twos,
        &b_tree_2_x_twos,
    )?;
    println!(
        "    After {} sweeps: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_2_x_twos, final_rel_2_x_twos
    );

    let (final_abs_state_2_x_twos, final_rel_state_2_x_twos) =
        compute_state_error(&x_2_x_twos, &x_true_twos_state)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_2_x_twos, final_rel_state_2_x_twos
    );

    print_bond_dims(&x_2_x_twos, "    x bond dimensions (final)");
    println!();

    // Test case: A = 2*X, x_true = all-pure imaginary i MPO state (i * all-ones)
    println!("=== Test case: A = 2*X, x_true = All-pure imaginary i MPO state ===");
    println!("Test setup: A = 2*X (real scalar * Pauli-X operator), x_true = all-pure imaginary i MPO state (i * all-ones), b = A*x_true.");
    println!("Then solve (2*X)*x = b with init=x_true (exact solution).\n");

    let imag_i = AnyScalar::from(Complex64::new(0.0, 1.0));
    let x_true_imag_state = scale_treetn(&x_true_ones_state, imag_i)?;
    print_bond_dims(
        &x_true_imag_state,
        "x_true (all-pure imaginary i MPO state) bond dimensions",
    );
    println!();

    let b_tree_2_x_imag =
        apply_linear_operator(&linop_2_x, &x_true_imag_state, ApplyOptions::default())?;
    print_bond_dims(&b_tree_2_x_imag, "b = (2*X)*x_true_imag bond dimensions");
    println!();

    println!("--- Solving (2*X)*x = b with init=x_true (all-pure imaginary i MPO state) ---");
    let mut x_2_x_imag = x_true_imag_state.clone();
    x_2_x_imag = x_2_x_imag.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let mut updater_2_x_imag = SquareLinsolveUpdater::with_index_mappings(
        operator_2_x.clone(),
        a_input_mapping_x.clone(),
        a_output_mapping_x.clone(),
        b_tree_2_x_imag.clone(),
        options.clone(),
    );
    let plan_2_x_imag = LocalUpdateSweepPlan::from_treetn(&x_2_x_imag, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let (init_abs_2_x_imag, init_rel_2_x_imag) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x_imag,
        &b_tree_2_x_imag,
    )?;
    println!(
        "    Initial residual: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        init_abs_2_x_imag, init_rel_2_x_imag
    );

    let (init_abs_state_2_x_imag, init_rel_state_2_x_imag) =
        compute_state_error(&x_2_x_imag, &x_true_imag_state)?;
    println!(
        "    Initial state error: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        init_abs_state_2_x_imag, init_rel_state_2_x_imag
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x_2_x_imag, &plan_2_x_imag, &mut updater_2_x_imag)?;
        if sweep == 1 || sweep % 5 == 0 || sweep == n_sweeps {
            let (_inter_abs, inter_rel) = compute_residual(
                &operator_2_x,
                &a_input_mapping_x,
                &a_output_mapping_x,
                0.0,
                1.0,
                &x_2_x_imag,
                &b_tree_2_x_imag,
            )?;
            let (_inter_abs_state, inter_rel_state) =
                compute_state_error(&x_2_x_imag, &x_true_imag_state)?;
            println!("    After {sweep} sweeps: |(2*X)x - b| / |b| = {:.6e}, |x - x_true| / |x_true| = {:.6e}", inter_rel, inter_rel_state);
        }
    }

    let (final_abs_2_x_imag, final_rel_2_x_imag) = compute_residual(
        &operator_2_x,
        &a_input_mapping_x,
        &a_output_mapping_x,
        0.0,
        1.0,
        &x_2_x_imag,
        &b_tree_2_x_imag,
    )?;
    println!(
        "    After {} sweeps: |(2*X)x - b| = {:.6e}, |(2*X)x - b| / |b| = {:.6e}",
        n_sweeps, final_abs_2_x_imag, final_rel_2_x_imag
    );

    let (final_abs_state_2_x_imag, final_rel_state_2_x_imag) =
        compute_state_error(&x_2_x_imag, &x_true_imag_state)?;
    println!(
        "    After {} sweeps: |x - x_true| = {:.6e}, |x - x_true| / |x_true| = {:.6e}",
        n_sweeps, final_abs_state_2_x_imag, final_rel_state_2_x_imag
    );

    print_bond_dims(&x_2_x_imag, "    x bond dimensions (final)");
    println!();

    println!("=== All tests completed ===");
    Ok(())
}
