//! Test: identity MPO A, random complex MPS x_true, b = A*x_true, solve A x = b (n=2)
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mps_identity_n2 --release

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{index::DynId, DynIndex, IdxTensor, IndexLike, TensorContractionLike};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

type MpoWithInternalIndices = (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>);

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

    for (i, bond) in bonds.iter().enumerate() {
        mps.connect(nodes[i], bond, nodes[i + 1], bond)?;
    }

    Ok(mps)
}

fn create_random_mps_chain_with_sites_real_c64(
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
        let mut data = Vec::with_capacity(nelem);
        for _ in 0..nelem {
            let r: f64 = rng.random();
            data.push(num_complex::Complex64::new(r, 0.0));
        }
        let t = IdxTensor::from_dense(indices, data).unwrap();
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for (i, bond) in bonds.iter().enumerate() {
        mps.connect(nodes[i], bond, nodes[i + 1], bond)?;
    }

    Ok(mps)
}

fn create_random_mps_chain_with_sites_imag_c64(
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
        let mut data = Vec::with_capacity(nelem);
        for _ in 0..nelem {
            let im: f64 = rng.random();
            data.push(num_complex::Complex64::new(0.0, im));
        }
        let t = IdxTensor::from_dense(indices, data).unwrap();
        let node = mps.add_tensor(make_node_name(i), t).unwrap();
        nodes.push(node);
    }

    for (i, bond) in bonds.iter().enumerate() {
        mps.connect(nodes[i], bond, nodes[i + 1], bond)?;
    }

    Ok(mps)
}

fn create_identity_mpo_with_internal_indices(
    n: usize,
    phys_dim: usize,
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<MpoWithInternalIndices> {
    anyhow::ensure!(n >= 1, "need at least 1 site");
    let mut mpo = TreeTN::<IdxTensor, String>::new();

    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();

    for i in 0..n {
        let name = make_node_name(i);
        // base identity on (out, in)
        let mut data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            data[k * phys_dim + k] = 1.0;
        }
        let base =
            IdxTensor::from_dense(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], data).unwrap();

        let tensor = if n == 1 {
            base
        } else if i == 0 {
            let ones = IdxTensor::from_dense(vec![bonds[i].clone()], vec![1.0_f64; 1]).unwrap();
            IdxTensor::outer_product(&base, &ones)?
        } else if i + 1 == n {
            let ones = IdxTensor::from_dense(vec![bonds[i - 1].clone()], vec![1.0_f64; 1]).unwrap();
            IdxTensor::outer_product(&ones, &base)?
        } else {
            let ones_left =
                IdxTensor::from_dense(vec![bonds[i - 1].clone()], vec![1.0_f64; 1]).unwrap();
            let ones_right =
                IdxTensor::from_dense(vec![bonds[i].clone()], vec![1.0_f64; 1]).unwrap();
            let t = IdxTensor::outer_product(&ones_left, &base)?;
            IdxTensor::outer_product(&t, &ones_right)?
        };

        mpo.add_tensor(name, tensor)?;
    }

    for (i, bond) in bonds.iter().enumerate() {
        let node_a = mpo.node_index(&make_node_name(i)).unwrap();
        let node_b = mpo.node_index(&make_node_name(i + 1)).unwrap();
        mpo.connect(node_a, bond, node_b, bond)?;
    }

    Ok((mpo, s_in_tmp, s_out_tmp))
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
    let ax_vec = ax_full.to_vec::<Complex64>()?;
    let b_vec = b_full.to_vec::<Complex64>()?;
    anyhow::ensure!(ax_vec.len() == b_vec.len(), "len mismatch");
    let mut diff2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for (a_i, b_i) in ax_vec.iter().zip(b_vec.iter()) {
        let r = a_i - *b_i;
        diff2 += r.norm_sqr();
        b2 += b_i.norm_sqr();
    }
    let abs = diff2.sqrt();
    let rel = if b2 > 0.0 { (diff2 / b2).sqrt() } else { 0.0 };
    Ok((abs, rel))
}

fn run_case(phys_dim: usize) -> anyhow::Result<()> {
    let n = 2usize;
    let bond_dim = 6usize;

    println!(
        "=== Test: identity MPO A, random MPS, n=2, phys_dim={} ===",
        phys_dim
    );

    let mut used = HashSet::<DynId>::new();
    let sites: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used, phys_dim))
        .collect();

    // operator: identity MPO
    let (mpo, s_in_tmp, s_out_tmp) =
        create_identity_mpo_with_internal_indices(n, phys_dim, &mut used)?;
    let (in_map, out_map) = {
        let mut im = HashMap::new();
        let mut om = HashMap::new();
        for i in 0..n {
            let name = make_node_name(i);
            im.insert(
                name.clone(),
                IndexMapping {
                    true_index: sites[i].clone(),
                    internal_index: s_in_tmp[i].clone(),
                },
            );
            om.insert(
                name.clone(),
                IndexMapping {
                    true_index: sites[i].clone(),
                    internal_index: s_out_tmp[i].clone(),
                },
            );
        }
        (im, om)
    };

    let operator = mpo;

    // x_true: random complex MPS
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut x_true =
        create_random_mps_chain_with_sites_c64(&mut rng, n, bond_dim, &sites, &mut used)?;
    let center = make_node_name(n / 2);
    x_true = x_true.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let b = apply_linear_operator(
        &LinearOperator::new(operator.clone(), in_map.clone(), out_map.clone()),
        &x_true,
        ApplyOptions::default(),
    )?;

    // Note: keep this example minimal. For debugging, add systematic tests instead of
    // extending examples on main.

    let options = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_bond_dim(bond_dim)
        .with_gmres_tol(1e-8);

    let mut x = x_true.clone();

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator.clone(),
        in_map.clone(),
        out_map.clone(),
        b.clone(),
        options.clone(),
    );
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("plan failed"))?;

    let (init_abs, init_rel) = compute_residual(&operator, &in_map, &out_map, &x, &b)?;
    println!(
            "    initial (identity MPO, random complex MPS, n=2, phys_dim={}): |Ax-b| = {:.6e}, rel = {:.6e}",
            phys_dim, init_abs, init_rel
        );

    for sweep in 1..=5 {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        if sweep == 1 || sweep == 5 {
            let (_abs, rel) = compute_residual(&operator, &in_map, &out_map, &x, &b)?;
            println!(
                "    After {sweep} sweeps (identity MPO, random complex MPS, n=2, phys_dim={}): |Ax-b|/|b| = {:.6e}",
                phys_dim, rel
            );
        }
    }

    // --- Now test with real random MPS (stored as Complex64 with zero imaginary parts) ---
    println!();
    println!(
        "=== Test: identity MPO A, real-random MPS (Complex64 imag=0), n=2, phys_dim={} ===",
        phys_dim
    );
    let mut used2 = used.clone();
    let mut rng2 = ChaCha8Rng::seed_from_u64(12345);
    let mut x_true_real =
        create_random_mps_chain_with_sites_real_c64(&mut rng2, n, bond_dim, &sites, &mut used2)?;
    x_true_real = x_true_real.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let b_real = apply_linear_operator(
        &LinearOperator::new(operator.clone(), in_map.clone(), out_map.clone()),
        &x_true_real,
        ApplyOptions::default(),
    )?;

    let mut x_real = x_true_real.clone();
    let mut updater_real = SquareLinsolveUpdater::with_index_mappings(
        operator.clone(),
        in_map.clone(),
        out_map.clone(),
        b_real.clone(),
        options.clone(),
    );
    let plan_real = LocalUpdateSweepPlan::from_treetn(&x_real, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("plan failed"))?;

    let (init_abs_r, init_rel_r) =
        compute_residual(&operator, &in_map, &out_map, &x_real, &b_real)?;
    println!(
        "    initial (identity MPO, real-random MPS, n=2, phys_dim={}): |Ax-b| = {:.6e}, rel = {:.6e}",
        phys_dim, init_abs_r, init_rel_r
    );

    for sweep in 1..=5 {
        apply_local_update_sweep(&mut x_real, &plan_real, &mut updater_real)?;
        if sweep == 1 || sweep == 5 {
            let (_abs, rel) = compute_residual(&operator, &in_map, &out_map, &x_real, &b_real)?;
            println!(
                "    After {sweep} sweeps (identity MPO, real-random MPS, n=2, phys_dim={}): |Ax-b|/|b| = {:.6e}",
                phys_dim, rel
            );
        }
    }

    // --- Fifth test: x_true all-pure-imaginary ---
    println!();
    println!(
        "=== Test: identity MPO A, pure-imaginary MPS (Complex64 imag!=0, real=0), n=2, phys_dim={} ===",
        phys_dim
    );
    let mut used3 = used.clone();
    let mut rng3 = ChaCha8Rng::seed_from_u64(98765);
    let mut x_true_imag =
        create_random_mps_chain_with_sites_imag_c64(&mut rng3, n, bond_dim, &sites, &mut used3)?;
    x_true_imag = x_true_imag.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let b_imag = apply_linear_operator(
        &LinearOperator::new(operator.clone(), in_map.clone(), out_map.clone()),
        &x_true_imag,
        ApplyOptions::default(),
    )?;

    let mut x_imag = x_true_imag.clone();
    let mut updater_imag = SquareLinsolveUpdater::with_index_mappings(
        operator.clone(),
        in_map.clone(),
        out_map.clone(),
        b_imag.clone(),
        options.clone(),
    );
    let plan_imag = LocalUpdateSweepPlan::from_treetn(&x_imag, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("plan failed"))?;

    let (init_abs_i, init_rel_i) =
        compute_residual(&operator, &in_map, &out_map, &x_imag, &b_imag)?;
    println!(
        "    initial (identity MPO, pure-imag MPS, n=2, phys_dim={}): |Ax-b| = {:.6e}, rel = {:.6e}",
        phys_dim, init_abs_i, init_rel_i
    );

    for sweep in 1..=5 {
        apply_local_update_sweep(&mut x_imag, &plan_imag, &mut updater_imag)?;
        if sweep == 1 || sweep == 5 {
            let (_abs, rel) = compute_residual(&operator, &in_map, &out_map, &x_imag, &b_imag)?;
            println!(
                "    After {sweep} sweeps (identity MPO, pure-imag MPS, n=2, phys_dim={}): |Ax-b|/|b| = {:.6e}",
                phys_dim, rel
            );
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    run_case(1)?;
    run_case(2)?;
    Ok(())
}
