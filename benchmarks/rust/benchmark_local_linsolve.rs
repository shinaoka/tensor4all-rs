// Benchmark isolated local GMRES and full two-site square_linsolve sweeps.
//
// Run:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release
//
// Optional args:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- <N> <state_bond_dim> <operator_bond_dim> <nsweeps> <gmres_max_restarts> <gmres_restart_dim> <step_index>

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{
    index::{DynId, Index},
    krylov::{gmres, GmresOptions},
    print_and_reset_contract_profile, reset_contract_profile,
    AnyScalar, DynIndex, IndexLike, SvdTruncationPolicy, TensorContractionLike, IdxTensor,
    TensorIndex,
};
use tensor4all_treetn::{
    square_linsolve, CanonicalizationOptions, IndexMapping, LinsolveOptions, LocalUpdateSweepPlan,
    ProjectedOperator, ProjectedState, TreeTN,
};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize, rng: &mut StdRng) -> DynIndex {
    loop {
        let id = DynId(rng.random());
        if used.insert(id) {
            return Index::new(id, dim);
        }
    }
}

fn chain_node_indices(n: usize, i: usize, bonds: &[DynIndex], sites: &[DynIndex]) -> Vec<DynIndex> {
    if n == 1 {
        vec![sites[i].clone()]
    } else if i == 0 {
        vec![sites[i].clone(), bonds[i].clone()]
    } else if i + 1 == n {
        vec![bonds[i - 1].clone(), sites[i].clone()]
    } else {
        vec![bonds[i - 1].clone(), sites[i].clone(), bonds[i].clone()]
    }
}

fn create_state_chain(
    n: usize,
    state_bond_dim: usize,
    acted_sites: &[DynIndex],
    spectator_sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    let mut tree = TreeTN::<IdxTensor, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, state_bond_dim, rng))
        .collect();

    let mut nodes = Vec::with_capacity(n);
    for (i, spectator_site) in spectator_sites.iter().enumerate().take(n) {
        let mut indices = chain_node_indices(n, i, &bonds, acted_sites);
        indices.insert(0, spectator_site.clone());
        let tensor = IdxTensor::random::<f64, _>(rng, indices)?;
        let node = tree.add_tensor(make_node_name(i), tensor)?;
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        tree.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])?;
    }

    Ok(tree)
}

#[allow(clippy::type_complexity)]
fn create_operator_chain(
    n: usize,
    phys_dim: usize,
    operator_bond_dim: usize,
    acted_sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    let mut tree = TreeTN::<IdxTensor, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, operator_bond_dim, rng))
        .collect();
    let s_in: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let indices = if n == 1 {
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
        };
        let tensor = IdxTensor::random::<f64, _>(rng, indices)?;
        let name = make_node_name(i);
        let node = tree.add_tensor(name.clone(), tensor)?;
        nodes.push(node);

        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: acted_sites[i].clone(),
                internal_index: s_in[i].clone(),
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: acted_sites[i].clone(),
                internal_index: s_out[i].clone(),
            },
        );
    }

    for i in 0..n.saturating_sub(1) {
        tree.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])?;
    }

    Ok((tree, input_mapping, output_mapping))
}

fn same_index_order(a: &[DynIndex], b: &[DynIndex]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(ai, bi)| ai == bi && ai.dim() == bi.dim())
}

fn same_index_set(a: &[DynIndex], b: &[DynIndex]) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|ai| b.iter().any(|bi| ai == bi && ai.dim() == bi.dim()))
}

fn align_rhs_to_init(init: &IdxTensor, rhs: IdxTensor) -> anyhow::Result<IdxTensor> {
    let init_indices = init.external_indices();
    let rhs_indices = rhs.external_indices();
    if !same_index_set(&init_indices, &rhs_indices) {
        anyhow::bail!(
            "RHS local index set does not match init: init={:?}, rhs={:?}",
            init_indices,
            rhs_indices
        );
    }
    if same_index_order(&init_indices, &rhs_indices) {
        Ok(rhs)
    } else {
        rhs.permuteinds(&init_indices).map_err(anyhow::Error::new)
    }
}

fn max_bond_dim(tree: &TreeTN<IdxTensor, String>) -> usize {
    tree.site_index_network()
        .edges()
        .filter_map(|(a, b)| tree.edge_between(&a, &b))
        .filter_map(|edge| tree.bond_index(edge))
        .map(|idx| idx.dim())
        .max()
        .unwrap_or(1)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_sites: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(38);
    let state_bond_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let operator_bond_dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let nfullsweeps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let gmres_max_restarts: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);
    let gmres_restart_dim: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(30);
    let step_index: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);

    anyhow::ensure!(
        n_sites >= 2,
        "N must be at least 2 for a two-site local step"
    );
    anyhow::ensure!(nfullsweeps > 0, "nsweeps must be greater than zero");
    anyhow::ensure!(gmres_max_restarts > 0, "gmres_max_restarts must be greater than zero");
    anyhow::ensure!(gmres_restart_dim > 0, "gmres_restart_dim must be greater than zero");

    let phys_dim = 2usize;
    let seed = 20260518_u64;
    let a0 = AnyScalar::new_real(1.0);
    let a1 = AnyScalar::new_real(0.01);
    let gmres_tol = 1.0e-30;
    let mut used_ids = HashSet::<DynId>::new();
    let mut rng = StdRng::seed_from_u64(seed);

    let acted_sites: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng))
        .collect();
    let spectator_sites: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng))
        .collect();

    let state_raw = create_state_chain(
        n_sites,
        state_bond_dim,
        &acted_sites,
        &spectator_sites,
        &mut used_ids,
        &mut rng,
    )?;
    let rhs = state_raw.clone();
    let (operator, input_mapping, output_mapping) = create_operator_chain(
        n_sites,
        phys_dim,
        operator_bond_dim,
        &acted_sites,
        &mut used_ids,
        &mut rng,
    )?;

    let center = make_node_name(n_sites / 2);
    let state = state_raw
        .clone()
        .canonicalize([center.clone()], CanonicalizationOptions::default())?;
    let reference_state = state.sim_link_indices()?;
    let plan = LocalUpdateSweepPlan::from_treetn(&state, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("failed to build two-site sweep plan"))?;
    let step = plan
        .steps
        .get(step_index % plan.steps.len())
        .ok_or_else(|| anyhow::anyhow!("empty sweep plan"))?;
    let local_tensor = state.extract_subtree(&step.nodes)?.contract_to_tensor()?;

    println!("=== Local GMRES / linsolve benchmark (Rust/tensor4all-rs) ===");
    println!("N = {n_sites}");
    println!("phys_dim = {phys_dim}");
    println!("state_bond_dim = {state_bond_dim}");
    println!("operator_bond_dim = {operator_bond_dim}");
    println!("nsweeps = {nfullsweeps}");
    println!("gmres_max_restarts = {gmres_max_restarts}");
    println!("gmres_restart_dim = {gmres_restart_dim}");
    println!("gmres_tol = {gmres_tol:.1e}");
    println!("coefficients = ({a0:?}, {a1:?})");
    println!("center = {center}");
    println!("sweep_plan_steps = {}", plan.steps.len());
    println!("step_index = {}", step_index % plan.steps.len());
    println!("step_nodes = {:?}", step.nodes);
    println!("local_dims = {:?}", local_tensor.dims());
    println!();

    reset_contract_profile();
    let mut projected_state = ProjectedState::new(rhs.clone());
    let rhs_start = Instant::now();
    let rhs_local_raw =
        projected_state.local_constant_term(&step.nodes, &state, state.site_index_network())?;
    let rhs_time = rhs_start.elapsed();
    let rhs_local = align_rhs_to_init(&local_tensor, rhs_local_raw)?;

    let projected_operator = RefCell::new(ProjectedOperator::with_index_mappings(
        operator.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
    ));
    let apply_count = Rc::new(Cell::new(0usize));
    let apply_time = Rc::new(RefCell::new(Duration::ZERO));
    let combine_time = Rc::new(RefCell::new(Duration::ZERO));
    let apply_count_ref = Rc::clone(&apply_count);
    let apply_time_ref = Rc::clone(&apply_time);
    let combine_time_ref = Rc::clone(&combine_time);

    let gmres_options = GmresOptions {
        max_iter: gmres_restart_dim,
        rtol: gmres_tol,
        max_restarts: gmres_max_restarts,
        verbose: false,
        check_true_residual: false,
    };

    let gmres_start = Instant::now();
    let gmres_result = gmres(
        |x: &IdxTensor| {
            apply_count_ref.set(apply_count_ref.get() + 1);
            let apply_start = Instant::now();
            let hx = projected_operator.borrow_mut().apply(
                black_box(x),
                &step.nodes,
                &state,
                &reference_state,
                state.site_index_network(),
            )?;
            *apply_time_ref.borrow_mut() += apply_start.elapsed();

            let combine_start = Instant::now();
            let y = x.axpby(a0.clone(), &hx, a1.clone())?;
            *combine_time_ref.borrow_mut() += combine_start.elapsed();
            Ok(y)
        },
        &rhs_local,
        &local_tensor,
        &gmres_options,
    )?;
    let gmres_time = gmres_start.elapsed();

    println!("--- Single local GMRES step ---");
    println!(
        "rhs projection: {:.3} ms, rhs_rank={}",
        rhs_time.as_secs_f64() * 1000.0,
        rhs_local.external_indices().len()
    );
    println!(
        "gmres total: {:.3} ms, iterations={}, converged={}, residual={:.3e}",
        gmres_time.as_secs_f64() * 1000.0,
        gmres_result.iterations,
        gmres_result.converged,
        gmres_result.residual_norm
    );
    println!("apply_count = {}", apply_count.get());
    println!(
        "projected apply inside GMRES: {:.3} ms",
        apply_time.borrow().as_secs_f64() * 1000.0
    );
    println!(
        "a0*x + a1*Hx combine: {:.3} ms",
        combine_time.borrow().as_secs_f64() * 1000.0
    );
    println!(
        "unaccounted GMRES/vector overhead: {:.3} ms",
        (gmres_time
            .saturating_sub(*apply_time.borrow())
            .saturating_sub(*combine_time.borrow()))
        .as_secs_f64()
            * 1000.0
    );
    println!();

    let options = LinsolveOptions::new(nfullsweeps)
        .with_coefficients(a0, a1)
        .with_gmres_tol(gmres_tol)
        .with_gmres_max_restarts(gmres_max_restarts)
        .with_gmres_restart_dim(gmres_restart_dim)
        .with_max_bond_dim(state_bond_dim)
        .with_svd_policy(SvdTruncationPolicy::new(0.0))
        .with_residual_check(false);

    let full_start = Instant::now();
    let full_result = square_linsolve(
        &operator,
        &rhs,
        state_raw,
        &center,
        options,
        Some(input_mapping),
        Some(output_mapping),
    )?;
    let full_time = full_start.elapsed();

    println!("--- Full two-site square_linsolve ---");
    println!(
        "total: {:.3} ms, sweeps={}, residual_reported={}",
        full_time.as_secs_f64() * 1000.0,
        full_result.sweeps,
        full_result.residual.is_some()
    );
    println!(
        "expected local update steps = {}",
        plan.steps.len() * nfullsweeps
    );
    println!(
        "solution max bond dim = {}",
        max_bond_dim(&full_result.solution)
    );
    print_and_reset_contract_profile();

    Ok(())
}
