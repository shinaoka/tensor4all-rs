use std::collections::HashMap;

use anyhow::Context;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{
    apply_local_update_sweep, ApplyOptions, CanonicalForm, CanonicalizationOptions, IndexMapping,
    LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater, TreeTN,
    TruncationOptions,
};

type MpsWithSiteIndices = (TreeTN<IdxTensor, String>, Vec<DynIndex>);
type MpoWithInternalIndices = (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>);

fn create_n_site_ones_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> anyhow::Result<MpsWithSiteIndices> {
    anyhow::ensure!(n_sites >= 2, "Need at least 2 sites");

    let mut mps = TreeTN::<IdxTensor, String>::new();

    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| {
            DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}"))
                .map_err(|e| anyhow::anyhow!("failed to create site index: {e:?}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| {
            DynIndex::new_dyn_with_tag(bond_dim, &format!("bond{i}"))
                .map_err(|e| anyhow::anyhow!("failed to create bond index: {e:?}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    for i in 0..n_sites {
        let name = format!("site{i}");
        let indices = if i == 0 {
            vec![site_indices[i].clone(), bond_indices[i].clone()]
        } else if i == n_sites - 1 {
            vec![bond_indices[i - 1].clone(), site_indices[i].clone()]
        } else {
            vec![
                bond_indices[i - 1].clone(),
                site_indices[i].clone(),
                bond_indices[i].clone(),
            ]
        };

        let nelem: usize = indices.iter().map(|idx| idx.dim).product();
        let tensor = IdxTensor::from_dense(indices, vec![1.0_f64; nelem]).unwrap();
        mps.add_tensor(name, tensor)?;
    }

    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mps.node_index(&name_i).context("missing node")?;
        let nj = mps.node_index(&name_j).context("missing node")?;
        mps.connect(ni, bond, nj, bond)?;
    }

    Ok((mps, site_indices))
}

fn create_identity_chain_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
) -> anyhow::Result<MpoWithInternalIndices> {
    anyhow::ensure!(n_sites >= 2, "Need at least 2 sites");

    let mut mpo = TreeTN::<IdxTensor, String>::new();

    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    let bond_indices: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let mut data = vec![0.0; phys_dim * phys_dim];
        for j in 0..phys_dim {
            data[j * phys_dim + j] = 1.0;
        }

        let tensor = if i == 0 {
            IdxTensor::from_dense(
                vec![
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else if i == n_sites - 1 {
            IdxTensor::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            IdxTensor::from_dense(
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
        mpo.add_tensor(name, tensor)?;
    }

    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mpo.node_index(&name_i).context("missing node")?;
        let nj = mpo.node_index(&name_j).context("missing node")?;
        mpo.connect(ni, bond, nj, bond)?;
    }

    Ok((mpo, s_in_tmp, s_out_tmp))
}

fn create_n_site_index_mappings(
    state_site_indices: &[DynIndex],
    s_in_tmp: &[DynIndex],
    s_out_tmp: &[DynIndex],
) -> (
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    let n_sites = state_site_indices.len();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for i in 0..n_sites {
        let site = format!("site{i}");
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

#[test]
fn issue192_regression_no_svd_nan_n5_identity_ones() -> anyhow::Result<()> {
    // This is the minimal reproducer found for the Issue #192 failure.
    let n_sites = 5usize;
    let phys_dim = 2usize;
    let bond_dim = 20usize;

    let a0 = 1.0_f64;
    let a1 = 0.0_f64;

    let n_sweeps = 2usize; // pre-fix it failed at sweep 2
    let cutoff = 1e-8_f64;
    let rtol = cutoff.sqrt();

    let gmres_tol = 1e-6_f64;
    let gmres_max_restarts = 20usize;
    let gmres_restart_dim = 30usize;

    let (rhs, site_indices) = create_n_site_ones_mps(n_sites, phys_dim, bond_dim)?;
    let (mpo, s_in_tmp, s_out_tmp) =
        create_identity_chain_mpo_with_internal_indices(n_sites, phys_dim)?;

    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    let linop = LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());

    // x0 = b
    let center = "site0".to_string();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x = rhs
        .clone()
        .canonicalize([center.clone()], canon_opts)
        .context("failed to canonicalize initial x0=b")?;

    let truncation = TruncationOptions::default()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
        .with_max_bond_dim(bond_dim);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(n_sweeps)
        .with_truncation(truncation)
        .with_gmres_tol(gmres_tol)
        .with_gmres_max_restarts(gmres_max_restarts)
        .with_gmres_restart_dim(gmres_restart_dim)
        .with_coefficients(a0, a1);

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        mpo,
        input_mapping,
        output_mapping,
        rhs.clone(),
        options,
    );

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)
            .with_context(|| format!("apply_local_update_sweep failed at sweep {sweep}"))?;
    }

    // Ensure the final state is finite.
    // If NaN/Inf sneaks in, contract_to_tensor().to_vec::<f64>() should reveal it.
    let x_full = x.contract_to_tensor().context("failed to contract x")?;
    let x_vec = x_full.to_vec::<f64>().context("failed to materialize x")?;
    anyhow::ensure!(x_vec.iter().all(|v| v.is_finite()), "x contains NaN/Inf");

    // Also ensure applying the linear operator does not introduce NaN/Inf.
    let ax = tensor4all_treetn::apply_linear_operator(&linop, &x, ApplyOptions::default())?;
    let ax_full = ax.contract_to_tensor()?;
    let ax_vec = ax_full.to_vec::<f64>()?;
    anyhow::ensure!(ax_vec.iter().all(|v| v.is_finite()), "Ax contains NaN/Inf");

    Ok(())
}
