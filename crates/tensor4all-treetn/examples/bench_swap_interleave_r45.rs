use std::collections::HashMap;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
use tensor4all_treetn::{SwapOptions, TreeTN};

fn build_tn(
    r: usize,
    bond_dim: usize,
) -> (TreeTN<TensorDynLen, String>, HashMap<DynIndex, String>) {
    let n = 2 * r;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let x_inds: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let y_inds: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..n - 1).map(|_| DynIndex::new_dyn(bond_dim)).collect();
    for i in 0..n {
        let site = if i < r {
            x_inds[i].clone()
        } else {
            y_inds[i - r].clone()
        };
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site);
        if i < n - 1 {
            indices.push(bonds[i].clone());
        }
        let size: usize = indices.iter().map(|idx| idx.dim()).product();
        let data: Vec<f64> = (0..size)
            .map(|_| (rng.next_u64() as f64) / (u64::MAX as f64))
            .collect();
        let t = TensorDynLen::from_dense(indices, data).unwrap();
        tn.add_tensor(i.to_string(), t).unwrap();
    }
    for (i, bond) in bonds.iter().enumerate() {
        let ni = tn.node_index(&i.to_string()).unwrap();
        let nj = tn.node_index(&(i + 1).to_string()).unwrap();
        tn.connect(ni, bond, nj, bond).unwrap();
    }
    let mut target = HashMap::new();
    for k in 0..r {
        target.insert(x_inds[k].clone(), (2 * k).to_string());
        target.insert(y_inds[k].clone(), (2 * k + 1).to_string());
    }
    (tn, target)
}

/// Frobenius relative error: ||after - before|| / ||before||
/// Uses sub() to align indices correctly before subtracting.
fn rel_error(before: &TensorDynLen, after: &TensorDynLen) -> f64 {
    let norm_before = before.norm().unwrap();
    if norm_before == 0.0 {
        return 0.0;
    }
    let diff = before.sub(after).expect("sub failed");
    diff.norm().unwrap() / norm_before
}

fn run_timing(r: usize, bond_dim: usize, options: &SwapOptions, label: &str) {
    let n = 2 * r;
    let (mut tn, target) = build_tn(r, bond_dim);
    let start = std::time::Instant::now();
    tn.swap_site_indices(&target, options).unwrap();
    let elapsed = start.elapsed();
    eprintln!("R={r:3} ({n:3} nodes) bond_dim={bond_dim} {label}: {elapsed:.3?}");
}

fn run_accuracy(r: usize, bond_dim: usize, options: &SwapOptions, label: &str) {
    let (mut tn, target) = build_tn(r, bond_dim);
    let before = tn.contract_to_tensor().unwrap();
    tn.swap_site_indices(&target, options).unwrap();
    let after = tn.contract_to_tensor().unwrap();
    let err = rel_error(&before, &after);
    eprintln!("R={r:3} bond_dim={bond_dim} {label}: max_rel_err={err:.2e}");
}

fn main() {
    eprintln!("=== timing: bond_dim=1, exact ===");
    for r in [5, 10, 15, 20, 25, 30, 45] {
        run_timing(r, 1, &SwapOptions::default(), "");
    }
    eprintln!("=== timing: bond_dim=5, exact (bond dims may grow) ===");
    for r in [5, 10, 15, 20, 25, 30] {
        run_timing(r, 5, &SwapOptions::default(), "");
    }
    eprintln!("=== timing: bond_dim=5, max_bond_dim=5 ===");
    for r in [5, 10, 15, 20, 25, 30, 45] {
        run_timing(
            r,
            5,
            &SwapOptions {
                max_bond_dim: Some(5),
                rtol: None,
            },
            "",
        );
    }
    eprintln!("=== timing: bond_dim=5, max_bond_dim=10 ===");
    for r in [5, 10, 15, 20, 25, 30, 45] {
        run_timing(
            r,
            5,
            &SwapOptions {
                max_bond_dim: Some(10),
                rtol: None,
            },
            "",
        );
    }

    eprintln!("=== accuracy (small R, contract full tensor) ===");
    for r in [3, 4, 5] {
        run_accuracy(r, 5, &SwapOptions::default(), "bond_dim=5 exact      ");
        run_accuracy(
            r,
            5,
            &SwapOptions {
                max_bond_dim: Some(5),
                rtol: None,
            },
            "bond_dim=5 max_bond_dim= 5",
        );
        run_accuracy(
            r,
            5,
            &SwapOptions {
                max_bond_dim: Some(10),
                rtol: None,
            },
            "bond_dim=5 max_bond_dim=10",
        );
    }
}
