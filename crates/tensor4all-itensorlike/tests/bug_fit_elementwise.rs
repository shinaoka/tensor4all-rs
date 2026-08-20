//! Bug: ContractOptions::fit() converges to wrong local minimum for element-wise
//! MPS products via diagonal embedding.
//!
//! fit() converges but to WRONG values, while zipup() is exact.
//! More sweeps don't help — error stays the same.
//!
//! This blocks using fit() for bubble computations in quanticsnegf-rs.

use tensor4all_core::{factorize, DynIndex, FactorizeOptions, IdxTensor, IndexLike};
use tensor4all_itensorlike::{ContractOptions, FitInitializer, TensorTrain, TruncateOptions};

/// Convert a column-major matrix buffer to quantics interleaved bit ordering.
fn matrix_to_quantics(nbit: usize, data: &[f64]) -> Vec<f64> {
    let n = 1 << nbit;
    assert_eq!(data.len(), n * n);
    let mut quantics = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut q_idx = 0;
            for b in 0..nbit {
                let r_bit = (row >> (nbit - 1 - b)) & 1;
                let c_bit = (col >> (nbit - 1 - b)) & 1;
                q_idx = q_idx * 2 + r_bit;
                q_idx = q_idx * 2 + c_bit;
            }
            quantics[q_idx] = data[row + n * col];
        }
    }
    quantics
}

/// Create a QTT representing a 2D function f(x,y) on [0,1)^2 grid
/// with interleaved quantics indices (row_0, col_0, row_1, col_1, ...).
fn create_function_2d_tt(
    row_sites: &[DynIndex],
    col_sites: &[DynIndex],
    f: impl Fn(f64, f64) -> f64,
    max_bond: usize,
) -> TensorTrain {
    let nbit = row_sites.len();
    let n = 1 << nbit;

    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i + n * j] = f(i as f64 / n as f64, j as f64 / n as f64);
        }
    }

    let quantics_data = matrix_to_quantics(nbit, &data);

    let mut all_sites = Vec::new();
    for i in 0..nbit {
        all_sites.push(row_sites[i].clone());
        all_sites.push(col_sites[i].clone());
    }

    let big = IdxTensor::from_dense(all_sites.clone(), quantics_data).unwrap();
    let opts = FactorizeOptions::qr().with_qr_rtol(0.0);
    let n_sites = all_sites.len();
    let mut remaining = big;
    let mut tensors = Vec::with_capacity(n_sites);

    for site_idx in all_sites.iter().take(n_sites - 1) {
        let remaining_indices = remaining.indices().to_vec();
        let mut left_inds = Vec::new();
        for idx in &remaining_indices {
            if idx.id() == site_idx.id() {
                left_inds.push(idx.clone());
                break;
            }
            left_inds.push(idx.clone());
        }
        let result = factorize(&remaining, &left_inds, &opts).unwrap();
        tensors.push(result.left);
        remaining = result.right;
    }
    tensors.push(remaining);
    let mut tt = TensorTrain::new(tensors).unwrap();
    if max_bond > 0 {
        let trunc = TruncateOptions::svd().with_max_bond_dim(max_bond);
        tt.truncate(&trunc).unwrap();
    }
    tt
}

/// Diagonalize: replace index `s` with [s_new1, s_new2], non-zero when s_new1==s_new2.
fn as_diagonal(
    tensor: &IdxTensor,
    s: &DynIndex,
    s_new1: &DynIndex,
    s_new2: &DynIndex,
) -> IdxTensor {
    let indices = tensor.indices();
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let dim = s.dim();
    let s_pos = indices.iter().position(|idx| idx.id() == s.id()).unwrap();

    let mut new_indices = indices.to_vec();
    new_indices.splice(s_pos..s_pos + 1, vec![s_new1.clone(), s_new2.clone()]);
    let mut new_dims = dims.clone();
    new_dims.splice(s_pos..s_pos + 1, vec![dim, dim]);

    let total_new: usize = new_dims.iter().product();
    let data = tensor.to_vec::<f64>().unwrap();
    let mut new_data = vec![0.0f64; total_new];

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut old_idx = vec![0usize; dims.len()];
        for i in 0..dims.len() {
            old_idx[i] = rem % dims[i];
            rem /= dims[i];
        }
        let s_val = old_idx[s_pos];
        let mut new_idx = old_idx;
        new_idx.splice(s_pos..s_pos + 1, vec![s_val, s_val]);
        let mut nf = 0;
        for i in (0..new_dims.len()).rev() {
            nf = nf * new_dims[i] + new_idx[i];
        }
        new_data[nf] = val;
    }
    IdxTensor::from_dense(new_indices, new_data).unwrap()
}

/// Extract diagonal: keep only s==s_result, remove s_result index.
fn extract_diagonal(tensor: &IdxTensor, s: &DynIndex, s_result: &DynIndex) -> IdxTensor {
    let indices = tensor.indices();
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let s_pos = indices.iter().position(|idx| idx.id() == s.id()).unwrap();
    let sr_pos = indices
        .iter()
        .position(|idx| idx.id() == s_result.id())
        .unwrap();

    let new_indices: Vec<DynIndex> = indices
        .iter()
        .filter(|idx| idx.id() != s_result.id())
        .cloned()
        .collect();
    let new_dims: Vec<usize> = new_indices.iter().map(|idx| idx.dim()).collect();
    let new_total: usize = new_dims.iter().product();
    let data = tensor.to_vec::<f64>().unwrap();
    let mut new_data = vec![0.0f64; new_total];

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut idx = vec![0usize; dims.len()];
        for i in 0..dims.len() {
            idx[i] = rem % dims[i];
            rem /= dims[i];
        }
        if idx[s_pos] != idx[sr_pos] {
            continue;
        }
        let new_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != sr_pos)
            .map(|(_, &v)| v)
            .collect();
        let mut nf = 0;
        for i in (0..new_dims.len()).rev() {
            nf = nf * new_dims[i] + new_idx[i];
        }
        new_data[nf] = val;
    }
    IdxTensor::from_dense(new_indices, new_data).unwrap()
}

/// Element-wise product via diagonal embedding + TT contraction.
///
/// For each site index s shared by m1 and m2:
///   m1: s → [s_result, s_contract] (diagonal)
///   m2: s → [s_contract, s] (diagonal)
/// Contract over s_contract, then extract diagonal s==s_result → result with [s].
fn elementwise_mul(
    m1: &TensorTrain,
    m2: &TensorTrain,
    sites: &[DynIndex],
    options: &ContractOptions,
) -> TensorTrain {
    let mut m1_prep = m1.clone();
    let mut m2_prep = m2.clone();
    let mut maps: Vec<(DynIndex, DynIndex)> = Vec::new();

    for s in sites {
        let s_contract = s.sim();
        let s_result = s.sim();

        let pos1 = m1_prep
            .site_indices()
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == s.id()))
            .unwrap();
        let pos2 = m2_prep
            .site_indices()
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == s.id()))
            .unwrap();

        let t1 = as_diagonal(m1_prep.tensor(pos1).unwrap(), s, &s_result, &s_contract);
        m1_prep.set_tensor(pos1, t1).unwrap();

        let t2 = as_diagonal(m2_prep.tensor(pos2).unwrap(), s, &s_contract, s);
        m2_prep.set_tensor(pos2, t2).unwrap();

        maps.push((s.clone(), s_result));
    }

    let mut result = m1_prep.contract(&m2_prep, options).unwrap();

    for (original, s_result) in &maps {
        let site_indices = result.site_indices();
        let pos = site_indices
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == original.id()))
            .unwrap();
        let t = result.tensor(pos).unwrap();
        let extracted = extract_diagonal(t, original, s_result);
        result.set_tensor(pos, extracted).unwrap();
    }
    result
}

/// Bug reproduction: fit() converges to WRONG answer for element-wise product
/// of structured QTTs with asymmetric bond dimensions.
///
/// Uses smooth 2D functions in QTT representation:
///   A(x,y) = exp(-3|x-y|)          (low rank, bond dim ~4)
///   B(x,y) = 1/((x-y)^2+0.01) - 1/((x-y)^2+0.1)  (higher rank, bond dim ~14)
///
/// Result: fit(A,B) consistently gives ~5e-4 relative error while
/// zipup gives near-zero error. The fit error is independent of maxdim,
/// indicating convergence to a wrong local minimum rather than truncation error.
///
/// This is the same mechanism causing O(1) errors in quanticsnegf-rs
/// bubble computations when using fit() for element-wise multiplication.
#[test]
fn test_fit_wrong_for_elementwise_structured() {
    let nbit = 3; // 8x8, 6 QTT sites

    let row_sites: Vec<DynIndex> = (0..nbit)
        .map(|i| DynIndex::new_dyn_with_tag(2, &format!("z1={}", i + 1)).unwrap())
        .collect();
    let col_sites: Vec<DynIndex> = (0..nbit)
        .map(|i| DynIndex::new_dyn_with_tag(2, &format!("z2={}", i + 1)).unwrap())
        .collect();

    let all_sites: Vec<DynIndex> = row_sites.iter().chain(col_sites.iter()).cloned().collect();

    // A: exponential decay (low rank)
    let tt_a = create_function_2d_tt(
        &row_sites,
        &col_sites,
        |x: f64, y: f64| (-3.0 * (x - y).abs()).exp(),
        10,
    );
    // B: rational function (higher rank, Green's function-like)
    let tt_b = create_function_2d_tt(
        &row_sites,
        &col_sites,
        |x: f64, y: f64| 1.0 / ((x - y).powi(2) + 0.01) - 1.0 / ((x - y).powi(2) + 0.1),
        30,
    );

    let a_max = (0..tt_a.len().saturating_sub(1))
        .map(|s| tt_a.linkind(s).map(|l| l.dim()).unwrap_or(0))
        .max()
        .unwrap_or(0);
    let b_max = (0..tt_b.len().saturating_sub(1))
        .map(|s| tt_b.linkind(s).map(|l| l.dim()).unwrap_or(0))
        .max()
        .unwrap_or(0);
    eprintln!("tt_a max bond: {a_max}, tt_b max bond: {b_max}");

    // Reference: zipup without truncation (exact)
    let result_ref = elementwise_mul(&tt_a, &tt_b, &all_sites, &ContractOptions::zipup());
    let ref_norm = result_ref.norm().unwrap();
    eprintln!("||ref|| = {:.6e}", ref_norm);

    // fit(A,B): this converges to wrong local minimum
    let result_fit = elementwise_mul(
        &tt_a,
        &tt_b,
        &all_sites,
        &ContractOptions::fit().with_initializer(FitInitializer::ZipUp),
    );
    let fit_err = result_fit
        .axpby(1.0.into(), &result_ref, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / ref_norm;

    // fit(B,A): swapped order should also work
    let result_fit_ba = elementwise_mul(
        &tt_b,
        &tt_a,
        &all_sites,
        &ContractOptions::fit().with_initializer(FitInitializer::ZipUp),
    );
    let fit_ba_err = result_fit_ba
        .axpby(1.0.into(), &result_ref, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / ref_norm;

    // fit(A,B) with explicit rtol=1e-30 (effectively zero, like ITensorMPS.jl benchmarks)
    let result_fit_rtol = elementwise_mul(
        &tt_a,
        &tt_b,
        &all_sites,
        &ContractOptions::fit()
            .with_initializer(FitInitializer::ZipUp)
            .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-30)),
    );
    let fit_rtol_err = result_fit_rtol
        .axpby(1.0.into(), &result_ref, (-1.0).into())
        .unwrap()
        .norm()
        .unwrap()
        / ref_norm;

    eprintln!("fit(A,B)            rel_err = {:.6e}", fit_err);
    eprintln!("fit(B,A)            rel_err = {:.6e}", fit_ba_err);
    eprintln!("fit(A,B,rtol=1e-30) rel_err = {:.6e}", fit_rtol_err);

    assert!(
        fit_err < 1e-4,
        "fit(A,B) converged to wrong local minimum: rel_err={:.6e}",
        fit_err
    );
    assert!(
        fit_ba_err < 1e-4,
        "fit(B,A) converged to wrong local minimum: rel_err={:.6e}",
        fit_ba_err
    );
    assert!(
        fit_rtol_err < 1e-4,
        "fit(A,B,rtol=1e-30) should not converge to wrong local minimum: rel_err={:.6e}",
        fit_rtol_err
    );
}
