//! CUDA SRC residency and parity matrix (issue #720 PR4b).
//!
//! Fixed/adaptive x chain/star-tree x final_svd on/off, all on two
//! same-context CUDA-resident trees: exact-context residency asserts plus
//! whole-result parity against the explicit-CPU reference. Rejection paths
//! (mixed host/CUDA, foreign contexts, legacy entry on resident inputs)
//! fail before algorithm work begins.

#![cfg(feature = "tenferro-cuda")]

use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    CudaExecutionContext, DynIndex, ExecutionContext, IdxTensor, TensorConstructionLike,
    TensorContractionLike,
};
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetn::contraction::{
    contract, contract_src_with_rng_in, ContractionOptions, SrcOptions,
};
use tensor4all_treetn::TreeTN;

/// Rebuild a host tree inside an explicit CPU context, preserving index
/// identities so topology reconnects. Supports f64 and Complex64 dense
/// fixtures.
fn scoped_cpu_copy(
    tree: &TreeTN<IdxTensor, String>,
    context: &ExecutionContext,
) -> TreeTN<IdxTensor, String> {
    use num_complex::Complex64;

    let names = tree.node_names();
    let tensors = names
        .iter()
        .map(|name| {
            let node = tree.node_index(name).unwrap();
            let tensor = tree.tensor(node).unwrap();
            let indices = tensor.indices().to_vec();
            if tensor.is_f64() {
                <IdxTensor as TensorConstructionLike>::from_dense_in(
                    context,
                    indices,
                    tensor.to_vec::<f64>().unwrap(),
                )
                .unwrap()
            } else {
                <IdxTensor as TensorConstructionLike>::from_dense_in(
                    context,
                    indices,
                    tensor.to_vec::<Complex64>().unwrap(),
                )
                .unwrap()
            }
        })
        .collect();
    TreeTN::from_tensors(tensors, names).unwrap()
}

fn cuda_context() -> (Arc<CudaExecutionContext>, ExecutionContext) {
    let cuda = Arc::new(CudaExecutionContext::new().expect("CUDA ordinal 0 must be available"));
    let context = ExecutionContext::Cuda(Arc::clone(&cuda));
    (cuda, context)
}

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

fn make_chain_pair() -> (TreeTN<IdxTensor, String>, TreeTN<IdxTensor, String>) {
    let shared = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let output_a = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let output_b = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds_a = (0..2).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds_b = (0..2).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let build = |bonds: &[DynIndex], outputs: &[DynIndex], num: Box<dyn Fn(usize) -> f64>| {
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(
                    vec![shared[0].clone(), outputs[0].clone(), bonds[0].clone()],
                    (0..8).map(|x| num(x)).collect(),
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![
                        bonds[0].clone(),
                        shared[1].clone(),
                        outputs[1].clone(),
                        bonds[1].clone(),
                    ],
                    (0..16).map(|x| num(x) / 3.0).collect(),
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![bonds[1].clone(), shared[2].clone(), outputs[2].clone()],
                    (0..8).map(|x| num(x) / 5.0).collect(),
                )
                .unwrap(),
            ],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap()
    };
    (
        build(&bonds_a, &output_a, Box::new(|x| x as f64 + 1.0)),
        build(&bonds_b, &output_b, Box::new(|x| x as f64 + 2.0)),
    )
}

fn make_star_pair() -> (TreeTN<IdxTensor, String>, TreeTN<IdxTensor, String>) {
    let shared = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    let output_a = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    let output_b = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    let bonds_a = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let bonds_b = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let names = vec!["C".to_string(), "L".to_string(), "R".to_string()];
    let build = |bonds: &[DynIndex; 2], outputs: &[DynIndex; 3], offset: f64| {
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(
                    vec![
                        shared[0].clone(),
                        outputs[0].clone(),
                        bonds[0].clone(),
                        bonds[1].clone(),
                    ],
                    (0..16).map(|i| offset + f64::from(i) / 10.0).collect(),
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![shared[1].clone(), outputs[1].clone(), bonds[0].clone()],
                    (0..8).map(|i| offset + f64::from(i) / 7.0).collect(),
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![shared[2].clone(), outputs[2].clone(), bonds[1].clone()],
                    (0..8).map(|i| offset + f64::from(i) / 6.0).collect(),
                )
                .unwrap(),
            ],
            names.clone(),
        )
        .unwrap()
    };
    (
        build(&bonds_a, &output_a, 1.0),
        build(&bonds_b, &output_b, 2.0),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_matrix_case(
    name: &str,
    tn_a: &TreeTN<IdxTensor, String>,
    tn_b: &TreeTN<IdxTensor, String>,
    center: &str,
    options: ContractionOptions,
    seed: u64,
    tolerance: f64,
) {
    let (cuda, context) = cuda_context();
    let cpu = cpu_context();

    let resident_a = tn_a.upload_cuda(&cuda).unwrap();
    let resident_b = tn_b.upload_cuda(&cuda).unwrap();
    resident_a.validate_context(&context).unwrap();
    resident_b.validate_context(&context).unwrap();

    // CUDA run.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let result = contract_src_with_rng_in(
        &resident_a,
        &resident_b,
        &center.to_string(),
        options.clone(),
        &mut rng,
        &context,
    )
    .unwrap_or_else(|error| panic!("{name}: CUDA SRC failed: {error:?}"));

    // Residency: every output node belongs to the exact caller-owned context,
    // and host reads fail without explicit download.
    result.validate_context(&context).unwrap();
    for node in result.node_indices() {
        let tensor = result.tensor(node).unwrap();
        assert!(
            tensor.to_vec::<f64>().is_err(),
            "{name}: node tensor is host-readable, residency lost"
        );
    }

    // Explicit-CPU reference with identical seeds (same scoped algorithm
    // family as the CUDA run; only the backend differs).
    let cpu_a = scoped_cpu_copy(tn_a, &cpu);
    let cpu_b = scoped_cpu_copy(tn_b, &cpu);
    let mut cpu_rng = ChaCha8Rng::seed_from_u64(seed);
    let reference = contract_src_with_rng_in(
        &cpu_a,
        &cpu_b,
        &center.to_string(),
        options,
        &mut cpu_rng,
        &cpu,
    )
    .unwrap_or_else(|error| panic!("{name}: CPU reference failed: {error:?}"));

    // Whole-result parity on complete index order. Values are compared as
    // host vectors (never combined across contexts); the downloaded tensor
    // is permuted onto the reference index order first.
    let got = result.download(&cuda).unwrap().to_dense().unwrap();
    let want = reference.to_dense().unwrap();
    let mut got_dims = got.dims();
    let mut want_dims = want.dims();
    got_dims.sort();
    want_dims.sort();
    assert_eq!(got_dims, want_dims, "{name}: dense shapes differ");
    let got = got.permuteinds(want.indices()).unwrap();
    assert_eq!(got.indices(), want.indices());
    let got_values = got.to_vec::<f64>().unwrap();
    let want_values = want.to_vec::<f64>().unwrap();
    assert_eq!(got_values.len(), want_values.len());
    let residual = got_values
        .iter()
        .zip(want_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        residual < tolerance,
        "{name}: CUDA/CPU parity residual {residual} exceeds {tolerance}"
    );
}

fn fixed_options(final_svd: bool) -> ContractionOptions {
    ContractionOptions::src()
        .with_max_bond_dim(4)
        .with_src_options(SrcOptions::fixed().with_final_svd(final_svd))
}

fn adaptive_options(final_svd: bool) -> ContractionOptions {
    ContractionOptions::src()
        .with_max_bond_dim(4)
        .with_src_options(
            SrcOptions::adaptive(1.0e-8, 4)
                .with_min_rank(1)
                .with_rank_increment(2)
                .with_final_svd(final_svd),
        )
}

#[test]
#[ignore]
fn cuda_src_fixed_chain_matches_cpu() {
    let (tn_a, tn_b) = make_chain_pair();
    run_matrix_case(
        "fixed/chain/no-final-svd",
        &tn_a,
        &tn_b,
        "C",
        fixed_options(false),
        11,
        1e-9,
    );
    run_matrix_case(
        "fixed/chain/final-svd",
        &tn_a,
        &tn_b,
        "C",
        fixed_options(true),
        11,
        1e-9,
    );
}

#[test]
#[ignore]
fn cuda_src_adaptive_chain_matches_cpu() {
    let (tn_a, tn_b) = make_chain_pair();
    run_matrix_case(
        "adaptive/chain/no-final-svd",
        &tn_a,
        &tn_b,
        "C",
        adaptive_options(false),
        13,
        1e-8,
    );
    run_matrix_case(
        "adaptive/chain/final-svd",
        &tn_a,
        &tn_b,
        "C",
        adaptive_options(true),
        13,
        1e-8,
    );
}

#[test]
#[ignore]
fn cuda_src_fixed_star_matches_cpu() {
    let (tn_a, tn_b) = make_star_pair();
    run_matrix_case(
        "fixed/star/no-final-svd",
        &tn_a,
        &tn_b,
        "C",
        fixed_options(false),
        17,
        1e-9,
    );
    run_matrix_case(
        "fixed/star/final-svd",
        &tn_a,
        &tn_b,
        "C",
        fixed_options(true),
        17,
        1e-9,
    );
}

#[test]
#[ignore]
fn cuda_src_adaptive_star_matches_cpu() {
    let (tn_a, tn_b) = make_star_pair();
    run_matrix_case(
        "adaptive/star/no-final-svd",
        &tn_a,
        &tn_b,
        "C",
        adaptive_options(false),
        19,
        1e-8,
    );
    run_matrix_case(
        "adaptive/star/final-svd",
        &tn_a,
        &tn_b,
        "C",
        adaptive_options(true),
        19,
        1e-8,
    );
}

#[test]
#[ignore]
fn cuda_src_rejects_mixed_and_foreign_inputs() {
    let (cuda, context) = cuda_context();
    let (tn_a, tn_b) = make_chain_pair();
    let resident_b = tn_b.upload_cuda(&cuda).unwrap();
    let options = fixed_options(false);

    // Mixed host/CUDA fails before work.
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    assert!(contract_src_with_rng_in(
        &tn_a,
        &resident_b,
        &"C".to_string(),
        options.clone(),
        &mut rng,
        &context
    )
    .is_err());

    // Foreign CUDA context fails before work.
    let foreign = ExecutionContext::Cuda(Arc::new(CudaExecutionContext::new().unwrap()));
    let resident_a = tn_a.upload_cuda(&cuda).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    assert!(contract_src_with_rng_in(
        &resident_a,
        &resident_b,
        &"C".to_string(),
        options.clone(),
        &mut rng,
        &foreign
    )
    .is_err());

    // Legacy context-free entry rejects resident inputs.
    assert!(contract(&resident_a, &resident_b, &"C".to_string(), options,).is_err());
}

#[test]
#[ignore]
fn cuda_src_c64_fixed_chain_probe() {
    use num_complex::Complex64;

    let (cuda, context) = cuda_context();
    let cpu = cpu_context();
    // Two-node C64 chain (mirrors the CPU complex fixture shape).
    let shared = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let output_a = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let output_b = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let bond_a = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let values = |offset: f64| {
        (0..8)
            .map(|i| Complex64::new(offset + f64::from(i), 0.25 * f64::from(i)))
            .collect::<Vec<_>>()
    };
    let build = |bonds: &[DynIndex; 1], outputs: &[DynIndex; 2], offset: f64| {
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(
                    vec![shared[0].clone(), outputs[0].clone(), bonds[0].clone()],
                    values(offset),
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![bonds[0].clone(), shared[1].clone(), outputs[1].clone()],
                    values(offset + 1.0),
                )
                .unwrap(),
            ],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap()
    };
    let tn_a = build(&[bond_a], &output_a, 1.0);
    let tn_b = build(&[bond_b], &output_b, 2.0);
    let resident_a = tn_a.upload_cuda(&cuda).unwrap();
    let resident_b = tn_b.upload_cuda(&cuda).unwrap();
    let options = ContractionOptions::src()
        .with_max_bond_dim(4)
        .with_src_options(SrcOptions::fixed());
    let mut rng = ChaCha8Rng::seed_from_u64(29);
    let result = contract_src_with_rng_in(
        &resident_a,
        &resident_b,
        &"B".to_string(),
        options.clone(),
        &mut rng,
        &context,
    );
    match result {
        Ok(result) => {
            result.validate_context(&context).unwrap();
            let cpu_a = scoped_cpu_copy(&tn_a, &cpu);
            let cpu_b = scoped_cpu_copy(&tn_b, &cpu);
            let mut cpu_rng = ChaCha8Rng::seed_from_u64(29);
            let reference = contract_src_with_rng_in(
                &cpu_a,
                &cpu_b,
                &"B".to_string(),
                options,
                &mut cpu_rng,
                &cpu,
            )
            .unwrap();
            let got: Vec<Complex64> = result
                .download(&cuda)
                .unwrap()
                .to_dense()
                .unwrap()
                .to_vec()
                .unwrap();
            let want: Vec<Complex64> = reference.to_dense().unwrap().to_vec().unwrap();
            assert_eq!(got.len(), want.len());
            let residual = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            println!("C64 chain CUDA/CPU parity residual: {residual:e}");
            assert!(residual < 1e-9, "C64 parity residual {residual}");
        }
        Err(error) => {
            panic!("C64 fixed CUDA SRC must be supported, got typed error: {error:?}");
        }
    }
}

/// Timing breakdown for the issue #720 evidence report: context setup,
/// upload, warm-up, synchronized steady-state, and download are reported
/// separately. Correctness/residency asserts stay outside the timed loops.
#[test]
#[ignore]
fn cuda_src_timing_breakdown() {
    use std::time::Instant;

    const REPS: usize = 5;

    for (name, options, seed) in [
        ("fixed/chain", fixed_options(false), 11_u64),
        ("adaptive/chain", adaptive_options(false), 13_u64),
    ] {
        let (tn_a, tn_b) = make_chain_pair();

        let start = Instant::now();
        let (cuda, context) = cuda_context();
        let setup_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let resident_a = tn_a.upload_cuda(&cuda).unwrap();
        let resident_b = tn_b.upload_cuda(&cuda).unwrap();
        let upload_ms = start.elapsed().as_secs_f64() * 1e3;

        // Warm-up (JIT + caches), discarded.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let warm = contract_src_with_rng_in(
            &resident_a,
            &resident_b,
            &"C".to_string(),
            options.clone(),
            &mut rng,
            &context,
        )
        .unwrap();
        cuda.synchronize().unwrap();
        let warm_ms = start.elapsed().as_secs_f64() * 1e3 - upload_ms;
        warm.validate_context(&context).unwrap();

        // Synchronized steady-state.
        let start = Instant::now();
        for _ in 0..REPS {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = contract_src_with_rng_in(
                &resident_a,
                &resident_b,
                &"C".to_string(),
                options.clone(),
                &mut rng,
                &context,
            )
            .unwrap();
            cuda.synchronize().unwrap();
            result.validate_context(&context).unwrap();
        }
        let steady_ms = start.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        // Explicit result download.
        let start = Instant::now();
        let back = warm.download(&cuda).unwrap();
        let _ = back.to_dense().unwrap();
        let download_ms = start.elapsed().as_secs_f64() * 1e3;

        // CPU steady-state reference (explicit context, same seeds).
        let cpu = cpu_context();
        let cpu_a = scoped_cpu_copy(&tn_a, &cpu);
        let cpu_b = scoped_cpu_copy(&tn_b, &cpu);
        let start = Instant::now();
        for _ in 0..REPS {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let _ = contract_src_with_rng_in(
                &cpu_a,
                &cpu_b,
                &"C".to_string(),
                options.clone(),
                &mut rng,
                &cpu,
            )
            .unwrap();
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        println!(
            "[{name}] setup={setup_ms:.1}ms upload={upload_ms:.1}ms warmup(first-run, incl. JIT)={warm_ms:.1}ms steady(CUDA,sync)={steady_ms:.2}ms download={download_ms:.1}ms cpu_steady={cpu_ms:.2}ms reps={REPS}"
        );
    }
}
