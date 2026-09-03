use std::hint::black_box;
use std::time::{Duration, Instant};

use tensor4all_core::{
    ContractionOptions, DynIndex, IdxTensor, PreparedContraction, TensorContractionLike,
};
use tensor4all_tensorbackend::{dense_native_tensor_from_col_major, einsum_native_tensors};

fn make_data(dims: &[usize], offset: usize) -> Vec<f64> {
    let len: usize = dims.iter().product();
    (0..len)
        .map(|i| (((i + offset) * 17 + 3) % 31) as f64 / 31.0 - 0.5)
        .collect()
}

fn make_dyn_indices(dims: &[usize]) -> Vec<DynIndex> {
    dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
}

fn time_best_of<R>(label: &str, repeats: usize, mut f: impl FnMut() -> R) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        for _ in 0..repeats {
            black_box(f());
        }
        best = best.min(started.elapsed());
    }

    let per_call_us = best.as_secs_f64() * 1e6 / repeats as f64;
    eprintln!(
        "  {label:<30} total={:.3}s  per_call={per_call_us:.3}us",
        best.as_secs_f64()
    );
    best
}

#[test]
#[ignore = "benchmark"]
fn bench_contract_fit_patterns_vs_native() {
    let env3_labels = make_dyn_indices(&[8, 8, 2, 16, 2, 8, 2, 8]);

    let env3_a_dims = [16, 8, 8];
    let env3_b_dims = [8, 2, 2, 8];
    let env3_c_dims = [8, 2, 2, 8];
    let env3_a_data = make_data(&env3_a_dims, 10);
    let env3_b_data = make_data(&env3_b_dims, 11);
    let env3_c_data = make_data(&env3_c_dims, 12);

    let env3_a = IdxTensor::from_dense(
        vec![
            env3_labels[3].clone(),
            env3_labels[0].clone(),
            env3_labels[1].clone(),
        ],
        env3_a_data.clone(),
    )
    .unwrap();
    let env3_b = IdxTensor::from_dense(
        vec![
            env3_labels[0].clone(),
            env3_labels[4].clone(),
            env3_labels[2].clone(),
            env3_labels[5].clone(),
        ],
        env3_b_data.clone(),
    )
    .unwrap();
    let env3_c = IdxTensor::from_dense(
        vec![
            env3_labels[1].clone(),
            env3_labels[2].clone(),
            env3_labels[6].clone(),
            env3_labels[7].clone(),
        ],
        env3_c_data.clone(),
    )
    .unwrap();

    let env3_a_native = dense_native_tensor_from_col_major(&env3_a_data, &env3_a_dims).unwrap();
    let env3_b_native = dense_native_tensor_from_col_major(&env3_b_data, &env3_b_dims).unwrap();
    let env3_c_native = dense_native_tensor_from_col_major(&env3_c_data, &env3_c_dims).unwrap();

    let env4_a_dims = [8, 2, 2, 8];
    let env4_b_dims = [8, 2, 2, 8];
    let env4_c_dims = [2, 2, 16, 16];
    let env4_d_dims = [8, 8, 16];
    let env4_a_data = make_data(&env4_a_dims, 20);
    let env4_b_data = make_data(&env4_b_dims, 21);
    let env4_c_data = make_data(&env4_c_dims, 22);
    let env4_d_data = make_data(&env4_d_dims, 23);

    let env4_labels = make_dyn_indices(&[2, 2, 8, 2, 8, 16, 8, 8, 16]);

    let env4_a = IdxTensor::from_dense(
        vec![
            env4_labels[6].clone(),
            env4_labels[1].clone(),
            env4_labels[0].clone(),
            env4_labels[2].clone(),
        ],
        env4_a_data.clone(),
    )
    .unwrap();
    let env4_b = IdxTensor::from_dense(
        vec![
            env4_labels[7].clone(),
            env4_labels[0].clone(),
            env4_labels[3].clone(),
            env4_labels[4].clone(),
        ],
        env4_b_data.clone(),
    )
    .unwrap();
    let env4_c = IdxTensor::from_dense(
        vec![
            env4_labels[1].clone(),
            env4_labels[3].clone(),
            env4_labels[5].clone(),
            env4_labels[8].clone(),
        ],
        env4_c_data.clone(),
    )
    .unwrap();
    let env4_d = IdxTensor::from_dense(
        vec![
            env4_labels[2].clone(),
            env4_labels[4].clone(),
            env4_labels[5].clone(),
        ],
        env4_d_data.clone(),
    )
    .unwrap();

    let env4_a_native = dense_native_tensor_from_col_major(&env4_a_data, &env4_a_dims).unwrap();
    let env4_b_native = dense_native_tensor_from_col_major(&env4_b_data, &env4_b_dims).unwrap();
    let env4_c_native = dense_native_tensor_from_col_major(&env4_c_data, &env4_c_dims).unwrap();
    let env4_d_native = dense_native_tensor_from_col_major(&env4_d_data, &env4_d_dims).unwrap();

    let env6_labels = make_dyn_indices(&[2, 8, 8, 8, 8, 2, 8, 8, 2, 2, 2, 2, 16, 16]);
    let env6_a_dims = [8, 2, 2, 8];
    let env6_b_dims = [8, 2, 2, 8];
    let env6_c_dims = [8, 2, 2, 8];
    let env6_d_dims = [8, 2, 2, 8];
    let env6_e_dims = [8, 8, 16];
    let env6_f_dims = [8, 8, 16];
    let env6_a_data = make_data(&env6_a_dims, 30);
    let env6_b_data = make_data(&env6_b_dims, 31);
    let env6_c_data = make_data(&env6_c_dims, 32);
    let env6_d_data = make_data(&env6_d_dims, 33);
    let env6_e_data = make_data(&env6_e_dims, 34);
    let env6_f_data = make_data(&env6_f_dims, 35);

    let env6_a = IdxTensor::from_dense(
        vec![
            env6_labels[2].clone(),
            env6_labels[8].clone(),
            env6_labels[0].clone(),
            env6_labels[1].clone(),
        ],
        env6_a_data.clone(),
    )
    .unwrap();
    let env6_b = IdxTensor::from_dense(
        vec![
            env6_labels[4].clone(),
            env6_labels[0].clone(),
            env6_labels[9].clone(),
            env6_labels[3].clone(),
        ],
        env6_b_data.clone(),
    )
    .unwrap();
    let env6_c = IdxTensor::from_dense(
        vec![
            env6_labels[1].clone(),
            env6_labels[10].clone(),
            env6_labels[5].clone(),
            env6_labels[6].clone(),
        ],
        env6_c_data.clone(),
    )
    .unwrap();
    let env6_d = IdxTensor::from_dense(
        vec![
            env6_labels[3].clone(),
            env6_labels[5].clone(),
            env6_labels[11].clone(),
            env6_labels[7].clone(),
        ],
        env6_d_data.clone(),
    )
    .unwrap();
    let env6_e = IdxTensor::from_dense(
        vec![
            env6_labels[2].clone(),
            env6_labels[4].clone(),
            env6_labels[12].clone(),
        ],
        env6_e_data.clone(),
    )
    .unwrap();
    let env6_f = IdxTensor::from_dense(
        vec![
            env6_labels[6].clone(),
            env6_labels[7].clone(),
            env6_labels[13].clone(),
        ],
        env6_f_data.clone(),
    )
    .unwrap();

    let env6_a_native = dense_native_tensor_from_col_major(&env6_a_data, &env6_a_dims).unwrap();
    let env6_b_native = dense_native_tensor_from_col_major(&env6_b_data, &env6_b_dims).unwrap();
    let env6_c_native = dense_native_tensor_from_col_major(&env6_c_data, &env6_c_dims).unwrap();
    let env6_d_native = dense_native_tensor_from_col_major(&env6_d_data, &env6_d_dims).unwrap();
    let env6_e_native = dense_native_tensor_from_col_major(&env6_e_data, &env6_e_dims).unwrap();
    let env6_f_native = dense_native_tensor_from_col_major(&env6_f_data, &env6_f_dims).unwrap();

    eprintln!("\n=== IdxTensor contract vs native einsum ===");
    let env3_plan =
        PreparedContraction::new(&[&env3_a, &env3_b, &env3_c], ContractionOptions::new()).unwrap();
    let env3_expected =
        <IdxTensor as TensorContractionLike>::contract(&[&env3_a, &env3_b, &env3_c]).unwrap();
    let env3_prepared_result = env3_plan.execute(&[&env3_a, &env3_b, &env3_c]).unwrap();
    assert!(env3_prepared_result
        .isapprox(&env3_expected, 1.0e-12, 0.0)
        .unwrap());
    let env3_contract = time_best_of("env3 IdxTensor::contract", 2_000, || {
        <IdxTensor as TensorContractionLike>::contract(&[&env3_a, &env3_b, &env3_c]).unwrap()
    });
    let env3_prepared = time_best_of("env3 PreparedContraction", 2_000, || {
        env3_plan.execute(&[&env3_a, &env3_b, &env3_c]).unwrap()
    });
    let env3_native = time_best_of("env3 native einsum", 2_000, || {
        einsum_native_tensors(
            &[
                (&env3_a_native, &[3, 0, 1]),
                (&env3_b_native, &[0, 4, 2, 5]),
                (&env3_c_native, &[1, 2, 6, 7]),
            ],
            &[3, 4, 5, 6, 7],
        )
        .unwrap()
    });

    let env4_contract = time_best_of("env4 IdxTensor::contract", 600, || {
        <IdxTensor as TensorContractionLike>::contract(&[&env4_a, &env4_b, &env4_c, &env4_d])
            .unwrap()
    });
    let env4_native = time_best_of("env4 native einsum", 600, || {
        einsum_native_tensors(
            &[
                (&env4_a_native, &[6, 1, 0, 2]),
                (&env4_b_native, &[7, 0, 3, 4]),
                (&env4_c_native, &[1, 3, 5, 8]),
                (&env4_d_native, &[2, 4, 5]),
            ],
            &[6, 7, 8],
        )
        .unwrap()
    });
    let env6_contract = time_best_of("env6 IdxTensor::contract", 400, || {
        <IdxTensor as TensorContractionLike>::contract(&[
            &env6_a, &env6_b, &env6_c, &env6_d, &env6_e, &env6_f,
        ])
        .unwrap()
    });
    let env6_native = time_best_of("env6 native einsum", 400, || {
        einsum_native_tensors(
            &[
                (&env6_a_native, &[2, 8, 0, 1]),
                (&env6_b_native, &[4, 0, 9, 3]),
                (&env6_c_native, &[1, 10, 5, 6]),
                (&env6_d_native, &[3, 5, 11, 7]),
                (&env6_e_native, &[2, 4, 12]),
                (&env6_f_native, &[6, 7, 13]),
            ],
            &[8, 9, 10, 11, 12, 13],
        )
        .unwrap()
    });

    eprintln!(
        "  ratio env3 IdxTensor/prepared     = {:.2}x",
        env3_contract.as_secs_f64() / env3_prepared.as_secs_f64()
    );
    eprintln!(
        "  ratio env3 IdxTensor/native       = {:.2}x",
        env3_contract.as_secs_f64() / env3_native.as_secs_f64()
    );
    eprintln!(
        "  ratio env4 IdxTensor/native       = {:.2}x",
        env4_contract.as_secs_f64() / env4_native.as_secs_f64()
    );
    eprintln!(
        "  ratio env6 IdxTensor/native       = {:.2}x",
        env6_contract.as_secs_f64() / env6_native.as_secs_f64()
    );
}
