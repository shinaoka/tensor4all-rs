// Controlled local-dimension experiment for issue #732. See benchmarks/README.md.
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use serde_json::{json, Value};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciOptions, TreeAciScalar};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEED: u64 = 732;
const TOLERANCE: f64 = 1e-8;
const CACHE_BYTES: usize = 16 * 1024 * 1024;

trait BenchScalar: TreeAciScalar {
    fn value(re: f64, im: f64) -> Self;
}
impl BenchScalar for f64 {
    fn value(re: f64, _: f64) -> Self {
        re
    }
}
impl BenchScalar for Complex64 {
    fn value(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }
}

// Fixed core coefficients across topologies. Only the factorization of their
// bond axes changes. Each fused leaf is exactly the tensor product of the same
// elementary leaves, so every degree represents the same global function.
fn fixture<T: BenchScalar>(
    degree: usize,
    atomic: &[usize; 4],
    d: usize,
    operand: usize,
    sites: &[DynIndex],
) -> Result<TreeTN<IdxTensor, usize>> {
    let mut groups: Vec<Vec<usize>> = (0..degree - 1).map(|axis| vec![axis]).collect();
    groups.push((degree - 1..4).collect());
    let bonds: Vec<_> = groups
        .iter()
        .map(|group| DynIndex::new_dyn(group.iter().map(|&axis| atomic[axis]).product()))
        .collect();
    let product: usize = atomic.iter().product();
    let mut hub_indices = vec![sites[0].clone()];
    hub_indices.extend(bonds.iter().cloned());
    let hub_values = (0..d * product)
        .map(|flat| {
            let phase = (flat + 19 * operand + SEED as usize) as f64;
            T::value(
                (0.17 * phase).sin() / (product as f64).sqrt(),
                0.3 * (0.31 * phase).cos() / (product as f64).sqrt(),
            )
        })
        .collect::<Vec<_>>();
    let mut tensors = vec![IdxTensor::from_dense(hub_indices, hub_values)?];
    for (arm, group) in groups.iter().enumerate() {
        let bond: usize = group.iter().map(|&axis| atomic[axis]).product();
        let physical = 1usize << group.len();
        let mut indices = vec![bonds[arm].clone()];
        indices.push(sites[arm + 1].clone());
        let values = (0..physical)
            .flat_map(|x| {
                (0..bond).map(move |a| {
                    let mut rest = a;
                    let mut value = T::value(1.0, 0.0);
                    for (position, &axis) in group.iter().enumerate() {
                        let coordinate = rest % atomic[axis];
                        rest /= atomic[axis];
                        let bit = (x >> position) & 1;
                        let phase = (coordinate * (bit + 1) + 7 * axis + 11 * operand) as f64;
                        value = value
                            * T::value(
                                0.5 + 0.4 * (0.37 * phase).cos(),
                                0.2 * (0.29 * phase).sin(),
                            );
                    }
                    value
                })
            })
            .collect::<Vec<_>>();
        tensors.push(IdxTensor::from_dense(indices, values)?);
    }
    Ok(TreeTN::from_tensors(tensors, (0..degree + 1).collect())?)
}

fn diagnostic_reset() {
    #[cfg(feature = "diagnostics")]
    tensor4all_treeaci::branch_diagnostics::reset();
}

fn diagnostic_rows() -> Vec<Value> {
    #[cfg(feature = "diagnostics")]
    {
        use tensor4all_treeaci::branch_diagnostics::{
            snapshot, BatchDiagnostics, KernelDiagnostics,
        };
        let batch = |b: BatchDiagnostics| json!({"calls":b.calls,"points":b.points,"min":b.min,"max":b.max});
        let kernel = |k: KernelDiagnostics| {
            json!({
            "setup_ns":k.setup_ns,"matmul_ns":k.matmul_ns,"accumulate_ns":k.accumulate_ns,
            "gather_ns":k.gather_ns,"matmul_calls":k.matmul_calls,"scalar_points":k.scalar_points,
            "prepared_hits":k.prepared_hits,"prepared_misses":k.prepared_misses,
            "prepared_refusals":k.prepared_refusals})
        };
        snapshot().into_iter().map(|row| json!({
            "node":row.node,"degree":row.coordination_number,"physical_dim":row.physical_dim,
            "bond_dims":row.bond_dims,"bond_product":row.bond_product,"local_elements":row.local_elements,
            "guard_ns":row.guard_ns,"frame_ns":row.frame_ns,"query_ns":row.query_ns,
            "guard_hits":row.guard_cache_hits,"guard_misses":row.guard_cache_misses,
            "frame_hits":row.frame_cache_hits,"frame_misses":row.frame_cache_misses,
            "guard_batches":batch(row.guard_batches),"frame_batches":batch(row.frame_batches),
            "query_batches":batch(row.query_batches),"guard_kernel":kernel(row.guard_kernel),
            "frame_kernel":kernel(row.frame_kernel),
            "query_cache": {
                "message_entries":row.query_cache.message_entries,
                "message_payload_bytes":row.query_cache.message_payload_bytes,
                "message_owned_bytes":row.query_cache.message_owned_bytes,
                "prepared_entries":row.query_cache.prepared_entries,
                "prepared_payload_bytes":row.query_cache.prepared_payload_bytes,
                "prepared_owned_bytes":row.query_cache.prepared_owned_bytes
            }
        })).collect()
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        Vec::new()
    }
}

fn run<T: BenchScalar>(
    degree: usize,
    profile: usize,
    d: usize,
    mode: &str,
    batch_size: usize,
    repeats: usize,
) -> Result<Value> {
    let profiles = [[2, 2, 2, 2], [2, 2, 4, 4], [2, 4, 4, 8], [4, 4, 8, 8]];
    let atomic = &profiles[profile];
    let local_dims: Vec<_> = std::iter::once(d)
        .chain(std::iter::repeat_n(2, degree - 1))
        .chain(std::iter::once(1usize << (5 - degree)))
        .collect();
    let sites: Vec<_> = local_dims.iter().copied().map(DynIndex::new_dyn).collect();
    let inputs = [
        fixture::<T>(degree, atomic, d, 0, &sites)?,
        fixture::<T>(degree, atomic, d, 1, &sites)?,
    ];
    let reference = inputs[0].to_dense()?.permute_indices(&sites)?;
    // A topology-independent oracle constructed at z=4. Materialize once.
    let reference_sites: Vec<_> = std::iter::once(d)
        .chain([2; 4])
        .map(DynIndex::new_dyn)
        .collect();
    let equivalent_values = fixture::<T>(4, atomic, d, 0, &reference_sites)?
        .to_dense()?
        .permute_indices(&reference_sites)?
        .to_vec::<T>()?;
    let equivalent = IdxTensor::from_dense(sites.clone(), equivalent_values)?;
    let scale = reference.maxabs()?;
    let topology_error = reference.sub(&equivalent)?.maxabs()? / scale;
    assert!(
        topology_error < 1e-11,
        "topology reshape residual {topology_error}"
    );
    let mut times = Vec::with_capacity(repeats);
    let mut max_error = topology_error;
    let mut evaluated_points = 0u64;
    let mut cache = json!({});
    let mut rows = Vec::new();

    if mode == "aci" {
        let right = inputs[1]
            .to_dense()?
            .permute_indices(&sites)?
            .to_vec::<T>()?;
        let expected = IdxTensor::from_dense(
            sites.clone(),
            reference
                .to_vec::<T>()?
                .into_iter()
                .zip(right)
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        )?;
        let options = TreeAciOptions {
            tolerance: TOLERANCE,
            min_sweeps: 2,
            max_sweeps: 12,
            rng_seed: SEED,
            nsearch_global_pivots: 30,
            max_nglobal_pivots: 30,
            message_cache_max_bytes: CACHE_BYTES,
            ..Default::default()
        };
        for iteration in 0..=repeats {
            diagnostic_reset();
            let started = Instant::now();
            let result = tree_elementwise_batched::<T, _, _>(
                |batch, output| {
                    for (point, out) in output.iter_mut().enumerate() {
                        *out = batch.get(0, point)? * batch.get(1, point)?;
                    }
                    Ok(())
                },
                black_box(&inputs),
                &options,
            )?;
            let elapsed = started.elapsed().as_nanos() as u64;
            let observed = diagnostic_rows();
            let error = result.tree.to_dense()?.sub(&expected)?.maxabs()? / expected.maxabs()?;
            assert!(error < 1e-7, "ACI residual {error}");
            black_box(&result.tree);
            max_error = max_error.max(error);
            if iteration > 0 {
                times.push(elapsed);
                rows.extend(observed);
                evaluated_points += result.diagnostics.evaluated_points;
            }
            cache = json!({"frame_records":result.diagnostics.frame_records,
                "frame_retained_bytes":result.diagnostics.frame_retained_bytes,
                "sample_arena_records":result.diagnostics.sample_arena_records,
                "sample_arena_retained_bytes":result.diagnostics.sample_arena_retained_bytes});
        }
    } else {
        let n_values = d * 16;
        let dense = reference.to_vec::<T>()?;
        let mut coordinates = Vec::with_capacity(local_dims.len() * batch_size);
        let mut expected_values = Vec::with_capacity(batch_size);
        for point in 0..batch_size {
            // Odd stride is a permutation when d=2; d=3 uses stride coprime to 48.
            let flat = (point * 7 + SEED as usize) % n_values;
            expected_values.push(dense[flat]);
            coordinates.push(flat % d);
            let mut tail = flat / d;
            for &dim in &local_dims[1..] {
                coordinates.push(tail % dim);
                tail /= dim;
            }
        }
        let point_index = DynIndex::new_dyn(batch_size);
        let expected = IdxTensor::from_dense(vec![point_index.clone()], expected_values)?;
        let batch_shape = [local_dims.len(), batch_size];
        let points = ColMajorArrayRef::new(&coordinates, &batch_shape)?;
        for iteration in 0..=repeats {
            let options = CachedEvaluatorOptions {
                center: Some(0),
                message_cache_max_bytes: CACHE_BYTES,
                branch_slice_cache_max_bytes: CACHE_BYTES,
                ..Default::default()
            };
            let mut evaluator = TreeTNCachedEvaluator::new(&inputs[0], &sites, options)?;
            if mode == "warm" {
                black_box(
                    evaluator.evaluate_batched_typed::<T>(points, EvaluationHint::around(0))?,
                );
            }
            diagnostic_reset();
            let started = Instant::now();
            let output = evaluator
                .evaluate_batched_typed::<T>(black_box(points), EvaluationHint::around(0))?;
            let elapsed = started.elapsed().as_nanos() as u64;
            black_box(&output);
            let observed = diagnostic_rows();
            let actual = IdxTensor::from_dense(vec![point_index.clone()], output)?;
            let error = actual.sub(&expected)?.maxabs()? / scale;
            assert!(error < 1e-11, "query residual {error}");
            max_error = max_error.max(error);
            if iteration > 0 {
                times.push(elapsed);
                rows.extend(observed);
                evaluated_points += batch_size as u64;
            }
        }
    }
    Ok(
        json!({"times_ns":times,"max_relative_error":max_error,"evaluated_points":evaluated_points,
        "cache":cache,"nodes":rows,"atomic_bonds":atomic,"hub_elements":d*atomic.iter().product::<usize>()}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "usage: benchmark_branch_cost DEGREE PROFILE D f64|c64 aci|cold|warm BATCH REPEATS"
                .into(),
        );
    }
    let degree: usize = args[0].parse()?;
    let profile: usize = args[1].parse()?;
    let d: usize = args[2].parse()?;
    let batch: usize = args[5].parse()?;
    let repeats: usize = args[6].parse()?;
    if !(2..=4).contains(&degree)
        || profile > 3
        || !(2..=3).contains(&d)
        || !(1..=256).contains(&batch)
        || !(1..=1000).contains(&repeats)
        || !["aci", "cold", "warm"].contains(&args[4].as_str())
    {
        return Err("case outside the bounded benchmark domain".into());
    }
    if cfg!(debug_assertions) {
        return Err("benchmark requires --release".into());
    }
    for key in [
        "RAYON_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        if std::env::var(key).as_deref() != Ok("1") {
            return Err(format!("set {key}=1").into());
        }
    }
    let mut result = match args[3].as_str() {
        "f64" => run::<f64>(degree, profile, d, &args[4], batch, repeats)?,
        "c64" => run::<Complex64>(degree, profile, d, &args[4], batch, repeats)?,
        _ => return Err("scalar must be f64 or c64".into()),
    };
    result["config"] = json!({"degree":degree,"profile":profile,"d":d,"scalar":args[3],
        "mode":args[4],"batch":batch,"repeats":repeats,"seed":SEED,"tolerance":TOLERANCE,
        "message_cache_budget":CACHE_BYTES,"diagnostics":cfg!(feature="diagnostics"),
        "build_commit":option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
        "release":true,"backend":"tenferro-cpu-faer"});
    println!("{result}");
    Ok(())
}
