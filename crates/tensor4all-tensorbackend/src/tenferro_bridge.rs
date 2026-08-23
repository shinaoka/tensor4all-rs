//! Bridge helpers between tensor4all storage snapshots and tenferro tensors.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use omeco::ScoreFunction;
use tenferro::{
    program::CoreSemanticOp, DType, GraphCompiler, Runtime, ScopedExecutionOutcome,
    ScopedReadInputs, Tensor as NativeTensor, TensorRead, TensorScalar, TensorSessionOpsExt,
    TensorView, TraceContext,
};
use tenferro_einsum::{
    ContractionOptimizerOptions, ContractionTree, EinsumSubscripts, Subscripts,
    TraceContextEinsumExt,
};
use tenferro_linalg::TensorLinalgExt;

use crate::any_scalar::promote_scalar_native;
/// Error returned by the storage/tensor bridge helpers.
///
/// Wraps the underlying tensor-element or backend diagnostic, preserving its
/// source chain.
///
/// # Remedies
/// - Dtype mismatch: extract with the element type matching the native tensor
///   dtype, or promote explicitly before the call.
/// - Shape mismatch: validate the native tensor shape against the payload
///   contract.
/// - Backend failure: the wrapped source chain identifies the failing stage.
#[derive(Debug, thiserror::Error)]
#[error("native tensor bridge operation failed: {source}")]
pub struct BridgeError {
    /// Original tensor-element or backend diagnostic.
    #[source]
    pub source: anyhow::Error,
}

impl From<anyhow::Error> for BridgeError {
    fn from(source: anyhow::Error) -> Self {
        Self { source }
    }
}

fn native_tensor_from_vec<T: TensorScalar>(
    shape: Vec<usize>,
    values: Vec<T>,
) -> std::result::Result<NativeTensor, BridgeError> {
    NativeTensor::from_vec_col_major(shape, values)
        .map_err(|source| BridgeError::from(anyhow::Error::new(source)))
}

use crate::context::{
    default_engine_buffer_pool_stats, reset_default_engine, reset_default_engine_buffer_pool,
    with_default_graph_runtime, with_default_session,
};
use crate::memory::release_process_allocator_cached_memory;
use crate::storage::Storage;
#[cfg(test)]
use crate::storage::StorageRepr;
use crate::tensor_element::TensorElement;
use crate::AnyScalar;

/// Read-only native tensor input that can either borrow external payload data
/// or own a temporary materialized tensor.
// Both tenferro handle variants are large; boxing the slightly larger one would
// add an allocation to the eager einsum path for little enum-size reduction.
#[allow(clippy::large_enum_variant)]
pub enum NativeTensorReadInput<'a> {
    /// Borrowed read-only tensor input.
    Borrowed(TensorRead<'a>),
    /// Owned temporary tensor input.
    Owned(NativeTensor),
}

impl<'a> NativeTensorReadInput<'a> {
    /// Return this input as a read-only tenferro tensor input.
    pub fn as_read(&'a self) -> TensorRead<'a> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
        }
    }

    /// Return the scalar dtype of this input.
    pub fn dtype(&self) -> DType {
        match self {
            Self::Borrowed(read) => read.dtype(),
            Self::Owned(tensor) => tensor.dtype(),
        }
    }

    /// Return the tensor shape of this input.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Borrowed(read) => read.shape(),
            Self::Owned(tensor) => tensor.shape(),
        }
    }
}

#[cfg(test)]
use std::cell::Cell;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum NativeEinsumPath {
    Owned,
    Borrowed,
    BorrowedWithConversions,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NativeOperandSignature {
    shape: Vec<usize>,
    ids: Vec<u32>,
    dtype: DType,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NativeEinsumSignature {
    path: NativeEinsumPath,
    operands: Vec<NativeOperandSignature>,
    output_ids: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
struct NativeEinsumProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static NATIVE_EINSUM_PROFILE_STATE: RefCell<HashMap<NativeEinsumSignature, NativeEinsumProfileEntry>> =
        RefCell::new(HashMap::new());
    static NATIVE_EINSUM_TRACE_STATE: RefCell<HashSet<NativeEinsumSignature>> =
        RefCell::new(HashSet::new());
}

#[cfg(test)]
thread_local! {
    static FORCE_NATIVE_EINSUM_PROFILE: Cell<bool> = const { Cell::new(false) };
}

fn native_einsum_profile_enabled() -> bool {
    #[cfg(test)]
    if FORCE_NATIVE_EINSUM_PROFILE.with(Cell::get) {
        return true;
    }
    env::var("T4A_PROFILE_NATIVE_EINSUM").is_ok()
}

fn native_einsum_path_trace_enabled() -> bool {
    env::var("T4A_TRACE_NATIVE_EINSUM_PATHS").is_ok()
}

fn native_einsum_path_trace_min_bytes() -> usize {
    env::var("T4A_TRACE_NATIVE_EINSUM_MIN_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn native_einsum_path_trace_max_signatures() -> usize {
    env::var("T4A_TRACE_NATIVE_EINSUM_MAX_SIGNATURES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64)
}

fn native_einsum_pool_trace_enabled() -> bool {
    env::var("T4A_TRACE_NATIVE_EINSUM_POOL").is_ok()
}

fn native_einsum_pool_trace_min_output_bytes() -> usize {
    env::var("T4A_TRACE_NATIVE_EINSUM_POOL_MIN_OUTPUT_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn native_einsum_pool_trace_min_retained_bytes() -> usize {
    env::var("T4A_TRACE_NATIVE_EINSUM_POOL_MIN_RETAINED_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn reset_native_einsum_engine_after_call() -> bool {
    env::var("T4A_RESET_NATIVE_EINSUM_ENGINE_AFTER_CALL").is_ok()
}

fn reset_native_einsum_buffer_pool_after_call() -> bool {
    env::var("T4A_RESET_NATIVE_EINSUM_BUFFER_POOL_AFTER_CALL").is_ok()
}

fn release_allocator_after_native_einsum_call() -> bool {
    env::var("T4A_RELEASE_ALLOCATOR_AFTER_NATIVE_EINSUM_CALL").is_ok()
}

#[cfg(test)]
pub(crate) fn set_native_einsum_profile_enabled_for_tests(enabled: bool) {
    FORCE_NATIVE_EINSUM_PROFILE.with(|slot| slot.set(enabled));
}

fn checked_native_einsum_labels(labels: &[usize]) -> Result<Vec<u32>> {
    labels
        .iter()
        .copied()
        .map(|label| {
            u32::try_from(label)
                .map_err(|_| anyhow!("native einsum label {label} exceeds the supported u32 range"))
        })
        .collect()
}

fn native_einsum_signature(
    path: NativeEinsumPath,
    operands: &[(&NativeTensor, &[u32])],
    output_ids: &[u32],
) -> NativeEinsumSignature {
    NativeEinsumSignature {
        path,
        operands: operands
            .iter()
            .map(|(tensor, ids)| NativeOperandSignature {
                shape: tensor.shape().to_vec(),
                ids: ids.to_vec(),
                dtype: tensor.dtype(),
            })
            .collect(),
        output_ids: output_ids.to_vec(),
    }
}

fn record_native_einsum_profile(
    path: NativeEinsumPath,
    operands: &[(&NativeTensor, &[u32])],
    output_ids: &[u32],
    elapsed: Duration,
) {
    if !native_einsum_profile_enabled() {
        return;
    }
    let signature = native_einsum_signature(path, operands, output_ids);
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(signature).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

fn native_slice<'a, T: TensorScalar>(
    tensor: &'a NativeTensor,
    label: &'static str,
) -> Result<&'a [T]> {
    tensor
        .as_slice::<T>()
        .map_err(|error| anyhow!("{label}: {error}"))
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::C32 => 8,
        DType::C64 => 16,
        DType::I32 => 4,
        DType::I64 => 8,
        DType::Bool => 1,
    }
}

fn native_tensor_bytes(tensor: &NativeTensor) -> usize {
    tensor
        .shape()
        .iter()
        .copied()
        .fold(1usize, usize::saturating_mul)
        .saturating_mul(dtype_size_bytes(tensor.dtype()))
}

fn format_label(label: u32) -> String {
    char::from_u32(label).map_or_else(|| label.to_string(), |label| label.to_string())
}

fn format_labels(labels: &[u32]) -> String {
    if labels.is_empty() {
        "scalar".to_string()
    } else {
        labels
            .iter()
            .map(|&label| format_label(label))
            .collect::<Vec<_>>()
            .join("")
    }
}

fn label_dims(subscripts: &Subscripts, shapes: &[Vec<usize>]) -> Result<HashMap<u32, usize>> {
    let mut dims = HashMap::new();
    for (labels, shape) in subscripts.inputs.iter().zip(shapes.iter()) {
        ensure!(
            labels.len() == shape.len(),
            "einsum labels {:?} do not match shape {:?}",
            labels,
            shape
        );
        for (&label, &dim) in labels.iter().zip(shape.iter()) {
            if let Some(previous) = dims.insert(label, dim) {
                ensure!(
                    previous == dim,
                    "inconsistent dimension for einsum label {}: {} vs {}",
                    format_label(label),
                    previous,
                    dim
                );
            }
        }
    }
    Ok(dims)
}

fn labels_size(labels: &[u32], dims: &HashMap<u32, usize>) -> usize {
    labels.iter().fold(1usize, |size, label| {
        size.saturating_mul(dims.get(label).copied().unwrap_or(1))
    })
}

fn union_labels(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut labels = Vec::new();
    for &label in lhs.iter().chain(rhs.iter()) {
        if seen.insert(label) {
            labels.push(label);
        }
    }
    labels
}

#[derive(Debug)]
struct NativeEinsumPlanReport {
    lines: Vec<String>,
    peak_intermediate_bytes: usize,
}

fn time_optimized_contraction_options() -> ContractionOptimizerOptions {
    ContractionOptimizerOptions {
        score: ScoreFunction::time_optimized(),
        ..ContractionOptimizerOptions::default()
    }
}

fn native_einsum_plan_report_with_options(
    signature: &NativeEinsumSignature,
    optimizer_name: &'static str,
    options: &ContractionOptimizerOptions,
) -> Result<NativeEinsumPlanReport> {
    let input_ids = signature
        .operands
        .iter()
        .map(|operand| operand.ids.as_slice())
        .collect::<Vec<_>>();
    let subscripts_string = build_einsum_subscripts(&input_ids, &signature.output_ids)?;
    let subscripts = Subscripts {
        inputs: input_ids.iter().map(|ids| ids.to_vec()).collect(),
        output: signature.output_ids.clone(),
    };
    let shapes = signature
        .operands
        .iter()
        .map(|operand| operand.shape.clone())
        .collect::<Vec<_>>();
    let shape_refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let tree = ContractionTree::optimize_with_options(&subscripts, &shape_refs, options)
        .map_err(|e| anyhow!("failed to optimize native einsum path: {e}"))?;
    let dims = label_dims(&subscripts, &shapes)?;
    let dtype = signature
        .operands
        .first()
        .map(|operand| operand.dtype)
        .unwrap_or(DType::F64);
    let dtype_size = dtype_size_bytes(dtype);

    let mut lines = Vec::new();
    lines.push(format!(
        "optimizer={optimizer_name} subscripts={subscripts_string} dtype={dtype:?} steps={}",
        tree.step_count()
    ));
    let mut peak_intermediate_elems = 1usize;
    for step in 0..tree.step_count() {
        let Some((left, right)) = tree.step_pair(step) else {
            continue;
        };
        let Some((lhs, rhs, out)) = tree.step_subscripts(step) else {
            continue;
        };
        let lhs_elems = labels_size(lhs, &dims);
        let rhs_elems = labels_size(rhs, &dims);
        let out_elems = labels_size(out, &dims);
        let flop_index_elems = labels_size(&union_labels(lhs, rhs), &dims);
        peak_intermediate_elems = peak_intermediate_elems.max(out_elems);
        lines.push(format!(
            "  step {step:02}: pair=({left},{right}) {}[{}] x {}[{}] -> {}[{}]  flop_index={}  intermediate={} elems ({:.3} MiB)",
            format_labels(lhs),
            lhs_elems,
            format_labels(rhs),
            rhs_elems,
            format_labels(out),
            out_elems,
            flop_index_elems,
            out_elems,
            out_elems as f64 * dtype_size as f64 / (1024.0 * 1024.0),
        ));
    }
    let peak_intermediate_bytes = peak_intermediate_elems.saturating_mul(dtype_size);
    lines.push(format!(
        "  peak_intermediate={} elems ({:.3} MiB)",
        peak_intermediate_elems,
        peak_intermediate_bytes as f64 / (1024.0 * 1024.0)
    ));

    Ok(NativeEinsumPlanReport {
        lines,
        peak_intermediate_bytes,
    })
}

fn native_einsum_time_optimized_plan_report(
    signature: &NativeEinsumSignature,
) -> Result<NativeEinsumPlanReport> {
    native_einsum_plan_report_with_options(
        signature,
        "time_optimized",
        &time_optimized_contraction_options(),
    )
}

fn native_einsum_balanced_plan_report(
    signature: &NativeEinsumSignature,
) -> Result<NativeEinsumPlanReport> {
    native_einsum_plan_report_with_options(
        signature,
        "balanced_default",
        &ContractionOptimizerOptions::default(),
    )
}

fn maybe_trace_native_einsum_path(
    path: NativeEinsumPath,
    operands: &[(&NativeTensor, &[u32])],
    output_ids: &[u32],
) {
    if !native_einsum_path_trace_enabled() {
        return;
    }
    let signature = native_einsum_signature(path, operands, output_ids);
    let report = match native_einsum_time_optimized_plan_report(&signature) {
        Ok(report) if report.peak_intermediate_bytes >= native_einsum_path_trace_min_bytes() => {
            report
        }
        Ok(_) => return,
        Err(err) => {
            eprintln!("native_einsum path trace failed: {err:#}");
            return;
        }
    };

    let max_signatures = native_einsum_path_trace_max_signatures();
    let should_trace = NATIVE_EINSUM_TRACE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.len() >= max_signatures || state.contains(&signature) {
            false
        } else {
            state.insert(signature.clone());
            true
        }
    });
    if !should_trace {
        return;
    }

    eprintln!("=== native_einsum Path Trace ===");
    eprintln!(
        "path={:?} output_ids={:?}",
        signature.path, signature.output_ids
    );
    for operand in &signature.operands {
        eprintln!(
            "  operand shape={:?} ids={:?} dtype={:?}",
            operand.shape, operand.ids, operand.dtype
        );
    }
    for line in report.lines {
        eprintln!("{line}");
    }
    if env::var("T4A_TRACE_NATIVE_EINSUM_COMPARE_BALANCED").is_ok() {
        match native_einsum_balanced_plan_report(&signature) {
            Ok(balanced) => {
                for line in balanced.lines {
                    eprintln!("{line}");
                }
            }
            Err(err) => eprintln!("balanced native_einsum path trace failed: {err:#}"),
        }
    }
}

/// Reset the aggregated native einsum profile.
pub fn reset_native_einsum_profile() {
    NATIVE_EINSUM_PROFILE_STATE.with(|state| state.borrow_mut().clear());
    NATIVE_EINSUM_TRACE_STATE.with(|state| state.borrow_mut().clear());
}

/// Print and clear the aggregated native einsum profile.
pub fn print_and_reset_native_einsum_profile() {
    if !native_einsum_profile_enabled() {
        return;
    }
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== native_einsum Profile ===");
        for (idx, (signature, entry)) in entries.into_iter().take(20).enumerate() {
            eprintln!(
                "#{idx:02} path={:?} calls={} total={:.3}s per_call={:.3}us output_ids={:?}",
                signature.path,
                entry.calls,
                entry.total_time.as_secs_f64(),
                entry.total_time.as_secs_f64() * 1e6 / entry.calls as f64,
                signature.output_ids,
            );
            for operand in &signature.operands {
                eprintln!(
                    "     shape={:?} ids={:?} dtype={:?}",
                    operand.shape, operand.ids, operand.dtype
                );
            }
            match native_einsum_time_optimized_plan_report(&signature) {
                Ok(report) => {
                    for line in report.lines {
                        eprintln!("     {line}");
                    }
                }
                Err(err) => eprintln!("     path report failed: {err:#}"),
            }
        }
    });
}

fn common_dtype(dtypes: &[DType]) -> DType {
    let has_f64 = dtypes.contains(&DType::F64);
    let has_c64 = dtypes.contains(&DType::C64);
    let has_c32 = dtypes.contains(&DType::C32);
    let has_i32 = dtypes.contains(&DType::I32);
    let has_i64 = dtypes.contains(&DType::I64);
    let has_bool = dtypes.contains(&DType::Bool);
    let has_complex = has_c64 || has_c32;
    if has_c64 || (has_f64 && has_complex) {
        DType::C64
    } else if has_c32 {
        DType::C32
    } else if has_f64 || has_i64 || has_i32 {
        DType::F64
    } else if has_bool {
        DType::Bool
    } else {
        DType::F32
    }
}

fn convert_tensor(tensor: &NativeTensor, to: DType) -> Result<NativeTensor> {
    if tensor.dtype() == to {
        return tensor
            .duplicate()
            .map_err(|e| anyhow!("tensor duplication failed: {e}"));
    }
    with_default_session(|session| tensor.convert(to, session))
        .map_err(|e| anyhow!("tensor conversion to {to:?} failed: {e}"))
}

fn ids_to_subscript(ids: &[u32]) -> Result<String> {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut out = String::with_capacity(ids.len());
    for &id in ids {
        let idx = usize::try_from(id).unwrap_or(usize::MAX);
        let letter = LETTERS
            .get(idx)
            .ok_or_else(|| anyhow!("einsum label {id} exceeds supported label range"))?;
        out.push(char::from(*letter));
    }
    Ok(out)
}

fn build_einsum_subscripts(operands: &[&[u32]], output_ids: &[u32]) -> Result<String> {
    let inputs = operands
        .iter()
        .map(|ids| ids_to_subscript(ids))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "{}->{}",
        inputs.join(","),
        ids_to_subscript(output_ids)?
    ))
}

fn compile_native_einsum_program(
    compiler: &mut GraphCompiler,
    input_specs: impl IntoIterator<Item = tenferro::program::ProgramInputSpec>,
    subscripts: &EinsumSubscripts,
) -> Result<tenferro::CompiledGraph> {
    let input_specs = input_specs.into_iter().collect::<Vec<_>>();
    let target = common_dtype(
        input_specs
            .iter()
            .map(|spec| spec.metadata().dtype())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mut trace = TraceContext::new();
    let trace_inputs = input_specs
        .into_iter()
        .map(|spec| {
            let dtype = spec.metadata().dtype();
            let input = trace
                .input(spec)
                .map_err(|e| anyhow!("native einsum input tracing failed: {e}"))?;
            if dtype == target {
                return Ok(input);
            }
            trace
                .add_op(
                    CoreSemanticOp::Convert {
                        from: dtype,
                        to: target,
                    },
                    &[input],
                )
                .map_err(|e| anyhow!("native einsum promotion tracing failed: {e}"))?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("native einsum promotion returned no output"))
        })
        .collect::<Result<Vec<_>>>()?;
    let output = trace
        .einsum_subscripts(&trace_inputs, subscripts)
        .map_err(|e| anyhow!("native einsum tracing failed: {e}"))?;
    let graph = trace
        .finish(&[output])
        .map_err(|e| anyhow!("native einsum graph finalization failed: {e}"))?;
    compiler
        .compile_traced_graph(&graph)
        .map_err(|e| anyhow!("native einsum graph compilation failed: {e}"))
}

fn run_cached_native_einsum(
    subscripts: &EinsumSubscripts,
    execute: impl FnOnce(&mut GraphCompiler, &Runtime) -> Result<NativeTensor>,
) -> Result<NativeTensor> {
    let trace_pool = native_einsum_pool_trace_enabled();
    let pool_before = trace_pool
        .then(default_engine_buffer_pool_stats)
        .transpose()?;
    let result =
        with_default_graph_runtime(|compiler, runtime, _backend| execute(compiler, runtime))??;
    if trace_pool {
        let pool_after = default_engine_buffer_pool_stats()?;
        let output_bytes = native_tensor_bytes(&result);
        let retained_threshold = native_einsum_pool_trace_min_retained_bytes();
        if pool_after != pool_before.unwrap_or_default()
            && pool_after.capacity_bytes >= retained_threshold
            || output_bytes >= native_einsum_pool_trace_min_output_bytes()
        {
            let before = pool_before.unwrap_or_default();
            eprintln!(
                "native_einsum pool subscripts={subscripts:?} before_buffers={} before_capacity={:.3} MiB after_buffers={} after_capacity={:.3} MiB output_shape={:?} output_bytes={:.3} MiB",
                before.buffers,
                before.capacity_bytes as f64 / (1024.0 * 1024.0),
                pool_after.buffers,
                pool_after.capacity_bytes as f64 / (1024.0 * 1024.0),
                result.shape(),
                output_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
    if reset_native_einsum_engine_after_call() {
        let before_reset = trace_pool
            .then(default_engine_buffer_pool_stats)
            .transpose()?;
        reset_default_engine()?;
        if trace_pool
            && before_reset.unwrap_or_default().capacity_bytes
                >= native_einsum_pool_trace_min_retained_bytes()
        {
            let before = before_reset.unwrap_or_default();
            let after = default_engine_buffer_pool_stats()?;
            eprintln!(
                "native_einsum engine_reset before_buffers={} before_capacity={:.3} MiB after_buffers={} after_capacity={:.3} MiB",
                before.buffers,
                before.capacity_bytes as f64 / (1024.0 * 1024.0),
                after.buffers,
                after.capacity_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    } else if reset_native_einsum_buffer_pool_after_call() {
        let before_clear = trace_pool
            .then(default_engine_buffer_pool_stats)
            .transpose()?;
        reset_default_engine_buffer_pool()?;
        if trace_pool
            && before_clear.unwrap_or_default().capacity_bytes
                >= native_einsum_pool_trace_min_retained_bytes()
        {
            let before = before_clear.unwrap_or_default();
            let after = default_engine_buffer_pool_stats()?;
            eprintln!(
                "native_einsum pool_reset before_buffers={} before_capacity={:.3} MiB after_buffers={} after_capacity={:.3} MiB",
                before.buffers,
                before.capacity_bytes as f64 / (1024.0 * 1024.0),
                after.buffers,
                after.capacity_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
    if release_allocator_after_native_einsum_call() {
        let report = release_process_allocator_cached_memory();
        if trace_pool && (report.released_bytes.unwrap_or(0) > 0 || report.success == Some(true)) {
            eprintln!(
                "native_einsum allocator_pressure_relief supported={} released_bytes={:?} success={:?}",
                report.supported,
                report.released_bytes,
                report.success,
            );
        }
    }
    Ok(result)
}

fn cached_einsum_native_tensors(
    inputs: &[&NativeTensor],
    subscripts: &EinsumSubscripts,
) -> Result<NativeTensor> {
    run_cached_native_einsum(subscripts, |compiler, runtime| {
        let program = compile_native_einsum_program(
            compiler,
            inputs.iter().map(|tensor| {
                tenferro::program::ProgramInputSpec::new(
                    tensor.dtype(),
                    tensor.shape().iter().copied().map(Into::into),
                )
            }),
            subscripts,
        )?;
        let mut outputs = runtime
            .run_compiled(&program, inputs)
            .map_err(|e| anyhow!("native einsum failed: {e}"))?;
        if outputs.len() != 1 {
            return Err(anyhow!(
                "native einsum returned {} outputs instead of one",
                outputs.len()
            ));
        }
        outputs
            .pop()
            .ok_or_else(|| anyhow!("native einsum returned no output"))
    })
}

fn cached_einsum_native_reads(
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
) -> Result<NativeTensor> {
    let views = inputs
        .iter()
        .map(|input| input.clone().tensor_view())
        .collect::<Vec<_>>();
    let einsum_subscripts = EinsumSubscripts::from(subscripts);
    run_cached_native_einsum(&einsum_subscripts, |compiler, runtime| {
        let program = compile_native_einsum_program(
            compiler,
            views.iter().map(|tensor| {
                tenferro::program::ProgramInputSpec::new(
                    tensor.dtype(),
                    tensor.shape().iter().copied().map(Into::into),
                )
            }),
            &einsum_subscripts,
        )?;
        let outcome = runtime
            .execute_scoped_read_only(&program, ScopedReadInputs::new(views))
            .map_err(|rejected| {
                let (error, _inputs) = rejected.into_parts();
                anyhow!("native einsum submission rejected: {error}")
            })?;
        let bundle = match outcome {
            ScopedExecutionOutcome::Completed(bundle) => bundle,
            ScopedExecutionOutcome::RetiredFailed { error, .. } => {
                return Err(anyhow!("native einsum failed: {error}"));
            }
        };
        bundle
            .into_owned_output(0)
            .map_err(|(_, error)| anyhow!("native einsum output extraction failed: {error}"))
    })
    .map_err(|e| anyhow!("native read einsum failed: {e}"))
}

/// Build native einsum ids for a binary contraction.
pub(crate) fn build_binary_einsum_ids(
    lhs_rank: usize,
    axes_a: &[usize],
    rhs_rank: usize,
    axes_b: &[usize],
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    ensure!(
        axes_a.len() == axes_b.len(),
        "contract axis length mismatch: lhs {:?}, rhs {:?}",
        axes_a,
        axes_b
    );

    let mut lhs_ids = vec![u32::MAX; lhs_rank];
    let mut rhs_ids = vec![u32::MAX; rhs_rank];
    let mut next_id = 0u32;

    let mut seen_lhs = vec![false; lhs_rank];
    let mut seen_rhs = vec![false; rhs_rank];

    for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
        ensure!(
            lhs_axis < lhs_rank,
            "lhs contract axis {lhs_axis} out of range"
        );
        ensure!(
            rhs_axis < rhs_rank,
            "rhs contract axis {rhs_axis} out of range"
        );
        ensure!(
            !seen_lhs[lhs_axis],
            "duplicate lhs contract axis {lhs_axis}"
        );
        ensure!(
            !seen_rhs[rhs_axis],
            "duplicate rhs contract axis {rhs_axis}"
        );
        seen_lhs[lhs_axis] = true;
        seen_rhs[rhs_axis] = true;
        lhs_ids[lhs_axis] = next_id;
        rhs_ids[rhs_axis] = next_id;
        next_id += 1;
    }

    let mut output_ids = Vec::with_capacity(lhs_rank + rhs_rank - 2 * axes_a.len());
    for (axis, slot) in lhs_ids.iter_mut().enumerate() {
        if *slot == u32::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        } else {
            let _ = axis;
        }
    }
    for slot in &mut rhs_ids {
        if *slot == u32::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        }
    }

    Ok((lhs_ids, rhs_ids, output_ids))
}

/// Build a dense native tensor from column-major data.
/// # Errors
///
/// Returns an error when the data length does not match the logical dimension product (a shape mismatch) or the backend conversion fails.
pub fn dense_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_dims: &[usize],
) -> Result<NativeTensor> {
    T::dense_native_tensor_from_col_major(data, logical_dims)
}

/// Build a dense native tensor whose logical values are diagonal.
/// # Errors
///
/// Returns an error when the diagonal payload is incompatible with the logical rank (a shape mismatch) or the backend conversion fails.
pub fn diag_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_rank: usize,
) -> Result<NativeTensor> {
    T::diag_native_tensor_from_col_major(data, logical_rank)
}

/// Convert storage to a dense native tensor.
/// # Errors
///
/// Returns an error when the storage cannot be converted to a native tensor (a scalar-kind mismatch or backend failure).
pub fn storage_to_native_tensor(
    storage: &Storage,
    logical_dims: &[usize],
) -> std::result::Result<NativeTensor, BridgeError> {
    if storage.is_c64() {
        dense_native_tensor_from_col_major(
            &storage
                .to_dense_c64_col_major_vec(logical_dims)
                .map_err(|e| anyhow!("dense c64 materialization failed: {e}"))?,
            logical_dims,
        )
        .map_err(BridgeError::from)
    } else {
        dense_native_tensor_from_col_major(
            &storage
                .to_dense_f64_col_major_vec(logical_dims)
                .map_err(|e| anyhow!("dense f64 materialization failed: {e}"))?,
            logical_dims,
        )
        .map_err(BridgeError::from)
    }
}

/// Build a read-only native tensor input over the compact storage payload.
///
/// Contiguous payloads are borrowed without copying. Non-contiguous payloads
/// are materialized into an owned native tensor.
/// # Errors
///
/// Returns an error when the storage payload cannot be read into a native buffer (a scalar-kind mismatch or backend failure).
pub fn storage_payload_native_read_input(
    storage: &Storage,
) -> std::result::Result<NativeTensorReadInput<'_>, BridgeError> {
    if storage.is_f64() {
        if let Some(view) = storage
            .payload_f64_col_major_view_if_contiguous()
            .map_err(anyhow::Error::msg)?
        {
            return Ok(NativeTensorReadInput::Borrowed(TensorRead::from_view(
                TensorView::f64(storage.payload_dims(), view)
                    .map_err(|e| BridgeError::from(anyhow::Error::new(e)))?,
            )));
        }
        native_tensor_from_vec(
            storage.payload_dims().to_vec(),
            storage
                .payload_f64_col_major_vec()
                .map_err(anyhow::Error::msg)?,
        )
        .map(NativeTensorReadInput::Owned)
    } else if storage.is_c64() {
        if let Some(view) = storage
            .payload_c64_col_major_view_if_contiguous()
            .map_err(anyhow::Error::msg)?
        {
            return Ok(NativeTensorReadInput::Borrowed(TensorRead::from_view(
                TensorView::c64(storage.payload_dims(), view)
                    .map_err(|e| BridgeError::from(anyhow::Error::new(e)))?,
            )));
        }
        native_tensor_from_vec(
            storage.payload_dims().to_vec(),
            storage
                .payload_c64_col_major_vec()
                .map_err(anyhow::Error::msg)?,
        )
        .map(NativeTensorReadInput::Owned)
    } else {
        Err(anyhow!("unsupported storage scalar type").into())
    }
}

/// Materialize a native tensor into dense storage.
/// # Errors
///
/// Returns an error when the native tensor cannot be converted to storage (a scalar-kind mismatch or backend failure).
pub fn native_tensor_primal_to_storage(
    tensor: &NativeTensor,
) -> std::result::Result<Storage, BridgeError> {
    match tensor.dtype() {
        DType::F32 => Storage::from_dense_col_major(
            native_slice::<f32>(tensor, "failed to read f32 native tensor")?
                .iter()
                .map(|&value| value as f64)
                .collect::<Vec<_>>(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::F64 => Storage::from_dense_col_major(
            native_slice::<f64>(tensor, "failed to read f64 native tensor")?.to_vec(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::I32 => Storage::from_dense_col_major(
            native_slice::<i32>(tensor, "failed to read i32 native tensor")?
                .iter()
                .map(|&value| value as f64)
                .collect::<Vec<_>>(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::I64 => Storage::from_dense_col_major(
            native_slice::<i64>(tensor, "failed to read i64 native tensor")?
                .iter()
                .map(|&value| value as f64)
                .collect::<Vec<_>>(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::Bool => Storage::from_dense_col_major(
            native_slice::<bool>(tensor, "failed to read bool native tensor")?
                .iter()
                .map(|&value| if value { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::C32 => Storage::from_dense_col_major(
            native_slice::<Complex32>(tensor, "failed to read c32 native tensor")?
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
        DType::C64 => Storage::from_dense_col_major(
            native_slice::<Complex64>(tensor, "failed to read c64 native tensor")?.to_vec(),
            tensor.shape(),
        )
        .map_err(|e| {
            BridgeError::from(anyhow!(
                "native tensor snapshot materialization failed: {e}"
            ))
        }),
    }
}

/// Materialize dense column-major values from a native tensor.
/// # Errors
///
/// Returns an error when the native tensor cannot be materialized as a dense
/// column-major buffer (a dtype mismatch or backend failure).
pub fn native_tensor_primal_to_dense_col_major<T: TensorElement>(
    tensor: &NativeTensor,
) -> std::result::Result<Vec<T>, BridgeError> {
    let target = <T as TensorScalar>::dtype();
    let tensor_is_real = matches!(
        tensor.dtype(),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool
    );
    let target_is_real = matches!(
        target,
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool
    );
    if tensor_is_real != target_is_real {
        return Err(anyhow!(
            "expected {} native tensor, got dtype {:?}",
            if target_is_real { "real" } else { "complex" },
            tensor.dtype()
        )
        .into());
    }
    <T as TensorElement>::dense_values_from_native_col_major(tensor).map_err(BridgeError::from)
}

/// Materialize diagonal values from a native tensor, promoting to the
/// matching real (`f64`) or complex (`Complex64`) dtype.
///
/// Real native tensors (`f32`, `f64`, `i32`, `i64`, `bool`) are promoted to
/// `f64`; complex tensors (`c32`, `c64`) are promoted to `Complex64`. The
/// scalar type `T` selects the promoted target.
/// # Errors
///
/// Returns an error when the native tensor is not compatible with the scalar
/// target (a dtype mismatch) or the materialization fails.
///
/// # Examples
/// ```
/// use tenferro::Tensor as NativeTensor;
/// use tensor4all_tensorbackend::native_tensor_primal_to_diag;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let native = NativeTensor::from_vec_col_major(vec![3, 3], vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;
/// let diag = native_tensor_primal_to_diag::<f64>(&native)?;
/// assert_eq!(diag, vec![1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub fn native_tensor_primal_to_diag<T: TensorElement>(
    tensor: &NativeTensor,
) -> std::result::Result<Vec<T>, BridgeError> {
    let promote_to = <T as TensorScalar>::dtype();
    let tensor_is_real = matches!(
        tensor.dtype(),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool
    );
    let target_is_real = matches!(
        promote_to,
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool
    );
    if tensor_is_real != target_is_real {
        return Err(anyhow!(
            "expected {} native tensor, got dtype {:?}",
            if target_is_real { "real" } else { "complex" },
            tensor.dtype()
        )
        .into());
    }
    let promoted = convert_tensor(tensor, promote_to)?;
    <T as TensorElement>::diag_values_from_native_temp(&promoted).map_err(BridgeError::from)
}

/// Reshape a native tensor without changing its column-major linearization.
/// # Errors
///
/// Returns an error when the native tensor cannot be reshaped to the requested dimensions (a shape mismatch) or the backend fails.
pub fn reshape_col_major_native_tensor(
    tensor: &NativeTensor,
    logical_dims: &[usize],
) -> Result<NativeTensor> {
    with_default_session(|session| tensor.reshape(logical_dims, session))
        .map_err(|e| anyhow!("native reshape failed: {e}"))
}

/// Compute a QR decomposition on a native tensor.
/// # Errors
///
/// Returns an error when the QR factorization fails (a backend or non-convergence failure).
pub fn qr_native_tensor(
    tensor: &NativeTensor,
) -> std::result::Result<(NativeTensor, NativeTensor), BridgeError> {
    let (q, r) = with_default_session(|session| tensor.qr(session))
        .map_err(|e| anyhow!("native QR failed: {e}"))?;
    Ok((q, r))
}

/// Compute an SVD on a native tensor.
/// # Errors
///
/// Returns an error when the SVD factorization fails (a backend or non-convergence failure).
pub fn svd_native_tensor(
    tensor: &NativeTensor,
) -> Result<(NativeTensor, NativeTensor, NativeTensor)> {
    let (u, s, vt) = with_default_session(|session| tensor.svd(session))
        .map_err(|e| anyhow!("native SVD failed: {e}"))?;
    Ok((u, s, vt))
}

/// Sum all elements of a native tensor, returning a dynamic scalar.
/// # Errors
///
/// Returns an error when the native reduction fails (a backend or dtype mismatch failure).
pub fn sum_native_tensor(tensor: &NativeTensor) -> std::result::Result<AnyScalar, BridgeError> {
    let reduced = if tensor.shape().is_empty() {
        tensor
            .duplicate()
            .map_err(|e| anyhow!("native scalar duplication failed: {e}"))?
    } else {
        let axes: Vec<usize> = (0..tensor.shape().len()).collect();
        with_default_session(|session| tensor.reduce_sum(&axes, session))
            .map_err(|e| anyhow!("native sum failed: {e}"))?
    };
    Ok(AnyScalar::from_native(reduced)?)
}

/// Return the tangent tensor when present.
///
/// Plain `Tensor` values do not carry tangent storage, so this bridge returns
/// `None`.
pub fn tangent_native_tensor(_tensor: &NativeTensor) -> Option<NativeTensor> {
    None
}

/// Multiply a native tensor by a dynamic scalar.
/// # Errors
///
/// Returns an error when the native scaling fails (a backend or dtype mismatch failure).
pub fn scale_native_tensor(
    tensor: &NativeTensor,
    scalar: &AnyScalar,
) -> std::result::Result<NativeTensor, BridgeError> {
    let target = common_dtype(&[tensor.dtype(), scalar.as_native().dtype()]);
    let tensor = convert_tensor(tensor, target)?;
    let scalar = promote_scalar_native(scalar.as_native(), target)?;

    match target {
        DType::F32 => {
            let factor = native_slice::<f32>(&scalar, "failed to read promoted f32 scalar")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar"))?;
            let values = native_slice::<f32>(&tensor, "failed to read promoted f32 tensor")?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            native_tensor_from_vec(tensor.shape().to_vec(), values)
        }
        DType::F64 => {
            let factor = native_slice::<f64>(&scalar, "failed to read promoted f64 scalar")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar"))?;
            let values = native_slice::<f64>(&tensor, "failed to read promoted f64 tensor")?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            native_tensor_from_vec(tensor.shape().to_vec(), values)
        }
        DType::C32 => {
            let factor = native_slice::<Complex32>(&scalar, "failed to read promoted c32 scalar")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar"))?;
            let values = native_slice::<Complex32>(&tensor, "failed to read promoted c32 tensor")?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            native_tensor_from_vec(tensor.shape().to_vec(), values)
        }
        DType::C64 => {
            let factor = native_slice::<Complex64>(&scalar, "failed to read promoted c64 scalar")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar"))?;
            let values = native_slice::<Complex64>(&tensor, "failed to read promoted c64 tensor")?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            native_tensor_from_vec(tensor.shape().to_vec(), values)
        }
        DType::I32 | DType::I64 | DType::Bool => {
            Err(anyhow!("scale_native_tensor does not support integer/bool tensors").into())
        }
    }
}

/// Compute `a * lhs + b * rhs`.
/// # Errors
///
/// Returns an error when the native axpby fails (a shape or dtype mismatch, or a backend failure).
pub fn axpby_native_tensor(
    lhs: &NativeTensor,
    a: &AnyScalar,
    rhs: &NativeTensor,
    b: &AnyScalar,
) -> std::result::Result<NativeTensor, BridgeError> {
    if lhs.shape() != rhs.shape() {
        return Err(BridgeError::from(anyhow!(
            "axpby requires matching tensor shapes, got lhs {:?} and rhs {:?}",
            lhs.shape(),
            rhs.shape()
        )));
    }

    let target = common_dtype(&[
        lhs.dtype(),
        rhs.dtype(),
        a.as_native().dtype(),
        b.as_native().dtype(),
    ]);
    let lhs = convert_tensor(lhs, target)?;
    let rhs = convert_tensor(rhs, target)?;
    let a = promote_scalar_native(a.as_native(), target)?;
    let b = promote_scalar_native(b.as_native(), target)?;

    match target {
        DType::F32 => {
            let a = native_slice::<f32>(&a, "failed to read promoted f32 scalar a")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar a"))?;
            let b = native_slice::<f32>(&b, "failed to read promoted f32 scalar b")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar b"))?;
            let lhs_values = native_slice::<f32>(&lhs, "failed to read promoted f32 lhs")?;
            let rhs_values = native_slice::<f32>(&rhs, "failed to read promoted f32 rhs")?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            native_tensor_from_vec(lhs.shape().to_vec(), values)
        }
        DType::F64 => {
            let a = native_slice::<f64>(&a, "failed to read promoted f64 scalar a")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar a"))?;
            let b = native_slice::<f64>(&b, "failed to read promoted f64 scalar b")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar b"))?;
            let lhs_values = native_slice::<f64>(&lhs, "failed to read promoted f64 lhs")?;
            let rhs_values = native_slice::<f64>(&rhs, "failed to read promoted f64 rhs")?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            native_tensor_from_vec(lhs.shape().to_vec(), values)
        }
        DType::C32 => {
            let a = native_slice::<Complex32>(&a, "failed to read promoted c32 scalar a")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar a"))?;
            let b = native_slice::<Complex32>(&b, "failed to read promoted c32 scalar b")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar b"))?;
            let lhs_values = native_slice::<Complex32>(&lhs, "failed to read promoted c32 lhs")?;
            let rhs_values = native_slice::<Complex32>(&rhs, "failed to read promoted c32 rhs")?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            native_tensor_from_vec(lhs.shape().to_vec(), values)
        }
        DType::C64 => {
            let a = native_slice::<Complex64>(&a, "failed to read promoted c64 scalar a")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar a"))?;
            let b = native_slice::<Complex64>(&b, "failed to read promoted c64 scalar b")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar b"))?;
            let lhs_values = native_slice::<Complex64>(&lhs, "failed to read promoted c64 lhs")?;
            let rhs_values = native_slice::<Complex64>(&rhs, "failed to read promoted c64 rhs")?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            native_tensor_from_vec(lhs.shape().to_vec(), values)
        }
        DType::I32 | DType::I64 | DType::Bool => {
            Err(anyhow!("axpby_native_tensor does not support integer/bool tensors").into())
        }
    }
}

/// Execute a cached einsum over owned native tensors.
///
/// This is the consuming bridge used by higher-level owned contraction APIs.
/// Inputs are promoted to a common dtype before tenferro evaluates the
/// contraction. Repeated calls with the same equation and shapes reuse
/// tenferro's process-global contraction path cache.
///
/// # Arguments
/// * `operands` - Native tensors paired with numeric einsum labels for each axis.
/// * `output_ids` - Numeric labels to keep in the result, in output axis order.
///
/// # Returns
/// The contracted native tensor in the promoted common dtype.
///
/// # Errors
/// Returns an error if the operand list is empty, any label list length does
/// not match its tensor rank, label generation exceeds the supported range, or
/// the backend contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::einsum_native_tensors_owned;
/// use tenferro::Tensor as NativeTensor;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///
/// let lhs = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
/// let rhs = NativeTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])?;
/// let result = einsum_native_tensors_owned(vec![(lhs, vec![0, 1]), (rhs, vec![1, 2])], &[0, 2])?;
///
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result.as_slice::<f64>()?, &[3.0, 3.0, 3.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub fn einsum_native_tensors_owned(
    operands: Vec<(NativeTensor, Vec<usize>)>,
    output_ids: &[usize],
) -> Result<NativeTensor> {
    ensure!(
        !operands.is_empty(),
        "native einsum requires at least one operand"
    );

    let target = common_dtype(
        &operands
            .iter()
            .map(|(tensor, _)| tensor.dtype())
            .collect::<Vec<_>>(),
    );

    let output_ids_u32 = checked_native_einsum_labels(output_ids)?;
    let mut converted = Vec::with_capacity(operands.len());
    let mut input_ids = Vec::with_capacity(operands.len());
    for (tensor, ids) in operands {
        ensure!(
            tensor.shape().len() == ids.len(),
            "einsum id list {:?} does not match tensor shape {:?}",
            ids,
            tensor.shape()
        );
        let checked_ids = checked_native_einsum_labels(&ids)?;
        let tensor = if tensor.dtype() == target {
            tensor
        } else {
            convert_tensor(&tensor, target)?
        };
        input_ids.push(checked_ids);
        converted.push(tensor);
    }

    let input_slices = input_ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let subscripts = EinsumSubscripts::new(&input_slices, &output_ids_u32);

    let input_refs = converted.iter().collect::<Vec<_>>();
    let trace_operands = input_refs
        .iter()
        .zip(input_ids.iter())
        .map(|(tensor, ids)| (*tensor, ids.as_slice()))
        .collect::<Vec<_>>();
    maybe_trace_native_einsum_path(NativeEinsumPath::Owned, &trace_operands, &output_ids_u32);
    let started = Instant::now();
    let result = cached_einsum_native_tensors(&input_refs, &subscripts)?;
    record_native_einsum_profile(
        NativeEinsumPath::Owned,
        &trace_operands,
        &output_ids_u32,
        started.elapsed(),
    );
    Ok(result)
}

/// Execute a cached einsum over borrowed native tensors.
///
/// Inputs are promoted to a common dtype before contraction. Operands that
/// already have the target dtype are passed to the backend by reference;
/// operands with another dtype are converted into temporary native tensors and
/// then borrowed for the contraction. Repeated calls with the same equation
/// and shapes reuse tenferro's process-global contraction path cache.
///
/// # Arguments
/// * `operands` - Native tensors paired with numeric einsum labels for each axis.
///
///   Each label slice must have the same length as the corresponding tensor rank.
/// * `output_ids` - Numeric labels to keep in the result, in output axis order.
///
/// # Returns
/// The contracted native tensor in the promoted common dtype.
///
/// # Errors
/// Returns an error if the operand list is empty, any label list length does
/// not match its tensor rank, label generation exceeds the supported range,
/// dtype conversion fails, or the backend contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::einsum_native_tensors;
/// use tenferro::Tensor as NativeTensor;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///
/// let lhs = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
/// let rhs = NativeTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])?;
/// let result = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])?;
///
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result.as_slice::<f64>()?, &[3.0, 3.0, 3.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub fn einsum_native_tensors(
    operands: &[(&NativeTensor, &[usize])],
    output_ids: &[usize],
) -> Result<NativeTensor> {
    ensure!(
        !operands.is_empty(),
        "native einsum requires at least one operand"
    );

    let target = common_dtype(
        &operands
            .iter()
            .map(|(tensor, _)| tensor.dtype())
            .collect::<Vec<_>>(),
    );
    let output_ids_u32 = checked_native_einsum_labels(output_ids)?;
    let mut converted = Vec::with_capacity(operands.len());
    let mut input_ids = Vec::with_capacity(operands.len());
    let mut has_conversions = false;
    let started = Instant::now();

    for (tensor, ids) in operands {
        ensure!(
            tensor.shape().len() == ids.len(),
            "einsum id list {:?} does not match tensor shape {:?}",
            ids,
            tensor.shape()
        );
        input_ids.push(checked_native_einsum_labels(ids)?);
        if tensor.dtype() == target {
            converted.push(None);
        } else {
            converted.push(Some(convert_tensor(tensor, target)?));
            has_conversions = true;
        }
    }

    let input_slices = input_ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let subscripts = EinsumSubscripts::new(&input_slices, &output_ids_u32);
    let input_refs = operands
        .iter()
        .zip(converted.iter())
        .map(|((tensor, _), converted)| converted.as_ref().unwrap_or(*tensor))
        .collect::<Vec<_>>();
    let trace_path = if has_conversions {
        NativeEinsumPath::BorrowedWithConversions
    } else {
        NativeEinsumPath::Borrowed
    };
    let trace_operands = input_refs
        .iter()
        .zip(input_ids.iter())
        .map(|(tensor, ids)| (*tensor, ids.as_slice()))
        .collect::<Vec<_>>();
    maybe_trace_native_einsum_path(trace_path, &trace_operands, &output_ids_u32);
    let result = cached_einsum_native_tensors(&input_refs, &subscripts)?;
    record_native_einsum_profile(
        trace_path,
        &trace_operands,
        &output_ids_u32,
        started.elapsed(),
    );
    Ok(result)
}

/// Execute a cached einsum over read-only native tensor inputs.
///
/// Backends consume borrowed host views inside their execution session. Mixed
/// dtypes are promoted by `Convert` nodes in the compiled einsum graph, so
/// non-contiguous operands remain borrowed until runtime execution.
/// # Errors
///
/// Returns an error when the native einsum fails (a shape or dtype mismatch, or a backend failure).
pub fn einsum_native_tensor_reads(
    operands: &[(&NativeTensorReadInput<'_>, &[usize])],
    output_ids: &[usize],
) -> Result<NativeTensor> {
    ensure!(
        !operands.is_empty(),
        "native einsum requires at least one operand"
    );

    let output_ids_u32 = checked_native_einsum_labels(output_ids)?;
    let mut input_ids = Vec::with_capacity(operands.len());
    let mut read_inputs = Vec::with_capacity(operands.len());

    for (tensor, ids) in operands {
        ensure!(
            tensor.shape().len() == ids.len(),
            "einsum id list {:?} does not match tensor shape {:?}",
            ids,
            tensor.shape()
        );
        input_ids.push(checked_native_einsum_labels(ids)?);
        read_inputs.push(tensor.as_read());
    }

    let subscripts = Subscripts {
        inputs: input_ids,
        output: output_ids_u32,
    };
    cached_einsum_native_reads(&read_inputs, &subscripts)
}

/// Permute axes of a native tensor.
/// # Errors
///
/// Returns an error when the native permutation fails (a shape or index mismatch, or a backend failure).
pub fn permute_native_tensor(
    tensor: &NativeTensor,
    perm: &[usize],
) -> std::result::Result<NativeTensor, BridgeError> {
    with_default_session(|session| tensor.transpose(perm, session))
        .map_err(|e| anyhow!("native permute failed: {e}"))
        .map_err(BridgeError::from)
}

/// Contract two native tensors along matching axes.
/// # Errors
///
/// Returns an error when the native contraction fails (a shape or index mismatch, or a backend failure).
pub fn contract_native_tensor(
    lhs: &NativeTensor,
    axes_a: &[usize],
    rhs: &NativeTensor,
    axes_b: &[usize],
) -> Result<NativeTensor> {
    let (lhs_ids, rhs_ids, output_ids) =
        build_binary_einsum_ids(lhs.shape().len(), axes_a, rhs.shape().len(), axes_b)?;
    let lhs_ids_usize = lhs_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let rhs_ids_usize = rhs_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let output_ids_usize = output_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let operands = [
        (lhs, lhs_ids_usize.as_slice()),
        (rhs, rhs_ids_usize.as_slice()),
    ];
    einsum_native_tensors(&operands, &output_ids_usize)
}

/// Compute the outer product of two native tensors.
/// # Errors
///
/// Returns an error when the native outer product fails (a shared-index or shape mismatch, or a backend failure).
pub fn outer_product_native_tensor(
    lhs: &NativeTensor,
    rhs: &NativeTensor,
) -> std::result::Result<NativeTensor, BridgeError> {
    contract_native_tensor(lhs, &[], rhs, &[]).map_err(BridgeError::from)
}

/// Conjugate a native tensor.
/// # Errors
///
/// Returns an error when the native conjugation fails (a dtype mismatch or backend failure).
pub fn conj_native_tensor(tensor: &NativeTensor) -> std::result::Result<NativeTensor, BridgeError> {
    match tensor.dtype() {
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => tensor
            .duplicate()
            .map_err(|e| anyhow!("native tensor duplication failed: {e}"))
            .map_err(BridgeError::from),
        DType::C32 => native_tensor_from_vec(
            tensor.shape().to_vec(),
            native_slice::<Complex32>(tensor, "failed to read c32 native tensor")?
                .iter()
                .map(|&value| value.conj())
                .collect::<Vec<_>>(),
        ),
        DType::C64 => native_tensor_from_vec(
            tensor.shape().to_vec(),
            native_slice::<Complex64>(tensor, "failed to read c64 native tensor")?
                .iter()
                .map(|&value| value.conj())
                .collect::<Vec<_>>(),
        ),
    }
}

/// Permute storage by round-tripping through native tensors.
/// # Errors
///
/// Returns an error when the native storage permutation fails (a shape or index mismatch, or a backend failure).
pub fn permute_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    perm: &[usize],
) -> std::result::Result<Storage, BridgeError> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let permuted = permute_native_tensor(&native, perm)?;
    native_tensor_primal_to_storage(&permuted)
}

/// Contract storages via native tensors.
/// # Errors
///
/// Returns an error when the native storage contraction fails (a shape or index mismatch, or a backend failure).
pub fn contract_storage_native(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    _result_dims: &[usize],
) -> std::result::Result<Storage, BridgeError> {
    let lhs = storage_to_native_tensor(storage_a, dims_a)?;
    let rhs = storage_to_native_tensor(storage_b, dims_b)?;
    let result = contract_native_tensor(&lhs, axes_a, &rhs, axes_b)?;
    native_tensor_primal_to_storage(&result)
}

/// Outer-product storages via native tensors.
/// # Errors
///
/// Returns an error when the native storage outer product fails (a shared-index or shape mismatch, or a backend failure).
pub fn outer_product_storage_native(
    lhs: &Storage,
    lhs_dims: &[usize],
    rhs: &Storage,
    rhs_dims: &[usize],
    _result_dims: &[usize],
) -> std::result::Result<Storage, BridgeError> {
    let lhs = storage_to_native_tensor(lhs, lhs_dims)?;
    let rhs = storage_to_native_tensor(rhs, rhs_dims)?;
    let result = outer_product_native_tensor(&lhs, &rhs)?;
    native_tensor_primal_to_storage(&result)
}

/// Scale storage by a scalar via native tensors.
/// # Errors
///
/// Returns an error when the native storage scaling fails (a dtype mismatch or backend failure).
pub fn scale_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    scalar: &AnyScalar,
) -> std::result::Result<Storage, BridgeError> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let scaled = scale_native_tensor(&native, scalar)?;
    native_tensor_primal_to_storage(&scaled)
}

/// Compute `a * lhs + b * rhs` over storages via native tensors.
/// # Errors
///
/// Returns an error when the native storage axpby fails (a shape or dtype mismatch, or a backend failure).
pub fn axpby_storage_native(
    lhs: &Storage,
    lhs_dims: &[usize],
    a: &AnyScalar,
    rhs: &Storage,
    rhs_dims: &[usize],
    b: &AnyScalar,
) -> std::result::Result<Storage, BridgeError> {
    let lhs = storage_to_native_tensor(lhs, lhs_dims)?;
    let rhs = storage_to_native_tensor(rhs, rhs_dims)?;
    let combined = axpby_native_tensor(&lhs, a, &rhs, b)?;
    native_tensor_primal_to_storage(&combined)
}

#[cfg(test)]
mod tests;
