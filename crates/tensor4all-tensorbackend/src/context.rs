//! Process-global tenferro CPU execution helpers.
//!
//! tensor4all-rs routes tenferro CPU execution through one process-global
//! `CpuContext`, matching tenferro's `cpu:0` default-global thread-pool model.
//! Plain tensor operations, cached traced execution, and eager AD currently use
//! separate `CpuBackend` values because tenferro does not expose a public API
//! for borrowing the backend owned by an `EagerRuntime`. All backends are
//! created from the same global CPU context, so thread-pool configuration is
//! shared.

use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{GraphCompiler, GraphExecutor};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::{buffer_pool::BufferPoolStats, CpuBackend, CpuContext};

static DEFAULT_CPU_CONTEXT: OnceLock<Arc<CpuContext>> = OnceLock::new();
static DEFAULT_BACKEND: OnceLock<Mutex<CpuBackend>> = OnceLock::new();
static DEFAULT_GRAPH_COMPILER: OnceLock<Mutex<GraphCompiler>> = OnceLock::new();
static DEFAULT_GRAPH_EXECUTOR: OnceLock<Mutex<GraphExecutor<CpuBackend>>> = OnceLock::new();
/// Error returned when the process-global eager AD runtime cannot be initialized.
///
/// The original backend error is retained as the [`std::error::Error::source`]
/// so callers can inspect the registration failure without parsing a string.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::EagerContextError;
///
/// let error = EagerContextError::Registration {
///     source: Arc::new(std::io::Error::other("registration failed")),
/// };
/// assert!(error.source().is_some());
/// assert!(error.to_string().contains("registration failed"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum EagerContextError {
    /// The tenferro linalg AD extension rule could not be registered.
    #[error("failed to register tenferro linalg AD rule: {source}")]
    Registration {
        /// Original diagnostic returned by tenferro.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

static DEFAULT_EAGER_RUNTIME: OnceLock<std::result::Result<Arc<EagerRuntime>, EagerContextError>> =
    OnceLock::new();

fn default_cpu_context() -> Arc<CpuContext> {
    DEFAULT_CPU_CONTEXT
        .get_or_init(|| Arc::new(CpuContext::from_env()))
        .clone()
}

fn default_backend() -> &'static Mutex<CpuBackend> {
    DEFAULT_BACKEND.get_or_init(|| Mutex::new(CpuBackend::from_context(default_cpu_context())))
}

fn default_graph_compiler() -> &'static Mutex<GraphCompiler> {
    DEFAULT_GRAPH_COMPILER.get_or_init(|| Mutex::new(GraphCompiler::new()))
}

fn default_graph_executor() -> &'static Mutex<GraphExecutor<CpuBackend>> {
    DEFAULT_GRAPH_EXECUTOR.get_or_init(|| {
        Mutex::new(GraphExecutor::new(CpuBackend::from_context(
            default_cpu_context(),
        )))
    })
}

fn lock_default_graph_compiler() -> std::sync::MutexGuard<'static, GraphCompiler> {
    match default_graph_compiler().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_default_graph_executor() -> std::sync::MutexGuard<'static, GraphExecutor<CpuBackend>> {
    match default_graph_executor().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Run a closure against the process-global CPU backend.
///
/// This is the canonical entry point for typed and untyped tenferro tensor
/// operations inside `tensor4all-tensorbackend`.
pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
    let mut backend = match default_backend().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut backend)
}

/// Run a closure against the process-global tenferro graph compiler/executor.
///
/// This is used for native tensor operations that benefit from tenferro's
/// persistent execution caches, such as N-ary einsum contraction paths.
pub(crate) fn with_default_graph_runtime<R>(
    f: impl FnOnce(&mut GraphCompiler, &mut GraphExecutor<CpuBackend>) -> R,
) -> R {
    let mut compiler = lock_default_graph_compiler();
    let mut executor = lock_default_graph_executor();
    f(&mut compiler, &mut executor)
}

/// Return retained-buffer statistics for the process-global graph executor.
pub(crate) fn default_engine_buffer_pool_stats() -> BufferPoolStats {
    lock_default_graph_executor().backend().buffer_pool_stats()
}

/// Reset retained buffers in the process-global graph executor.
pub(crate) fn reset_default_engine_buffer_pool() {
    let mut executor = lock_default_graph_executor();
    *executor = GraphExecutor::new(CpuBackend::from_context(default_cpu_context()));
}

/// Drop and recreate the process-global graph compiler/executor.
///
/// This releases tenferro's retained execution buffers and cached contraction
/// paths. It is intended for diagnostics and memory-pressure recovery, not for
/// normal hot loops where the caches are valuable.
pub(crate) fn reset_default_engine() {
    let mut compiler = lock_default_graph_compiler();
    *compiler = GraphCompiler::new();
    let mut executor = lock_default_graph_executor();
    *executor = GraphExecutor::new(CpuBackend::from_context(default_cpu_context()));
}

/// Return the process-global eager context used for reverse-mode AD.
///
/// This context owns a separate `CpuBackend` from [`with_default_backend`] and
/// the cached graph executor, but all backends share the same process-global
/// tenferro CPU context.
///
/// # Errors
///
/// Returns [`EagerContextError::Registration`] if the tenferro linalg AD rule
/// cannot be registered.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::default_eager_ctx;
/// use std::sync::Arc;
///
/// let first = default_eager_ctx().unwrap();
/// let second = default_eager_ctx().unwrap();
/// assert!(Arc::ptr_eq(&first, &second));
/// ```
pub fn default_eager_ctx() -> std::result::Result<Arc<EagerRuntime>, EagerContextError> {
    match DEFAULT_EAGER_RUNTIME.get_or_init(|| {
        tenferro_linalg::register_extension_rule()
            .map(|_| {
                EagerRuntime::with_cpu_backend(CpuBackend::from_context(default_cpu_context()))
            })
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })
    }) {
        Ok(context) => Ok(Arc::clone(context)),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_context_error_preserves_source_chain() {
        let source = std::io::Error::other("forced registration failure");
        let error = EagerContextError::Registration {
            source: Arc::new(source),
        };

        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "forced registration failure"
        );
    }

    #[test]
    fn eager_context_has_typed_error_contract() {
        let result: std::result::Result<Arc<EagerRuntime>, EagerContextError> = default_eager_ctx();
        assert!(result.is_ok());
    }

    #[test]
    fn eager_context_is_process_global() {
        let first = default_eager_ctx().unwrap();
        let second = default_eager_ctx().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn eager_context_is_shared_across_threads() {
        let main_context = default_eager_ctx().unwrap();
        let worker_context = std::thread::spawn(|| default_eager_ctx().unwrap())
            .join()
            .expect("worker thread should complete");

        assert!(Arc::ptr_eq(&main_context, &worker_context));
    }

    #[test]
    fn default_backend_is_shared_across_threads() {
        let main_threads = with_default_backend(|backend| backend.num_threads());
        let worker_threads =
            std::thread::spawn(|| with_default_backend(|backend| backend.num_threads()))
                .join()
                .expect("worker thread should complete");

        assert_eq!(main_threads, worker_threads);
    }

    #[test]
    fn default_engine_is_shared_across_threads() {
        let main_threads =
            with_default_graph_runtime(|_, executor| executor.backend().num_threads());
        let worker_threads = std::thread::spawn(|| {
            with_default_graph_runtime(|_, executor| executor.backend().num_threads())
        })
        .join()
        .expect("worker thread should complete");

        assert_eq!(main_threads, worker_threads);
    }
}
