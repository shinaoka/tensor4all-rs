//! Process-global tenferro CPU execution helpers.
//!
//! tensor4all-rs routes tenferro CPU execution through one process-global
//! `CpuContext`, matching tenferro's `cpu:0` default-global thread-pool model.
//! Plain tensor operations, cached traced execution, and eager AD currently use
//! separate `CpuBackend` values because tenferro does not expose a public API
//! for borrowing the backend owned by an `EagerRuntime`. All backends are
//! created from the same global CPU context, so thread-pool configuration is
//! shared.

#[cfg(test)]
use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::anyhow;
use tenferro::{GraphCompiler, Runtime};
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::{BufferPoolStats, CpuBackend, CpuContext};

static DEFAULT_CPU_CONTEXT: OnceLock<Arc<CpuContext>> = OnceLock::new();
static DEFAULT_BACKEND: OnceLock<Mutex<CpuBackend>> = OnceLock::new();

struct DefaultGraphRuntime {
    compiler: GraphCompiler,
    runtime: Runtime,
    backend: CpuBackend,
}

static DEFAULT_GRAPH_RUNTIME: OnceLock<std::result::Result<Mutex<DefaultGraphRuntime>, String>> =
    OnceLock::new();
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

#[cfg(test)]
thread_local! {
    static FORCE_EAGER_CONTEXT_FAILURE: Cell<bool> = const { Cell::new(false) };
}

fn default_cpu_context() -> Arc<CpuContext> {
    DEFAULT_CPU_CONTEXT
        .get_or_init(|| Arc::new(CpuContext::from_env()))
        .clone()
}

fn default_backend() -> &'static Mutex<CpuBackend> {
    DEFAULT_BACKEND.get_or_init(|| Mutex::new(CpuBackend::from_context(default_cpu_context())))
}

fn build_graph_runtime(backend: &CpuBackend) -> std::result::Result<Runtime, String> {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration(backend).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    builder.build().map_err(|e| e.to_string())
}

fn default_graph_runtime() -> anyhow::Result<&'static Mutex<DefaultGraphRuntime>> {
    match DEFAULT_GRAPH_RUNTIME.get_or_init(|| {
        let backend = CpuBackend::from_context(default_cpu_context());
        build_graph_runtime(&backend).map(|runtime| {
            Mutex::new(DefaultGraphRuntime {
                compiler: GraphCompiler::new(),
                runtime,
                backend,
            })
        })
    }) {
        Ok(runtime) => Ok(runtime),
        Err(error) => Err(anyhow!("failed to initialize graph runtime: {error}")),
    }
}

fn lock_default_graph_runtime(
) -> anyhow::Result<std::sync::MutexGuard<'static, DefaultGraphRuntime>> {
    match default_graph_runtime()?.lock() {
        Ok(guard) => Ok(guard),
        Err(poisoned) => Ok(poisoned.into_inner()),
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
    f: impl FnOnce(&mut GraphCompiler, &Runtime, &mut CpuBackend) -> R,
) -> anyhow::Result<R> {
    let mut graph = lock_default_graph_runtime()?;
    let graph = &mut *graph;
    let compiler = &mut graph.compiler;
    let runtime = &graph.runtime;
    let backend = &mut graph.backend;
    Ok(f(compiler, runtime, backend))
}

/// Return retained-buffer statistics for the process-global graph runtime.
pub(crate) fn default_engine_buffer_pool_stats() -> anyhow::Result<BufferPoolStats> {
    let graph = lock_default_graph_runtime()?;
    graph
        .backend
        .buffer_pool_stats()
        .map_err(|e| anyhow!("failed to read graph buffer-pool statistics: {e}"))
}

/// Reset retained buffers in the process-global graph runtime.
pub(crate) fn reset_default_engine_buffer_pool() -> anyhow::Result<()> {
    let mut graph = lock_default_graph_runtime()?;
    graph
        .backend
        .reset_buffer_pool()
        .map_err(|e| anyhow!("failed to reset graph buffer pool: {e}"))
}

/// Drop and recreate the process-global graph compiler/runtime.
///
/// This releases tenferro's retained execution buffers and cached contraction
/// paths. It is intended for diagnostics and memory-pressure recovery, not for
/// normal hot loops where the caches are valuable.
pub(crate) fn reset_default_engine() -> anyhow::Result<()> {
    let mut graph = lock_default_graph_runtime()?;
    let runtime = build_graph_runtime(&graph.backend)
        .map_err(|e| anyhow!("failed to reset graph runtime: {e}"))?;
    graph.compiler = GraphCompiler::new();
    graph.runtime = runtime;
    graph
        .backend
        .reset_buffer_pool()
        .map_err(|e| anyhow!("failed to reset graph buffer pool: {e}"))
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
    #[cfg(test)]
    if FORCE_EAGER_CONTEXT_FAILURE.with(Cell::get) {
        return Err(EagerContextError::Registration {
            source: Arc::new(std::io::Error::other(
                "forced default eager context registration failure",
            )),
        });
    }

    match DEFAULT_EAGER_RUNTIME.get_or_init(|| {
        let ad_context = AdContext::builder()
            .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().map_err(
                |source| EagerContextError::Registration {
                    source: Arc::new(source),
                },
            )?)
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })?
            .build()
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })?;
        EagerRuntime::with_cpu_backend_and_ad_context(
            CpuBackend::from_context(default_cpu_context()),
            &ad_context,
        )
        .map_err(|source| EagerContextError::Registration {
            source: Arc::new(source),
        })
    }) {
        Ok(context) => Ok(Arc::clone(context)),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
pub(crate) fn with_forced_eager_context_failure<T>(f: impl FnOnce() -> T) -> T {
    let previous = FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.replace(true));
    let result = f();
    FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.set(previous));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_cpu::linalg_interop::PoolScalar;

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
    fn eager_context_failure_uses_production_path_and_preserves_source() {
        with_forced_eager_context_failure(|| {
            let error = match default_eager_ctx() {
                Ok(_) => panic!("forced eager context failure unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(matches!(error, EagerContextError::Registration { .. }));
            let source = std::error::Error::source(&error).unwrap();
            assert_eq!(
                source.to_string(),
                "forced default eager context registration failure"
            );
            assert!(source.source().is_none());
        });
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
    fn reset_default_engine_releases_retained_backend_buffers() {
        reset_default_engine_buffer_pool().unwrap();
        with_default_graph_runtime(|_, _, backend| {
            backend.with_linalg_pool(|_, pool| {
                let buffer = <f64 as PoolScalar>::pool_acquire_zeroed(pool, 1024);
                <f64 as PoolScalar>::pool_release(pool, buffer);
                Ok(())
            })
        })
        .unwrap()
        .unwrap();

        let before = default_engine_buffer_pool_stats().unwrap();
        assert!(
            before.capacity_bytes > 0,
            "operation should retain a buffer"
        );
        reset_default_engine().unwrap();
        let after = default_engine_buffer_pool_stats().unwrap();
        assert_eq!(after.buffers, 0);
        assert_eq!(after.capacity_bytes, 0);
    }

    #[test]
    fn default_engine_is_shared_across_threads() {
        let main_threads =
            with_default_graph_runtime(|_, _, backend| backend.num_threads()).unwrap();
        let worker_threads = std::thread::spawn(|| {
            with_default_graph_runtime(|_, _, backend| backend.num_threads()).unwrap()
        })
        .join()
        .expect("worker thread should complete");

        assert_eq!(main_threads, worker_threads);
    }
}
