//! Explicit and optional process-global tenferro CPU execution contexts.

use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{CompiledGraph, GraphCompiler, Runtime, Tensor, TracedGraph};
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::{BufferPoolStats, CpuBackend};
use tenferro_tensor::{BackendSession, BackendSessionHost};

/// Caller-owned execution domain used by context-aware tensor algorithms.
///
/// Values are validated against the exact runtime represented by the selected
/// context; no implicit host/device transfer is performed by this enum.
#[derive(Clone, Debug)]
pub enum ExecutionContext {
    /// Host execution through one caller-owned CPU context.
    Cpu(Arc<CpuExecutionContext>),
    /// CUDA execution through one caller-owned CUDA context.
    #[cfg(feature = "tenferro-cuda")]
    Cuda(Arc<crate::cuda::CudaExecutionContext>),
}

/// Error returned by explicit CPU context graph or eager-runtime operations.
///
/// The original tenferro diagnostic is retained as the error source.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::CpuExecutionContextError;
///
/// let error = CpuExecutionContextError::Initialization {
///     component: "graph runtime",
///     source: Arc::new(std::io::Error::other("registration failed")),
/// };
/// assert!(error.source().is_some());
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CpuExecutionContextError {
    /// A context-owned graph or eager runtime could not be initialized.
    #[error("failed to initialize {component}: {source}")]
    Initialization {
        /// Context component being initialized.
        component: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// Graph compilation or execution failed.
    #[error("CPU graph {operation} failed: {source}")]
    Graph {
        /// Graph operation that failed.
        operation: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

const CANONICAL_SESSION_REENTRY_MESSAGE: &str = "recursive tensorbackend canonical session entry";

thread_local! {
    static CANONICAL_SESSION_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct CanonicalSessionGuard {
    previous: bool,
}

impl CanonicalSessionGuard {
    fn assert_inactive() {
        CANONICAL_SESSION_ACTIVE.with(|active| {
            assert!(!active.get(), "{CANONICAL_SESSION_REENTRY_MESSAGE}");
        });
    }

    fn enter() -> Self {
        Self::assert_inactive();
        CANONICAL_SESSION_ACTIVE.with(|active| Self {
            previous: active.replace(true),
        })
    }
}

impl Drop for CanonicalSessionGuard {
    fn drop(&mut self) {
        CANONICAL_SESSION_ACTIVE.with(|active| active.set(self.previous));
    }
}

impl CpuExecutionContextError {
    fn initialization(
        component: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Initialization {
            component,
            source: Arc::new(source),
        }
    }

    fn graph(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Graph {
            operation,
            source: Arc::new(source),
        }
    }
}

struct GraphState {
    compiler: GraphCompiler,
    runtime: Runtime,
    backend: CpuBackend,
}

/// Caller-owned CPU execution domain for plain, graph, and eager-AD work.
///
/// The supplied backend is the only source of CPU execution resources. Backend
/// clones preserve its runtime identity; this constructor never uses
/// `CpuBackend::new`, `CpuContext::from_env`, or a process-global fallback.
/// Graph preparation caches and the eager runtime are owned by this context and
/// are released when it is dropped.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let threads = context.with_backend(|backend| backend.num_threads());
/// assert_eq!(threads, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuExecutionContext {
    backend: Mutex<CpuBackend>,
    graph: OnceLock<Result<Mutex<GraphState>, CpuExecutionContextError>>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CpuExecutionContextError>>,
}

impl std::fmt::Debug for CpuExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionContext")
            .field("graph_initialized", &self.graph.get().is_some())
            .field("eager_initialized", &self.eager.get().is_some())
            .finish_non_exhaustive()
    }
}

impl CpuExecutionContext {
    /// Create an execution context from a caller-selected CPU backend.
    ///
    /// Runtime construction is lazy, so creating a context cannot fail and does
    /// not allocate another executor or consult environment configuration.
    pub fn from_backend(backend: CpuBackend) -> Self {
        Self {
            backend: Mutex::new(backend),
            graph: OnceLock::new(),
            eager: OnceLock::new(),
        }
    }

    /// Run a plain tensor operation with this context's backend.
    ///
    /// The closure runs while the context-local backend lock is held. A poisoned
    /// lock is recovered because tenferro validates every new backend session.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut backend)
    }

    pub(crate) fn with_session<R: Send>(
        &self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        CanonicalSessionGuard::assert_inactive();
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        backend.with_backend_session(|session| {
            let _guard = CanonicalSessionGuard::enter();
            f(session)
        })
    }

    fn backend_clone(&self) -> CpuBackend {
        self.with_backend(|backend| backend.clone())
    }

    fn graph_state(&self) -> Result<&Mutex<GraphState>, CpuExecutionContextError> {
        self.graph
            .get_or_init(|| {
                let backend = self.backend_clone();
                build_graph_runtime(&backend).map(|runtime| {
                    Mutex::new(GraphState {
                        compiler: GraphCompiler::new(),
                        runtime,
                        backend,
                    })
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn with_graph_state<R>(
        &self,
        f: impl FnOnce(&mut GraphCompiler, &mut Runtime, &mut CpuBackend) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        let mut graph = match self.graph_state()?.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let GraphState {
            compiler,
            runtime,
            backend,
        } = &mut *graph;
        Ok(f(compiler, runtime, backend))
    }

    /// Compile a backend-neutral traced graph using this context's compiler cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph-runtime initialization or
    /// graph compilation fails.
    pub fn compile_graph(
        &self,
        graph: &TracedGraph,
    ) -> Result<CompiledGraph, CpuExecutionContextError> {
        self.with_graph_state(|compiler, _, _| compiler.compile_traced_graph(graph))?
            .map_err(|source| CpuExecutionContextError::graph("compilation", source))
    }

    /// Execute a compiled graph in this context's runtime and prepared-plan cache.
    ///
    /// `CompiledGraph` is backend-neutral. Backend-prepared executables and
    /// workspaces never leave this context-owned runtime.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when runtime initialization,
    /// preparation, or execution fails.
    pub fn run_graph(
        &self,
        graph: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>, CpuExecutionContextError> {
        self.with_graph_state(|_, runtime, _| runtime.run_compiled(graph, inputs))?
            .map_err(|source| CpuExecutionContextError::graph("execution", source))
    }

    /// Return this context's eager reverse-AD runtime.
    ///
    /// Repeated calls return the same runtime and therefore the same eager
    /// compilation cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when linalg AD-rule or eager-runtime
    /// registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
        self.eager
            .get_or_init(|| build_eager_runtime(self.backend_clone()))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    /// Return statistics for this context's runtime-owned graph caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or the
    /// cache statistics query fails.
    pub fn graph_cache_stats(
        &self,
    ) -> Result<tenferro::RuntimeCacheStats, CpuExecutionContextError> {
        self.with_graph_state(|_, runtime, _| runtime.cache_stats())?
            .map_err(|source| CpuExecutionContextError::graph("cache statistics", source))
    }

    /// Return retained-buffer statistics for this context's graph backend.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or the
    /// backend statistics query fails.
    pub fn graph_buffer_pool_stats(&self) -> Result<BufferPoolStats, CpuExecutionContextError> {
        self.with_graph_state(|_, _, backend| backend.buffer_pool_stats())?
            .map_err(|source| CpuExecutionContextError::graph("buffer-pool statistics", source))
    }

    /// Release retained buffers owned by this context's graph backend.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or reset
    /// fails.
    pub fn reset_graph_buffer_pool(&self) -> Result<(), CpuExecutionContextError> {
        self.with_graph_state(|_, _, backend| backend.reset_buffer_pool())?
            .map_err(|source| CpuExecutionContextError::graph("buffer-pool reset", source))
    }

    /// Recreate this context's graph runtime and release its prepared caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when runtime reconstruction or
    /// buffer release fails.
    pub fn reset_graph_runtime(&self) -> Result<(), CpuExecutionContextError> {
        self.with_graph_state(|compiler, runtime, backend| {
            let replacement = build_graph_runtime(backend)?;
            *compiler = GraphCompiler::new();
            let old = std::mem::replace(runtime, replacement);
            drop(old);
            backend
                .reset_buffer_pool()
                .map_err(|source| CpuExecutionContextError::graph("buffer-pool reset", source))
        })??;
        Ok(())
    }
}

fn build_graph_runtime(backend: &CpuBackend) -> Result<Runtime, CpuExecutionContextError> {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration(backend).map_err(|source| {
                CpuExecutionContextError::initialization("graph CPU engine", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("graph CPU engine", source))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().map_err(|source| {
                    CpuExecutionContextError::initialization("einsum extension", source)
                })?,
            )
            .map_err(|source| {
                CpuExecutionContextError::initialization("einsum extension", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    builder
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("graph runtime", source))
}

fn build_eager_runtime(backend: CpuBackend) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
    let ad_context = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().map_err(|source| {
            CpuExecutionContextError::initialization("linalg AD rules", source)
        })?)
        .map_err(|source| CpuExecutionContextError::initialization("linalg AD rules", source))?
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("AD context", source))?;
    EagerRuntime::with_cpu_backend_and_ad_context(backend, &ad_context)
        .map_err(|source| CpuExecutionContextError::initialization("eager runtime", source))
}

#[cfg(feature = "global-defaults")]
mod defaults {
    use super::*;
    use tenferro_cpu::CpuContext;

    static DEFAULT_CONTEXT: OnceLock<Arc<CpuExecutionContext>> = OnceLock::new();

    #[cfg(test)]
    thread_local! {
        static FORCE_EAGER_CONTEXT_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[cfg(test)]
    static DEFAULT_CONTEXT_HITS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn default_context() -> &'static Arc<CpuExecutionContext> {
        DEFAULT_CONTEXT.get_or_init(|| {
            #[cfg(test)]
            DEFAULT_CONTEXT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Arc::new(CpuExecutionContext::from_backend(CpuBackend::from_context(
                Arc::new(CpuContext::from_env()),
            )))
        })
    }

    /// Error returned when the process-global eager AD runtime cannot be initialized.
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

    /// Run a closure against the optional process-global CPU backend.
    pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        default_context().with_backend(f)
    }

    pub(crate) fn with_default_session<R: Send>(
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        default_context().with_session(f)
    }

    pub(crate) fn with_default_graph_runtime<R>(
        f: impl FnOnce(&mut GraphCompiler, &Runtime, &mut CpuBackend) -> R,
    ) -> anyhow::Result<R> {
        default_context()
            .with_graph_state(|compiler, runtime, backend| f(compiler, runtime, backend))
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn default_engine_buffer_pool_stats() -> anyhow::Result<BufferPoolStats> {
        default_context()
            .graph_buffer_pool_stats()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine_buffer_pool() -> anyhow::Result<()> {
        default_context()
            .reset_graph_buffer_pool()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine() -> anyhow::Result<()> {
        default_context()
            .reset_graph_runtime()
            .map_err(anyhow::Error::new)
    }

    /// Return the optional process-global eager context used by convenience APIs.
    ///
    /// # Errors
    ///
    /// Returns [`EagerContextError::Registration`] when eager runtime
    /// initialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::default_eager_ctx;
    ///
    /// let first = default_eager_ctx().unwrap();
    /// let second = default_eager_ctx().unwrap();
    /// assert!(Arc::ptr_eq(&first, &second));
    /// ```
    pub fn default_eager_ctx() -> Result<Arc<EagerRuntime>, EagerContextError> {
        #[cfg(test)]
        if FORCE_EAGER_CONTEXT_FAILURE.with(std::cell::Cell::get) {
            return Err(EagerContextError::Registration {
                source: Arc::new(std::io::Error::other(
                    "forced default eager context registration failure",
                )),
            });
        }
        default_context()
            .eager_runtime()
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })
    }

    #[cfg(test)]
    pub(crate) fn default_context_hits() -> usize {
        DEFAULT_CONTEXT_HITS.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn with_forced_eager_context_failure<T>(f: impl FnOnce() -> T) -> T {
        let previous = FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.replace(true));
        let result = f();
        FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.set(previous));
        result
    }
}

#[cfg(all(test, feature = "global-defaults"))]
pub(crate) use defaults::with_forced_eager_context_failure;
#[cfg(feature = "global-defaults")]
pub use defaults::{default_eager_ctx, with_default_backend, EagerContextError};
#[cfg(feature = "global-defaults")]
pub(crate) use defaults::{
    default_engine_buffer_pool_stats, reset_default_engine, reset_default_engine_buffer_pool,
    with_default_graph_runtime, with_default_session,
};

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    use tenferro::{DType, TensorSessionOpsExt, TraceContext};
    use tenferro_ad::EagerTensor;
    use tenferro_cpu::{CpuContext, ExternalCpuDomain};
    use tenferro_tensor::CpuDomainId;

    fn context() -> CpuExecutionContext {
        CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap())
    }

    #[test]
    fn explicit_session_runs_a_concrete_operation() {
        let context = context();
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
        let result = context
            .with_session(|session| lhs.matmul(&rhs, session))
            .unwrap();

        assert_eq!(result.as_slice::<f64>().unwrap(), &[23.0, 34.0]);
    }

    #[test]
    fn recursive_session_entry_fails_before_lock_and_restores_guard() {
        let context = context();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.with_session(|_| context.with_session(|_| ()))
        }))
        .expect_err("recursive canonical session entry should panic");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .expect("recursive entry panic should contain a string message");
        assert_eq!(message, CANONICAL_SESSION_REENTRY_MESSAGE);

        assert_eq!(context.with_session(|_| 7usize), 7);
    }

    #[test]
    fn explicit_plain_graph_and_eager_paths_share_only_the_supplied_backend() {
        let context = context();
        assert!(format!("{context:?}").contains("graph_initialized: false"));
        assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);

        let mut trace = TraceContext::new();
        let input = trace
            .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
            .unwrap();
        let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
        let graph = trace.finish(&[output]).unwrap();
        let compiled = context.compile_graph(&graph).unwrap();
        let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
        let output = context.run_graph(&compiled, &[&input]).unwrap();
        assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
        context.run_graph(&compiled, &[&input]).unwrap();
        let cached = context.graph_cache_stats().unwrap().prepared_plans;
        assert!(cached.entries > 0);
        assert!(cached.hits > 0);
        context.reset_graph_runtime().unwrap();
        assert_eq!(
            context.graph_cache_stats().unwrap().prepared_plans.entries,
            0
        );

        let eager = context.eager_runtime().unwrap();
        assert!(Arc::ptr_eq(&eager, &context.eager_runtime().unwrap()));
    }

    #[test]
    fn separate_eager_contexts_reject_cross_context_operations() {
        let first = context().eager_runtime().unwrap();
        let second = context().eager_runtime().unwrap();
        let a = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
            first,
        )
        .unwrap();
        let b = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
            second,
        )
        .unwrap();
        assert!(matches!(
            a.add(&b),
            Err(tenferro_ad::Error::ContextMismatch { .. })
        ));
    }

    #[test]
    fn caller_managed_backend_remains_caller_owned_after_context_drop() {
        let executor = Arc::new(CpuContext::with_threads(1).unwrap());
        let id = CpuDomainId::new(7);
        let domain =
            ExternalCpuDomain::new_caller_managed(id, executor.clone(), NonZeroUsize::MIN).unwrap();
        let backend = CpuBackend::from_external_managed_domains(id, [domain]).unwrap();
        let context = CpuExecutionContext::from_backend(backend);
        assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);
        drop(context);
        assert_eq!(executor.num_threads(), 1);
    }

    #[test]
    fn independent_contexts_do_not_share_a_backend_mutex() {
        let first = Arc::new(context());
        let second = Arc::new(context());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let handles = [first, second].map(|context| {
            let entered_tx = entered_tx.clone();
            let release_rx = Arc::clone(&release_rx);
            std::thread::spawn(move || {
                context.with_backend(|_| {
                    entered_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv().unwrap();
                });
            })
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        release_tx.send(()).unwrap();
        release_tx.send(()).unwrap();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[cfg(feature = "global-defaults")]
    #[test]
    fn explicit_paths_do_not_initialize_the_default_context() {
        let before = defaults::default_context_hits();
        let context = context();
        context.with_backend(|backend| assert_eq!(backend.num_threads(), 1));
        context.eager_runtime().unwrap();
        assert_eq!(defaults::default_context_hits(), before);
    }
}
