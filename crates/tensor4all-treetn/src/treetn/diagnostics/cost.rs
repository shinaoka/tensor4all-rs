//! Shape and measurement values shared by the diagnostic producers.

use std::time::Duration;

/// Actual local tensor dimensions used to group [`super::NodeDiagnostics`].
///
/// Recording sorts bond dimensions as a multiset; their order is not axis order.
/// Changing an output rank produces a separate row rather than overwriting it.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::NodeShape;
/// let shape = NodeShape { physical_dim: 2, bond_dims: vec![2, 3, 4] };
/// assert_eq!(shape.local_elements(), Some(48));
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeShape {
    /// Product of local physical dimensions; one for no physical legs.
    pub physical_dim: usize,
    /// Actual dimensions of every incident bond, including the outgoing bond.
    pub bond_dims: Vec<usize>,
}

impl NodeShape {
    /// Returns the incident bond product, or `None` on overflow.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::diagnostics::NodeShape;
    /// let shape = NodeShape { physical_dim: 2, bond_dims: vec![3, 4] };
    /// assert_eq!(shape.bond_product(), Some(12));
    /// ```
    pub fn bond_product(&self) -> Option<usize> {
        self.bond_dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
    }

    /// Returns the local tensor-size proxy, or `None` on overflow.
    ///
    /// This is `physical_dim * product(bond_dims)`, not a FLOP count.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::diagnostics::NodeShape;
    /// let shape = NodeShape { physical_dim: 2, bond_dims: vec![usize::MAX] };
    /// assert_eq!(shape.local_elements(), None);
    /// ```
    pub fn local_elements(&self) -> Option<usize> {
        self.bond_product()?.checked_mul(self.physical_dim)
    }
}

/// Disjoint kernel buckets inside a message/frame measurement.
///
/// Subtract [`super::kernel_snapshot`] values to attribute work to a node.
/// Scalar and uninstrumented fallback work remains in the enclosing elapsed
/// time: zero in a bucket does not mean zero total contraction cost.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::KernelDiagnostics;
/// let first = KernelDiagnostics { matmul_ns: 5, ..Default::default() };
/// let last = KernelDiagnostics { matmul_ns: 12, ..Default::default() };
/// assert_eq!(last.since(first).matmul_ns, 7);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelDiagnostics {
    /// Core-slice preparation nanoseconds.
    pub setup_ns: u64,
    /// Backend matrix multiplication nanoseconds.
    pub matmul_ns: u64,
    /// Final contraction/scatter nanoseconds.
    pub accumulate_ns: u64,
    /// Child-message packing nanoseconds.
    pub gather_ns: u64,
    /// Matrix multiplication jobs; a grouped backend call counts each job.
    pub matmul_calls: u64,
    /// Points sent to scalar message routes.
    pub scalar_points: u64,
    /// Prepared branch-slice cache hits.
    pub prepared_hits: u64,
    /// Prepared branch-slice cache misses.
    pub prepared_misses: u64,
    /// Prepared slices computed without retention because of the budget.
    pub prepared_refusals: u64,
}

macro_rules! kernel_fields {
    ($this:ident, $other:ident, $operation:ident) => {
        Self {
            setup_ns: $this.setup_ns.$operation($other.setup_ns),
            matmul_ns: $this.matmul_ns.$operation($other.matmul_ns),
            accumulate_ns: $this.accumulate_ns.$operation($other.accumulate_ns),
            gather_ns: $this.gather_ns.$operation($other.gather_ns),
            matmul_calls: $this.matmul_calls.$operation($other.matmul_calls),
            scalar_points: $this.scalar_points.$operation($other.scalar_points),
            prepared_hits: $this.prepared_hits.$operation($other.prepared_hits),
            prepared_misses: $this.prepared_misses.$operation($other.prepared_misses),
            prepared_refusals: $this.prepared_refusals.$operation($other.prepared_refusals),
        }
    };
}

impl KernelDiagnostics {
    /// Subtracts an earlier snapshot, saturating if a reset intervened.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::diagnostics::KernelDiagnostics;
    /// let before = KernelDiagnostics { prepared_hits: 3, ..Default::default() };
    /// assert_eq!(KernelDiagnostics::default().since(before).prepared_hits, 0);
    /// ```
    pub fn since(self, earlier: Self) -> Self {
        kernel_fields!(self, earlier, saturating_sub)
    }

    pub(super) fn plus(self, other: Self) -> Self {
        kernel_fields!(self, other, saturating_add)
    }
}

/// Aggregate batch sizes for one phase of [`super::NodeDiagnostics`].
///
/// Message points are unique component assignments; frame points are candidate
/// samples; query points are full assignments supplied by the caller.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::BatchDiagnostics;
/// let b = BatchDiagnostics::default();
/// assert_eq!((b.calls, b.points, b.min, b.max), (0, 0, 0, 0));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchDiagnostics {
    /// Number of nonempty batches.
    pub calls: u64,
    /// Sum of batch lengths.
    pub points: u64,
    /// Minimum nonempty length, or zero for no batches.
    pub min: u64,
    /// Maximum length, or zero for no batches.
    pub max: u64,
}

impl BatchDiagnostics {
    pub(super) fn record(&mut self, points: u64) {
        if points == 0 {
            return;
        }
        self.min = if self.calls == 0 {
            points
        } else {
            self.min.min(points)
        };
        self.max = self.max.max(points);
        self.calls = self.calls.saturating_add(1);
        self.points = self.points.saturating_add(points);
    }
}

/// One observation submitted to [`super::record_guard`] or [`super::record_frame`].
///
/// `hits + misses` is its batch length; misses count newly computed samples.
/// Elapsed Guard time and kernel counters exclude recursive child work.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::PhaseMeasurement;
/// let sample = PhaseMeasurement { hits: 2, misses: 3, ..Default::default() };
/// assert_eq!(sample.hits + sample.misses, 5);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseMeasurement {
    /// Local elapsed time including lookup and contraction, excluding children.
    pub elapsed: Duration,
    /// Cache hits, counted once per requested sample.
    pub hits: u64,
    /// Cache misses, including computations with retention disabled.
    pub misses: u64,
    /// Kernel work in this observation only.
    pub kernel: KernelDiagnostics,
}

/// Whole-evaluator cache high-water observations, attached to query centers.
///
/// Do not sum these across nodes or shapes. Owned estimates include payloads,
/// spare message capacity and hash-table buckets; they exclude backend allocator
/// arenas and heap allocations within generic node labels. No new numerical
/// cache is introduced. See [`super::NodeDiagnostics::query_cache`].
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::CacheDiagnostics;
/// let cache = CacheDiagnostics::default();
/// assert_eq!((cache.message_entries, cache.prepared_entries), (0, 0));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheDiagnostics {
    /// Maximum retained directed message columns.
    pub message_entries: usize,
    /// Maximum logical message payload bytes across all directed caches.
    pub message_payload_bytes: usize,
    /// Maximum estimated owned message bytes, including spare capacity and keys.
    pub message_owned_bytes: usize,
    /// Maximum retained prepared slices across scalar kinds.
    pub prepared_entries: usize,
    /// Maximum logical prepared matrix payload bytes.
    pub prepared_payload_bytes: usize,
    /// Maximum prepared payload plus estimated map metadata bytes.
    pub prepared_owned_bytes: usize,
}

impl CacheDiagnostics {
    pub(super) fn record(&mut self, other: Self) {
        self.message_entries = self.message_entries.max(other.message_entries);
        self.message_payload_bytes = self.message_payload_bytes.max(other.message_payload_bytes);
        self.message_owned_bytes = self.message_owned_bytes.max(other.message_owned_bytes);
        self.prepared_entries = self.prepared_entries.max(other.prepared_entries);
        self.prepared_payload_bytes = self
            .prepared_payload_bytes
            .max(other.prepared_payload_bytes);
        self.prepared_owned_bytes = self.prepared_owned_bytes.max(other.prepared_owned_bytes);
    }
}
