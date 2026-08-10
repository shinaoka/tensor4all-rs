//! Opaque types and enums for the reduced C API surface.

use std::ffi::c_void;

#[cfg(test)]
use tensor4all_core::TensorStorageError;
use tensor4all_core::{
    DynIndex, FactorizeAlg, SingularValueMeasure, SvdTruncationPolicy, TensorDynLen,
    ThresholdScale, TruncationRule,
};
use tensor4all_quanticstransform::BoundaryCondition as QuanticsBoundaryCondition;
use tensor4all_tensorbackend::StorageKind;
use tensor4all_treetn::treetn::contraction::ContractionMethod;
use tensor4all_treetn::{CanonicalForm as TreeCanonicalForm, DefaultTreeTN};

/// Internal dynamic index type wrapped by `t4a_index`.
pub(crate) type InternalIndex = DynIndex;

/// Internal tensor type wrapped by `t4a_tensor`.
pub(crate) type InternalTensor = TensorDynLen;

/// Internal tree tensor network type wrapped by `t4a_treetn`.
pub(crate) type InternalTreeTN = DefaultTreeTN<usize>;

/// Internal C API evaluator handle state.
#[derive(Clone)]
pub(crate) struct InternalTreeTNEvaluatorHandle {
    pub(crate) tree: InternalTreeTN,
    pub(crate) indices: Vec<InternalIndex>,
    pub(crate) center: Option<usize>,
    pub(crate) scalar_kind: t4a_scalar_kind,
}

/// Opaque index type for the C API.
#[repr(C)]
pub struct t4a_index {
    pub(crate) _private: *const c_void,
}

impl t4a_index {
    /// Create a wrapper from an internal index value.
    pub(crate) fn new(index: InternalIndex) -> Self {
        Self {
            _private: Box::into_raw(Box::new(index)) as *const c_void,
        }
    }

    /// Borrow the wrapped index.
    pub(crate) fn inner(&self) -> &InternalIndex {
        unsafe { &*(self._private as *const InternalIndex) }
    }
}

impl Clone for t4a_index {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_index {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalIndex);
            }
        }
    }
}

unsafe impl Send for t4a_index {}
unsafe impl Sync for t4a_index {}

/// Scalar kind used by the reduced tensor surface.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_scalar_kind {
    /// Real-valued `f64` data.
    F64 = 0,
    /// Complex-valued `Complex64` data represented as interleaved doubles.
    C64 = 1,
}

impl t4a_scalar_kind {
    /// Classify a tensor by the scalar family exposed through the C API.
    pub(crate) fn from_tensor(tensor: &TensorDynLen) -> Self {
        if tensor.is_complex() {
            Self::C64
        } else {
            Self::F64
        }
    }
}

/// Storage layout kind used by tensor payload inspection APIs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_storage_kind {
    /// Dense storage whose payload axes match the logical tensor axes.
    Dense = 0,
    /// Diagonal storage whose logical axes all share one payload axis.
    Diagonal = 1,
    /// General structured storage with explicit payload-axis classes.
    Structured = 2,
}

impl From<StorageKind> for t4a_storage_kind {
    fn from(kind: StorageKind) -> Self {
        match kind {
            StorageKind::Dense => Self::Dense,
            StorageKind::Diagonal => Self::Diagonal,
            StorageKind::Structured => Self::Structured,
        }
    }
}

/// Opaque tensor type for the C API.
#[repr(C)]
pub struct t4a_tensor {
    pub(crate) _private: *const c_void,
    #[cfg(test)]
    pub(crate) test_storage_error: Option<TensorStorageError>,
}

impl t4a_tensor {
    /// Create a wrapper from an internal tensor value.
    pub(crate) fn new(tensor: InternalTensor) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tensor)) as *const c_void,
            #[cfg(test)]
            test_storage_error: None,
        }
    }

    /// Borrow the wrapped tensor.
    pub(crate) fn inner(&self) -> &InternalTensor {
        unsafe { &*(self._private as *const InternalTensor) }
    }

    #[cfg(test)]
    pub(crate) fn with_test_storage_error(
        tensor: InternalTensor,
        error: TensorStorageError,
    ) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tensor)) as *const c_void,
            test_storage_error: Some(error),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_storage_error(&self) -> Option<TensorStorageError> {
        self.test_storage_error.clone()
    }
}

impl Clone for t4a_tensor {
    fn clone(&self) -> Self {
        #[cfg(test)]
        if let Some(error) = self.test_storage_error() {
            return Self::with_test_storage_error(self.inner().clone(), error);
        }
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_tensor {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalTensor);
            }
        }
    }
}

unsafe impl Send for t4a_tensor {}
unsafe impl Sync for t4a_tensor {}

/// Opaque tree tensor network type for the C API.
#[repr(C)]
pub struct t4a_treetn {
    pub(crate) _private: *const c_void,
}

impl t4a_treetn {
    /// Create a wrapper from an internal TreeTN value.
    pub(crate) fn new(treetn: InternalTreeTN) -> Self {
        Self {
            _private: Box::into_raw(Box::new(treetn)) as *const c_void,
        }
    }

    /// Borrow the wrapped TreeTN.
    pub(crate) fn inner(&self) -> &InternalTreeTN {
        unsafe { &*(self._private as *const InternalTreeTN) }
    }

    /// Mutably borrow the wrapped TreeTN.
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTreeTN {
        unsafe { &mut *(self._private as *mut InternalTreeTN) }
    }
}

impl Clone for t4a_treetn {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_treetn {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalTreeTN);
            }
        }
    }
}

unsafe impl Send for t4a_treetn {}
unsafe impl Sync for t4a_treetn {}

/// Opaque reusable TreeTN evaluator type for the C API.
#[repr(C)]
pub struct t4a_treetn_evaluator {
    pub(crate) _private: *const c_void,
}

impl t4a_treetn_evaluator {
    /// Create a wrapper from internal reusable evaluator state.
    pub(crate) fn new(handle: InternalTreeTNEvaluatorHandle) -> Self {
        Self {
            _private: Box::into_raw(Box::new(handle)) as *const c_void,
        }
    }

    /// Borrow the wrapped reusable evaluator state.
    pub(crate) fn inner(&self) -> &InternalTreeTNEvaluatorHandle {
        unsafe { &*(self._private as *const InternalTreeTNEvaluatorHandle) }
    }

    /// Mutably borrow the wrapped reusable evaluator state.
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTreeTNEvaluatorHandle {
        unsafe { &mut *(self._private as *mut InternalTreeTNEvaluatorHandle) }
    }
}

impl Clone for t4a_treetn_evaluator {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_treetn_evaluator {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalTreeTNEvaluatorHandle);
            }
        }
    }
}

unsafe impl Send for t4a_treetn_evaluator {}
unsafe impl Sync for t4a_treetn_evaluator {}

/// Canonical form used for orthogonalization and canonicalization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_canonical_form {
    /// Unitary / QR-based canonicalization.
    Unitary = 0,
    /// LU-based canonicalization.
    LU = 1,
    /// Cross-interpolation-based canonicalization.
    CI = 2,
}

impl From<TreeCanonicalForm> for t4a_canonical_form {
    fn from(form: TreeCanonicalForm) -> Self {
        match form {
            TreeCanonicalForm::Unitary => Self::Unitary,
            TreeCanonicalForm::LU => Self::LU,
            TreeCanonicalForm::CI => Self::CI,
        }
    }
}

impl From<t4a_canonical_form> for TreeCanonicalForm {
    fn from(form: t4a_canonical_form) -> Self {
        match form {
            t4a_canonical_form::Unitary => Self::Unitary,
            t4a_canonical_form::LU => Self::LU,
            t4a_canonical_form::CI => Self::CI,
        }
    }
}

/// Threshold scaling used by the C API SVD truncation policy.
///
/// Related types:
/// - [`t4a_singular_value_measure`] selects whether the threshold is applied to
///   singular values or squared singular values.
/// - [`t4a_truncation_rule`] selects whether the threshold is checked per value
///   or against the discarded suffix sum.
/// - [`t4a_svd_truncation_policy`] combines all three knobs with the numeric
///   threshold.
///
/// # Examples
///
/// ```
/// use tensor4all_capi::t4a_threshold_scale;
/// use tensor4all_core::ThresholdScale;
///
/// assert_eq!(
///     ThresholdScale::from(t4a_threshold_scale::Relative),
///     ThresholdScale::Relative
/// );
/// assert_eq!(
///     ThresholdScale::from(t4a_threshold_scale::Absolute),
///     ThresholdScale::Absolute
/// );
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_threshold_scale {
    /// Compare the threshold to a singular-value-derived reference scale.
    #[default]
    Relative = 0,
    /// Compare the threshold directly to the measured singular-value quantity.
    Absolute = 1,
}

impl From<ThresholdScale> for t4a_threshold_scale {
    fn from(scale: ThresholdScale) -> Self {
        match scale {
            ThresholdScale::Relative => Self::Relative,
            ThresholdScale::Absolute => Self::Absolute,
        }
    }
}

impl From<t4a_threshold_scale> for ThresholdScale {
    fn from(scale: t4a_threshold_scale) -> Self {
        match scale {
            t4a_threshold_scale::Relative => Self::Relative,
            t4a_threshold_scale::Absolute => Self::Absolute,
        }
    }
}

/// Singular-value quantity used by the C API SVD truncation policy.
///
/// Related types:
/// - [`t4a_threshold_scale`] selects whether the threshold is relative or
///   absolute.
/// - [`t4a_truncation_rule`] selects whether truncation is per value or based
///   on the discarded tail sum.
/// - [`t4a_svd_truncation_policy`] stores the full C-facing SVD policy.
///
/// # Examples
///
/// ```
/// use tensor4all_capi::t4a_singular_value_measure;
/// use tensor4all_core::SingularValueMeasure;
///
/// assert_eq!(
///     SingularValueMeasure::from(t4a_singular_value_measure::Value),
///     SingularValueMeasure::Value
/// );
/// assert_eq!(
///     SingularValueMeasure::from(t4a_singular_value_measure::SquaredValue),
///     SingularValueMeasure::SquaredValue
/// );
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_singular_value_measure {
    /// Measure singular values directly.
    #[default]
    Value = 0,
    /// Measure squared singular values.
    SquaredValue = 1,
}

impl From<SingularValueMeasure> for t4a_singular_value_measure {
    fn from(measure: SingularValueMeasure) -> Self {
        match measure {
            SingularValueMeasure::Value => Self::Value,
            SingularValueMeasure::SquaredValue => Self::SquaredValue,
        }
    }
}

impl From<t4a_singular_value_measure> for SingularValueMeasure {
    fn from(measure: t4a_singular_value_measure) -> Self {
        match measure {
            t4a_singular_value_measure::Value => Self::Value,
            t4a_singular_value_measure::SquaredValue => Self::SquaredValue,
        }
    }
}

/// Rank-selection rule used by the C API SVD truncation policy.
///
/// Related types:
/// - [`t4a_threshold_scale`] sets whether the threshold is relative or
///   absolute.
/// - [`t4a_singular_value_measure`] sets whether values or squared values are
///   measured.
/// - [`t4a_svd_truncation_policy`] combines the rule with the other SVD
///   truncation knobs.
///
/// # Examples
///
/// ```
/// use tensor4all_capi::t4a_truncation_rule;
/// use tensor4all_core::TruncationRule;
///
/// assert_eq!(
///     TruncationRule::from(t4a_truncation_rule::PerValue),
///     TruncationRule::PerValue
/// );
/// assert_eq!(
///     TruncationRule::from(t4a_truncation_rule::DiscardedTailSum),
///     TruncationRule::DiscardedTailSum
/// );
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_truncation_rule {
    /// Apply the threshold independently to each singular value.
    #[default]
    PerValue = 0,
    /// Apply the threshold to the cumulative discarded tail.
    DiscardedTailSum = 1,
}

impl From<TruncationRule> for t4a_truncation_rule {
    fn from(rule: TruncationRule) -> Self {
        match rule {
            TruncationRule::PerValue => Self::PerValue,
            TruncationRule::DiscardedTailSum => Self::DiscardedTailSum,
        }
    }
}

impl From<t4a_truncation_rule> for TruncationRule {
    fn from(rule: t4a_truncation_rule) -> Self {
        match rule {
            t4a_truncation_rule::PerValue => Self::PerValue,
            t4a_truncation_rule::DiscardedTailSum => Self::DiscardedTailSum,
        }
    }
}

/// Explicit SVD truncation policy exposed through the C API.
///
/// This is the ABI-facing mirror of [`tensor4all_core::SvdTruncationPolicy`].
/// Pass it to SVD-based C API entry points to choose the threshold value, the
/// threshold scale, whether singular values are squared first, and whether the
/// truncation rule is per value or tail-sum based.
///
/// Related types:
/// - [`t4a_threshold_scale`] controls relative vs absolute thresholding.
/// - [`t4a_singular_value_measure`] controls value vs squared-value
///   measurement.
/// - [`t4a_truncation_rule`] controls per-value vs discarded-tail-sum
///   truncation.
///
/// # Examples
///
/// ```
/// use tensor4all_capi::{
///     t4a_singular_value_measure, t4a_svd_truncation_policy, t4a_threshold_scale,
///     t4a_truncation_rule,
/// };
/// use tensor4all_core::{SingularValueMeasure, SvdTruncationPolicy, ThresholdScale, TruncationRule};
///
/// let ffi_policy = t4a_svd_truncation_policy {
///     threshold: 1e-8,
///     scale: t4a_threshold_scale::Absolute,
///     measure: t4a_singular_value_measure::SquaredValue,
///     rule: t4a_truncation_rule::DiscardedTailSum,
/// };
/// let core_policy = SvdTruncationPolicy::from(ffi_policy);
/// assert_eq!(core_policy.threshold, 1e-8);
/// assert_eq!(core_policy.scale, ThresholdScale::Absolute);
/// assert_eq!(core_policy.measure, SingularValueMeasure::SquaredValue);
/// assert_eq!(core_policy.rule, TruncationRule::DiscardedTailSum);
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct t4a_svd_truncation_policy {
    /// Threshold value used by the selected scale/rule combination.
    pub threshold: f64,
    /// Relative or absolute threshold interpretation.
    pub scale: t4a_threshold_scale,
    /// Singular-value quantity used for thresholding.
    pub measure: t4a_singular_value_measure,
    /// Rank-selection rule.
    pub rule: t4a_truncation_rule,
}

impl Default for t4a_svd_truncation_policy {
    fn default() -> Self {
        Self::from(SvdTruncationPolicy::new(1e-12))
    }
}

impl From<SvdTruncationPolicy> for t4a_svd_truncation_policy {
    fn from(policy: SvdTruncationPolicy) -> Self {
        Self {
            threshold: policy.threshold,
            scale: policy.scale.into(),
            measure: policy.measure.into(),
            rule: policy.rule.into(),
        }
    }
}

impl From<t4a_svd_truncation_policy> for SvdTruncationPolicy {
    fn from(policy: t4a_svd_truncation_policy) -> Self {
        SvdTruncationPolicy {
            threshold: policy.threshold,
            scale: policy.scale.into(),
            measure: policy.measure.into(),
            rule: policy.rule.into(),
        }
    }
}

/// Factorization algorithms exposed through the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_factorize_alg {
    /// Singular value decomposition.
    #[default]
    SVD = 0,
    /// QR decomposition.
    QR = 1,
    /// Rank-revealing LU decomposition.
    LU = 2,
    /// Cross interpolation.
    CI = 3,
}

impl From<FactorizeAlg> for t4a_factorize_alg {
    fn from(alg: FactorizeAlg) -> Self {
        match alg {
            FactorizeAlg::SVD => Self::SVD,
            FactorizeAlg::QR => Self::QR,
            FactorizeAlg::LU => Self::LU,
            FactorizeAlg::CI => Self::CI,
        }
    }
}

impl From<t4a_factorize_alg> for FactorizeAlg {
    fn from(alg: t4a_factorize_alg) -> Self {
        match alg {
            t4a_factorize_alg::SVD => Self::SVD,
            t4a_factorize_alg::QR => Self::QR,
            t4a_factorize_alg::LU => Self::LU,
            t4a_factorize_alg::CI => Self::CI,
        }
    }
}

/// TreeTN contraction method exposed through the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_contract_method {
    /// Zip-up contraction with on-the-fly truncation.
    #[default]
    Zipup = 0,
    /// Variational fit-based contraction.
    Fit = 1,
    /// Naive dense/reference contraction for generic TreeTN contraction.
    ///
    /// `t4a_treetn_contract` requires an explicit nonzero dense-element limit
    /// for this method. `t4a_treetn_apply_operator_chain` uses a dedicated
    /// local-exact apply path instead of generic full-dense TreeTN contraction.
    Naive = 2,
}

impl From<ContractionMethod> for t4a_contract_method {
    fn from(method: ContractionMethod) -> Self {
        match method {
            ContractionMethod::Zipup => Self::Zipup,
            ContractionMethod::Fit => Self::Fit,
            ContractionMethod::Naive => Self::Naive,
        }
    }
}

impl From<t4a_contract_method> for ContractionMethod {
    fn from(method: t4a_contract_method) -> Self {
        match method {
            t4a_contract_method::Zipup => Self::Zipup,
            t4a_contract_method::Fit => Self::Fit,
            t4a_contract_method::Naive => Self::Naive,
        }
    }
}

/// Boundary-condition choices for quantics transform materialization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_boundary_condition {
    /// Periodic boundary conditions.
    Periodic = 0,
    /// Open boundary conditions.
    Open = 1,
    /// Anti-periodic boundary conditions.
    AntiPeriodic = 2,
}

impl From<t4a_boundary_condition> for QuanticsBoundaryCondition {
    fn from(value: t4a_boundary_condition) -> Self {
        match value {
            t4a_boundary_condition::Periodic => Self::Periodic,
            t4a_boundary_condition::AntiPeriodic => Self::AntiPeriodic,
            t4a_boundary_condition::Open => Self::Open,
        }
    }
}

impl From<QuanticsBoundaryCondition> for t4a_boundary_condition {
    fn from(value: QuanticsBoundaryCondition) -> Self {
        match value {
            QuanticsBoundaryCondition::Periodic => Self::Periodic,
            QuanticsBoundaryCondition::AntiPeriodic => Self::AntiPeriodic,
            QuanticsBoundaryCondition::Open => Self::Open,
        }
    }
}

/// Canonical binary QTT layout kinds supported by the reduced C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_qtt_layout_kind {
    /// Level-major interleaved layout.
    Interleaved = 0,
    /// Fused-per-level layout.
    Fused = 1,
}

/// Internal immutable descriptor for canonical binary QTT layouts.
#[derive(Debug, Clone)]
pub(crate) struct InternalQttLayout {
    kind: t4a_qtt_layout_kind,
    variable_resolutions: Vec<usize>,
    nsites: usize,
}

impl InternalQttLayout {
    /// Validate and build a canonical layout descriptor.
    pub(crate) fn new(
        kind: t4a_qtt_layout_kind,
        variable_resolutions: Vec<usize>,
    ) -> Result<Self, String> {
        if variable_resolutions.is_empty() {
            return Err("nvariables must be greater than zero".to_string());
        }
        if variable_resolutions.contains(&0) {
            return Err("variable_resolutions must all be greater than zero".to_string());
        }

        let first = variable_resolutions[0];
        let all_equal = variable_resolutions.iter().all(|&r| r == first);
        let nsites = match kind {
            t4a_qtt_layout_kind::Interleaved => {
                if !all_equal {
                    return Err(
                        "interleaved layouts require all variable_resolutions to match".to_string(),
                    );
                }
                first
                    .checked_mul(variable_resolutions.len())
                    .ok_or_else(|| "interleaved layout site count overflows usize".to_string())?
            }
            t4a_qtt_layout_kind::Fused => {
                if !all_equal {
                    return Err(
                        "fused layouts require all variable_resolutions to match".to_string()
                    );
                }
                first
            }
        };

        Ok(Self {
            kind,
            variable_resolutions,
            nsites,
        })
    }

    /// Number of logical variables.
    pub(crate) fn nvariables(&self) -> usize {
        self.variable_resolutions.len()
    }

    /// Number of physical sites in the canonical layout.
    pub(crate) fn nsites(&self) -> usize {
        self.nsites
    }

    /// Bit resolution of a single logical variable.
    pub(crate) fn resolution(&self, variable: usize) -> usize {
        self.variable_resolutions[variable]
    }

    /// Layout kind.
    pub(crate) fn kind(&self) -> t4a_qtt_layout_kind {
        self.kind
    }
}

/// Opaque canonical QTT layout descriptor.
#[repr(C)]
pub struct t4a_qtt_layout {
    pub(crate) _private: *const c_void,
}

impl t4a_qtt_layout {
    /// Wrap an internal layout descriptor.
    pub(crate) fn new(layout: InternalQttLayout) -> Self {
        Self {
            _private: Box::into_raw(Box::new(layout)) as *const c_void,
        }
    }

    /// Borrow the wrapped layout descriptor.
    pub(crate) fn inner(&self) -> &InternalQttLayout {
        unsafe { &*(self._private as *const InternalQttLayout) }
    }
}

impl Clone for t4a_qtt_layout {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_qtt_layout {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalQttLayout);
            }
        }
    }
}

unsafe impl Send for t4a_qtt_layout {}
unsafe impl Sync for t4a_qtt_layout {}

#[cfg(test)]
mod tests;
