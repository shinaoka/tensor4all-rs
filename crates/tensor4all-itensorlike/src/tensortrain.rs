//! Main Tensor Train type as a wrapper around TreeTN.
//!
//! This module provides the `TensorTrain` type, which represents a Tensor Train
//! (also known as MPS) with orthogonality tracking, inspired by ITensorMPS.jl.
//!
//! Internally, TensorTrain is implemented as a thin wrapper around
//! `TreeTN<TensorDynLen, usize>` where node names are site indices (0, 1, 2, ...).

use num_complex::Complex64;
use std::any::TypeId;
use std::env;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tensor4all_core::{
    common_inds, contract_pair, contract_pair_with_operand_options, hascommoninds, DynIndex,
    IndexLike, PairwiseContractionOptions,
};
use tensor4all_core::{
    AnyScalar, Canonical, CommonScalar, DirectSumResult, FactorizeAlg, FactorizeError,
    FactorizeOptions, FactorizeResult, LinearizationOrder, TensorConstructionLike,
    TensorContractionLike, TensorDynLen, TensorDynLenError, TensorElement, TensorFactorizationLike,
    TensorIndex, TensorVectorSpace,
};
use tensor4all_treetn::{CanonicalizationOptions, TreeTN, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_svd_truncation_options, CanonicalForm, TruncateOptions};

#[derive(Debug, Default)]
struct TensorTrainInnerProfile {
    sim_internal_inds: Duration,
    node_lookup: Duration,
    right_tensor_clone: Duration,
    conj: Duration,
    contract: Duration,
    final_dims: Duration,
    sum: Duration,
}

fn tensortrain_inner_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("T4A_PROFILE_TT_INNER").is_ok())
}

fn profile_tt_inner_section<T>(enabled: bool, slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    if !enabled {
        return f();
    }
    let started = Instant::now();
    let result = f();
    *slot += started.elapsed();
    result
}

fn print_tt_inner_profile(profile: &TensorTrainInnerProfile, length: usize) {
    let total = profile.sim_internal_inds
        + profile.node_lookup
        + profile.right_tensor_clone
        + profile.conj
        + profile.contract
        + profile.final_dims
        + profile.sum;
    eprintln!(
        "tt_inner_profile,L={length},total_ms={:.6},sim_internal_inds_ms={:.6},node_lookup_ms={:.6},right_tensor_clone_ms={:.6},conj_ms={:.6},contract_ms={:.6},final_dims_ms={:.6},sum_ms={:.6}",
        total.as_secs_f64() * 1.0e3,
        profile.sim_internal_inds.as_secs_f64() * 1.0e3,
        profile.node_lookup.as_secs_f64() * 1.0e3,
        profile.right_tensor_clone.as_secs_f64() * 1.0e3,
        profile.conj.as_secs_f64() * 1.0e3,
        profile.contract.as_secs_f64() * 1.0e3,
        profile.final_dims.as_secs_f64() * 1.0e3,
        profile.sum.as_secs_f64() * 1.0e3,
    );
}

/// Tensor Train with orthogonality tracking.
/// This type represents a tensor train as a sequence of tensors with tracked
/// orthogonality limits. It is inspired by ITensorMPS.jl but uses
/// 0-indexed sites (Rust convention).
/// Unlike traditional MPS which assumes one physical index per site, this
/// implementation allows each site to have multiple site indices.
/// # Orthogonality Tracking
/// The tensor train tracks orthogonality using `ortho_region` from the underlying TreeTN:
/// - When `ortho_region` is empty, no orthogonality is assumed
/// - When `ortho_region` contains a single site, that site is the orthogonality center
/// # Implementation
/// Internally wraps `TreeTN<TensorDynLen, usize>` where node names are site indices.
/// This allows reuse of TreeTN's canonicalization and contraction algorithms.
/// # Examples
/// Build a 2-site tensor train and query its properties:
/// ```
/// use tensor4all_itensorlike::TensorTrain;
/// use tensor4all_core::{DynIndex, TensorDynLen, Index};
/// use tensor4all_core::DynId;
/// // Site indices and link index
/// let s0 = Index::new_with_size(DynId(0), 2);
/// let link = Index::new_with_size(DynId(1), 3);
/// let s1 = Index::new_with_size(DynId(2), 2);
/// let t0 = TensorDynLen::from_dense(
///     vec![s0.clone(), link.clone()],
///     (0..6).map(|i| i as f64).collect(),
/// ).unwrap();
/// let t1 = TensorDynLen::from_dense(
///     vec![link.clone(), s1.clone()],
///     (0..6).map(|i| i as f64).collect(),
/// ).unwrap();
/// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
/// assert_eq!(tt.len(), 2);
/// assert_eq!(tt.maxbonddim(), 3);
/// assert!(!tt.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// The underlying TreeTN with linear chain topology.
    /// Node names are usize (0, 1, 2, ...) representing site indices.
    pub(crate) treetn: TreeTN<TensorDynLen, usize>,
    /// The canonical form used (if known).
    canonical_form: Option<CanonicalForm>,
}

#[derive(Debug)]
struct PackedSiteTensor<T> {
    left_dim: usize,
    physical_dim: usize,
    right_dim: usize,
    data: Vec<T>,
}

impl<T: Copy> PackedSiteTensor<T> {
    fn get(&self, left: usize, physical: usize, right: usize) -> T {
        debug_assert!(left < self.left_dim);
        debug_assert!(physical < self.physical_dim);
        debug_assert!(right < self.right_dim);

        let idx = left + self.left_dim * (physical + self.physical_dim * right);
        self.data[idx]
    }
}

trait NormAccumScalar: CommonScalar {
    fn into_nonnegative_real(self) -> f64;
}

impl NormAccumScalar for f64 {
    fn into_nonnegative_real(self) -> f64 {
        self.max(0.0)
    }
}

impl NormAccumScalar for Complex64 {
    fn into_nonnegative_real(self) -> f64 {
        self.re.max(0.0)
    }
}

impl TensorTrain {
    /// Create a new tensor train from a vector of tensors.
    ///
    /// The tensor train is created with no assumed orthogonality.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors representing the tensor train
    ///
    /// # Returns
    ///
    /// A new tensor train with no orthogonality.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor's site dimensions are incompatible with
    /// /// its neighbors (a shape mismatch) or the chain is structurally
    /// /// inconsistent (an invalid-state failure).
    ///
    pub fn new(tensors: Vec<TensorDynLen>) -> Result<Self> {
        if tensors.is_empty() {
            // Create an empty TreeTN
            let treetn = TreeTN::<TensorDynLen, usize>::new();
            return Ok(Self {
                treetn,
                canonical_form: None,
            });
        }

        // Validate that adjacent tensors share exactly one common index (the link)
        for i in 0..tensors.len() - 1 {
            let left = &tensors[i];
            let right = &tensors[i + 1];

            let common = common_inds(left.indices(), right.indices());
            if common.is_empty() {
                return Err(TensorTrainError::InvalidStructure {
                    message: format!(
                        "No common index between tensors at sites {} and {}",
                        i,
                        i + 1
                    ),
                });
            }
            if common.len() > 1 {
                return Err(TensorTrainError::InvalidStructure {
                    message: format!(
                        "Multiple common indices ({}) between tensors at sites {} and {}",
                        common.len(),
                        i,
                        i + 1
                    ),
                });
            }
        }

        // Create node names: 0, 1, 2, ..., n-1
        let node_names: Vec<usize> = (0..tensors.len()).collect();

        // Create TreeTN with from_tensors (auto-connects by shared index IDs)
        let treetn =
            TreeTN::<TensorDynLen, usize>::from_tensors(tensors, node_names).map_err(|e| {
                TensorTrainError::InvalidStructure {
                    message: format!("Failed to create TreeTN: {}", e),
                }
            })?;

        let tt = Self {
            treetn,
            canonical_form: None,
        };
        Ok(tt)
    }

    /// Create a new tensor train with specified orthogonality center.
    ///
    /// This is useful when constructing a tensor train that is already in canonical form.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors representing the tensor train
    /// * `llim` - Left orthogonality limit (for compatibility; only used to compute center)
    /// * `rlim` - Right orthogonality limit (for compatibility; only used to compute center)
    /// * `canonical_form` - The method used for canonicalization (if any)
    /// # Errors
    ///
    /// Returns an error when the orthogonality center is out of range (an
    /// /// out of bounds failure) or orthogonalization fails.
    ///
    pub fn with_ortho(
        tensors: Vec<TensorDynLen>,
        llim: i32,
        rlim: i32,
        canonical_form: Option<CanonicalForm>,
    ) -> Result<Self> {
        let mut tt = Self::new(tensors)?;

        // Convert llim/rlim to ortho center
        // When llim + 2 == rlim, ortho center is at llim + 1
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < tt.len() as i32 {
            let center = (llim + 1) as usize;
            tt.treetn.set_canonical_region(vec![center]).map_err(|e| {
                TensorTrainError::InvalidStructure {
                    message: format!("Failed to set ortho region: {}", e),
                }
            })?;
        }

        tt.canonical_form = canonical_form;
        Ok(tt)
    }

    /// Create a TensorTrain from an existing TreeTN and canonical form.
    ///
    /// This is a crate-internal constructor used by `contract` and `linsolve`.
    pub(crate) fn from_inner(
        treetn: TreeTN<TensorDynLen, usize>,
        canonical_form: Option<CanonicalForm>,
    ) -> Result<Self> {
        let mut node_names = treetn.node_names();
        node_names.sort_unstable();
        let mut tt = Self {
            treetn,
            canonical_form,
        };
        for (site, old_name) in node_names.into_iter().enumerate() {
            if old_name != site {
                tt.treetn.rename_node(&old_name, site).map_err(|e| {
                    TensorTrainError::InvalidStructure {
                        message: format!("Failed to renumber TensorTrain sites: {}", e),
                    }
                })?;
            }
        }
        Ok(tt)
    }

    /// Create a tensor train from a linear-chain [`TreeTN`].
    ///
    /// The input tree must use `usize` node names and represent a tensor-train
    /// chain. Node names are renumbered to `0..len` when necessary. Site tensor
    /// index order is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error when the input TreeTN is not a linear chain (an
    /// /// invalid-topology failure) or a site conversion fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    /// use tensor4all_treetn::TreeTN;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let site0 = DynIndex::new_dyn(2);
    /// let link = DynIndex::new_bond(1)?;
    /// let site1 = DynIndex::new_dyn(2);
    /// let t0 = TensorDynLen::from_dense(vec![site0, link.clone()], vec![1.0, 0.0])?;
    /// let t1 = TensorDynLen::from_dense(vec![link, site1], vec![2.0, 0.0])?;
    /// let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize])?;
    ///
    /// let tt = TensorTrain::from_treetn(tree)?;
    /// assert_eq!(tt.len(), 2);
    /// assert_eq!(
    ///     tt.siteinds()
    ///         .into_iter()
    ///         .map(|indices| indices.into_iter().map(|idx| idx.size()).collect::<Vec<_>>())
    ///         .collect::<Vec<_>>(),
    ///     vec![vec![2], vec![2]]
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_treetn(treetn: TreeTN<TensorDynLen, usize>) -> Result<Self> {
        Self::from_inner(treetn, None)
    }

    /// Consume this tensor train and return its underlying [`TreeTN`].
    ///
    /// Use this when a chain MPS must be passed to APIs that operate on general
    /// tree tensor networks. The returned tree preserves the tensor and index
    /// metadata stored in the tensor train.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let site = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![site], vec![1.0, 2.0])?;
    /// let tt = TensorTrain::new(vec![tensor])?;
    ///
    /// let tree = tt.into_treetn();
    /// assert_eq!(tree.node_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_treetn(self) -> TreeTN<TensorDynLen, usize> {
        self.treetn
    }

    /// Get a reference to the underlying TreeTN.
    ///
    /// This is a crate-internal accessor used by `contract` and `linsolve`.
    pub(crate) fn as_treetn(&self) -> &TreeTN<TensorDynLen, usize> {
        &self.treetn
    }

    /// Number of sites (tensors) in the tensor train.
    #[inline]
    pub fn len(&self) -> usize {
        self.treetn.node_count()
    }

    /// Check if the tensor train is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.treetn.node_count() == 0
    }

    /// Left orthogonality limit.
    ///
    /// Sites `0..llim` are guaranteed to be left-orthogonal.
    /// Returns -1 if no sites are left-orthogonal.
    #[inline]
    pub fn llim(&self) -> i32 {
        match self.orthocenter() {
            Some(center) => center as i32 - 1,
            None => -1,
        }
    }

    /// Right orthogonality limit.
    ///
    /// Sites `rlim..len()` are guaranteed to be right-orthogonal.
    /// Returns `len() + 1` if no sites are right-orthogonal.
    #[inline]
    pub fn rlim(&self) -> i32 {
        match self.orthocenter() {
            Some(center) => center as i32 + 1,
            None => self.len() as i32 + 1,
        }
    }

    /// Set the left orthogonality limit.
    #[inline]
    pub fn set_llim(&mut self, llim: i32) {
        // Convert to ortho center if possible
        let rlim = self.rlim();
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < self.len() as i32 {
            let center = (llim + 1) as usize;
            let _ = self.treetn.set_canonical_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.treetn.set_canonical_region(Vec::<usize>::new());
        }
    }

    /// Set the right orthogonality limit.
    #[inline]
    pub fn set_rlim(&mut self, rlim: i32) {
        // Convert to ortho center if possible
        let llim = self.llim();
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < self.len() as i32 {
            let center = (llim + 1) as usize;
            let _ = self.treetn.set_canonical_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.treetn.set_canonical_region(Vec::<usize>::new());
        }
    }

    /// Get the orthogonality center range.
    ///
    /// Returns the range of sites that may not be orthogonal.
    /// If the tensor train is fully left-orthogonal, returns an empty range at the end.
    /// If the tensor train is fully right-orthogonal, returns an empty range at the beginning.
    pub fn ortho_lims(&self) -> Range<usize> {
        let llim = self.llim();
        let rlim = self.rlim();
        let start = (llim + 1).max(0) as usize;
        let end = rlim.max(0) as usize;
        start..end.min(self.len())
    }

    /// Check if the tensor train has a single orthogonality center.
    ///
    /// Returns true if there is exactly one site that is not guaranteed to be orthogonal.
    #[inline]
    pub fn isortho(&self) -> bool {
        self.treetn.canonical_region().len() == 1
    }

    /// Get the orthogonality center (0-indexed).
    ///
    /// Returns `Some(site)` if the tensor train has a single orthogonality center,
    /// `None` otherwise.
    pub fn orthocenter(&self) -> Option<usize> {
        let region = self.treetn.canonical_region();
        if region.len() == 1 {
            // Node name IS the site index since V = usize
            region.iter().next().copied()
        } else {
            None
        }
    }

    /// Get the canonicalization method used.
    #[inline]
    pub fn canonical_form(&self) -> Option<CanonicalForm> {
        self.canonical_form
    }

    /// Set the canonicalization method.
    #[inline]
    pub fn set_canonical_form(&mut self, method: Option<CanonicalForm>) {
        self.canonical_form = method;
    }

    /// Get a reference to the tensor at the given site.
    ///
    #[inline]
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure).
    ///
    pub fn tensor(&self, site: usize) -> Result<&TensorDynLen> {
        self.tensor_checked(site)
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure).
    ///
    pub fn tensor_checked(&self, site: usize) -> Result<&TensorDynLen> {
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }
        let node_idx =
            self.treetn
                .node_index(&site)
                .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                    site,
                    length: self.len(),
                })?;
        self.treetn
            .tensor(node_idx)
            .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            })
    }

    /// Get a mutable reference to the tensor at the given site.
    ///
    #[inline]
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure).
    ///
    pub fn tensor_mut(&mut self, site: usize) -> Result<&mut TensorDynLen> {
        self.tensor_mut_checked(site)
    }

    /// Get a mutable reference to the tensor at the given site.
    ///
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynId, Index, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let link = Index::new_with_size(DynId(1), 3);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone(), link.clone()], vec![1.0; 6]).unwrap();
    /// let t1 = TensorDynLen::from_dense(vec![link, s1], vec![2.0; 6]).unwrap();
    /// let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// assert_eq!(tt.tensor_mut_checked(0).unwrap().indices()[0], s0);
    /// assert!(tt.tensor_mut_checked(2).is_err());
    /// ```
    pub fn tensor_mut_checked(&mut self, site: usize) -> Result<&mut TensorDynLen> {
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }
        let length = self.len();
        let node_idx = self
            .treetn
            .node_index(&site)
            .ok_or(TensorTrainError::SiteOutOfBounds { site, length })?;
        self.treetn
            .tensor_mut(node_idx)
            .ok_or_else(|| TensorTrainError::InvalidStructure {
                message: format!("missing tensor storage for site {site}"),
            })
    }

    /// Get a reference to all tensors.
    #[inline]
    pub fn tensors(&self) -> Vec<&TensorDynLen> {
        (0..self.len())
            .filter_map(|site| {
                let node_idx = self.treetn.node_index(&site)?;
                self.treetn.tensor(node_idx)
            })
            .collect()
    }

    /// Get a mutable reference to all tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when the internal site-to-node mapping is inconsistent (a
    /// graph consistency failure).
    #[inline]
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn tensors_mut(&mut self) -> Result<Vec<&mut TensorDynLen>> {
        self.tensors_mut_checked()
    }

    /// Get mutable references to all tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynId, Index, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let link = Index::new_with_size(DynId(1), 3);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    /// let t0 = TensorDynLen::from_dense(vec![s0, link.clone()], vec![1.0; 6]).unwrap();
    /// let t1 = TensorDynLen::from_dense(vec![link, s1], vec![2.0; 6]).unwrap();
    /// let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let tensors = tt.tensors_mut_checked().unwrap();
    /// assert_eq!(tensors.len(), 2);
    /// assert_eq!(tensors[0].indices().len(), 2);
    /// assert_eq!(tensors[1].indices().len(), 2);
    /// ```
    pub fn tensors_mut_checked(&mut self) -> Result<Vec<&mut TensorDynLen>> {
        let length = self.len();
        let node_indices: Vec<_> = (0..length)
            .map(|site| {
                self.treetn
                    .node_index(&site)
                    .ok_or(TensorTrainError::SiteOutOfBounds { site, length })
            })
            .collect::<Result<_>>()?;
        let mut tensor_ptrs = Vec::with_capacity(node_indices.len());
        for (site, node_idx) in node_indices.into_iter().enumerate() {
            let tensor = self.treetn.tensor_mut(node_idx).ok_or_else(|| {
                TensorTrainError::InvalidStructure {
                    message: format!("missing tensor storage for site {site}"),
                }
            })?;
            tensor_ptrs.push(tensor as *mut TensorDynLen);
        }

        // SAFETY: TensorTrain site names are unique, so each site resolves to a
        // distinct TreeTN node. We collect at most one pointer per node and do
        // not mutate the network structure before converting those pointers back
        // into mutable references.
        Ok(unsafe { tensor_ptrs.into_iter().map(|tensor| &mut *tensor).collect() })
    }

    /// Get the link index between sites `i` and `i+1`.
    ///
    /// Returns `None` if `i >= len() - 1` or if no common index exists.
    pub fn linkind(&self, i: usize) -> Option<DynIndex> {
        if i >= self.len().saturating_sub(1) {
            return None;
        }

        let left_node = self.treetn.node_index(&i)?;
        let right_node = self.treetn.node_index(&(i + 1))?;
        let left = self.treetn.tensor(left_node)?;
        let right = self.treetn.tensor(right_node)?;
        let common = common_inds(left.indices(), right.indices());
        common.into_iter().next()
    }

    /// Get all link indices.
    ///
    /// Returns a vector of length `len() - 1` containing the link indices.
    pub fn linkinds(&self) -> Vec<DynIndex> {
        (0..self.len().saturating_sub(1))
            .filter_map(|i| self.linkind(i))
            .collect()
    }

    /// Create a copy with all link indices replaced by new unique IDs.
    ///
    /// This is useful for computing inner products where two tensor trains
    /// share link indices. By simulating (replacing) the link indices in one
    /// of the tensor trains, they can be contracted over site indices only.
    ///
    /// # Returns
    ///
    /// A tensor train with the same site indices and fresh link indices.
    ///
    /// # Errors
    ///
    /// Returns an error when the link-index relabeling fails (an invalid-index
    /// /// failure).
    ///
    pub fn sim_linkinds(&self) -> Result<Self> {
        if self.len() <= 1 {
            return Ok(self.clone());
        }

        // Build replacement pairs: (old_link, new_link) for each link index
        let old_links = self.linkinds();
        let new_links: Vec<_> = old_links.iter().map(|idx| idx.sim()).collect();
        let replacements: Vec<_> = old_links
            .iter()
            .cloned()
            .zip(new_links.iter().cloned())
            .collect();

        // Replace link indices in each tensor and rebuild
        let mut new_tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            let tensor = self.tensor_checked(site)?;
            let mut new_tensor = tensor.clone();
            for (old_idx, new_idx) in &replacements {
                new_tensor = new_tensor.replaceind(old_idx, new_idx).map_err(|err| {
                    TensorTrainError::operation_source(
                        "failed to replace simulated link index",
                        err,
                    )
                })?;
            }
            new_tensors.push(new_tensor);
        }

        Self::new(new_tensors)
    }

    fn normalize_site_tensor_orders(&mut self) -> Result<()> {
        for site in 0..self.len() {
            self.normalize_site_tensor_order(site)?;
        }
        Ok(())
    }

    fn has_simple_linear_links(&self) -> Result<bool> {
        if self.len() <= 1 {
            return Ok(true);
        }

        for site in 0..self.len() - 1 {
            let left = self.tensor_checked(site)?;
            let right = self.tensor_checked(site + 1)?;
            if common_inds(left.indices(), right.indices()).len() > 1 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn can_normalize_site_tensor_order(&self, site: usize) -> Result<bool> {
        let left_ok = if site > 0 {
            let left = self.tensor_checked(site - 1)?;
            let current = self.tensor_checked(site)?;
            common_inds(left.indices(), current.indices()).len() <= 1
        } else {
            true
        };
        let right_ok = if site + 1 < self.len() {
            let current = self.tensor_checked(site)?;
            let right = self.tensor_checked(site + 1)?;
            common_inds(current.indices(), right.indices()).len() <= 1
        } else {
            true
        };
        Ok(left_ok && right_ok)
    }

    fn with_explicit_unit_links(&self) -> Result<Self> {
        if self.len() <= 1 {
            return Ok(self.clone());
        }

        let mut tensors = (0..self.len())
            .map(|site| self.tensor_checked(site).cloned())
            .collect::<Result<Vec<_>>>()?;
        for site in 0..tensors.len() - 1 {
            let common = common_inds(tensors[site].indices(), tensors[site + 1].indices());
            if common.len() > 1 {
                let fused_dim = common.iter().try_fold(1usize, |acc, index| {
                    acc.checked_mul(index.dim())
                        .ok_or_else(|| TensorTrainError::InvalidStructure {
                            message: "parallel link fusion would overflow index dimension"
                                .to_string(),
                        })
                })?;
                let fused_link = DynIndex::new_dyn(fused_dim);
                tensors[site] = tensors[site]
                    .fuse_indices(&common, fused_link.clone(), LinearizationOrder::ColumnMajor)
                    .map_err(|e| {
                        TensorTrainError::operation_source("failed to fuse parallel TT links", e)
                    })?;
                tensors[site + 1] = tensors[site + 1]
                    .fuse_indices(&common, fused_link, LinearizationOrder::ColumnMajor)
                    .map_err(|e| {
                        TensorTrainError::operation_source("failed to fuse parallel TT links", e)
                    })?;
                continue;
            }
            if common.len() == 1 {
                continue;
            }

            let link = DynIndex::new_dyn(1);
            let left_link =
                <TensorDynLen as TensorConstructionLike>::ones(std::slice::from_ref(&link))
                    .map_err(|e| {
                        TensorTrainError::operation_source(
                            "failed to build implicit unit link tensor",
                            anyhow::Error::new(e),
                        )
                    })?;
            tensors[site] = tensors[site].outer_product(&left_link).map_err(|e| {
                TensorTrainError::operation_source(
                    "failed to attach implicit unit link",
                    anyhow::Error::new(e),
                )
            })?;

            let right_link =
                <TensorDynLen as TensorConstructionLike>::ones(&[link]).map_err(|e| {
                    TensorTrainError::operation_source(
                        "failed to build implicit unit link tensor",
                        anyhow::Error::new(e),
                    )
                })?;
            tensors[site + 1] = tensors[site + 1].outer_product(&right_link).map_err(|e| {
                TensorTrainError::operation_source(
                    "failed to attach implicit unit link",
                    anyhow::Error::new(e),
                )
            })?;
        }

        Self::new(tensors)
    }

    fn normalize_site_tensor_order(&mut self, site: usize) -> Result<()> {
        if !self.can_normalize_site_tensor_order(site)? {
            return Ok(());
        }

        let tensor = self.tensor_checked(site)?.clone();
        let current = tensor.indices().to_vec();
        let left = if site > 0 {
            self.linkind(site - 1)
        } else {
            None
        };
        let right = if site + 1 < self.len() {
            self.linkind(site)
        } else {
            None
        };

        let mut desired = Vec::with_capacity(current.len());
        if let Some(ref left_link) = left {
            desired.push(left_link.clone());
        }
        desired.extend(
            current
                .iter()
                .filter(|idx| Some(*idx) != left.as_ref() && Some(*idx) != right.as_ref())
                .cloned(),
        );
        if let Some(ref right_link) = right {
            desired.push(right_link.clone());
        }

        if desired == current {
            return Ok(());
        }

        let normalized =
            tensor
                .permuteinds(&desired)
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!(
                        "Failed to normalize site tensor index order at site {}: {}",
                        site, e
                    ),
                })?;
        self.set_tensor_raw(site, normalized)
    }

    /// Get the site indices (non-link indices) for all sites.
    ///
    /// For each site, returns a vector of indices that are not shared with
    /// adjacent tensors (i.e., the "physical" or "site" indices).
    pub fn siteinds(&self) -> Vec<Vec<DynIndex>> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let Ok(tensor) = self.tensor_checked(i) else {
                return Vec::new();
            };
            let mut site_inds: Vec<DynIndex> = tensor.indices().to_vec();

            // Remove link to left neighbor
            if i > 0 {
                if let Some(link) = self.linkind(i - 1) {
                    site_inds.retain(|idx| idx != &link);
                }
            }

            // Remove link to right neighbor
            if i < self.len() - 1 {
                if let Some(link) = self.linkind(i) {
                    site_inds.retain(|idx| idx != &link);
                }
            }

            result.push(site_inds);
        }

        result
    }

    /// Get the bond dimension at link `i` (between sites `i` and `i+1`).
    ///
    /// Returns `None` if `i >= len() - 1`.
    pub fn bond_dim(&self, i: usize) -> Option<usize> {
        self.linkind(i).map(|idx| idx.size())
    }

    /// Get all bond dimensions.
    ///
    /// Returns a vector of length `len() - 1`.
    pub fn bond_dims(&self) -> Vec<usize> {
        self.linkinds().iter().map(|idx| idx.size()).collect()
    }

    /// Get the maximum bond dimension.
    pub fn maxbonddim(&self) -> usize {
        self.bond_dims().into_iter().max().unwrap_or(1)
    }

    /// Check if two adjacent tensors share an index.
    pub fn haslink(&self, i: usize) -> bool {
        if i >= self.len().saturating_sub(1) {
            return false;
        }
        let left_node = self.treetn.node_index(&i);
        let right_node = self.treetn.node_index(&(i + 1));
        match (left_node, right_node) {
            (Some(l), Some(r)) => {
                let left = self.treetn.tensor(l);
                let right = self.treetn.tensor(r);
                match (left, right) {
                    (Some(l), Some(r)) => hascommoninds(l.indices(), r.indices()),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking.
    fn set_tensor_raw(&mut self, site: usize, tensor: TensorDynLen) -> Result<()> {
        let node_idx =
            self.treetn
                .node_index(&site)
                .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                    site,
                    length: self.len(),
                })?;
        self.treetn.replace_tensor(node_idx, tensor).map_err(|e| {
            TensorTrainError::InvalidStructure {
                message: format!("Failed to replace tensor at site {}: {}", site, e),
            }
        })?;
        Ok(())
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking.
    ///
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure) or
    /// /// the new tensor has incompatible dimensions (a shape mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynId, Index, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let link = Index::new_with_size(DynId(1), 3);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone(), link.clone()], vec![1.0; 6]).unwrap();
    /// let t1 = TensorDynLen::from_dense(vec![link.clone(), s1], vec![2.0; 6]).unwrap();
    /// let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let replacement = TensorDynLen::from_dense(vec![s0, link], vec![3.0; 6]).unwrap();
    /// tt.set_tensor(0, replacement).unwrap();
    /// assert_eq!(tt.tensor(0).unwrap().to_vec::<f64>().unwrap(), vec![3.0; 6]);
    /// ```
    pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen) -> Result<()> {
        self.set_tensor_checked(site, tensor)
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking.
    ///
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure) or
    /// /// the new tensor has incompatible dimensions (a shape mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynId, Index, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let link = Index::new_with_size(DynId(1), 3);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone(), link.clone()], vec![1.0; 6]).unwrap();
    /// let t1 = TensorDynLen::from_dense(vec![link.clone(), s1], vec![2.0; 6]).unwrap();
    /// let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let replacement = TensorDynLen::from_dense(vec![s0, link], vec![4.0; 6]).unwrap();
    /// tt.set_tensor_checked(0, replacement).unwrap();
    /// assert!(tt.set_tensor_checked(2, tt.tensor(0).unwrap().clone()).is_err());
    /// ```
    pub fn set_tensor_checked(&mut self, site: usize, tensor: TensorDynLen) -> Result<()> {
        self.set_tensor_raw(site, tensor)?;
        // Invalidate orthogonality
        self.treetn
            .set_canonical_region(Vec::<usize>::new())
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Failed to clear canonical region: {}", e),
            })?;
        Ok(())
    }

    /// Orthogonalize the tensor train to have orthogonality center at the given site.
    ///
    /// This function performs a series of factorizations to make the tensor train
    /// canonical with orthogonality center at `site`.
    ///
    /// # Arguments
    ///
    /// * `site` - The target site for the orthogonality center (0-indexed)
    ///
    /// # Errors
    ///
    /// Returns an error if the factorization fails or if the site is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::TensorTrain;
    /// use tensor4all_core::{DynIndex, TensorDynLen, Index, DynId};
    ///
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let link = Index::new_with_size(DynId(1), 3);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    ///
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), link.clone()],
    ///     (0..6).map(|i| i as f64).collect(),
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![link.clone(), s1.clone()],
    ///     (0..6).map(|i| i as f64).collect(),
    /// ).unwrap();
    ///
    /// let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    /// assert!(!tt.isortho());
    ///
    /// // Orthogonalize to site 0
    /// tt.orthogonalize(0).unwrap();
    /// assert!(tt.isortho());
    /// assert_eq!(tt.orthocenter(), Some(0));
    /// ```
    pub fn orthogonalize(&mut self, site: usize) -> Result<()> {
        self.orthogonalize_with(site, CanonicalForm::Unitary)
    }

    /// Orthogonalize with a specified canonical form.
    ///
    /// # Arguments
    ///
    /// * `site` - The target site for the orthogonality center (0-indexed)
    /// * `form` - The canonical form to use:
    ///   - `Unitary`: Uses QR decomposition, each tensor is isometric
    ///   - `LU`: Uses LU decomposition, one factor has unit diagonal
    ///   - `CI`: Uses Cross Interpolation
    /// # Errors
    ///
    /// Returns an error when `site` is out of range (an out of bounds failure) or
    /// /// orthogonalization fails (a backend or non-convergence failure).
    ///
    pub fn orthogonalize_with(&mut self, site: usize, form: CanonicalForm) -> Result<()> {
        if self.is_empty() {
            return Err(TensorTrainError::Empty);
        }
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }

        // Use TreeTN's canonicalize (accepts node names and CanonicalizationOptions)
        // Since V = usize, node names are site indices
        let options = CanonicalizationOptions::forced().with_form(form);
        self.treetn = std::mem::take(&mut self.treetn)
            .canonicalize(vec![site], options)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Canonicalize failed: {}", e),
            })?;

        self.canonical_form = Some(form);
        Ok(())
    }

    /// Truncate the tensor train bond dimensions.
    ///
    /// This delegates to the TreeTN's truncate_mut method, which performs a
    /// two-site sweep with Euler tour traversal for optimal truncation.
    ///
    /// Note: The `site_range` option in `TruncateOptions` is currently ignored
    /// as the underlying TreeTN truncation operates on the full network.
    ///
    /// # Errors
    ///
    /// Returns an error when truncation fails (a backend or SVD failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::{TensorTrain, TruncateOptions};
    /// use tensor4all_core::{DynIndex, TensorDynLen, Index, DynId};
    ///
    /// // Build a 3-site tensor train with bond dimension 4
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let l01 = Index::new_with_size(DynId(1), 4);
    /// let s1 = Index::new_with_size(DynId(2), 2);
    /// let l12 = Index::new_with_size(DynId(3), 4);
    /// let s2 = Index::new_with_size(DynId(4), 2);
    ///
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), l01.clone()],
    ///     (0..8).map(|i| i as f64).collect(),
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![l01.clone(), s1.clone(), l12.clone()],
    ///     (0..32).map(|i| i as f64).collect(),
    /// ).unwrap();
    /// let t2 = TensorDynLen::from_dense(
    ///     vec![l12.clone(), s2.clone()],
    ///     (0..8).map(|i| i as f64).collect(),
    /// ).unwrap();
    ///
    /// let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
    /// assert_eq!(tt.maxbonddim(), 4);
    ///
    /// // Truncate bond dimension to at most 2
    /// let opts = TruncateOptions::svd().with_max_rank(2);
    /// tt.truncate(&opts).unwrap();
    /// assert!(tt.maxbonddim() <= 2);
    /// ```
    pub fn truncate(&mut self, options: &TruncateOptions) -> Result<()> {
        if self.len() <= 1 {
            return Ok(());
        }

        validate_svd_truncation_options(options.max_rank(), options.svd_policy())?;

        // Convert TruncateOptions to TruncationOptions
        let mut treetn_options = TruncationOptions::new();
        if let Some(policy) = options.svd_policy() {
            treetn_options = treetn_options.with_svd_policy(policy);
        }
        if let Some(max_rank) = options.max_rank() {
            treetn_options = treetn_options.with_max_rank(max_rank);
        }

        // Truncate towards the last site (rightmost) as the canonical center
        // This matches ITensor convention where truncation sweeps left-to-right
        let center = self.len() - 1;

        self.treetn
            .truncate_mut([center], treetn_options)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("TreeTN truncation failed: {}", e),
            })?;

        self.canonical_form = Some(CanonicalForm::Unitary);

        Ok(())
    }

    /// Compute the inner product (dot product) of two tensor trains.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both tensor trains must have the same site indices (same IDs).
    /// Link indices may differ between the two tensor trains.
    ///
    /// # Errors
    ///
    /// Returns an error when the two tensor trains have incompatible site spaces
    /// /// (a shape mismatch) or the contraction fails (a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::TensorTrain;
    /// use tensor4all_core::{DynIndex, TensorDynLen, Index, DynId, AnyScalar};
    ///
    /// // Single-site tensor train with values [1.0, 0.0]
    /// let s0 = Index::new_with_size(DynId(0), 2);
    /// let t = TensorDynLen::from_dense(
    ///     vec![s0.clone()],
    ///     vec![1.0_f64, 0.0],
    /// ).unwrap();
    ///
    /// let tt = TensorTrain::new(vec![t]).unwrap();
    ///
    /// // <tt | tt> = 1.0^2 + 0.0^2 = 1.0
    /// let result = tt.inner(&tt).unwrap();
    /// assert!((result.real() - 1.0).abs() < 1e-10);
    /// ```
    pub fn inner(&self, other: &Self) -> Result<AnyScalar> {
        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidStructure {
                message: format!(
                    "Tensor trains must have the same length for inner product: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(AnyScalar::new_real(0.0));
        }

        // Sequential bra-ket contraction along the chain: O(N·D²·d).
        // TreeTN::inner() uses contract_naive which is O(d^N) and OOMs for large N.
        let profile_enabled = tensortrain_inner_profile_enabled();
        let mut profile = TensorTrainInnerProfile::default();
        let other_sim =
            profile_tt_inner_section(profile_enabled, &mut profile.sim_internal_inds, || {
                other.treetn.sim_internal_inds()
            });

        let node_idx = |ttn: &TreeTN<TensorDynLen, usize>, site: usize| {
            ttn.node_index(&site)
                .ok_or_else(|| TensorTrainError::InvalidStructure {
                    message: format!("missing node for site {site}"),
                })
        };

        // Start with leftmost tensors - contract over site indices only
        let mut env = {
            let a0 = profile_tt_inner_section(profile_enabled, &mut profile.node_lookup, || {
                self.tensor_checked(0)
            })?;
            let b0_node =
                profile_tt_inner_section(profile_enabled, &mut profile.node_lookup, || {
                    node_idx(&other_sim, 0)
                })?;
            let b0 =
                profile_tt_inner_section(profile_enabled, &mut profile.right_tensor_clone, || {
                    Ok::<TensorDynLen, TensorTrainError>(
                        other_sim
                            .tensor(b0_node)
                            .ok_or_else(|| TensorTrainError::InvalidStructure {
                                message: "missing tensor for site 0 in simulated right operand"
                                    .to_string(),
                            })?
                            .clone(),
                    )
                })?;
            profile_tt_inner_section(profile_enabled, &mut profile.contract, || {
                contract_pair_with_operand_options(
                    a0,
                    &b0,
                    PairwiseContractionOptions::new().with_lhs_conj(true),
                )
                .map_err(|err| {
                    TensorTrainError::operation_source("failed to contract leftmost tensors", err)
                })
            })?
        };

        // Sweep through remaining sites
        for i in 1..self.len() {
            let ai = profile_tt_inner_section(profile_enabled, &mut profile.node_lookup, || {
                self.tensor_checked(i)
            })?;
            let bi_node =
                profile_tt_inner_section(profile_enabled, &mut profile.node_lookup, || {
                    node_idx(&other_sim, i)
                })?;
            let bi = profile_tt_inner_section(profile_enabled, &mut profile.node_lookup, || {
                other_sim
                    .tensor(bi_node)
                    .ok_or_else(|| TensorTrainError::InvalidStructure {
                        message: format!("missing tensor for site {i} in simulated right operand"),
                    })
            })?;

            // Contract: env * conj(A_i) (over self's link index)
            env = profile_tt_inner_section(profile_enabled, &mut profile.contract, || {
                contract_pair_with_operand_options(
                    &env,
                    ai,
                    PairwiseContractionOptions::new().with_rhs_conj(true),
                )
                .map_err(|err| {
                    TensorTrainError::operation_source(
                        format!("failed to contract environment with site {i}"),
                        err,
                    )
                })
            })?;
            // Contract: result * B_i (over other's link index and site indices)
            env = profile_tt_inner_section(profile_enabled, &mut profile.contract, || {
                contract_pair(&env, bi).map_err(|err| {
                    TensorTrainError::operation_source(
                        format!("failed to contract right operand at site {i}"),
                        err,
                    )
                })
            })?;
        }

        // Result should be a scalar (0-dimensional tensor)
        let dims =
            profile_tt_inner_section(profile_enabled, &mut profile.final_dims, || env.dims());
        let total_size: usize = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        if total_size != 1 {
            return Err(TensorTrainError::InvalidStructure {
                message: format!(
                    "inner product did not contract to a scalar: got dims {:?}",
                    dims
                ),
            });
        }
        let result = profile_tt_inner_section(profile_enabled, &mut profile.sum, || {
            env.sum().map_err(|err| {
                TensorTrainError::operation_source("failed to sum scalar inner-product tensor", err)
            })
        });
        if profile_enabled {
            print_tt_inner_profile(&profile, self.len());
        }
        result
    }

    /// Compute the squared norm of the tensor train.
    ///
    /// Returns `<self | self>` = ||self||^2.
    ///
    /// # Errors
    /// Returns a [`TensorTrainError`] when storage or contraction diagnostics
    /// prevent evaluating the norm.
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![site], vec![3.0_f64, 4.0])?;
    /// let tt = TensorTrain::new(vec![tensor])?;
    /// assert!((tt.norm_squared()? - 25.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Note
    /// For linear tensor trains with one site index per site, this uses a
    /// specialized chain contraction instead of the generic inner-product path.
    /// Due to numerical errors, the final scalar can be very slightly negative,
    /// so the returned value is clamped to be non-negative.
    pub fn norm_squared(&self) -> Result<f64> {
        match self.norm_squared_fast_path()? {
            Some(value) if value.is_nan() => Err(TensorTrainError::TensorDynLen {
                source: TensorDynLenError::NaNInput {
                    operation: "norm_squared",
                },
            }),
            Some(value) => Ok(value),
            None => self.inner(self).and_then(|value| {
                let value = value.real();
                if value.is_nan() {
                    Err(TensorTrainError::TensorDynLen {
                        source: TensorDynLenError::NaNInput {
                            operation: "norm_squared",
                        },
                    })
                } else {
                    Ok(value.max(0.0))
                }
            }),
        }
    }

    /// Compute the norm of the tensor train.
    ///
    /// Returns ||self|| = sqrt(<self | self>).
    ///
    /// # Errors
    /// Returns a [`TensorTrainError`] when storage or contraction diagnostics
    /// prevent evaluating the norm.
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![site], vec![3.0_f64, 4.0])?;
    /// let tt = TensorTrain::new(vec![tensor])?;
    /// assert!((tt.norm()? - 5.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self) -> Result<f64> {
        Ok(self.norm_squared()?.sqrt())
    }

    fn norm_squared_fast_path(&self) -> Result<Option<f64>> {
        if self.is_empty() {
            return Ok(Some(0.0));
        }
        if !self.has_simple_linear_links()? {
            return Ok(None);
        }
        if self
            .siteinds()
            .iter()
            .any(|site_indices| site_indices.len() != 1)
        {
            return Ok(None);
        }

        let mut normalized = self.clone();
        normalized.normalize_site_tensor_orders()?;

        if let Some(sites) = Self::pack_normalized_sites::<f64>(&normalized)? {
            return Ok(Some(Self::norm_squared_from_packed_sites(&sites)));
        }
        if let Some(sites) = Self::pack_normalized_sites::<Complex64>(&normalized)? {
            return Ok(Some(Self::norm_squared_from_packed_sites(&sites)));
        }

        Ok(None)
    }

    fn pack_normalized_sites<T: TensorElement>(
        tt: &Self,
    ) -> Result<Option<Vec<PackedSiteTensor<T>>>> {
        let mut sites = Vec::with_capacity(tt.len());

        for site in 0..tt.len() {
            let tensor = tt.tensor_checked(site)?;
            let left_dim = if site == 0 {
                1
            } else {
                match tt.linkind(site - 1) {
                    Some(link) => link.size(),
                    None => return Ok(None),
                }
            };
            let right_dim = if site + 1 == tt.len() {
                1
            } else {
                match tt.linkind(site) {
                    Some(link) => link.size(),
                    None => return Ok(None),
                }
            };
            let total_size: usize = tensor.dims().iter().product();
            let boundary_size = match left_dim.checked_mul(right_dim) {
                Some(size) => size,
                None => return Ok(None),
            };
            if boundary_size == 0 || !total_size.is_multiple_of(boundary_size) {
                return Ok(None);
            }

            let dtype_matches = if TypeId::of::<T>() == TypeId::of::<f64>() {
                tensor.is_f64()
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                tensor.is_c64()
            } else {
                false
            };
            if !dtype_matches {
                return Ok(None);
            }

            sites.push(PackedSiteTensor {
                left_dim,
                physical_dim: total_size / boundary_size,
                right_dim,
                data: tensor
                    .to_vec::<T>()
                    .map_err(|err| TensorTrainError::TensorDynLen {
                        source: TensorDynLenError::Materialization {
                            source: Arc::from(err.into_boxed_dyn_error()),
                        },
                    })?,
            });
        }

        Ok(Some(sites))
    }

    fn norm_squared_from_packed_sites<T: NormAccumScalar>(sites: &[PackedSiteTensor<T>]) -> f64 {
        if sites.is_empty() {
            return 0.0;
        }

        let first = &sites[0];
        let mut current = vec![T::zero(); first.right_dim * first.right_dim];

        for physical in 0..first.physical_dim {
            for right in 0..first.right_dim {
                let value = first.get(0, physical, right);
                for right_conj in 0..first.right_dim {
                    let idx = right * first.right_dim + right_conj;
                    current[idx] = current[idx] + value * first.get(0, physical, right_conj).conj();
                }
            }
        }

        for site in &sites[1..] {
            let mut next = vec![T::zero(); site.right_dim * site.right_dim];

            for left in 0..site.left_dim {
                for left_conj in 0..site.left_dim {
                    let env = current[left * site.left_dim + left_conj];
                    for physical in 0..site.physical_dim {
                        for right in 0..site.right_dim {
                            let value = site.get(left, physical, right);
                            for right_conj in 0..site.right_dim {
                                let idx = right * site.right_dim + right_conj;
                                next[idx] = next[idx]
                                    + env
                                        * value
                                        * site.get(left_conj, physical, right_conj).conj();
                            }
                        }
                    }
                }
            }

            current = next;
        }

        current[0].into_nonnegative_real()
    }

    /// Convert the tensor train to a single dense tensor.
    ///
    /// This contracts all tensors in the train along their link indices,
    /// producing a single tensor with only site indices.
    ///
    /// # Warning
    /// This operation can be very expensive for large tensor trains,
    /// as the result size grows exponentially with the number of sites.
    ///
    /// # Returns
    /// A single tensor containing all site indices, or an error if the
    /// tensor train is empty.
    ///
    /// # Errors
    ///
    /// Returns an error when the dense materialization fails (a materialization
    /// /// or backend failure).
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let s0 = DynIndex::new_dyn(2);
    /// let link = DynIndex::new_dyn(1);
    /// let s1 = DynIndex::new_dyn(2);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone(), link.clone()], vec![1.0, 2.0])?;
    /// let t1 = TensorDynLen::from_dense(vec![link.clone(), s1.clone()], vec![3.0, 4.0])?;
    ///
    /// let tt = TensorTrain::new(vec![t0, t1])?;
    /// let dense = tt.to_dense()?;
    ///
    /// assert_eq!(dense.dims(), vec![2, 2]);
    /// assert_eq!(dense.to_vec::<f64>()?, vec![3.0, 6.0, 4.0, 8.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_dense(&self) -> Result<TensorDynLen> {
        if self.is_empty() {
            return Err(TensorTrainError::InvalidStructure {
                message: "Cannot convert empty tensor train to dense".to_string(),
            });
        }

        self.treetn.contract_to_tensor().map_err(|source| {
            TensorTrainError::operation_source("Failed to contract to dense", source)
        })
    }

    /// Compute an explicit dense maximum absolute value.
    ///
    /// This method first materializes the full tensor train with
    /// [`Self::to_dense`], then computes the dense tensor's L-infinity norm.
    /// Use it only for small reference/debug checks. Long tensor-train
    /// comparisons should use scalable residual norms such as
    /// `tt1.axpby(1, tt2, -1)?.norm() / tt2.norm()`.
    ///
    /// # Returns
    /// The maximum absolute element in the dense tensor represented by this
    /// tensor train.
    ///
    /// # Errors
    ///
    /// Returns an error when the dense materialization fails (a materialization
    /// /// or backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let site = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![site], vec![-2.0, 3.0])?;
    /// let tt = TensorTrain::new(vec![tensor])?;
    /// assert_eq!(tt.dense_maxabs()?, 3.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn dense_maxabs(&self) -> Result<f64> {
        self.to_dense()?
            .maxabs()
            .map_err(|source| TensorTrainError::TensorDynLen { source })
    }

    /// Add two tensor trains using direct-sum construction.
    ///
    /// This creates a new tensor train where each tensor is the direct sum of the
    /// corresponding tensors from self and other, with bond dimensions merged.
    /// The result has bond dimensions equal to the sum of the input bond dimensions.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to add
    ///
    /// # Returns
    /// A new tensor train representing the sum.
    ///
    /// # Errors
    ///
    /// Returns an error when the two tensor trains have incompatible site spaces
    /// /// (a shape mismatch) or the direct-sum construction fails.
    ///
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.is_empty() && other.is_empty() {
            return Ok(Self::default());
        }

        if self.is_empty() {
            return Ok(other.clone());
        }

        if other.is_empty() {
            return Ok(self.clone());
        }

        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidStructure {
                message: format!(
                    "Tensor trains must have the same length for addition: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        let result_inner =
            self.treetn
                .add(&other.treetn)
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!("TT addition failed: {}", e),
                })?;

        Self::from_inner(result_inner, None)?.with_explicit_unit_links()
    }

    /// Add two tensor trains after reindexing `other` to this tensor train's site space.
    ///
    /// This method is useful when two tensor trains represent the same logical
    /// vector space but carry distinct site-index IDs, for example after
    /// independent contractions. It pairs site indices site-by-site by
    /// dimension, rewrites `other` to use `self`'s site-index IDs, then performs
    /// strict tensor-train addition.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor train to reindex and add. It must have the same
    ///   chain length and compatible site dimensions as `self`.
    ///
    /// # Returns
    ///
    /// A tensor train representing `self + other`, with site indices matching
    /// `self`.
    ///
    /// # Errors
    ///
    /// Returns an error when the two tensor trains have incompatible site spaces
    /// /// (a shape mismatch) or the reindexed addition fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynId, Index, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// fn one_site(id: u64, values: Vec<f64>) -> TensorTrain {
    ///     let site = Index::new_with_size(DynId(id), 2);
    ///     let tensor = TensorDynLen::from_dense(vec![site], values).unwrap();
    ///     TensorTrain::new(vec![tensor]).unwrap()
    /// }
    ///
    /// let lhs = one_site(0, vec![1.0, 2.0]);
    /// let rhs = one_site(1, vec![3.0, 4.0]);
    /// let sum = lhs.add_reindexed_like_self(&rhs).unwrap();
    ///
    /// let dense = sum.to_dense().unwrap();
    /// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![4.0, 6.0]);
    /// assert_eq!(dense.indices()[0], lhs.siteinds()[0][0]);
    /// ```
    pub fn add_reindexed_like_self(&self, other: &Self) -> Result<Self> {
        if self.is_empty() && other.is_empty() {
            return Ok(Self::default());
        }

        if self.is_empty() {
            return Ok(other.clone());
        }

        if other.is_empty() {
            return Ok(self.clone());
        }

        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidStructure {
                message: format!(
                    "Tensor trains must have the same length for reindexed addition: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        let lhs = self.with_explicit_unit_links()?;
        let rhs = other.with_explicit_unit_links()?;

        let result_inner = lhs
            .treetn
            .add_reindexed_like_self(&rhs.treetn)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("TT reindexed addition failed: {}", e),
            })?;

        Self::from_inner(result_inner, None)
    }

    /// Scale the tensor train by a scalar.
    ///
    /// Only one tensor (the first non-empty site) is scaled to avoid scalar^n scaling.
    /// This is correct because the tensor train represents a product of tensors,
    /// so scaling one factor scales the entire product.
    ///
    /// # Arguments
    /// * `scalar` - The scalar to multiply by
    ///
    /// # Returns
    /// A new tensor train scaled by the given scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when the scaling fails (a dtype mismatch or backend
    /// /// failure).
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
    /// use tensor4all_itensorlike::TensorTrain;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let s0 = DynIndex::new_dyn(2);
    /// let tt = TensorTrain::new(vec![TensorDynLen::from_dense(
    ///     vec![s0.clone()],
    ///     vec![1.0, 2.0],
    /// )?])?;
    ///
    /// let scaled = tt.scale(AnyScalar::new_real(2.0))?;
    /// assert_eq!(scaled.to_dense()?.to_vec::<f64>()?, vec![2.0, 4.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            let tensor = self.tensor_checked(site)?;
            if site == 0 {
                // Scale only the first tensor
                let scaled = tensor.scale(scalar.clone()).map_err(|e| {
                    TensorTrainError::operation_source("failed to scale tensor at site 0", e)
                })?;
                tensors.push(scaled);
            } else {
                tensors.push(tensor.clone());
            }
        }

        Self::new(tensors)
    }

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// This is equivalent to `self.scale(a)?.add(&other.scale(b)?)`.
    ///
    /// # Arguments
    /// * `a` - Scalar coefficient for self
    /// * `other` - The other tensor train
    /// * `b` - Scalar coefficient for other
    ///
    /// # Returns
    /// A new tensor train representing `a * self + b * other`.
    ///
    /// # Note
    /// The bond dimension of the result is the sum of the bond dimensions
    /// of the two input tensor trains (before any truncation).
    /// # Errors
    ///
    /// Returns an error when the two tensor trains have incompatible site spaces
    /// /// (a shape mismatch) or the axpby computation fails (a backend failure).
    ///
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        let scaled_self = self.scale(a)?;
        let scaled_other = other.scale(b)?;
        scaled_self.add(&scaled_other)
    }
}

// Implement Default for TensorTrain to allow std::mem::take
impl Default for TensorTrain {
    fn default() -> Self {
        Self {
            treetn: TreeTN::new(),
            canonical_form: None,
        }
    }
}

// ============================================================================
// TensorIndex implementation for TensorTrain
// ============================================================================

impl TensorIndex for TensorTrain {
    type Index = DynIndex;
    type Error = TensorTrainError;

    fn external_indices(&self) -> Vec<Self::Index> {
        // Delegate to the internal TreeTN's TensorIndex implementation
        self.treetn.external_indices()
    }

    fn num_external_indices(&self) -> usize {
        self.treetn.num_external_indices()
    }

    fn replaceind(
        &self,
        old: &Self::Index,
        new: &Self::Index,
    ) -> std::result::Result<Self, Self::Error> {
        // Delegate to the internal TreeTN's replaceind
        // After replacement, canonical form may be invalid, so set to None
        let treetn = self
            .treetn
            .replaceind(old, new)
            .map_err(anyhow::Error::new)?;
        Self::from_inner(treetn, None).map_err(Self::Error::from)
    }

    fn replaceinds(
        &self,
        old: &[Self::Index],
        new: &[Self::Index],
    ) -> std::result::Result<Self, Self::Error> {
        let treetn = self
            .treetn
            .replaceinds(old, new)
            .map_err(anyhow::Error::new)?;
        Self::from_inner(treetn, None).map_err(Self::Error::from)
    }
}

// ============================================================================
// TensorLike implementation for TensorTrain
// ============================================================================

impl TensorVectorSpace for TensorTrain {
    // ========================================================================
    // GMRES-required methods (fully supported)
    // ========================================================================

    fn axpby(
        &self,
        a: AnyScalar,
        other: &Self,
        b: AnyScalar,
    ) -> std::result::Result<Self, Self::Error> {
        TensorTrain::axpby(self, a, other, b).map_err(Self::Error::from)
    }

    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error> {
        TensorTrain::scale(self, scalar).map_err(Self::Error::from)
    }

    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error> {
        self.inner(other).map_err(Self::Error::from)
    }

    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        TensorTrain::norm_squared(self)
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support TensorVectorSpace::maxabs without explicit dense materialization; use TensorTrain::dense_maxabs() for small reference checks or norm-based residuals for long tensor trains".to_string(),
        })
    }
}

impl TensorContractionLike for TensorTrain {
    // ========================================================================
    // Tensor network operations
    // ========================================================================

    fn conj(&self) -> Self {
        let mut result = self.clone();
        if let Ok(tensors) = result.tensors_mut_checked() {
            for tensor in tensors {
                let conjugated = tensor.conj();
                *tensor = conjugated;
            }
        }
        result
    }

    fn contract(_tensors: &[&Self]) -> std::result::Result<Self, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support TensorContractionLike::contract; use TensorTrain::contract() method instead".to_string(),
        })
    }

    fn direct_sum(
        &self,
        _other: &Self,
        _pairs: &[(Self::Index, Self::Index)],
    ) -> std::result::Result<DirectSumResult<Self>, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support direct_sum; use add() instead".to_string(),
        })
    }

    fn outer_product(&self, _other: &Self) -> std::result::Result<Self, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support outer_product".to_string(),
        })
    }

    fn permuteinds(&self, _new_order: &[Self::Index]) -> std::result::Result<Self, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support permuteinds".to_string(),
        })
    }

    fn fuse_indices(
        &self,
        _old_indices: &[Self::Index],
        _new_index: Self::Index,
        _order: LinearizationOrder,
    ) -> std::result::Result<Self, Self::Error> {
        Err(TensorTrainError::OperationError {
            message: "TensorTrain does not support TensorContractionLike::fuse_indices".to_string(),
        })
    }
}

impl TensorFactorizationLike for TensorTrain {
    fn factorize(
        &self,
        _left_inds: &[Self::Index],
        _options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::UnsupportedStorage(
            "TensorTrain does not support factorize; use orthogonalize() instead",
        ))
    }

    fn factorize_full_rank(
        &self,
        _left_inds: &[Self::Index],
        _alg: FactorizeAlg,
        _canonical: Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::UnsupportedStorage(
            "TensorTrain does not support factorize_full_rank; use orthogonalize() instead",
        ))
    }
}

impl TensorConstructionLike for TensorTrain {
    fn diagonal(
        input: &Self::Index,
        output: &Self::Index,
    ) -> std::result::Result<Self, Self::Error> {
        // Create a single-site TensorTrain with an identity tensor
        let delta = TensorDynLen::diagonal(input, output)?;
        Ok(Self::new(vec![delta])?)
    }

    fn scalar_one() -> std::result::Result<Self, Self::Error> {
        // Empty tensor train represents scalar 1
        Ok(Self::new(vec![])?)
    }

    fn ones(indices: &[Self::Index]) -> std::result::Result<Self, Self::Error> {
        let t = TensorDynLen::ones(indices)?;
        Ok(Self::new(vec![t])?)
    }

    fn onehot(index_vals: &[(Self::Index, usize)]) -> std::result::Result<Self, Self::Error> {
        let t = TensorDynLen::onehot(index_vals)?;
        Ok(Self::new(vec![t])?)
    }
}

#[cfg(test)]
mod tests;
