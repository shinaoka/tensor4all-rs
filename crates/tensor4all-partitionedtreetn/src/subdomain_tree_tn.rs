//! Eagerly projected TreeTN subdomains.

use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{
    contraction::{self, ContractionMethod, ContractionOptions},
    SiteIndexNetwork, TreeTN, TruncationOptions,
};

/// A TreeTN together with the projector defining its subdomain.
///
/// The stored TreeTN is always eagerly masked. Every original site axis is
/// retained, and values outside the projector are zero. Rebuilding from local
/// tensors also clears canonical and orthogonality metadata; a subdomain does
/// not store a canonical center.
///
/// `V` names TreeTN nodes. The tensor storage is intentionally fixed to
/// [`IdxTensor`] because projection uses its differentiable `mask_index` API.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])?;
/// let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
/// let subdomain = SubDomainTreeTN::new(
///     tree,
///     Projector::from_pairs([(site.clone(), 1)])?,
/// )?;
///
/// assert_eq!(subdomain.node_count(), 1);
/// assert_eq!(subdomain.data().tensor(subdomain.data().node_index(&0).unwrap())
///     .unwrap().to_vec::<f64>()?, vec![0.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct SubDomainTreeTN<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    data: TreeTN<IdxTensor, V>,
    projector: Projector,
    budget_squared: Option<f64>,
}

impl<V> SubDomainTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Return the eagerly projected TreeTN by immutable reference.
    pub fn data(&self) -> &TreeTN<IdxTensor, V> {
        &self.data
    }

    /// Return the projector by immutable reference.
    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    /// Consume the subdomain and return its eagerly projected TreeTN.
    pub fn into_data(self) -> TreeTN<IdxTensor, V> {
        self.data
    }

    /// Return the number of TreeTN nodes.
    pub fn node_count(&self) -> usize {
        self.data.node_count()
    }

    /// Return whether the TreeTN has no nodes.
    pub fn is_empty(&self) -> bool {
        self.data.node_count() == 0
    }

    /// Return all full site indices, including every axis on every node.
    ///
    /// The order is unspecified because TreeTN site spaces are sets. Full
    /// index equality, rather than ID-only equality, is used by the stored
    /// network and projector.
    pub fn all_indices(&self) -> Vec<DynIndex> {
        self.data
            .node_names()
            .into_iter()
            .flat_map(|name| {
                self.data
                    .site_space(&name)
                    .into_iter()
                    .flat_map(|indices| indices.iter().cloned())
            })
            .collect()
    }

    /// Return the TreeTN site-index network describing node topology and site axes.
    pub fn site_index_network(&self) -> &SiteIndexNetwork<V, DynIndex> {
        self.data.site_index_network()
    }

    /// Return whether `index` is fixed by this subdomain projector.
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.projector.is_projected_at(index)
    }

    fn validate_projector(&self, projector: &Projector) -> Result<()>
    where
        V: Ord,
    {
        validate_projector_against_data(&self.data, projector)
    }
}

impl<V> SubDomainTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Return the largest internal bond dimension, or one for a bondless tree.
    pub fn max_bond_dim(&self) -> usize {
        self.data.link_dims().into_iter().fold(1, usize::max)
    }

    pub(crate) fn validate_invariants(&self) -> Result<()> {
        validate_data(&self.data)?;
        self.validate_projector(&self.projector)?;
        if let Some(budget_squared) = self.budget_squared {
            validate_budget_squared(budget_squared)?;
        }
        Ok(())
    }

    pub(crate) fn budget_squared(&self) -> Option<f64> {
        self.budget_squared
    }

    pub(crate) fn with_budget_squared(mut self, budget_squared: f64) -> Result<Self> {
        validate_budget_squared(budget_squared)?;
        self.budget_squared = Some(budget_squared);
        Ok(self)
    }

    pub(crate) fn scalar_kind(&self) -> Result<Option<ScalarKind>> {
        validate_data(&self.data)
    }

    pub(crate) fn from_masked_data(
        data: TreeTN<IdxTensor, V>,
        projector: Projector,
        budget_squared: Option<f64>,
    ) -> Result<Self> {
        validate_data(&data)?;
        validate_projector_against_data(&data, &projector)?;
        if let Some(budget_squared) = budget_squared {
            validate_budget_squared(budget_squared)?;
        }
        Ok(Self {
            data,
            projector,
            budget_squared,
        })
    }

    /// Construct an eagerly projected subdomain from a TreeTN and projector.
    ///
    /// The input topology must be one connected acyclic TreeTN. Every
    /// projector index must match a full site-index identity in the TreeTN;
    /// coordinates are checked against the TreeTN dimension, not merely the
    /// dimension carried by the projector key. All nodes must use one
    /// supported homogeneous `IdxTensor` scalar dtype.
    ///
    /// Construction masks each selected local tensor with
    /// [`IdxTensor::mask_index`], retains all site axes, and rebuilds the
    /// TreeTN. It therefore does not materialize the full network and does not
    /// retain canonical metadata from the input.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidTopology`] for a non-tree
    /// input, [`PartitionedTreeTNError::DTypeMismatch`] for mixed node dtypes,
    /// [`PartitionedTreeTNError::ProjectorIndexNotFound`] for an absent full
    /// site identity, [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`]
    /// for an invalid coordinate, or a typed backend/TreeTN error when local
    /// masking or rebuilding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::new(
    ///     tree,
    ///     Projector::from_pairs([(site.clone(), 0)])?,
    /// )?;
    /// assert_eq!(subdomain.projector().get(&site), Some(0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(data: TreeTN<IdxTensor, V>, projector: Projector) -> Result<Self> {
        let _ = validate_data(&data)?;
        validate_projector_against_data(&data, &projector)?;
        let data = rebuild_masked_tree(&data, &projector)?;
        Ok(Self {
            data,
            projector,
            budget_squared: None,
        })
    }

    /// Construct an eagerly rebuilt subdomain with an empty projector.
    ///
    /// Rebuilding also clears canonical and orthogonality metadata present on
    /// the input TreeTN.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidTopology`] for a non-tree,
    /// [`PartitionedTreeTNError::DTypeMismatch`] or
    /// [`PartitionedTreeTNError::UnsupportedDType`] for invalid node dtypes,
    /// [`PartitionedTreeTNError::TensorStorage`] or
    /// [`PartitionedTreeTNError::TensorConstruction`] when rebuilding fails,
    /// and [`PartitionedTreeTNError::TreeTN`] for other TreeTN validation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!(subdomain.projector().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_treetn(data: TreeTN<IdxTensor, V>) -> Result<Self> {
        Self::new(data, Projector::new())
    }

    /// Apply compatible new restrictions and return another eagerly masked value.
    ///
    /// Existing restrictions are already represented in `self.data`, so only
    /// newly added projector entries are sent through `mask_index`. A
    /// conflicting projector returns `Ok(None)` without changing `self`.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorIndexNotFound`] or
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for invalid
    /// restrictions, [`PartitionedTreeTNError::TensorStorage`] or
    /// [`PartitionedTreeTNError::TensorConstruction`] when masking fails, and
    /// [`PartitionedTreeTNError::TreeTN`] when site-space validation or rebuilding
    /// the TreeTN fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
    ///
    /// let left = DynIndex::new_dyn(2);
    /// let right = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![left.clone(), right.clone()],
    ///         vec![1.0_f64, 2.0, 3.0, 4.0],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let source = SubDomainTreeTN::new(tree, Projector::from_pairs([(left.clone(), 0)])?)?;
    /// let result = source.project(&Projector::from_pairs([(right.clone(), 1)])?)?;
    /// let result = result.ok_or("conflicting projector")?;
    /// assert_eq!(result.projector().get(&left), Some(0));
    /// assert_eq!(result.projector().get(&right), Some(1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn project(&self, projector: &Projector) -> Result<Option<Self>> {
        self.validate_projector(projector)?;
        let Some(merged_projector) = self.projector.intersection(projector) else {
            return Ok(None);
        };
        self.validate_projector(&merged_projector)?;

        let mut additions = Projector::new();
        for (index, &value) in projector.iter() {
            if self.projector.get(index) != Some(value) {
                additions.insert(index.clone(), value)?;
            }
        }

        let data = rebuild_masked_tree(&self.data, &additions)?;
        Ok(Some(Self {
            data,
            projector: merged_projector,
            budget_squared: self.budget_squared,
        }))
    }

    /// Add two subdomains with the same projector using strict TreeTN addition.
    ///
    /// The stored TreeTNs are already eagerly masked, so this method performs no
    /// projector pass. The result retains every site axis and the left budget
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorMismatch`] when the projector
    /// keys differ, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] when the TreeTN structures
    /// differ, [`PartitionedTreeTNError::DTypeMismatch`] for mixed scalar dtypes,
    /// or [`PartitionedTreeTNError::TreeTN`] for strict addition failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = |values| {
    ///     TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], values)?],
    ///         vec![0usize],
    ///     )
    /// };
    /// let left = SubDomainTreeTN::from_treetn(make(vec![1.0_f64, 2.0])?)?;
    /// let right = SubDomainTreeTN::from_treetn(make(vec![3.0_f64, 4.0])?)?;
    /// let sum = left.add(&right)?;
    /// let node = sum.data().node_index(&0).ok_or("missing node")?;
    /// assert_eq!(sum.data().tensor(node).ok_or("missing tensor")?.to_vec::<f64>()?,
    ///            vec![4.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_tree_structure(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        if self.projector != other.projector {
            return Err(PartitionedTreeTNError::ProjectorMismatch);
        }

        let data = self
            .data
            .add(&other.data)
            .map_err(PartitionedTreeTNError::from)?;
        ensure_same_tree_structure(&self.data, &data)?;
        Self::from_masked_data(data, self.projector.clone(), self.budget_squared)
    }

    /// Truncate this eagerly masked subdomain towards an explicit TreeTN center.
    ///
    /// The operation works on a clone and commits only after TreeTN truncation
    /// and invariant validation succeed. It preserves all site axes, the
    /// projector, and the internal squared budget; the center is not stored.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent,
    /// [`PartitionedTreeTNError::InvalidOptions`] when `options` requests an
    /// invalid bond dimension, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] if truncation changes the
    /// site structure, or [`PartitionedTreeTNError::TreeTN`] for canonicalization,
    /// SVD, or backend failures. The receiver is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec!["center".to_string()],
    /// )?;
    /// let mut subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// subdomain.truncate(&"center".to_string(), TruncationOptions::default())?;
    /// assert_eq!(subdomain.node_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn truncate(&mut self, center: &V, options: TruncationOptions) -> Result<()> {
        validate_truncation_options(&options)?;
        self.validate_invariants()?;
        ensure_center(&self.data, center)?;

        let original = self.data.clone();
        let mut data = original.clone();
        data.truncate_mut([center.clone()], options)
            .map_err(PartitionedTreeTNError::from)?;
        ensure_same_tree_structure(&original, &data)?;
        validate_data(&data)?;
        validate_projector_against_data(&data, &self.projector)?;
        self.data = data;
        Ok(())
    }

    /// Contract two compatible subdomains at an explicit TreeTN center.
    ///
    /// The operation contracts the already masked stored TreeTNs directly.
    /// Compatible projectors are merged, then entries for site indices absent
    /// from the contracted output are removed. The returned subdomain retains
    /// the left budget metadata and never stores `center`.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TopologyMismatch`] when named node
    /// topologies differ, [`PartitionedTreeTNError::SiteIndexMismatch`] when a
    /// contractable site index is assigned to different nodes,
    /// [`PartitionedTreeTNError::DTypeMismatch`] for mixed dtypes,
    /// [`PartitionedTreeTNError::InvalidCenter`] for an absent center,
    /// [`PartitionedTreeTNError::InvalidOptions`] for forbidden dense/reference
    /// options, or [`PartitionedTreeTNError::TreeTN`] for contraction failures.
    /// Returns `Ok(None)` when the projector regions conflict.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
    ///
    /// let left_site = DynIndex::new_dyn(2);
    /// let right_site = DynIndex::new_dyn(2);
    /// let left = SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![left_site], vec![1.0_f64, 2.0])?],
    ///     vec![0usize],
    /// )?)?;
    /// let right = SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![right_site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?)?;
    /// let result = left.contract(&right, &0, ContractionOptions::default())?;
    /// assert!(result.is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contract(
        &self,
        other: &Self,
        center: &V,
        options: ContractionOptions,
    ) -> Result<Option<Self>> {
        validate_contraction_options(&options)?;
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_topology(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        validate_contraction_site_assignment(&self.data, &other.data)?;
        ensure_center(&self.data, center)?;
        if !self.projector.is_compatible_with(&other.projector) {
            return Ok(None);
        }

        let merged_projector = self
            .projector
            .intersection(&other.projector)
            .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
        let data = contraction::contract(&self.data, &other.data, center, options)
            .map_err(PartitionedTreeTNError::from)?;
        let (site_indices, _) = data
            .all_site_indices()
            .map_err(PartitionedTreeTNError::from)?;
        let projector = merged_projector.filter_indices(&site_indices);
        Self::from_masked_data(data, projector, self.budget_squared).map(Some)
    }

    /// Compute the Frobenius norm without mutating the stored TreeTN.
    ///
    /// This clones the TreeTN wrapper and allows the clone to canonicalize
    /// internally. The cost is one TreeTN metadata/tensor clone plus the
    /// underlying canonicalization sweep; it does not materialize the full
    /// site tensor.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TreeTN`] when TreeTN canonicalization
    /// or norm evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!((subdomain.norm()? - 5.0).abs() < 1.0e-12);
    /// assert!(subdomain.data().canonical_region().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn norm(&self) -> Result<f64> {
        let mut data = self.data.clone();
        data.norm().map_err(PartitionedTreeTNError::from)
    }

    /// Compute the squared Frobenius norm without mutating the stored TreeTN.
    ///
    /// This clones the TreeTN wrapper and lets the clone canonicalize, with the
    /// same linear-network clone and sweep cost as [`Self::norm`]. No full dense
    /// materialization is performed.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TreeTN`] when TreeTN canonicalization
    /// or norm evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!((subdomain.norm_squared()? - 25.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn norm_squared(&self) -> Result<f64> {
        let mut data = self.data.clone();
        data.norm_squared().map_err(PartitionedTreeTNError::from)
    }

    /// Compute the inner product using the already masked stored TreeTNs.
    ///
    /// No projector is reapplied and no dense materialization is performed.
    ///
    /// # Errors
    ///
    /// Returns a dtype mismatch, site-space mismatch, TreeTN, or backend error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = || {
    ///     TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])?],
    ///         vec![0usize],
    ///     )
    /// };
    /// let left = SubDomainTreeTN::from_treetn(make()?)?;
    /// let right = SubDomainTreeTN::from_treetn(make()?)?;
    /// assert!((left.inner(&right)?.real() - 25.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn inner(&self, other: &Self) -> Result<tensor4all_core::AnyScalar> {
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_tree_structure(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        self.data
            .inner(&other.data)
            .map_err(PartitionedTreeTNError::from)
    }
}

pub(crate) fn ensure_same_topology<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if left.same_topology(right) {
        Ok(())
    } else {
        Err(PartitionedTreeTNError::TopologyMismatch)
    }
}

pub(crate) fn ensure_same_tree_structure<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    ensure_same_topology(left, right)?;
    if left.share_equivalent_site_index_network(right) {
        Ok(())
    } else {
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    }
}

pub(crate) fn ensure_same_dtype(
    expected: Option<ScalarKind>,
    actual: Option<ScalarKind>,
) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(PartitionedTreeTNError::DTypeMismatch {
        expected: expected
            .map(|dtype| dtype.name().to_string())
            .unwrap_or_else(|| "empty".to_string()),
        actual: actual
            .map(|dtype| dtype.name().to_string())
            .unwrap_or_else(|| "empty".to_string()),
    })
}

pub(crate) fn ensure_center<V>(data: &TreeTN<IdxTensor, V>, center: &V) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    if data.node_index(center).is_some() {
        Ok(())
    } else {
        Err(PartitionedTreeTNError::InvalidCenter)
    }
}

fn validate_budget_squared(budget_squared: f64) -> Result<()> {
    if !budget_squared.is_finite() || budget_squared < 0.0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "squared patch budgets must be finite and non-negative",
        });
    }
    Ok(())
}

pub(crate) fn validate_truncation_options(options: &TruncationOptions) -> Result<()> {
    if options.max_bond_dim() == Some(0) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "truncate",
            reason: "max_bond_dim must be greater than zero",
        });
    }
    Ok(())
}

pub(crate) fn validate_contraction_options(options: &ContractionOptions) -> Result<()> {
    if options.max_bond_dim == Some(0) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "max_bond_dim must be greater than zero",
        });
    }
    if matches!(options.method, ContractionMethod::Naive) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "dense/reference contraction is not allowed in partition algebra",
        });
    }
    if matches!(options.method, ContractionMethod::Fit) && options.nfullsweeps == 0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "Fit contraction requires at least one full sweep",
        });
    }
    if options
        .qr_rtol
        .is_some_and(|rtol| !rtol.is_finite() || rtol < 0.0)
    {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "qr_rtol must be finite and non-negative",
        });
    }
    if options
        .convergence_tol
        .is_some_and(|tol| !tol.is_finite() || tol < 0.0)
    {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "convergence_tol must be finite and non-negative",
        });
    }
    Ok(())
}

pub(crate) fn validate_contraction_site_assignment<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (left_indices, left_nodes) = left
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    let (right_indices, right_nodes) = right
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;

    for (left_index, left_node) in left_indices.iter().zip(&left_nodes) {
        for (right_index, right_node) in right_indices.iter().zip(&right_nodes) {
            if left_index.is_contractable(right_index) && left_node != right_node {
                return Err(PartitionedTreeTNError::SiteIndexMismatch);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    F32,
    F64,
    C32,
    C64,
}

impl ScalarKind {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C32 => "Complex32",
            Self::C64 => "Complex64",
        }
    }
}

fn scalar_kind(tensor: &IdxTensor) -> Result<ScalarKind> {
    if tensor.is_f32() {
        Ok(ScalarKind::F32)
    } else if tensor.is_f64() {
        Ok(ScalarKind::F64)
    } else if tensor.is_c32() {
        Ok(ScalarKind::C32)
    } else if tensor.is_c64() {
        Ok(ScalarKind::C64)
    } else {
        Err(PartitionedTreeTNError::UnsupportedDType {
            dtype: format!("{:?}", tensor.storage_kind()),
        })
    }
}

fn validate_data<V>(data: &TreeTN<IdxTensor, V>) -> Result<Option<ScalarKind>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    data.validate_tree()
        .map_err(PartitionedTreeTNError::invalid_topology)?;

    let mut dtype: Option<ScalarKind> = None;
    for name in data.node_names() {
        let node = data
            .node_index(&name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let tensor = data
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let actual = scalar_kind(tensor)?;
        if let Some(expected) = dtype {
            if expected != actual {
                return Err(PartitionedTreeTNError::DTypeMismatch {
                    expected: expected.name().to_string(),
                    actual: actual.name().to_string(),
                });
            }
        } else {
            dtype = Some(actual);
        }
    }
    Ok(dtype)
}

fn validate_projector_against_data<V>(
    data: &TreeTN<IdxTensor, V>,
    projector: &Projector,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (site_indices, _) = data
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    for (index, &value) in projector.iter() {
        let Some(matched) = site_indices.iter().find(|candidate| *candidate == index) else {
            return Err(PartitionedTreeTNError::ProjectorIndexNotFound {
                index: index.clone(),
            });
        };
        if value >= matched.dim {
            return Err(PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
                index: index.clone(),
                value,
                dim: matched.dim,
            });
        }
    }
    Ok(())
}

fn rebuild_masked_tree<V>(
    data: &TreeTN<IdxTensor, V>,
    projector: &Projector,
) -> Result<TreeTN<IdxTensor, V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let node_names = data.node_names();
    let mut tensors = Vec::with_capacity(node_names.len());

    for name in &node_names {
        let node = data
            .node_index(name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let source = data
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let mut tensor = source.clone();
        for (index, &value) in projector.iter() {
            if tensor.indices().iter().any(|candidate| candidate == index) {
                tensor = tensor.mask_index(index, value)?;
            }
        }
        tensors.push(tensor);
    }

    TreeTN::from_tensors(tensors, node_names).map_err(PartitionedTreeTNError::from)
}
