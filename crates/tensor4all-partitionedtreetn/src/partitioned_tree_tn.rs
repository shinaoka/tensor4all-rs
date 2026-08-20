//! Collections of eagerly projected [`SubDomainTreeTN`] patches.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use crate::subdomain_tree_tn::{
    ensure_center, ensure_same_dtype, ensure_same_topology, ensure_same_tree_structure,
    validate_contraction_options, validate_contraction_site_assignment,
    validate_truncation_options, SubDomainTreeTN,
};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN, TruncationOptions};

/// A collection of pairwise-disjoint, eagerly masked TreeTN patches.
///
/// All patches in one collection have the same named topology, exact full
/// site-index assignment at every node, and one homogeneous scalar dtype.
/// Projector keys are immutable through this type; use [`Self::insert`] or
/// [`Self::append`] for validated transactional updates.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let data = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?],
///     vec![0usize],
/// )?;
/// let patch = SubDomainTreeTN::new(data, Projector::from_pairs([(site, 0)])?)?;
/// let partition = PartitionedTreeTN::from_subdomain(patch)?;
/// assert_eq!(partition.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct PartitionedTreeTN<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    data: HashMap<Projector, SubDomainTreeTN<V>>,
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Create an empty partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::PartitionedTreeTN;
    ///
    /// let partition = PartitionedTreeTN::<usize>::new();
    /// assert!(partition.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Return the number of stored patches.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return whether no patches are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over projector keys in unspecified hash-map order.
    pub fn projectors(&self) -> impl Iterator<Item = &Projector> {
        self.data.keys()
    }

    /// Get the patch stored under an exact projector key.
    pub fn get(&self, projector: &Projector) -> Option<&SubDomainTreeTN<V>> {
        self.data.get(projector)
    }

    /// Return whether an exact projector key is present.
    pub fn contains(&self, projector: &Projector) -> bool {
        self.data.contains_key(projector)
    }

    /// Iterate over immutable `(projector, patch)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Projector, &SubDomainTreeTN<V>)> {
        self.data.iter()
    }

    /// Iterate over immutable patches.
    pub fn values(&self) -> impl Iterator<Item = &SubDomainTreeTN<V>> {
        self.data.values()
    }
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Construct a partition from validated patches.
    ///
    /// Equal projector keys replace earlier patches in input order. Different
    /// compatible projectors are rejected; all topology, site-space, dtype,
    /// projector, and overlap checks finish before the returned map is built.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] for different
    /// overlapping keys, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] for inconsistent patch
    /// structure, [`PartitionedTreeTNError::DTypeMismatch`] for mixed scalar
    /// dtypes, or the construction/validation source errors from a patch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = |value| -> Result<_, Box<dyn std::error::Error>> {
    ///     let data = TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], vec![value, value])?],
    ///         vec![0usize],
    ///     )?;
    ///     Ok(SubDomainTreeTN::new(
    ///         data,
    ///         Projector::from_pairs([(site.clone(), value as usize)])?,
    ///     )?)
    /// };
    /// let partition = PartitionedTreeTN::from_subdomains(vec![make(0.0)?, make(1.0)?])?;
    /// assert_eq!(partition.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_subdomains(subdomains: Vec<SubDomainTreeTN<V>>) -> Result<Self> {
        let references: Vec<_> = subdomains.iter().collect();
        validate_candidate_set(&references)?;

        let mut data = HashMap::with_capacity(subdomains.len());
        for subdomain in subdomains {
            let projector = subdomain.projector().clone();
            data.remove(&projector);
            data.insert(projector, subdomain);
        }
        Ok(Self { data })
    }

    /// Construct a partition containing one patch.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorIndexNotFound`] or
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for an invalid
    /// stored projector, [`PartitionedTreeTNError::InvalidTopology`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], or
    /// [`PartitionedTreeTNError::UnsupportedDType`] for invalid TreeTN data, and
    /// typed [`PartitionedTreeTNError::TreeTN`],
    /// [`PartitionedTreeTNError::TensorStorage`], or
    /// [`PartitionedTreeTNError::TensorConstruction`] source failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let data = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 0.0])?],
    ///     vec![0usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(data)?)?;
    /// assert_eq!(partition.len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_subdomain(subdomain: SubDomainTreeTN<V>) -> Result<Self> {
        Self::from_subdomains(vec![subdomain])
    }

    /// Insert or exactly replace one patch transactionally.
    ///
    /// An equal projector key replaces both the old key and value. A different
    /// compatible key is rejected. No mutation occurs until the candidate has
    /// passed all partition invariant checks.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], or a typed patch validation
    /// error. On error, the partition is unchanged.
    pub fn insert(&mut self, subdomain: SubDomainTreeTN<V>) -> Result<()> {
        self.validate_contents()?;
        subdomain.validate_invariants()?;
        if let Some(template) = self.data.values().next() {
            ensure_same_tree_structure(template.data(), subdomain.data())?;
            ensure_same_dtype(template.scalar_kind()?, subdomain.scalar_kind()?)?;
        }
        self.insert_prevalidated(subdomain)
    }

    /// Insert a subdomain that has already satisfied every partition invariant.
    ///
    /// This private incremental path skips re-validating all existing patches
    /// and the candidate itself, which internal algebra already produces from
    /// validated operands, and checks only projector-key overlap. Repeatedly
    /// calling the public [`Self::insert`] while building a result is cubic in
    /// the patch count because each call rescans the full partition.
    fn insert_prevalidated(&mut self, subdomain: SubDomainTreeTN<V>) -> Result<()> {
        let projector = subdomain.projector().clone();
        for existing in self.data.keys() {
            if existing != &projector && existing.is_compatible_with(&projector) {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
        self.data.remove(&projector);
        self.data.insert(projector, subdomain);
        Ok(())
    }

    /// Append another partition transactionally.
    ///
    /// Equal keys replace existing patches. Different overlapping keys are
    /// rejected, including overlaps between the two inputs. Every candidate is
    /// validated before this partition is mutated.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] for an invalid
    /// layout, [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] for incompatible patch
    /// metadata. On error, `self` is unchanged.
    pub fn append(&mut self, other: Self) -> Result<()> {
        self.validate_contents()?;
        other.validate_contents()?;

        if let (Some(left), Some(right)) = (self.data.values().next(), other.data.values().next()) {
            ensure_same_tree_structure(left.data(), right.data())?;
            ensure_same_dtype(left.scalar_kind()?, right.scalar_kind()?)?;
        }

        for right_projector in other.data.keys() {
            for left_projector in self.data.keys() {
                if right_projector != left_projector
                    && right_projector.is_compatible_with(left_projector)
                {
                    return Err(PartitionedTreeTNError::OverlappingProjectors);
                }
            }
        }

        for (projector, subdomain) in other.data {
            self.data.remove(&projector);
            self.data.insert(projector, subdomain);
        }
        Ok(())
    }

    /// Append a vector of patches using the same transactional checks as
    /// [`Self::append`].
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] for incompatible patches, plus
    /// the typed projector, topology, dtype, tensor, and TreeTN validation errors
    /// documented by [`Self::from_subdomain`]. `self` is unchanged on failure.
    pub fn append_subdomains(&mut self, subdomains: Vec<SubDomainTreeTN<V>>) -> Result<()> {
        let other = Self::from_subdomains(subdomains)?;
        self.append(other)
    }

    /// Compute the squared Frobenius norm of all eager patches.
    ///
    /// This sums each stored patch's squared norm without remasking or dense
    /// materialization. The receiver is borrowed immutably; each patch's norm
    /// implementation clones its TreeTN wrapper for canonicalization.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] when stored partition metadata
    /// is inconsistent, the validation errors documented by
    /// [`Self::from_subdomain`], or [`PartitionedTreeTNError::TreeTN`] when a
    /// patch norm cannot be evaluated.
    pub fn norm_squared(&self) -> Result<f64> {
        self.validate_contents()?;
        self.data
            .values()
            .try_fold(0.0, |sum, subdomain| Ok(sum + subdomain.norm_squared()?))
    }

    /// Compute the Frobenius norm of all eager patches.
    ///
    /// This returns the square root of [`Self::norm_squared`]. The receiver is
    /// borrowed immutably and no full dense tensor is materialized.
    ///
    /// # Errors
    ///
    /// Returns the concrete validation variants documented by
    /// [`Self::norm_squared`], including
    /// [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], and
    /// [`PartitionedTreeTNError::TreeTN`].
    pub fn norm(&self) -> Result<f64> {
        Ok(self.norm_squared()?.sqrt())
    }

    /// Add two partitions patch-wise at an explicit truncation center.
    ///
    /// Identical projector keys are added with strict [`TreeTN::add`] semantics
    /// and truncated with `options`. A key present on only one side is copied as
    /// the other side's zero contribution. Different overlapping layouts are
    /// rejected; no projector refinement is performed.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before patch shortcuts,
    /// [`PartitionedTreeTNError::InvalidCenter`] for an absent center,
    /// [`PartitionedTreeTNError::InvalidOptions`] for an invalid truncation
    /// option, [`PartitionedTreeTNError::OverlappingProjectors`] for a different
    /// overlapping layout, or a typed TreeTN error for addition/truncation.
    pub fn add(&self, other: &Self, center: &V, options: TruncationOptions) -> Result<Self> {
        validate_truncation_options(&options)?;
        self.validate_contents()?;
        other.validate_contents()?;
        let (left_template, right_template) = nonempty_templates(self, other)?;
        ensure_same_tree_structure(left_template.data(), right_template.data())?;
        ensure_same_dtype(left_template.scalar_kind()?, right_template.scalar_kind()?)?;
        ensure_center(left_template.data(), center)?;
        ensure_center(right_template.data(), center)?;
        ensure_union_disjoint(&self.data, &other.data)?;

        let mut result = Self::new();
        for (projector, left) in &self.data {
            if let Some(right) = other.data.get(projector) {
                let mut sum = left.add(right)?;
                sum.truncate(center, options)?;
                result.insert_prevalidated(sum)?;
            } else {
                result.insert_prevalidated(left.clone())?;
            }
        }
        for (projector, right) in &other.data {
            if !self.data.contains_key(projector) {
                result.insert_prevalidated(right.clone())?;
            }
        }
        Ok(result)
    }

    /// Contract two partitions at an explicit center.
    ///
    /// V1 requires the same named topology. Shared contractable site indices
    /// must remain assigned to the same named node; distinct site indices are
    /// retained as external output axes. Compatible patch pairs are contracted
    /// directly from their eager stored data. Projector entries absent from the
    /// output site space are pruned, and duplicate output keys are combined by
    /// strict subdomain addition followed by the requested bond truncation.
    ///
    /// # Known limitation: no output-region refinement
    ///
    /// Output regions are the projector intersections of each contracted pair,
    /// with contracted site indices removed. Two individually valid, internally
    /// disjoint input partitions can contract into output regions that are not
    /// mutually disjoint (for example `{i=0}, {i=1,a=0}, {i=1,a=1}` contracted
    /// with `{i=0,b=0}, {i=0,b=1}, {i=1}`), because a patch such as `{a=0}` and
    /// `{b=0}` are distinct keys that still intersect in the full site space.
    /// Such outputs are rejected with
    /// [`PartitionedTreeTNError::OverlappingProjectors`] rather than silently
    /// corrupted; combine over a refined disjoint partition of the output
    /// space before contracting if overlapping outputs are possible.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before pair iteration,
    /// [`PartitionedTreeTNError::InvalidCenter`],
    /// [`PartitionedTreeTNError::InvalidOptions`],
    /// [`PartitionedTreeTNError::OverlappingProjectors`] for incompatible
    /// non-identical output layouts (see the limitation note above), or a
    /// typed TreeTN/backend error. The operation returns a new partition, so
    /// failure cannot mutate either input.
    pub fn contract(&self, other: &Self, center: &V, options: ContractionOptions) -> Result<Self> {
        validate_contraction_options(&options)?;
        self.validate_contents()?;
        other.validate_contents()?;
        let (left_template, right_template) = nonempty_templates(self, other)?;
        ensure_same_topology(left_template.data(), right_template.data())?;
        ensure_same_dtype(left_template.scalar_kind()?, right_template.scalar_kind()?)?;
        validate_contraction_site_assignment(left_template.data(), right_template.data())?;
        ensure_center(left_template.data(), center)?;
        ensure_center(right_template.data(), center)?;

        // Collect pairwise contraction contributions grouped by output
        // projector. Left/right patches are visited in canonical projector
        // order and groups are sorted, so the within-group exact-add order (and
        // therefore the result) is independent of the input HashMap iteration
        // order.
        let mut left_patches: Vec<_> = self.data.values().collect();
        let mut right_patches: Vec<_> = other.data.values().collect();
        left_patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));
        right_patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));
        let mut groups: Vec<(Projector, Vec<SubDomainTreeTN<V>>)> = Vec::new();
        let mut positions: HashMap<Projector, usize> = HashMap::new();
        for left in left_patches {
            for right in &right_patches {
                let Some(contracted) = left.contract(right, center, options.clone())? else {
                    continue;
                };
                let projector = contracted.projector().clone();
                match positions.get(&projector) {
                    Some(&position) => groups[position].1.push(contracted),
                    None => {
                        positions.insert(projector.clone(), groups.len());
                        groups.push((projector, vec![contracted]));
                    }
                }
            }
        }

        // Exact-add each completed group with strict TreeTN addition, then
        // truncate the group once (a single contribution is already the
        // per-pair contraction result). This replaces repeated
        // add-and-truncate, which was order dependent and re-approximated the
        // sum at every contribution.
        groups.sort_by(|(left, _), (right, _)| left.canonical_cmp(right));
        let duplicate_options = truncation_options_from_contraction(&options);
        let mut subdomains = Vec::with_capacity(groups.len());
        for (_, contributions) in groups {
            let mut contributions = contributions.into_iter();
            let first = contributions.next().ok_or(PartitionedTreeTNError::Empty)?;
            let mut combined = first;
            let mut contribution_count = 1usize;
            for contribution in contributions {
                combined = combined.add(&contribution)?;
                contribution_count += 1;
            }
            if contribution_count > 1 {
                combined.truncate(center, duplicate_options)?;
            }
            subdomains.push(combined);
        }

        // Build and validate the complete result partition once at the end;
        // overlapping non-identical output projectors are rejected here with
        // the documented `OverlappingProjectors` limitation.
        PartitionedTreeTN::from_subdomains(subdomains)
    }

    /// Sum eager patches in deterministic canonical projector order.
    ///
    /// The result uses strict high-level [`TreeTN::add`] and never materializes
    /// the full tensor densely. Equal numerical inputs therefore have stable
    /// patch traversal and direct-sum construction order.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] when there are no patches,
    /// typed invariant errors for invalid metadata, or [`PartitionedTreeTNError::TreeTN`]
    /// when strict TreeTN addition fails.
    pub fn to_treetn(&self) -> Result<TreeTN<tensor4all_core::IdxTensor, V>> {
        self.validate_contents()?;
        if self.is_empty() {
            return Err(PartitionedTreeTNError::Empty);
        }

        let mut patches: Vec<_> = self.data.iter().collect();
        patches.sort_by(|(left, _), (right, _)| left.canonical_cmp(right));
        let (_, first) = patches.first().ok_or(PartitionedTreeTNError::Empty)?;
        let mut result = first.data().clone();
        for (_, patch) in patches.into_iter().skip(1) {
            result = result
                .add(patch.data())
                .map_err(PartitionedTreeTNError::from)?;
        }
        Ok(result)
    }

    pub(crate) fn validate_contents(&self) -> Result<()> {
        let references: Vec<_> = self.data.values().collect();
        validate_candidate_set(&references)
    }
}

fn validate_candidate_set<V>(candidates: &[&SubDomainTreeTN<V>]) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    for candidate in candidates {
        candidate.validate_invariants()?;
    }

    if let Some(template) = candidates.first() {
        for candidate in candidates.iter().skip(1) {
            ensure_same_tree_structure(template.data(), candidate.data())?;
            ensure_same_dtype(template.scalar_kind()?, candidate.scalar_kind()?)?;
        }
    }

    for (position, left) in candidates.iter().enumerate() {
        for right in candidates.iter().skip(position + 1) {
            if left.projector() != right.projector()
                && left.projector().is_compatible_with(right.projector())
            {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
    }
    Ok(())
}

fn nonempty_templates<'a, V>(
    left: &'a PartitionedTreeTN<V>,
    right: &'a PartitionedTreeTN<V>,
) -> Result<(&'a SubDomainTreeTN<V>, &'a SubDomainTreeTN<V>)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let left_template = left
        .data
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    let right_template = right
        .data
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    Ok((left_template, right_template))
}

fn ensure_union_disjoint<V>(
    left: &HashMap<Projector, SubDomainTreeTN<V>>,
    right: &HashMap<Projector, SubDomainTreeTN<V>>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    for left_projector in left.keys() {
        for right_projector in right.keys() {
            if left_projector != right_projector
                && left_projector.is_compatible_with(right_projector)
            {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
    }
    Ok(())
}

fn truncation_options_from_contraction(options: &ContractionOptions) -> TruncationOptions {
    let mut truncation = TruncationOptions::default();
    if let Some(max_bond_dim) = options.max_bond_dim {
        truncation = truncation.with_max_bond_dim(max_bond_dim);
    }
    if let Some(policy) = options.svd_policy {
        truncation = truncation.with_svd_policy(policy);
    }
    truncation
}

impl<V> IntoIterator for PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    type Item = (Projector, SubDomainTreeTN<V>);
    type IntoIter = std::collections::hash_map::IntoIter<Projector, SubDomainTreeTN<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, V> IntoIterator for &'a PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    type Item = (&'a Projector, &'a SubDomainTreeTN<V>);
    type IntoIter = std::collections::hash_map::Iter<'a, Projector, SubDomainTreeTN<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}
