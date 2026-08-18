//! PartitionedTT: A collection of non-overlapping SubDomainTTs
//!
//! A `PartitionedTT` represents a tensor train that is decomposed into
//! multiple independent sub-components, each associated with a projector
//! onto a subdomain of the full index set.

use std::collections::HashMap;

use crate::error::{PartitionedTTError, Result};
use crate::projector::Projector;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_core::DynIndex;
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// A partitioned tensor train: a collection of non-overlapping SubDomainTTs.
///
/// Each SubDomainTT covers a disjoint region of the index space defined by
/// its projector. The projectors must be mutually disjoint (non-overlapping).
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{PartitionedTT, Projector, SubDomainTT, TensorTrain};
/// use tensor4all_partitionedtt::{DynIndex, IdxTensor};
/// use tensor4all_core::index::Index;
///
/// fn make_tt(s0: &DynIndex, bond: &DynIndex, s1: &DynIndex) -> TensorTrain {
///     let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
///     let t1 = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![3.0, 4.0]).unwrap();
///     TensorTrain::new(vec![t0, t1]).unwrap()
/// }
///
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let tt = make_tt(&s0, &bond, &s1);
///
/// // Create a PartitionedTT with one patch projected to s0=0
/// let proj = Projector::from_pairs([(s0.clone(), 0)]).unwrap();
/// let subdomain = SubDomainTT::new(tt, proj).unwrap();
/// let ptt = PartitionedTT::from_subdomain(subdomain).unwrap();
///
/// assert_eq!(ptt.len(), 1);
/// assert!(!ptt.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct PartitionedTT {
    /// Map from projector to subdomain
    data: HashMap<Projector, SubDomainTT>,
}

impl PartitionedTT {
    /// Create an empty partitioned tensor train.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::PartitionedTT;
    ///
    /// let ptt = PartitionedTT::new();
    /// assert!(ptt.is_empty());
    /// assert_eq!(ptt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a PartitionedTT from a vector of SubDomainTTs.
    ///
    /// Returns an error if different projectors overlap. Equal projectors
    /// replace earlier values in input order.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::OverlappingProjectors`] when two
    /// different projectors describe overlapping regions or when a subdomain
    /// fails projector validation. No partially built value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::{PartitionedTT, Projector, SubDomainTT, TensorTrain};
    /// use tensor4all_partitionedtt::{DynIndex, IdxTensor};
    /// use tensor4all_core::index::Index;
    ///
    /// let s0 = Index::new_dyn(2);
    /// let bond = Index::new_dyn(1);
    /// let s1 = Index::new_dyn(2);
    /// let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
    /// let t1 = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![3.0, 4.0]).unwrap();
    /// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// // Two disjoint patches: s0=0 and s0=1
    /// let proj0 = Projector::from_pairs([(s0.clone(), 0)]).unwrap();
    /// let proj1 = Projector::from_pairs([(s0.clone(), 1)]).unwrap();
    /// let sd0 = SubDomainTT::new(tt.clone(), proj0).unwrap();
    /// let sd1 = SubDomainTT::new(tt, proj1).unwrap();
    ///
    /// let ptt = PartitionedTT::from_subdomains(vec![sd0, sd1]).unwrap();
    /// assert_eq!(ptt.len(), 2);
    /// ```
    pub fn from_subdomains(subdomains: Vec<SubDomainTT>) -> Result<Self> {
        let mut partitioned = Self::new();
        for subdomain in subdomains {
            partitioned.insert(subdomain)?;
        }
        Ok(partitioned)
    }

    /// Create a `PartitionedTT` from one validated subdomain.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::ProjectorIndexNotFound`] if the projector
    /// contains an index absent from the tensor train, or
    /// [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] if a coordinate is
    /// invalid for the matched tensor-train index. No partition is returned on
    /// failure.
    pub fn from_subdomain(subdomain: SubDomainTT) -> Result<Self> {
        let mut partitioned = Self::new();
        partitioned.insert(subdomain)?;
        Ok(partitioned)
    }

    /// Number of subdomains (patches).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get projectors as an iterator.
    pub fn projectors(&self) -> impl Iterator<Item = &Projector> {
        self.data.keys()
    }

    /// Get subdomain by projector.
    pub fn get(&self, projector: &Projector) -> Option<&SubDomainTT> {
        self.data.get(projector)
    }

    /// Check if a projector exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::{PartitionedTT, Projector, SubDomainTT, TensorTrain};
    /// use tensor4all_partitionedtt::{DynIndex, IdxTensor};
    /// use tensor4all_core::index::Index;
    ///
    /// let s0 = Index::new_dyn(2);
    /// let bond = Index::new_dyn(1);
    /// let s1 = Index::new_dyn(2);
    /// let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
    /// let t1 = IdxTensor::from_dense(vec![bond, s1], vec![3.0, 4.0]).unwrap();
    /// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let proj = Projector::from_pairs([(s0.clone(), 0)]).unwrap();
    /// let subdomain = SubDomainTT::new(tt, proj.clone()).unwrap();
    /// let ptt = PartitionedTT::from_subdomain(subdomain).unwrap();
    ///
    /// assert!(ptt.contains(&proj));
    /// let absent = Projector::from_pairs([(s0, 1)]).unwrap();
    /// assert!(!ptt.contains(&absent));
    /// ```
    pub fn contains(&self, projector: &Projector) -> bool {
        self.data.contains_key(projector)
    }

    /// Iterate over (projector, subdomain) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Projector, &SubDomainTT)> {
        self.data.iter()
    }

    /// Iterate over subdomains.
    pub fn values(&self) -> impl Iterator<Item = &SubDomainTT> {
        self.data.values()
    }

    /// Insert a subdomain transactionally.
    ///
    /// An exactly equal projector replaces the existing key and value together.
    /// A different projector must be disjoint from every stored projector.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::OverlappingProjectors`] for a different
    /// compatible projector, or a typed projector-validation error when the
    /// subdomain's data and projector are inconsistent. On error, this
    /// partition is unchanged.
    pub fn insert(&mut self, subdomain: SubDomainTT) -> Result<()> {
        subdomain.validate_invariants()?;
        let projector = subdomain.projector().clone();
        for existing in self.data.keys() {
            if existing != &projector && existing.is_compatible_with(&projector) {
                return Err(PartitionedTTError::OverlappingProjectors);
            }
        }

        // HashMap::insert retains an equal old key. Remove first so the key and
        // value remain the coherent pair supplied by the caller.
        self.data.remove(&projector);
        self.data.insert(projector, subdomain);
        Ok(())
    }

    /// Append another PartitionedTT (must have non-overlapping projectors).
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::OverlappingProjectors`] when a different
    /// projector overlaps an existing projector,
    /// [`PartitionedTTError::ProjectorIndexNotFound`] when an appended
    /// subdomain refers to an absent tensor-train index, or
    /// [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] when its coordinate
    /// is invalid. Equal keys replace existing values. All checks happen before
    /// this partition is mutated.
    ///
    pub fn append(&mut self, other: Self) -> Result<()> {
        for subdomain in other.data.values() {
            subdomain.validate_invariants()?;
        }
        for projector in other.data.keys() {
            for existing in self.data.keys() {
                if projector != existing && projector.is_compatible_with(existing) {
                    return Err(PartitionedTTError::OverlappingProjectors);
                }
            }
        }

        for (projector, subdomain) in other.data {
            self.data.remove(&projector);
            self.data.insert(projector, subdomain);
        }
        Ok(())
    }

    /// Append subdomains.
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::OverlappingProjectors`] when different
    /// candidate or existing projectors overlap,
    /// [`PartitionedTTError::ProjectorIndexNotFound`] when a candidate refers to
    /// an absent tensor-train index, or
    /// [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] when its coordinate
    /// is invalid. The complete candidate set is validated before this partition
    /// is mutated.
    ///
    pub fn append_subdomains(&mut self, subdomains: Vec<SubDomainTT>) -> Result<()> {
        let other = Self::from_subdomains(subdomains)?;
        self.append(other)
    }

    /// Compute the total Frobenius norm (sqrt of sum of squared norms).
    ///
    /// # Errors
    ///
    /// Returns an error when the norm cannot be evaluated (a materialization or
    /// /// backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::{PartitionedTT, Projector, SubDomainTT, TensorTrain};
    /// use tensor4all_partitionedtt::{DynIndex, IdxTensor};
    /// use tensor4all_core::index::Index;
    ///
    /// let s0 = Index::new_dyn(2);
    /// let bond = Index::new_dyn(1);
    /// let s1 = Index::new_dyn(2);
    /// let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 0.0]).unwrap();
    /// let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0]).unwrap();
    /// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let proj = Projector::from_pairs([(s0, 0)]).unwrap();
    /// let subdomain = SubDomainTT::new(tt, proj).unwrap();
    /// let ptt = PartitionedTT::from_subdomain(subdomain).unwrap();
    ///
    /// let n = ptt.norm().unwrap();
    /// assert!(n >= 0.0);
    /// ```
    pub fn norm(&self) -> Result<f64> {
        let sum_sq = self
            .data
            .values()
            .map(SubDomainTT::norm_squared)
            .try_fold(0.0, |sum, value| {
                Ok::<f64, PartitionedTTError>(sum + value?)
            })?;
        Ok(sum_sq.sqrt())
    }

    /// Contract with another PartitionedTT.
    ///
    /// Performs pairwise contraction of compatible SubDomainTTs and combines results.
    /// # Errors
    ///
    /// Returns an error when the contraction fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        let mut result = Self::new();

        // Build contraction tasks
        let tasks = self.contraction_tasks(other)?;

        // Execute contractions
        for (proj, m1, m2) in tasks {
            if let Some(contracted) = m1.contract(&m2, options)? {
                // Check if we already have a subdomain with the same projector
                if let Some(existing) = result.data.get_mut(&proj) {
                    // Sum the subdomains using TT addition
                    let mut summed_tt = existing
                        .data()
                        .add_reindexed_like_self(contracted.data())
                        .map_err(|source| PartitionedTTError::TensorTrain { source })?;
                    // Truncate after addition using the same truncation params as contraction
                    let mut truncate_opts = TruncateOptions::svd();
                    if let Some(policy) = options.svd_policy() {
                        truncate_opts = truncate_opts.with_svd_policy(policy);
                    }
                    if let Some(max_bond_dim) = options.max_bond_dim() {
                        truncate_opts = truncate_opts.with_max_bond_dim(max_bond_dim);
                    }
                    summed_tt
                        .truncate(&truncate_opts)
                        .map_err(|source| PartitionedTTError::TensorTrain { source })?;
                    *existing = SubDomainTT::new(summed_tt, proj.clone())?;
                } else {
                    result.insert(contracted)?;
                }
            }
        }

        Ok(result)
    }

    /// Add another PartitionedTT patch-by-patch.
    ///
    /// Both PartitionedTTs must have compatible patch structures:
    /// - The union of all projectors from both must be pairwise disjoint
    /// - Missing patches in either side are allowed (treated as zero)
    ///
    /// For each projector present in both, the corresponding TTs are added
    /// and then truncated according to the provided options.
    ///
    /// # Errors
    ///
    /// Returns an error when the partitions are incompatible (a shape mismatch)
    /// /// or the addition fails.
    ///
    pub fn add(&self, other: &Self, options: &TruncateOptions) -> Result<Self> {
        // Collect unique projectors from both (union)
        let mut unique_projectors: std::collections::HashSet<Projector> =
            self.projectors().cloned().collect();
        unique_projectors.extend(other.projectors().cloned());
        let all_projectors: Vec<Projector> = unique_projectors.into_iter().collect();

        // Check that all unique projectors are pairwise disjoint
        if !Projector::are_disjoint(&all_projectors) {
            return Err(PartitionedTTError::IncompatibleProjectors(
                "Projectors must be pairwise disjoint for patch-wise addition".to_string(),
            ));
        }

        let mut result = Self::new();

        // Process projectors from self
        for (proj, subdomain) in self.iter() {
            if let Some(other_subdomain) = other.get(proj) {
                // Both have this projector: add and truncate
                let mut summed_tt = subdomain
                    .data()
                    .add(other_subdomain.data())
                    .map_err(|source| PartitionedTTError::TensorTrain { source })?;
                summed_tt
                    .truncate(options)
                    .map_err(|source| PartitionedTTError::TensorTrain { source })?;
                result.insert(SubDomainTT::new(summed_tt, proj.clone())?)?;
            } else {
                // Only self has this projector: clone it
                result.insert(subdomain.clone())?;
            }
        }

        // Process projectors only in other (not in self)
        for (proj, subdomain) in other.iter() {
            if !self.contains(proj) {
                result.insert(subdomain.clone())?;
            }
        }

        Ok(result)
    }

    /// Build contraction tasks for two PartitionedTTs.
    fn contraction_tasks(
        &self,
        other: &Self,
    ) -> Result<Vec<(Projector, SubDomainTT, SubDomainTT)>> {
        let mut tasks = Vec::new();

        for m1 in self.data.values() {
            for m2 in other.data.values() {
                // Check if projectors are compatible
                if m1.projector().is_compatible_with(m2.projector()) {
                    // Compute the projector after contraction
                    let indices1: std::collections::HashSet<_> =
                        m1.all_indices().into_iter().collect();
                    let indices2: std::collections::HashSet<_> =
                        m2.all_indices().into_iter().collect();

                    // External indices
                    let common: std::collections::HashSet<_> =
                        indices1.intersection(&indices2).cloned().collect();
                    let all: std::collections::HashSet<_> =
                        indices1.union(&indices2).cloned().collect();
                    let external: std::collections::HashSet<DynIndex> =
                        all.difference(&common).cloned().collect();

                    // Build projector for external indices
                    let mut proj_data = Vec::new();
                    for idx in &external {
                        if let Some(val) = m1.projector().get(idx) {
                            proj_data.push((idx.clone(), val));
                        } else if let Some(val) = m2.projector().get(idx) {
                            proj_data.push((idx.clone(), val));
                        }
                    }
                    let proj_after = Projector::from_pairs(proj_data)?;

                    // SubDomainTT::contract already applies each input projector.
                    // Pre-projecting here is redundant and can attach projector
                    // metadata for indices that are not present in a subdomain.
                    tasks.push((proj_after, m1.clone(), m2.clone()));
                }
            }
        }

        Ok(tasks)
    }

    /// Convert to a single TensorTrain by summing all subdomains.
    ///
    /// Uses direct-sum (block) addition to combine all SubDomainTT tensors.
    /// The result has bond dimension equal to the sum of individual bond dimensions.
    ///
    /// Subdomains are processed in a deterministic order (sorted by projector)
    /// to ensure reproducible results.
    ///
    /// # Errors
    ///
    /// Returns an error when the conversion fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn to_tensor_train(&self) -> Result<TensorTrain> {
        if self.is_empty() {
            return Err(PartitionedTTError::Empty);
        }

        // Sort subdomains by projector for deterministic ordering
        let mut sorted: Vec<_> = self.data.iter().collect();
        sorted.sort_by(|(p1, _), (p2, _)| p1.canonical_cmp(p2));

        let mut iter = sorted.into_iter().map(|(_, subdomain)| subdomain);
        let first = iter.next().ok_or(PartitionedTTError::Empty)?;
        let mut result = first.data().clone();

        for subdomain in iter {
            result = result
                .add(subdomain.data())
                .map_err(|source| PartitionedTTError::TensorTrain { source })?;
        }

        Ok(result)
    }
}

impl IntoIterator for PartitionedTT {
    type Item = (Projector, SubDomainTT);
    type IntoIter = std::collections::hash_map::IntoIter<Projector, SubDomainTT>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a PartitionedTT {
    type Item = (&'a Projector, &'a SubDomainTT);
    type IntoIter = std::collections::hash_map::Iter<'a, Projector, SubDomainTT>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests;
