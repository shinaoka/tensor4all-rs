use crate::error::{Result as TreeTciResult, TreeTciError};
use crate::{OwnedGlobalIndexBatch, SubtreeKey};

/// Multi-index type used by TreeTCI.
///
/// A `MultiIndex` is a vector of local indices, one per site in site order.
pub type MultiIndex = Vec<usize>;

/// Assemble one global site-order point from subtree-local assignments and
/// central site values.
///
/// Each subtree assignment maps a [`SubtreeKey`] to its local multi-index.
/// Each central assignment is a `(site, value)` pair.
///
/// Returns the assembled global multi-index with one entry per site.
///
/// # Errors
///
/// Returns an error when the construction or conversion fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{assemble_global_point, SubtreeKey};
///
/// let left = SubtreeKey::new(vec![0, 1]);
/// let right = SubtreeKey::new(vec![2]);
///
/// let point = assemble_global_point(
///     3,
///     &[(&left, &vec![10, 20]), (&right, &vec![30])],
///     &[],
/// ).unwrap();
/// assert_eq!(point, vec![10, 20, 30]);
///
/// // With a central site assignment
/// let key = SubtreeKey::new(vec![0]);
/// let point = assemble_global_point(
///     3,
///     &[(&key, &vec![5])],
///     &[(1, 7), (2, 9)],
/// ).unwrap();
/// assert_eq!(point, vec![5, 7, 9]);
/// ```
pub fn assemble_global_point(
    n_sites: usize,
    subtree_assignments: &[(&SubtreeKey, &MultiIndex)],
    central_assignments: &[(usize, usize)],
) -> TreeTciResult<MultiIndex> {
    let mut point = vec![usize::MAX; n_sites];

    for &(key, values) in subtree_assignments {
        if !(key.as_slice().len() == values.len()) {
            return Err(anyhow::anyhow!(
                "subtree key of length {} cannot be filled from multi-index of length {}",
                key.as_slice().len(),
                values.len()
            )
            .into());
        };
        for (&site, &value) in key.as_slice().iter().zip(values.iter()) {
            if site >= n_sites {
                return Err(TreeTciError::IndexOutOfBounds {
                    message: format!("site {site} is out of bounds for {n_sites} sites"),
                });
            }
            if !(point[site] == usize::MAX) {
                return Err(TreeTciError::IndexOutOfBounds {
                    message: format!("site {site} was assigned more than once"),
                });
            };
            point[site] = value;
        }
    }

    for &(site, value) in central_assignments {
        if site >= n_sites {
            return Err(TreeTciError::IndexOutOfBounds {
                message: format!("site {site} is out of bounds for {n_sites} sites"),
            });
        }
        if !(point[site] == usize::MAX) {
            return Err(TreeTciError::IndexOutOfBounds {
                message: format!("site {site} was assigned more than once"),
            });
        };
        point[site] = value;
    }

    if !(point.iter().all(|&value| value != usize::MAX)) {
        return Err(TreeTciError::IndexOutOfBounds {
            message: "global point assembly left some sites unassigned".to_string(),
        });
    };
    Ok(point)
}

/// Pack global points into column-major `(n_sites, n_points)` storage.
///
/// All points must have the same length (the number of sites).
///
/// # Errors
///
/// Returns an error when the construction or conversion fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::assemble_points_column_major;
///
/// let points = vec![vec![0, 1], vec![1, 0], vec![0, 0]];
/// let batch = assemble_points_column_major(&points).unwrap();
/// let view = batch.as_view();
///
/// assert_eq!(view.n_sites(), 2);
/// assert_eq!(view.n_points(), 3);
/// assert_eq!(view.get(0, 0), Some(0)); // point 0, site 0
/// assert_eq!(view.get(1, 0), Some(1)); // point 0, site 1
/// assert_eq!(view.get(0, 2), Some(0)); // point 2, site 0
/// ```
pub fn assemble_points_column_major(points: &[MultiIndex]) -> TreeTciResult<OwnedGlobalIndexBatch> {
    let n_points = points.len();
    let n_sites = points.first().map_or(0, Vec::len);
    if !(n_sites > 0) {
        return Err(anyhow::anyhow!("at least one point with one site is required").into());
    };
    if !(n_points > 0) {
        return Err(anyhow::anyhow!("at least one point is required").into());
    };
    if !(points.iter().all(|point| point.len() == n_sites)) {
        return Err(anyhow::anyhow!("all points must have the same site count").into());
    };

    let mut data = Vec::with_capacity(n_sites * n_points);
    for point in points {
        data.extend_from_slice(point);
    }
    OwnedGlobalIndexBatch::new(data, n_sites, n_points)
}

#[cfg(test)]
mod tests;
