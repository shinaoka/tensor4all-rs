//! Inherent discrete grid implementation

use crate::error::{QuanticsGridError, Result};
use crate::{IndexTable, LookupEntry, UnfoldingScheme};

/// A discrete grid for quantics tensor train representations.
///
/// This structure maps between three coordinate systems:
/// - **Quantics indices**: Integer values for each tensor core (1-indexed)
/// - **Grid indices**: Integer positions in each dimension (1-indexed)
/// - **Original coordinates**: Integer positions based on origin and step
///
/// # Example
/// ```
/// use quanticsgrids::{InherentDiscreteGrid, UnfoldingScheme};
///
/// let grid = InherentDiscreteGrid::builder(&[3, 2])
///     .with_variable_names(&["x", "y"])
///     .with_unfolding_scheme(UnfoldingScheme::Fused)
///     .build()
///     .unwrap();
///
/// // Convert grid indices to quantics
/// let quantics = grid.grididx_to_quantics(&[5, 2]).unwrap();
/// // Convert back
/// let grididx = grid.quantics_to_grididx(&quantics).unwrap();
/// assert_eq!(grididx, vec![5, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct InherentDiscreteGrid {
    /// Number of dimensions
    ndims: usize,
    /// Resolution (number of bits) per dimension
    rs: Vec<usize>,
    /// Origin in each dimension (1-indexed by default)
    origin: Vec<i64>,
    /// Step size in each dimension
    step: Vec<i64>,
    /// Variable names for each dimension
    variable_names: Vec<String>,
    /// Numeric base (default 2)
    base: usize,
    /// Index table defining tensor structure
    index_table: IndexTable,
    /// Lookup table: lookup_table[dim][bit] -> (site_index, position_in_site)
    lookup_table: Vec<Vec<LookupEntry>>,
    /// Maximum grid index per dimension (base^R)
    max_grididx: Vec<i64>,
}

impl InherentDiscreteGrid {
    /// Create a new builder for an InherentDiscreteGrid.
    ///
    /// # Example
    /// ```
    /// use quanticsgrids::InherentDiscreteGrid;
    ///
    /// let grid = InherentDiscreteGrid::builder(&[3, 2])
    ///     .with_variable_names(&["x", "y"])
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder(rs: &[usize]) -> InherentDiscreteGridBuilder {
        InherentDiscreteGridBuilder::new(rs)
    }

    /// Create a grid from an explicit index table.
    pub fn from_index_table(
        variable_names: &[&str],
        index_table: IndexTable,
    ) -> InherentDiscreteGridBuilder {
        InherentDiscreteGridBuilder::from_index_table(variable_names, index_table)
    }

    /// Number of dimensions
    pub fn ndims(&self) -> usize {
        self.ndims
    }

    /// Number of tensor sites (cores)
    pub fn len(&self) -> usize {
        self.index_table.len()
    }

    /// Returns true if the grid has no tensor sites
    pub fn is_empty(&self) -> bool {
        self.index_table.is_empty()
    }

    /// Resolution (number of bits) per dimension
    pub fn rs(&self) -> &[usize] {
        &self.rs
    }

    /// Origin in each dimension
    pub fn origin(&self) -> &[i64] {
        &self.origin
    }

    /// Step size in each dimension
    pub fn step(&self) -> &[i64] {
        &self.step
    }

    /// Variable names
    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Numeric base
    pub fn base(&self) -> usize {
        self.base
    }

    /// Index table
    pub fn index_table(&self) -> &IndexTable {
        &self.index_table
    }

    /// Maximum grid index per dimension
    pub fn max_grididx(&self) -> &[i64] {
        &self.max_grididx
    }

    /// Local dimension of a tensor site
    pub fn site_dim(&self, site: usize) -> Result<usize> {
        if site >= self.index_table.len() {
            return Err(QuanticsGridError::SiteIndexOutOfBounds {
                site,
                max: self.index_table.len(),
            });
        }
        Ok(self.base.pow(self.index_table[site].len() as u32))
    }

    /// Local dimensions of all tensor sites
    pub fn local_dimensions(&self) -> Vec<usize> {
        self.index_table
            .iter()
            .map(|site| self.base.pow(site.len() as u32))
            .collect()
    }

    /// Minimum grid coordinate (origin)
    pub fn grid_min(&self) -> Vec<i64> {
        self.origin.clone()
    }

    /// Maximum grid coordinate
    pub fn grid_max(&self) -> Vec<i64> {
        self.origin
            .iter()
            .zip(self.step.iter())
            .zip(self.max_grididx.iter())
            .map(|((&o, &s), &m)| o + s * (m - 1))
            .collect()
    }

    // ========================================================================
    // Core conversion functions
    // ========================================================================

    /// Convert quantics indices to grid indices.
    ///
    /// # Arguments
    /// * `quantics` - Quantics values for each tensor site (1-indexed)
    ///
    /// # Returns
    /// Grid indices for each dimension (1-indexed)
    pub fn quantics_to_grididx(&self, quantics: &[i64]) -> Result<Vec<i64>> {
        self.validate_quantics(quantics)?;

        if self.base == 2 {
            Ok(self.quantics_to_grididx_base2(quantics))
        } else {
            Ok(self.quantics_to_grididx_general(quantics))
        }
    }

    /// Convert grid indices to quantics indices.
    ///
    /// # Arguments
    /// * `grididx` - Grid indices for each dimension (1-indexed)
    ///
    /// # Returns
    /// Quantics values for each tensor site (1-indexed)
    pub fn grididx_to_quantics(&self, grididx: &[i64]) -> Result<Vec<i64>> {
        let grididx = self.expand_grididx(grididx)?;
        self.validate_grididx(&grididx)?;

        let mut result = vec![1i64; self.index_table.len()];
        if self.base == 2 {
            self.grididx_to_quantics_base2(&mut result, &grididx);
        } else {
            self.grididx_to_quantics_general(&mut result, &grididx);
        }
        Ok(result)
    }

    /// Convert grid indices to original coordinates.
    ///
    /// # Arguments
    /// * `grididx` - Grid indices for each dimension (1-indexed)
    ///
    /// # Returns
    /// Original integer coordinates
    pub fn grididx_to_origcoord(&self, grididx: &[i64]) -> Result<Vec<i64>> {
        let grididx = self.expand_grididx(grididx)?;
        self.validate_grididx(&grididx)?;

        Ok(self
            .origin
            .iter()
            .zip(grididx.iter())
            .zip(self.step.iter())
            .map(|((&o, &g), &s)| o + (g - 1) * s)
            .collect())
    }

    /// Convert original coordinates to grid indices.
    ///
    /// # Arguments
    /// * `coord` - Original integer coordinates
    ///
    /// # Returns
    /// Grid indices for each dimension (1-indexed)
    pub fn origcoord_to_grididx(&self, coord: &[i64]) -> Result<Vec<i64>> {
        let coord = self.expand_coord(coord)?;
        self.validate_origcoord(&coord)?;

        let indices: Vec<i64> = self
            .origin
            .iter()
            .zip(coord.iter())
            .zip(self.step.iter())
            .zip(self.max_grididx.iter())
            .map(|(((&o, &c), &s), &m)| ((c - o) / s + 1).clamp(1, m))
            .collect();

        Ok(indices)
    }

    /// Convert original coordinates to quantics indices.
    pub fn origcoord_to_quantics(&self, coord: &[i64]) -> Result<Vec<i64>> {
        let grididx = self.origcoord_to_grididx(coord)?;
        self.grididx_to_quantics(&grididx)
    }

    /// Convert quantics indices to original coordinates.
    pub fn quantics_to_origcoord(&self, quantics: &[i64]) -> Result<Vec<i64>> {
        let grididx = self.quantics_to_grididx(quantics)?;
        self.grididx_to_origcoord(&grididx)
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn validate_quantics(&self, quantics: &[i64]) -> Result<()> {
        if quantics.len() != self.index_table.len() {
            return Err(QuanticsGridError::WrongQuanticsLength {
                expected: self.index_table.len(),
                actual: quantics.len(),
            });
        }

        for (site, &val) in quantics.iter().enumerate() {
            let max = self.site_dim(site)? as i64;
            if val < 1 || val > max {
                return Err(QuanticsGridError::QuanticsOutOfRange {
                    site,
                    value: val,
                    max,
                });
            }
        }

        Ok(())
    }

    fn validate_grididx(&self, grididx: &[i64]) -> Result<()> {
        for (dim, (&val, &max)) in grididx.iter().zip(self.max_grididx.iter()).enumerate() {
            if val < 1 || val > max {
                return Err(QuanticsGridError::GridIndexOutOfBounds {
                    dim,
                    value: val,
                    max,
                });
            }
        }
        Ok(())
    }

    fn validate_origcoord(&self, coord: &[i64]) -> Result<()> {
        let min = self.grid_min();
        let max = self.grid_max();
        for (dim, ((&c, &lo), &hi)) in coord.iter().zip(min.iter()).zip(max.iter()).enumerate() {
            if c < lo || c > hi {
                return Err(QuanticsGridError::CoordinateOutOfBounds {
                    dim,
                    value: c as f64,
                    lower: lo as f64,
                    upper: hi as f64,
                });
            }
        }
        Ok(())
    }

    /// Expand a scalar or lower-dimensional input to full dimensions
    fn expand_grididx(&self, grididx: &[i64]) -> Result<Vec<i64>> {
        if grididx.len() == 1 && self.ndims > 1 {
            Ok(vec![grididx[0]; self.ndims])
        } else if grididx.len() == self.ndims {
            Ok(grididx.to_vec())
        } else {
            Err(QuanticsGridError::DimensionMismatch {
                expected: self.ndims,
                actual: grididx.len(),
            })
        }
    }

    fn expand_coord(&self, coord: &[i64]) -> Result<Vec<i64>> {
        if coord.len() == 1 && self.ndims > 1 {
            Ok(vec![coord[0]; self.ndims])
        } else if coord.len() == self.ndims {
            Ok(coord.to_vec())
        } else {
            Err(QuanticsGridError::DimensionMismatch {
                expected: self.ndims,
                actual: coord.len(),
            })
        }
    }

    fn quantics_to_grididx_base2(&self, quantics: &[i64]) -> Vec<i64> {
        (0..self.ndims)
            .map(|d| {
                let r_d = self.rs[d];
                let mut grididx = 0i64;

                for bitnumber in 0..r_d {
                    let (site_idx, pos_in_site) = self.lookup_table[d][bitnumber];
                    let bit_position = self.index_table[site_idx].len() - 1 - pos_in_site;
                    let digit = ((quantics[site_idx] - 1) >> bit_position) & 1;
                    grididx |= digit << (r_d - 1 - bitnumber);
                }
                grididx + 1
            })
            .collect()
    }

    fn quantics_to_grididx_general(&self, quantics: &[i64]) -> Vec<i64> {
        let base = self.base as i64;

        (0..self.ndims)
            .map(|d| {
                let r_d = self.rs[d];
                let mut grididx = 1i64;

                for bitnumber in 0..r_d {
                    let (site_idx, pos_in_site) = self.lookup_table[d][bitnumber];
                    let site_len = self.index_table[site_idx].len();

                    let mut temp = quantics[site_idx] - 1;
                    for _ in 0..(site_len - 1 - pos_in_site) {
                        temp /= base;
                    }
                    let digit = temp % base;

                    grididx += digit * base.pow((r_d - 1 - bitnumber) as u32);
                }
                grididx
            })
            .collect()
    }

    fn grididx_to_quantics_base2(&self, result: &mut [i64], grididx: &[i64]) {
        for (&grid_val, (&r_d, lookup)) in grididx
            .iter()
            .zip(self.rs.iter().zip(self.lookup_table.iter()))
            .take(self.ndims)
        {
            let zero_based_idx = grid_val - 1;

            for (bitnumber, &(site_idx, pos_in_site)) in lookup.iter().enumerate().take(r_d) {
                let site_length = self.index_table[site_idx].len();

                let bit_position = r_d - 1 - bitnumber;
                let digit = (zero_based_idx >> bit_position) & 1;

                let power = site_length - 1 - pos_in_site;
                result[site_idx] += digit << power;
            }
        }
    }

    fn grididx_to_quantics_general(&self, result: &mut [i64], grididx: &[i64]) {
        let base = self.base as i64;

        for (&grid_val, (&r_d, lookup)) in grididx
            .iter()
            .zip(self.rs.iter().zip(self.lookup_table.iter()))
            .take(self.ndims)
        {
            let zero_based_idx = grid_val - 1;

            for (bitnumber, &(site_idx, pos_in_site)) in lookup.iter().enumerate().take(r_d) {
                let site_length = self.index_table[site_idx].len();

                let bit_position = r_d - 1 - bitnumber;
                let digit = (zero_based_idx / base.pow(bit_position as u32)) % base;

                let power = site_length - 1 - pos_in_site;
                result[site_idx] += digit * base.pow(power as u32);
            }
        }
    }
}

/// Builder for InherentDiscreteGrid
#[derive(Debug, Clone)]
pub struct InherentDiscreteGridBuilder {
    rs: Vec<usize>,
    origin: Option<Vec<i64>>,
    step: Option<Vec<i64>>,
    variable_names: Option<Vec<String>>,
    base: usize,
    unfolding_scheme: UnfoldingScheme,
    index_table: Option<IndexTable>,
}

impl InherentDiscreteGridBuilder {
    /// Create a new builder with given resolutions
    pub fn new(rs: &[usize]) -> Self {
        Self {
            rs: rs.to_vec(),
            origin: None,
            step: None,
            variable_names: None,
            base: 2,
            unfolding_scheme: UnfoldingScheme::Fused,
            index_table: None,
        }
    }

    /// Create a builder from an explicit index table
    pub fn from_index_table(variable_names: &[&str], index_table: IndexTable) -> Self {
        // Compute Rs from index table
        let rs: Vec<usize> = variable_names
            .iter()
            .map(|&name| {
                index_table
                    .iter()
                    .flatten()
                    .filter(|(var, _)| var == name)
                    .count()
            })
            .collect();

        Self {
            rs,
            origin: None,
            step: None,
            variable_names: Some(variable_names.iter().map(|s| s.to_string()).collect()),
            base: 2,
            unfolding_scheme: UnfoldingScheme::Fused,
            index_table: Some(index_table),
        }
    }

    /// Set the origin for each dimension
    pub fn with_origin(mut self, origin: &[i64]) -> Self {
        self.origin = Some(origin.to_vec());
        self
    }

    /// Set the step size for each dimension
    pub fn with_step(mut self, step: &[i64]) -> Self {
        self.step = Some(step.to_vec());
        self
    }

    /// Set variable names
    pub fn with_variable_names(mut self, names: &[&str]) -> Self {
        self.variable_names = Some(names.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set the numeric base (default 2)
    pub fn with_base(mut self, base: usize) -> Self {
        self.base = base;
        self
    }

    /// Set the unfolding scheme
    pub fn with_unfolding_scheme(mut self, scheme: UnfoldingScheme) -> Self {
        self.unfolding_scheme = scheme;
        self
    }

    /// Build the InherentDiscreteGrid
    pub fn build(self) -> Result<InherentDiscreteGrid> {
        let ndims = self.rs.len();
        if ndims == 0 {
            return Err(QuanticsGridError::NoResolutions);
        }

        if self.base < 2 {
            return Err(QuanticsGridError::InvalidBase(self.base));
        }

        // Validate resolutions
        for (d, &r) in self.rs.iter().enumerate() {
            if !rangecheck_r(r, self.base) {
                return Err(QuanticsGridError::ResolutionTooLarge { r, base: self.base });
            }
            if let Some(ref step) = self.step {
                if step.len() > d && step[d] < 1 {
                    return Err(QuanticsGridError::InvalidStep {
                        dim: d,
                        value: step[d],
                    });
                }
            }
        }

        // Default values
        let origin = self.origin.unwrap_or_else(|| vec![1; ndims]);
        let step = self.step.unwrap_or_else(|| vec![1; ndims]);
        let variable_names = self.variable_names.unwrap_or_else(|| {
            (1..=ndims).map(|i| i.to_string()).collect()
        });

        // Validate dimensions match
        if origin.len() != ndims {
            return Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: origin.len(),
            });
        }
        if step.len() != ndims {
            return Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: step.len(),
            });
        }
        if variable_names.len() != ndims {
            return Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: variable_names.len(),
            });
        }

        // Check for duplicate variable names
        for (i, name) in variable_names.iter().enumerate() {
            for other in variable_names.iter().skip(i + 1) {
                if name == other {
                    return Err(QuanticsGridError::DuplicateVariableName(name.clone()));
                }
            }
        }

        // Build or use provided index table
        let index_table = match self.index_table {
            Some(table) => table,
            None => build_index_table(&variable_names, &self.rs, self.unfolding_scheme),
        };

        // Build lookup table
        let lookup_table = build_lookup_table(&self.rs, &index_table, &variable_names)?;

        // Compute max grid indices
        let max_grididx: Vec<i64> = self.rs.iter().map(|&r| (self.base as i64).pow(r as u32)).collect();

        Ok(InherentDiscreteGrid {
            ndims,
            rs: self.rs,
            origin,
            step,
            variable_names,
            base: self.base,
            index_table,
            lookup_table,
            max_grididx,
        })
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if base^R fits in i64
fn rangecheck_r(r: usize, base: usize) -> bool {
    let mut result: i64 = 1;
    for _ in 0..r {
        if result > i64::MAX / (base as i64) {
            return false;
        }
        result *= base as i64;
    }
    true
}

/// Build an index table from variable names and resolutions
fn build_index_table(
    variable_names: &[String],
    rs: &[usize],
    scheme: UnfoldingScheme,
) -> IndexTable {
    let max_r = *rs.iter().max().unwrap_or(&0);
    let mut index_table = IndexTable::new();

    for bitnumber in 0..max_r {
        match scheme {
            UnfoldingScheme::Interleaved => {
                add_interleaved_indices(&mut index_table, variable_names, rs, bitnumber);
            }
            UnfoldingScheme::Fused => {
                add_fused_indices(&mut index_table, variable_names, rs, bitnumber);
            }
        }
    }

    index_table
}

fn add_interleaved_indices(
    index_table: &mut IndexTable,
    variable_names: &[String],
    rs: &[usize],
    bitnumber: usize,
) {
    for (d, name) in variable_names.iter().enumerate() {
        if bitnumber < rs[d] {
            let qindex = (name.clone(), bitnumber + 1); // 1-indexed bit number
            index_table.push(vec![qindex]);
        }
    }
}

fn add_fused_indices(
    index_table: &mut IndexTable,
    variable_names: &[String],
    rs: &[usize],
    bitnumber: usize,
) {
    let mut indices_bitnumber = Vec::new();
    // Add dimensions in reverse order to match Julia convention
    for d in (0..variable_names.len()).rev() {
        if bitnumber < rs[d] {
            let qindex = (variable_names[d].clone(), bitnumber + 1); // 1-indexed
            indices_bitnumber.push(qindex);
        }
    }
    if !indices_bitnumber.is_empty() {
        index_table.push(indices_bitnumber);
    }
}

/// Build lookup table from index table
fn build_lookup_table(
    rs: &[usize],
    index_table: &IndexTable,
    variable_names: &[String],
) -> Result<Vec<Vec<LookupEntry>>> {
    let mut lookup_table: Vec<Vec<LookupEntry>> = rs
        .iter()
        .map(|&r| vec![(0, 0); r])
        .collect();

    let mut index_visited: Vec<Vec<bool>> = rs.iter().map(|&r| vec![false; r]).collect();

    for (site_idx, quantics_indices) in index_table.iter().enumerate() {
        for (pos_in_site, (var_name, bitnumber)) in quantics_indices.iter().enumerate() {
            let var_idx = variable_names
                .iter()
                .position(|n| n == var_name)
                .ok_or_else(|| QuanticsGridError::UnknownVariable {
                    variable: var_name.clone(),
                    valid: variable_names.to_vec(),
                })?;

            // bitnumber is 1-indexed in QuanticsIndex
            let bit_idx = bitnumber - 1;

            if *bitnumber > rs[var_idx] {
                return Err(QuanticsGridError::InvalidBitNumber {
                    variable: var_name.clone(),
                    bit: *bitnumber,
                    resolution: rs[var_idx],
                });
            }

            if index_visited[var_idx][bit_idx] {
                return Err(QuanticsGridError::DuplicateIndexEntry {
                    variable: var_name.clone(),
                    bit: *bitnumber,
                });
            }

            lookup_table[var_idx][bit_idx] = (site_idx, pos_in_site);
            index_visited[var_idx][bit_idx] = true;
        }
    }

    // Check all indices are covered
    for (var_idx, visited) in index_visited.iter().enumerate() {
        for (bit_idx, &v) in visited.iter().enumerate() {
            if !v {
                return Err(QuanticsGridError::MissingIndexEntry {
                    variable: variable_names[var_idx].clone(),
                    bit: bit_idx + 1, // 1-indexed
                });
            }
        }
    }

    Ok(lookup_table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_1d_grid() {
        let grid = InherentDiscreteGrid::builder(&[3]).build().unwrap();
        assert_eq!(grid.ndims(), 1);
        assert_eq!(grid.len(), 3); // 3 sites for fused scheme
        assert_eq!(grid.base(), 2);
        assert_eq!(grid.rs(), &[3]);
        assert_eq!(grid.max_grididx(), &[8]);
    }

    #[test]
    fn test_basic_2d_grid() {
        let grid = InherentDiscreteGrid::builder(&[3, 2])
            .with_variable_names(&["x", "y"])
            .build()
            .unwrap();

        assert_eq!(grid.ndims(), 2);
        assert_eq!(grid.variable_names(), &["x", "y"]);
        assert_eq!(grid.max_grididx(), &[8, 4]);
    }

    #[test]
    fn test_grididx_to_quantics_roundtrip() {
        let grid = InherentDiscreteGrid::builder(&[3, 2])
            .with_variable_names(&["a", "b"])
            .build()
            .unwrap();

        let grididx = vec![5i64, 2];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let back = grid.quantics_to_grididx(&quantics).unwrap();
        assert_eq!(back, grididx);
    }

    #[test]
    fn test_all_grididx_roundtrip() {
        let grid = InherentDiscreteGrid::builder(&[2, 2]).build().unwrap();

        for x in 1..=4 {
            for y in 1..=4 {
                let grididx = vec![x, y];
                let quantics = grid.grididx_to_quantics(&grididx).unwrap();
                let back = grid.quantics_to_grididx(&quantics).unwrap();
                assert_eq!(back, grididx, "Failed for grididx {:?}", grididx);
            }
        }
    }

    #[test]
    fn test_interleaved_scheme() {
        let grid = InherentDiscreteGrid::builder(&[2, 2])
            .with_unfolding_scheme(UnfoldingScheme::Interleaved)
            .build()
            .unwrap();

        // Interleaved: [1_1], [2_1], [1_2], [2_2]
        assert_eq!(grid.len(), 4);
        for x in 1..=4 {
            for y in 1..=4 {
                let grididx = vec![x, y];
                let quantics = grid.grididx_to_quantics(&grididx).unwrap();
                let back = grid.quantics_to_grididx(&quantics).unwrap();
                assert_eq!(back, grididx);
            }
        }
    }

    #[test]
    fn test_base3_grid() {
        let grid = InherentDiscreteGrid::builder(&[2])
            .with_base(3)
            .build()
            .unwrap();

        assert_eq!(grid.base(), 3);
        assert_eq!(grid.max_grididx(), &[9]); // 3^2 = 9

        for x in 1..=9 {
            let grididx = vec![x];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx);
        }
    }

    #[test]
    fn test_origcoord_conversion() {
        let grid = InherentDiscreteGrid::builder(&[2])
            .with_origin(&[0])
            .with_step(&[1])
            .build()
            .unwrap();

        // origin=0, step=1, max_grididx=4
        // grididx 1 -> origcoord 0
        // grididx 4 -> origcoord 3
        let coord = grid.grididx_to_origcoord(&[1]).unwrap();
        assert_eq!(coord, vec![0]);

        let coord = grid.grididx_to_origcoord(&[4]).unwrap();
        assert_eq!(coord, vec![3]);

        let idx = grid.origcoord_to_grididx(&[2]).unwrap();
        assert_eq!(idx, vec![3]);
    }

    #[test]
    fn test_error_invalid_base() {
        let result = InherentDiscreteGrid::builder(&[3]).with_base(1).build();
        assert!(matches!(result, Err(QuanticsGridError::InvalidBase(1))));
    }

    #[test]
    fn test_error_duplicate_variable_names() {
        let result = InherentDiscreteGrid::builder(&[2, 2])
            .with_variable_names(&["x", "x"])
            .build();
        assert!(matches!(
            result,
            Err(QuanticsGridError::DuplicateVariableName(_))
        ));
    }

    #[test]
    fn test_error_quantics_out_of_range() {
        let grid = InherentDiscreteGrid::builder(&[2]).build().unwrap();
        // Rs=[2] with Fused creates 2 sites, each with dim 2
        // So quantics should have length 2, and each value should be in [1, 2]
        let result = grid.quantics_to_grididx(&[5, 1]); // 5 is out of range [1, 2]
        assert!(matches!(
            result,
            Err(QuanticsGridError::QuanticsOutOfRange { .. })
        ));
    }

    #[test]
    fn test_error_grididx_out_of_bounds() {
        let grid = InherentDiscreteGrid::builder(&[2]).build().unwrap();
        let result = grid.grididx_to_quantics(&[5]); // max is 4
        assert!(matches!(
            result,
            Err(QuanticsGridError::GridIndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_local_dimensions() {
        let grid = InherentDiscreteGrid::builder(&[3, 2])
            .with_unfolding_scheme(UnfoldingScheme::Fused)
            .build()
            .unwrap();

        let dims = grid.local_dimensions();
        // Fused scheme with Rs=(3,2): sites have 2, 2, 1 indices each -> dims 4, 4, 2
        // Actually: bitnumber 0: [b, a] -> dim 4
        //           bitnumber 1: [b, a] -> dim 4
        //           bitnumber 2: [a] -> dim 2
        assert_eq!(dims, vec![4, 4, 2]);
    }

    #[test]
    fn test_from_index_table() {
        // Create a custom index table like in Julia: [[(:a, 1), (:b, 2)], [(:a, 2)], [(:b, 1), (:a, 3)]]
        let index_table = vec![
            vec![("a".to_string(), 1), ("b".to_string(), 2)],
            vec![("a".to_string(), 2)],
            vec![("b".to_string(), 1), ("a".to_string(), 3)],
        ];

        let grid = InherentDiscreteGrid::from_index_table(&["a", "b"], index_table)
            .build()
            .unwrap();

        assert_eq!(grid.ndims(), 2);
        assert_eq!(grid.rs(), &[3, 2]); // a has 3 bits, b has 2 bits
        assert_eq!(grid.len(), 3);

        // Test roundtrip
        for x in 1..=8 {
            for y in 1..=4 {
                let grididx = vec![x, y];
                let quantics = grid.grididx_to_quantics(&grididx).unwrap();
                let back = grid.quantics_to_grididx(&quantics).unwrap();
                assert_eq!(back, grididx);
            }
        }
    }
}
