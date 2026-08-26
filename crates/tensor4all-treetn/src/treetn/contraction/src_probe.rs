//! Gaussian probe generation and column-major probe batches for SRC.

use anyhow::Result;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use tensor4all_core::IndexLike;

/// Reusable Gaussian probe columns indexed by physical tensor legs.
///
/// For each index, coefficients are stored as a column-major `dim × width`
/// matrix. Extending the bank advances one persistent RNG and appends columns,
/// so an adaptive run observes exactly the same prefix as a fixed-width run
/// with the same seed and index order.
pub(super) struct ProbeBank<I> {
    indices: Vec<I>,
    coefficients: HashMap<I, Vec<f64>>,
    rng: rand::rngs::StdRng,
    width: usize,
}

impl<I> ProbeBank<I>
where
    I: IndexLike,
{
    /// Construct a bank with `width` Gaussian columns.
    pub(super) fn new(indices: Vec<I>, width: usize, seed: u64) -> Result<Self> {
        if width == 0 {
            anyhow::bail!("SRC probe bank width must be at least 1");
        }
        let mut seen = HashSet::with_capacity(indices.len());
        let mut coefficients = HashMap::with_capacity(indices.len());
        for index in &indices {
            if index.dim() == 0 {
                anyhow::bail!("SRC probe index {:?} has zero dimension", index);
            }
            if !seen.insert(index.clone()) {
                anyhow::bail!("SRC probe index {:?} occurs more than once", index);
            }
            let capacity = index
                .dim()
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("SRC probe bank size overflow"))?;
            coefficients.insert(index.clone(), Vec::with_capacity(capacity));
        }

        let mut bank = Self {
            indices,
            coefficients,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            width: 0,
        };
        bank.extend_to(width)?;
        Ok(bank)
    }

    /// Return the number of columns currently stored in the bank.
    pub(super) fn width(&self) -> usize {
        self.width
    }

    /// Return the column-major coefficient matrix for one physical index.
    pub(super) fn coefficients(&self, index: &I) -> Option<&[f64]> {
        self.coefficients.get(index).map(Vec::as_slice)
    }

    /// Append Gaussian columns until the bank reaches `target_width`.
    pub(super) fn extend_to(&mut self, target_width: usize) -> Result<()> {
        if target_width <= self.width {
            return Ok(());
        }

        for _ in self.width..target_width {
            for index in &self.indices {
                let values = self
                    .coefficients
                    .get_mut(index)
                    .ok_or_else(|| anyhow::anyhow!("SRC probe index is missing from bank"))?;
                values.extend((0..index.dim()).map(|_| standard_normal(&mut self.rng)));
            }
        }
        self.width = target_width;
        Ok(())
    }
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::ProbeBank;
    use tensor4all_core::{DynIndex, IndexLike};

    #[test]
    fn probe_bank_extension_preserves_the_existing_prefix() {
        let first = DynIndex::new_dyn(3);
        let second = DynIndex::new_dyn(2);
        let indices = vec![first.clone(), second.clone()];

        let mut extended = ProbeBank::new(indices.clone(), 2, 17).unwrap();
        let prefix_first = extended.coefficients(&first).unwrap().to_vec();
        let prefix_second = extended.coefficients(&second).unwrap().to_vec();
        extended.extend_to(5).unwrap();

        let reference = ProbeBank::new(indices, 5, 17).unwrap();
        assert_eq!(extended.width(), 5);
        assert_eq!(
            &extended.coefficients(&first).unwrap()[..6],
            &prefix_first[..]
        );
        assert_eq!(
            &extended.coefficients(&second).unwrap()[..4],
            &prefix_second[..]
        );
        assert_eq!(
            extended.coefficients(&first).unwrap(),
            reference.coefficients(&first).unwrap()
        );
        assert_eq!(
            extended.coefficients(&second).unwrap(),
            reference.coefficients(&second).unwrap()
        );
        assert_eq!(first.dim(), 3);
        assert_eq!(second.dim(), 2);
    }

    #[test]
    fn probe_bank_rejects_zero_width_and_zero_dimensional_indices() {
        let index = DynIndex::new_dyn(2);
        assert!(ProbeBank::new(vec![index.clone()], 0, 0).is_err());

        let zero_dimensional = DynIndex::new_dyn(0);
        assert!(ProbeBank::new(vec![zero_dimensional], 1, 0).is_err());
    }
}
