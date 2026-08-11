use crate::scalar::{sample_standard_normal, AciScalar};
use crate::validation::{validate_inputs, validate_options};
use crate::{AciError, AciOptions, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain, Tensor3, Tensor3Ops,
};

// Protective internal allocation guard for generated or supplied guesses; this
// is not a public tuning knob.
pub(crate) const MAX_INITIAL_GUESS_CORE_ENTRIES: usize = 10_000_000;
pub(crate) const MAX_INITIAL_GUESS_TOTAL_ENTRIES: usize = 10_000_000;

pub(crate) fn initial_guess<T: AciScalar>(
    inputs: &[SimpleTensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<SimpleTensorTrain<T>> {
    let site_dims = validate_inputs(inputs)?;
    validate_options(options)?;

    if let Some(guess) = &options.initial_guess {
        validate_existing_initial_guess(guess, &site_dims, options.max_bond_dim)?;
        return Ok(guess.clone());
    }

    let link_dims = default_link_dims(inputs, &site_dims, options.max_bond_dim)?;
    let core_dims = initial_guess_core_dims(&site_dims, &link_dims);
    initial_guess_total_entry_count(&core_dims)?;

    let mut rng = ChaCha8Rng::seed_from_u64(options.rng_seed);
    let mut cores = Vec::with_capacity(core_dims.len());

    for (left_dim, site_dim, right_dim) in core_dims {
        cores.push(random_core(left_dim, site_dim, right_dim, &mut rng)?);
    }

    Ok(SimpleTensorTrain::new(cores)?)
}

fn validate_existing_initial_guess<T: AciScalar>(
    guess: &SimpleTensorTrain<T>,
    site_dims: &[usize],
    max_bond_dim: Option<usize>,
) -> Result<()> {
    let guess_site_dims = guess.site_dims();
    if guess_site_dims != site_dims {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "site dimensions must match inputs: expected {:?}, got {:?}",
                site_dims, guess_site_dims
            ),
        });
    }

    for (site, core) in guess.site_tensors().iter().enumerate() {
        if core.left_dim() == 0 || core.site_dim() == 0 || core.right_dim() == 0 {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "initial guess core {site} dimensions must be positive: got \
                     ({}, {}, {})",
                    core.left_dim(),
                    core.site_dim(),
                    core.right_dim()
                ),
            });
        }
    }

    for (bond, link_dim) in guess.link_dims().into_iter().enumerate() {
        if max_bond_dim.is_some_and(|cap| link_dim > cap) {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "initial guess bond dimension at bond {bond} exceeds max_bond_dim: \
                     got {link_dim}, max {max_bond_dim:?}"
                ),
            });
        }
    }

    initial_guess_existing_entry_count(guess)?;
    Ok(())
}

fn initial_guess_core_dims(site_dims: &[usize], link_dims: &[usize]) -> Vec<(usize, usize, usize)> {
    site_dims
        .iter()
        .enumerate()
        .map(|(site, &site_dim)| {
            let left_dim = if site == 0 { 1 } else { link_dims[site - 1] };
            let right_dim = link_dims.get(site).copied().unwrap_or(1);
            (left_dim, site_dim, right_dim)
        })
        .collect()
}

fn default_link_dims<T: AciScalar>(
    inputs: &[SimpleTensorTrain<T>],
    site_dims: &[usize],
    max_bond_dim: Option<usize>,
) -> Result<Vec<usize>> {
    if site_dims.len() <= 1 {
        return Ok(Vec::new());
    }

    let mut left_products = Vec::with_capacity(site_dims.len() - 1);
    let mut left_product = 1usize;
    for &site_dim in &site_dims[..site_dims.len() - 1] {
        left_product = checked_mul(left_product, site_dim, "site dimension product")?;
        left_products.push(left_product);
    }

    let mut right_products = vec![1usize; site_dims.len() - 1];
    let mut right_product = 1usize;
    for bond in (0..site_dims.len() - 1).rev() {
        right_product = checked_mul(right_product, site_dims[bond + 1], "site dimension product")?;
        right_products[bond] = right_product;
    }

    let mut link_dims = Vec::with_capacity(site_dims.len() - 1);
    for bond in 0..site_dims.len() - 1 {
        let min_input_link_dim = inputs
            .iter()
            .map(|input| input.link_dim(bond))
            .min()
            .unwrap_or(usize::MAX);
        let link_dim = max_bond_dim
            .map_or(left_products[bond].min(right_products[bond]), |cap| {
                left_products[bond].min(right_products[bond]).min(cap)
            })
            .min(min_input_link_dim)
            .max(1);
        link_dims.push(link_dim);
    }

    Ok(link_dims)
}

fn random_core<T: AciScalar>(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Tensor3<T>> {
    let len = initial_guess_core_entry_count(left_dim, site_dim, right_dim)?;
    let data = (0..len)
        .map(|_| sample_standard_normal(rng))
        .collect::<Vec<_>>();
    Ok(tensor3_from_data(data, left_dim, site_dim, right_dim)?)
}

pub(crate) fn initial_guess_core_entry_count(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<usize> {
    let len = checked_mul(
        checked_mul(left_dim, site_dim, "initial guess core size")?,
        right_dim,
        "initial guess core size",
    )?;
    if len > MAX_INITIAL_GUESS_CORE_ENTRIES {
        return Err(AciError::InvalidOptions {
            message: format!(
                "initial guess core size {len} exceeds internal limit of \
                 {MAX_INITIAL_GUESS_CORE_ENTRIES} entries"
            ),
        });
    }
    Ok(len)
}

pub(crate) fn initial_guess_total_entry_count(
    core_dims: &[(usize, usize, usize)],
) -> Result<usize> {
    let mut total = 0usize;
    for &(left_dim, site_dim, right_dim) in core_dims {
        let len = initial_guess_core_entry_count(left_dim, site_dim, right_dim)?;
        total = checked_add(total, len, "initial guess total size")?;
        if total > MAX_INITIAL_GUESS_TOTAL_ENTRIES {
            return Err(AciError::InvalidOptions {
                message: format!(
                    "initial guess total size {total} exceeds internal limit of \
                     {MAX_INITIAL_GUESS_TOTAL_ENTRIES} entries"
                ),
            });
        }
    }
    Ok(total)
}

pub(crate) fn initial_guess_existing_entry_count<T: AciScalar>(
    guess: &SimpleTensorTrain<T>,
) -> Result<usize> {
    let mut total = 0usize;
    for core in guess.site_tensors() {
        let len =
            initial_guess_core_entry_count(core.left_dim(), core.site_dim(), core.right_dim())?;
        total = checked_add(total, len, "initial guess total size")?;
        if total > MAX_INITIAL_GUESS_TOTAL_ENTRIES {
            return Err(AciError::InvalidOptions {
                message: format!(
                    "initial guess total size {total} exceeds internal limit of \
                     {MAX_INITIAL_GUESS_TOTAL_ENTRIES} entries"
                ),
            });
        }
    }
    Ok(total)
}

fn checked_add(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AciError::InvalidOptions {
            message: format!("{description} overflows usize"),
        })
}

fn checked_mul(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidOptions {
            message: format!("{description} overflows usize"),
        })
}
