use crate::{AciError, Result};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar, Tensor3Ops};

pub(crate) fn validate_options<T: TTScalar>(options: &crate::AciOptions<T>) -> Result<()> {
    if options.max_iters == 0 {
        return Err(AciError::InvalidOptions {
            message: "max_iters must be at least 1".to_string(),
        });
    }

    if options.min_iters == 0 {
        return Err(AciError::InvalidOptions {
            message: "min_iters must be at least 1".to_string(),
        });
    }

    if options.max_bond_dim == Some(0) {
        return Err(AciError::InvalidOptions {
            message: "max_bond_dim must be at least 1".to_string(),
        });
    }

    if options.min_iters > options.max_iters {
        return Err(AciError::InvalidOptions {
            message: format!(
                "min_iters ({}) must be less than or equal to max_iters ({})",
                options.min_iters, options.max_iters
            ),
        });
    }

    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(AciError::InvalidOptions {
            message: "tolerance must be finite and non-negative".to_string(),
        });
    }

    if !options.tol_margin_global_search.is_finite() || options.tol_margin_global_search < 0.0 {
        return Err(AciError::InvalidOptions {
            message: "tol_margin_global_search must be finite and non-negative".to_string(),
        });
    }

    Ok(())
}

pub(crate) fn validate_inputs<T: TTScalar>(inputs: &[SimpleTensorTrain<T>]) -> Result<Vec<usize>> {
    let Some(first) = inputs.first() else {
        return Err(AciError::EmptyInputs);
    };

    let site_dims = first.site_dims();
    let expected_len = site_dims.len();
    if expected_len == 0 {
        return Err(AciError::InvalidOptions {
            message: "input tensor trains must have at least one site".to_string(),
        });
    }
    validate_positive_site_dims(&site_dims)?;
    validate_positive_core_dims(first, 0)?;

    for (input_index, input) in inputs[1..].iter().enumerate() {
        let input_index = input_index + 1;
        if input.len() != expected_len {
            return Err(AciError::LengthMismatch {
                expected: expected_len,
                got: input.len(),
            });
        }

        let input_site_dims = input.site_dims();
        validate_positive_site_dims(&input_site_dims)?;
        validate_positive_core_dims(input, input_index)?;

        for (site, (expected, got)) in site_dims.iter().zip(input_site_dims).enumerate() {
            if *expected != got {
                return Err(AciError::SiteDimMismatch {
                    site,
                    expected: *expected,
                    got,
                });
            }
        }
    }

    Ok(site_dims)
}

fn validate_positive_site_dims(site_dims: &[usize]) -> Result<()> {
    for (site, &site_dim) in site_dims.iter().enumerate() {
        if site_dim == 0 {
            return Err(AciError::InvalidOptions {
                message: format!("site {site} dimension must be positive, got 0"),
            });
        }
    }
    Ok(())
}

fn validate_positive_core_dims<T: TTScalar>(
    input: &SimpleTensorTrain<T>,
    input_index: usize,
) -> Result<()> {
    for site in 0..input.len() {
        let core = input.site_tensor(site);
        if core.left_dim() == 0 || core.site_dim() == 0 || core.right_dim() == 0 {
            return Err(AciError::InvalidOptions {
                message: format!(
                    "input {input_index} site {site} core bond dimension must be positive, \
                     got ({}, {}, {})",
                    core.left_dim(),
                    core.site_dim(),
                    core.right_dim()
                ),
            });
        }
    }
    Ok(())
}
