//! Regression coverage for issue #618: the rank of an elementwise product must
//! follow the rank of the function, not the number of sites.
//!
//! The target is a product of two trigonometric sums on a base-four quantics
//! grid. Each factor `sum_j w_j cos(2 pi m_j x)` is a rank-`2K` tensor train for
//! every site count `R`, and the product of a `K`-term and a `K'`-term sum is a
//! `2 K K'`-term sum, so the exact rank of the product is bounded by `4 K K'`
//! independently of `R`. A run whose output rank tracks the algebraic bound of
//! the site space instead (`4^(R/2)`, the behavior reported in the issue) is
//! therefore visible as rank growth between two site counts.
//!
//! The factors also carry a large amplitude, which is what makes the effect
//! reproducible in a few seconds: with a scale-blind tolerance the requested
//! `1e-12` becomes a relative `1e-18` against an output magnitude of `1e6`,
//! below double-precision rounding, and the sweep then adds pivots for rounding
//! noise until it runs out of index space.

use std::f64::consts::TAU;
use tensor4all_aci::{elementwise, AciOptions};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const SITE_DIM: usize = 4;

/// Value of the grid coordinate `x` in `[0, 1)` for a base-four multi-index.
fn coordinate(index: &[usize]) -> f64 {
    index
        .iter()
        .enumerate()
        .map(|(site, &digit)| digit as f64 * (SITE_DIM as f64).powi(-(site as i32 + 1)))
        .sum()
}

/// `scale * sum_j weights[j] * cos(TAU * freqs[j] * x)` as an exact rank-`2K`
/// tensor train over `r` base-four sites.
///
/// Site `n` advances the phase of term `j` by `TAU * freqs[j] * digit *
/// 4^-(n+1)` through a two-by-two rotation, so the running state
/// `[cos(phase), sin(phase)]` reaches `[cos(TAU f x), sin(TAU f x)]` at the last
/// site, where it is contracted with the weight.
fn cosine_sum_tt(freqs: &[f64], weights: &[f64], scale: f64, r: usize) -> SimpleTensorTrain<f64> {
    assert_eq!(freqs.len(), weights.len());
    let terms = freqs.len();
    let rank = 2 * terms;
    let rotation = |term: usize, site: usize, digit: usize| {
        let phase = TAU * freqs[term] * digit as f64 * (SITE_DIM as f64).powi(-(site as i32 + 1));
        let (sin, cos) = phase.sin_cos();
        [[cos, sin], [-sin, cos]]
    };

    let mut cores = Vec::with_capacity(r);
    #[allow(clippy::needless_range_loop)]
    for site in 0..r {
        let left = if site == 0 { 1 } else { rank };
        let right = if site == r - 1 { 1 } else { rank };
        let mut data = vec![0.0; left * SITE_DIM * right];
        for digit in 0..SITE_DIM {
            for term in 0..terms {
                let rot = rotation(term, site, digit);
                for (row, rot_row) in rot.iter().enumerate() {
                    for (col, &value) in rot_row.iter().enumerate() {
                        // Start from [cos(0), sin(0)] = [1, 0] at the first
                        // site and project onto the weighted cosine component
                        // at the last one.
                        let (l, r_idx, entry) = match (site == 0, site == r - 1) {
                            (true, true) => {
                                if row != 0 || col != 0 {
                                    continue;
                                }
                                (0, 0, scale * weights[term] * value)
                            }
                            (true, false) => {
                                if row != 0 {
                                    continue;
                                }
                                (0, 2 * term + col, value)
                            }
                            (false, true) => {
                                if col != 0 {
                                    continue;
                                }
                                (2 * term + row, 0, scale * weights[term] * value)
                            }
                            (false, false) => (2 * term + row, 2 * term + col, value),
                        };
                        data[l + left * (digit + SITE_DIM * r_idx)] += entry;
                    }
                }
            }
        }
        cores.push(tensor3_from_data(data, left, SITE_DIM, right).unwrap());
    }
    SimpleTensorTrain::new(cores).unwrap()
}

fn cosine_sum_value(freqs: &[f64], weights: &[f64], scale: f64, x: f64) -> f64 {
    scale
        * freqs
            .iter()
            .zip(weights)
            .map(|(&f, &w)| w * (TAU * f * x).cos())
            .sum::<f64>()
}

/// Deterministic grid points spread over the index space.
fn sample_indices(r: usize, count: usize) -> Vec<Vec<usize>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..count)
        .map(|_| {
            (0..r)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % SITE_DIM as u64) as usize
                })
                .collect()
        })
        .collect()
}

struct ProductRun {
    rank: usize,
    max_rel_error: f64,
}

fn run_product(r: usize, options: &AciOptions<f64>) -> TestResult<ProductRun> {
    let f_freqs = [1.0, 5.0, 11.0];
    let f_weights = [1.0, 0.6, 0.3];
    let g_freqs = [2.0, 7.0, 13.0];
    let g_weights = [0.9, 0.5, 0.2];
    let scale = 1.0e3;

    let f = cosine_sum_tt(&f_freqs, &f_weights, scale, r);
    let g = cosine_sum_tt(&g_freqs, &g_weights, scale, r);
    let result = elementwise(|values| values[0] * values[1], &[f, g], options)?;

    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for index in sample_indices(r, 96) {
        let x = coordinate(&index);
        let want = cosine_sum_value(&f_freqs, &f_weights, scale, x)
            * cosine_sum_value(&g_freqs, &g_weights, scale, x);
        let got = result.tensor_train.evaluate(&index)?;
        max_abs = max_abs.max((got - want).abs());
        max_ref = max_ref.max(want.abs());
    }

    Ok(ProductRun {
        rank: result.tensor_train.rank(),
        max_rel_error: max_abs / max_ref,
    })
}

#[test]
fn product_rank_does_not_grow_with_the_site_count() -> TestResult {
    // Three terms per factor: the product has at most 2 * 3 * 3 = 18 cosine
    // terms, so its exact rank is at most 36 for every site count.
    const EXACT_RANK_BOUND: usize = 36;

    let options = AciOptions::<f64>::default();
    let short = run_product(6, &options)?;
    let long = run_product(8, &options)?;

    for (r, run) in [(6, &short), (8, &long)] {
        assert!(
            run.max_rel_error < 1e-10,
            "r = {r}: relative error {:.3e} is above 1e-10",
            run.max_rel_error
        );
        assert!(
            run.rank <= 6 * EXACT_RANK_BOUND / 5,
            "r = {r}: rank {} exceeds 1.2 times the exact rank bound {EXACT_RANK_BOUND}",
            run.rank
        );
    }

    assert!(
        long.rank * 5 <= short.rank * 6,
        "rank grew with the site count: {} at r = 6, {} at r = 8",
        short.rank,
        long.rank
    );
    Ok(())
}
