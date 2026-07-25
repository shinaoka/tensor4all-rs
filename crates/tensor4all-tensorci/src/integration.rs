//! Numerical integration using TCI and Gauss-Kronrod quadrature.
//!
//! [`integrate`] approximates a multi-dimensional integral over a hypercube
//! by building a tensor-train representation of the integrand on
//! Gauss-Kronrod nodes and then summing the tensor train.
//!
//! Supported quadrature orders: 15, 31, 41, 51, and 61 (standard fixed
//! Gauss-Kronrod rules). Julia's `TensorCrossInterpolation.jl` obtains rules
//! from `QuadGK.kronrod`; Rust intentionally uses this deterministic embedded
//! rule set.

use crate::error::{Result, TCIError};
use crate::tensorci2::{crossinterpolate2, TCI2Options};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar};
use tensor4all_tcicore::{MatrixLuciScalar, MultiIndex, Scalar};

/// Gauss-Kronrod 15-point rule: nodes on [-1, 1]
const GK15_NODES: [f64; 15] = [
    -0.991_455_371_120_812_6,
    -0.949_107_912_342_758_5,
    -0.864_864_423_359_769_1,
    -0.741_531_185_599_394_5,
    -0.586_087_235_467_691_1,
    -0.405_845_151_377_397_2,
    -0.207_784_955_007_898_48,
    0.000000000000000000000000000000000,
    0.207_784_955_007_898_48,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_5,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

/// Gauss-Kronrod 15-point rule: weights on [-1, 1]
const GK15_WEIGHTS: [f64; 15] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_19,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_89,
    0.209_482_141_084_727_82,
    0.204_432_940_075_298_89,
    0.190_350_578_064_785_42,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_19,
    0.063_092_092_629_978_56,
    0.022_935_322_010_529_224,
];

/// Gauss-Kronrod 31-point rule: nodes on [-1, 1]
/// Source: QUADPACK (Piessens et al., 1983)
const GK31_NODES: [f64; 31] = [
    -0.998_002_298_693_397_1,
    -0.987_992_518_020_485_4,
    -0.967_739_075_679_139_1,
    -0.937_273_392_400_706,
    -0.897_264_532_344_081_9,
    -0.848_206_583_410_427_2,
    -0.790_418_501_442_466,
    -0.724_417_731_360_170_1,
    -0.650_996_741_297_416_9,
    -0.570_972_172_608_539,
    -0.485_081_863_640_239_7,
    -0.394_151_347_077_563_4,
    -0.299_180_007_153_168_8,
    -0.201_194_093_997_434_5,
    -0.101_142_066_918_717_5,
    0.0,
    0.101_142_066_918_717_5,
    0.201_194_093_997_434_5,
    0.299_180_007_153_168_8,
    0.394_151_347_077_563_4,
    0.485_081_863_640_239_7,
    0.570_972_172_608_539,
    0.650_996_741_297_416_9,
    0.724_417_731_360_170_1,
    0.790_418_501_442_466,
    0.848_206_583_410_427_2,
    0.897_264_532_344_081_9,
    0.937_273_392_400_706,
    0.967_739_075_679_139_1,
    0.987_992_518_020_485_4,
    0.998_002_298_693_397_1,
];

/// Gauss-Kronrod 31-point rule: weights on [-1, 1]
const GK31_WEIGHTS: [f64; 31] = [
    0.005_377_479_872_923_349,
    0.015_007_947_329_316_122,
    0.025_460_847_326_715_32,
    0.035_346_360_791_375_85,
    0.044_589_751_324_764_88,
    0.053_481_524_690_928_09,
    0.062_009_567_800_670_64,
    0.069_854_121_318_728_26,
    0.076_849_680_757_720_38,
    0.083_080_502_823_133_17,
    0.088_564_443_056_211_77,
    0.093_126_598_170_825_32,
    0.096_642_726_983_623_68,
    0.099_173_598_721_791_96,
    0.100_769_845_523_875_6,
    0.101_330_007_014_791_55,
    0.100_769_845_523_875_6,
    0.099_173_598_721_791_96,
    0.096_642_726_983_623_68,
    0.093_126_598_170_825_32,
    0.088_564_443_056_211_77,
    0.083_080_502_823_133_17,
    0.076_849_680_757_720_38,
    0.069_854_121_318_728_26,
    0.062_009_567_800_670_64,
    0.053_481_524_690_928_09,
    0.044_589_751_324_764_88,
    0.035_346_360_791_375_85,
    0.025_460_847_326_715_32,
    0.015_007_947_329_316_122,
    0.005_377_479_872_923_349,
];

/// Gauss-Kronrod 41-point rule: nodes on [-1, 1]
/// Source: QUADPACK `dqk41`.
const GK41_NODES: [f64; 41] = [
    -0.9988590315882777,
    -0.9931285991850949,
    -0.9815078774502503,
    -0.9639719272779138,
    -0.9408226338317548,
    -0.912234428251326,
    -0.878276811252282,
    -0.8391169718222188,
    -0.7950414288375512,
    -0.7463319064601508,
    -0.6932376563347514,
    -0.636053680726515,
    -0.5751404468197103,
    -0.5108670019508271,
    -0.4435931752387251,
    -0.37370608871541955,
    -0.301627868114913,
    -0.22778585114164507,
    -0.15260546524092267,
    -0.07652652113349734,
    0.0,
    0.07652652113349734,
    0.15260546524092267,
    0.22778585114164507,
    0.301627868114913,
    0.37370608871541955,
    0.4435931752387251,
    0.5108670019508271,
    0.5751404468197103,
    0.636053680726515,
    0.6932376563347514,
    0.7463319064601508,
    0.7950414288375512,
    0.8391169718222188,
    0.878276811252282,
    0.912234428251326,
    0.9408226338317548,
    0.9639719272779138,
    0.9815078774502503,
    0.9931285991850949,
    0.9988590315882777,
];

/// Gauss-Kronrod 41-point rule: weights on [-1, 1]
const GK41_WEIGHTS: [f64; 41] = [
    0.0030735837185205317,
    0.008600269855642943,
    0.014626169256971253,
    0.020388373461266523,
    0.02588213360495116,
    0.0312873067770328,
    0.036600169758200796,
    0.041668873327973686,
    0.04643482186749767,
    0.05094457392372869,
    0.05519510534828599,
    0.05911140088063957,
    0.06265323755478117,
    0.06583459713361842,
    0.06864867292852161,
    0.07105442355344407,
    0.07303069033278667,
    0.07458287540049918,
    0.07570449768455667,
    0.07637786767208074,
    0.07660071191799965,
    0.07637786767208074,
    0.07570449768455667,
    0.07458287540049918,
    0.07303069033278667,
    0.07105442355344407,
    0.06864867292852161,
    0.06583459713361842,
    0.06265323755478117,
    0.05911140088063957,
    0.05519510534828599,
    0.05094457392372869,
    0.04643482186749767,
    0.041668873327973686,
    0.036600169758200796,
    0.0312873067770328,
    0.02588213360495116,
    0.020388373461266523,
    0.014626169256971253,
    0.008600269855642943,
    0.0030735837185205317,
];

/// Gauss-Kronrod 51-point rule: nodes on [-1, 1]
/// Source: QUADPACK `dqk51`.
const GK51_NODES: [f64; 51] = [
    -0.9992621049926098,
    -0.9955569697904981,
    -0.9880357945340772,
    -0.9766639214595175,
    -0.9616149864258425,
    -0.9429745712289743,
    -0.9207471152817016,
    -0.8949919978782753,
    -0.8658470652932756,
    -0.833442628760834,
    -0.7978737979985001,
    -0.7592592630373576,
    -0.7177664068130843,
    -0.6735663684734684,
    -0.6268100990103174,
    -0.577_662_930_241_223,
    -0.5263252843347191,
    -0.473002731445715,
    -0.4178853821930377,
    -0.36117230580938786,
    -0.30308953893110785,
    -0.24386688372098844,
    -0.1837189394210489,
    -0.1228646926107104,
    -0.06154448300568508,
    0.0,
    0.06154448300568508,
    0.1228646926107104,
    0.1837189394210489,
    0.24386688372098844,
    0.30308953893110785,
    0.36117230580938786,
    0.4178853821930377,
    0.473002731445715,
    0.5263252843347191,
    0.577_662_930_241_223,
    0.6268100990103174,
    0.6735663684734684,
    0.7177664068130843,
    0.7592592630373576,
    0.7978737979985001,
    0.833442628760834,
    0.8658470652932756,
    0.8949919978782753,
    0.9207471152817016,
    0.9429745712289743,
    0.9616149864258425,
    0.9766639214595175,
    0.9880357945340772,
    0.9955569697904981,
    0.9992621049926098,
];

/// Gauss-Kronrod 51-point rule: weights on [-1, 1]
const GK51_WEIGHTS: [f64; 51] = [
    0.001987383892330316,
    0.005561932135356714,
    0.009473973386174152,
    0.013236229195571676,
    0.0168478177091283,
    0.020435371145882834,
    0.024009945606953216,
    0.027_475_317_587_851_74,
    0.03079230016738749,
    0.03400213027432933,
    0.03711627148341554,
    0.04008382550403238,
    0.04287284502017005,
    0.04550291304992179,
    0.04798253713883671,
    0.05027767908071567,
    0.05236288580640747,
    0.05425112988854549,
    0.055950811220412316,
    0.057437116361567835,
    0.058689680022394206,
    0.05972034032417406,
    0.06053945537604586,
    0.061128509717053046,
    0.061471189871425316,
    0.061580818067832936,
    0.061471189871425316,
    0.061128509717053046,
    0.06053945537604586,
    0.05972034032417406,
    0.058689680022394206,
    0.057437116361567835,
    0.055950811220412316,
    0.05425112988854549,
    0.05236288580640747,
    0.05027767908071567,
    0.04798253713883671,
    0.04550291304992179,
    0.04287284502017005,
    0.04008382550403238,
    0.03711627148341554,
    0.03400213027432933,
    0.03079230016738749,
    0.027_475_317_587_851_74,
    0.024009945606953216,
    0.020435371145882834,
    0.0168478177091283,
    0.013236229195571676,
    0.009473973386174152,
    0.005561932135356714,
    0.001987383892330316,
];

/// Gauss-Kronrod 61-point rule: nodes on [-1, 1]
/// Source: QUADPACK `dqk61`.
const GK61_NODES: [f64; 61] = [
    -0.9994844100504906,
    -0.9968934840746495,
    -0.9916309968704046,
    -0.9836681232797472,
    -0.9731163225011262,
    -0.9600218649683075,
    -0.94437444474856,
    -0.9262000474292743,
    -0.9055733076999079,
    -0.8825605357920527,
    -0.8572052335460611,
    -0.8295657623827684,
    -0.799727835821839,
    -0.7677774321048262,
    -0.7337900624532267,
    -0.6978504947933159,
    -0.6600610641266269,
    -0.6205261829892429,
    -0.5793452358263617,
    -0.5366241481420199,
    -0.49248046786177857,
    -0.44703376953808915,
    -0.4004012548303944,
    -0.3527047255308781,
    -0.30407320227362505,
    -0.25463692616788985,
    -0.2045251166823099,
    -0.15386991360858354,
    -0.10280693796673702,
    -0.0514718425553177,
    0.0,
    0.0514718425553177,
    0.10280693796673702,
    0.15386991360858354,
    0.2045251166823099,
    0.25463692616788985,
    0.30407320227362505,
    0.3527047255308781,
    0.4004012548303944,
    0.44703376953808915,
    0.49248046786177857,
    0.5366241481420199,
    0.5793452358263617,
    0.6205261829892429,
    0.6600610641266269,
    0.6978504947933159,
    0.7337900624532267,
    0.7677774321048262,
    0.799727835821839,
    0.8295657623827684,
    0.8572052335460611,
    0.8825605357920527,
    0.9055733076999079,
    0.9262000474292743,
    0.94437444474856,
    0.9600218649683075,
    0.9731163225011262,
    0.9836681232797472,
    0.9916309968704046,
    0.9968934840746495,
    0.9994844100504906,
];

/// Gauss-Kronrod 61-point rule: weights on [-1, 1]
const GK61_WEIGHTS: [f64; 61] = [
    0.0013890136986770077,
    0.003890461127099884,
    0.006630703915931293,
    0.009273279659517764,
    0.011823015253496341,
    0.014369729507045804,
    0.016920889189053273,
    0.01941414119394238,
    0.021828035821609194,
    0.0241911620780806,
    0.0265099548823331,
    0.028_754_048_765_041_29,
    0.03090725756238776,
    0.03298144705748372,
    0.034979338028060024,
    0.03688236465182123,
    0.038678945624727595,
    0.04037453895153596,
    0.04196981021516424,
    0.04345253970135607,
    0.04481480013316266,
    0.04605923827100699,
    0.04718554656929915,
    0.04818586175708713,
    0.04905543455502978,
    0.04979568342707421,
    0.05040592140278235,
    0.05088179589874961,
    0.051221547849258774,
    0.05142612853745902,
    0.05149472942945157,
    0.05142612853745902,
    0.051221547849258774,
    0.05088179589874961,
    0.05040592140278235,
    0.04979568342707421,
    0.04905543455502978,
    0.04818586175708713,
    0.04718554656929915,
    0.04605923827100699,
    0.04481480013316266,
    0.04345253970135607,
    0.04196981021516424,
    0.04037453895153596,
    0.038678945624727595,
    0.03688236465182123,
    0.034979338028060024,
    0.03298144705748372,
    0.03090725756238776,
    0.028_754_048_765_041_29,
    0.0265099548823331,
    0.0241911620780806,
    0.021828035821609194,
    0.01941414119394238,
    0.016920889189053273,
    0.014369729507045804,
    0.011823015253496341,
    0.009273279659517764,
    0.006630703915931293,
    0.003890461127099884,
    0.0013890136986770077,
];

const SUPPORTED_GK_ORDERS: &[usize] = &[15, 31, 41, 51, 61];

fn supported_gk_orders_message() -> String {
    SUPPORTED_GK_ORDERS
        .iter()
        .map(|order| order.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Get Gauss-Kronrod nodes and weights for order `gk_order`.
///
/// Supported orders: 15, 31, 41, 51, and 61.
fn gk_nodes_weights(gk_order: usize) -> Result<(&'static [f64], &'static [f64])> {
    match gk_order {
        15 => Ok((&GK15_NODES, &GK15_WEIGHTS)),
        31 => Ok((&GK31_NODES, &GK31_WEIGHTS)),
        41 => Ok((&GK41_NODES, &GK41_WEIGHTS)),
        51 => Ok((&GK51_NODES, &GK51_WEIGHTS)),
        61 => Ok((&GK61_NODES, &GK61_WEIGHTS)),
        _ => Err(TCIError::InvalidOperation {
            message: format!(
                "GK order {} not supported. Supported orders: {}.",
                gk_order,
                supported_gk_orders_message()
            ),
        }),
    }
}

/// Integrate a function over a hypercube `[a, b]` using TCI and
/// Gauss-Kronrod quadrature.
///
/// The integrand is sampled at Gauss-Kronrod nodes in each dimension,
/// and a tensor-train approximation is built via [`crossinterpolate2`].
/// The integral is then computed as a weighted sum of the tensor train.
///
/// # Arguments
///
/// * `f` -- function to integrate, takes a coordinate slice `&[f64]`
/// * `a` -- lower bounds for each dimension
/// * `b` -- upper bounds for each dimension
/// * `gk_order` -- Gauss-Kronrod quadrature order (15, 31, 41, 51, or 61)
/// * `tci_options` -- options for the TCI approximation
///
/// # Returns
///
/// The approximate integral value.
///
/// # Errors
///
/// Returns an error if `a` and `b` have different lengths, or if
/// `gk_order` is not one of the embedded fixed rules.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::integration::integrate;
/// use tensor4all_tensorci::TCI2Options;
///
/// // Integrate (x^2 + y^2) over [0,1]^2 = 2/3
/// let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
/// let a = vec![0.0, 0.0];
/// let b = vec![1.0, 1.0];
/// let options = TCI2Options { tolerance: 1e-10, ..TCI2Options::default() };
///
/// let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
/// assert!((result - 2.0 / 3.0).abs() < 1e-8);
/// ```
pub fn integrate<T, F>(
    f: &F,
    a: &[f64],
    b: &[f64],
    gk_order: usize,
    tci_options: TCI2Options,
) -> Result<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&[f64]) -> T,
{
    if a.len() != b.len() {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "Lower bounds ({}) and upper bounds ({}) must have same length",
                a.len(),
                b.len()
            ),
        });
    }

    let ndims = a.len();
    let (nodes_ref, weights_ref) = gk_nodes_weights(gk_order)?;
    let n_nodes = nodes_ref.len();

    // Transform nodes from [-1, 1] to [a_i, b_i] and weights
    let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(ndims);
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let mut dim_nodes = Vec::with_capacity(n_nodes);
        let mut dim_weights = Vec::with_capacity(n_nodes);
        for k in 0..n_nodes {
            // x = (b - a) * (node + 1) / 2 + a
            dim_nodes.push((b[d] - a[d]) * (nodes_ref[k] + 1.0) / 2.0 + a[d]);
            // w = (b - a) * weight / 2
            dim_weights.push((b[d] - a[d]) * weights_ref[k] / 2.0);
        }
        nodes.push(dim_nodes);
        weights.push(dim_weights);
    }

    let normalization = (gk_order as f64).powi(ndims as i32);

    // Create wrapper function: F(indices) = w_prod * f(x) * normalization
    let wrapper = |idx: &MultiIndex| -> T {
        let x: Vec<f64> = (0..ndims).map(|d| nodes[d][idx[d]]).collect();
        let w: f64 = (0..ndims).map(|d| weights[d][idx[d]]).product();
        let fval = f(&x);
        // fval * (w * normalization)
        let scale = <T as Scalar>::from_f64(w * normalization);
        fval * scale
    };

    let local_dims = vec![n_nodes; ndims];

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<T, _, fn(&[MultiIndex]) -> Vec<T>>(
        wrapper,
        None,
        local_dims,
        vec![],
        tci_options,
    )?;

    let tt = tci.to_tensor_train()?;
    let sum = tt.sum();

    // Return sum / normalization
    let inv_norm = <T as Scalar>::from_f64(1.0 / normalization);
    Ok(sum * inv_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        // Integral of 1.0 over [0, 1]^2 = 1.0
        let f = |_x: &[f64]| 1.0f64;
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!((result - 1.0).abs() < 1e-8, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_integrate_polynomial() {
        // Integral of x*y over [0, 1]^2 = 1/4
        let f = |x: &[f64]| x[0] * x[1];
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - 0.25).abs() < 1e-8,
            "Expected 0.25, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_5d_polynomial() {
        // Integral of product of polynomials over [0, 1]^5
        // p(x) = 1 + x + x^2, integral = 1 + 1/2 + 1/3 = 11/6
        // 5D integral = (11/6)^5
        let polynomial = |x: f64| 1.0 + x + x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 5;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        let expected = (11.0 / 6.0_f64).powi(5);
        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_integrate_with_gk31() {
        // Same polynomial test with GK31 — should be more accurate
        let polynomial = |x: f64| 1.0 + x + x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 3;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let options = TCI2Options {
            tolerance: 1e-12,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 31, options).unwrap();
        let expected = (11.0 / 6.0_f64).powi(3);
        assert!(
            (result - expected).abs() < 1e-10,
            "GK31: Expected {}, got {}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );
    }

    #[test]
    fn test_gk_order_unsupported() {
        let f = |_x: &[f64]| 1.0f64;
        let a = vec![0.0];
        let b = vec![1.0];
        let options = TCI2Options::default();
        let result: std::result::Result<f64, _> = integrate(&f, &a, &b, 7, options);
        assert!(result.is_err());
    }

    #[test]
    fn test_gk_supported_orders_are_reported() {
        assert!(gk_nodes_weights(15).is_ok());
        assert!(gk_nodes_weights(31).is_ok());
        assert!(gk_nodes_weights(41).is_ok());
        assert!(gk_nodes_weights(51).is_ok());
        assert!(gk_nodes_weights(61).is_ok());
    }

    #[test]
    fn test_gk_order_error_lists_supported_orders() {
        let err = gk_nodes_weights(21).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("21"));
        assert!(message.contains("15, 31, 41, 51, 61"));
    }

    #[test]
    fn test_gk_weights_sum_to_two() {
        for order in [15, 31, 41, 51, 61] {
            let (_nodes, weights) = gk_nodes_weights(order).unwrap();
            let total: f64 = weights.iter().sum();
            assert!(
                (total - 2.0).abs() < 1e-14,
                "GK{order} weights sum to {total}"
            );
        }
    }

    #[test]
    fn test_gk_nodes_are_symmetric_and_sorted() {
        for order in [15, 31, 41, 51, 61] {
            let (nodes, weights) = gk_nodes_weights(order).unwrap();
            assert_eq!(nodes.len(), order);
            assert_eq!(weights.len(), order);
            for pair in nodes.windows(2) {
                assert!(pair[0] < pair[1], "GK{order} nodes are not sorted");
            }
            for i in 0..nodes.len() {
                let j = nodes.len() - 1 - i;
                assert!((nodes[i] + nodes[j]).abs() < 1e-14);
                assert!((weights[i] - weights[j]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_integrate_polynomial_with_gk61() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let options = TCI2Options {
            tolerance: 1e-10,
            max_iter: 20,
            seed: Some(42),
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &[0.0, 0.0], &[1.0, 1.0], 61, options).unwrap();

        assert!((result - 2.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_one_dimensional_reports_tci_requirement() {
        let f = |x: &[f64]| x[0] * x[0];
        let err = integrate::<f64, _>(&f, &[0.0], &[1.0], 15, TCI2Options::default()).unwrap_err();

        assert!(err
            .to_string()
            .contains("local_dims should have at least 2 elements"));
    }

    /// Port of Julia test_integration.jl: "Integrate real polynomials" with arbitrary bounds
    #[test]
    fn test_integrate_arbitrary_bounds() {
        // p(x) = 1 + 0.5*x + 0.3*x^2
        // integral of p from a to b = [x + 0.25*x^2 + 0.1*x^3]_a^b
        let polynomial = |x: f64| 1.0 + 0.5 * x + 0.3 * x * x;
        let poly_integral = |x: f64| x + 0.25 * x * x + 0.1 * x * x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 3;
        let a = vec![0.2, 0.5, 0.1];
        let b = vec![0.8, 0.9, 0.7];
        let expected: f64 = (0..n)
            .map(|i| poly_integral(b[i]) - poly_integral(a[i]))
            .product();

        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };
        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - expected).abs() < 1e-8,
            "Arbitrary bounds: expected {expected}, got {result}"
        );
    }

    /// Port of Julia test_integration.jl: "Integrate complex polynomials"
    #[test]
    fn test_integrate_complex_polynomial() {
        use num_complex::Complex64;

        let polynomial = |x: f64| Complex64::new(1.0 + x, 0.5 * x * x);
        let poly_integral = |x: f64| Complex64::new(x + 0.5 * x * x, 0.5 * x * x * x / 3.0);
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<Complex64>();

        let n = 3;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let expected: Complex64 = (0..n)
            .map(|_| poly_integral(1.0) - poly_integral(0.0))
            .product();

        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };
        let result: Complex64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - expected).norm() < 1e-8,
            "Complex: expected {expected}, got {result}"
        );
    }
}
