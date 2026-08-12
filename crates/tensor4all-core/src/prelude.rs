//! Commonly used traits, types, and functions for tensor construction,
//! contraction, and factorization.
//!
//! ```rust
//! use tensor4all_core::prelude::*;
//!
//! let i = DynIndex::new_dyn(3);
//! let j = DynIndex::new_dyn(4);
//! let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
//! let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
//! let result = factorize(&tensor, &[i], &FactorizeOptions::svd()).unwrap();
//! let recovered = result.left.contract_pair(&result.right).unwrap();
//! assert!(tensor.distance(&recovered).unwrap() < 1e-12);
//! ```

pub use crate::{
    contract, contract_pair, direct_sum, factorize, outer_product, qr, svd, tensordot, Canonical,
    CommonScalar, ContractionOptions, DecompositionAlg, DirectSumResult, DynId, DynIndex,
    FactorizeAlg, FactorizeOptions, FactorizeResult, IdxTensor, Index, IndexLike,
    LinearizationOrder, PairwiseContractionOptions, SingularValueMeasure, SvdOptions,
    SvdTruncationPolicy, Tag, TagSet, TagSetLike, TensorConstructionLike, TensorContractionLike,
    TensorElement, TensorFactorizationLike, TensorIndex, TensorLike, TensorVectorSpace,
    ThresholdScale, TruncationRule,
};
