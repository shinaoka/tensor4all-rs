//! Simultaneous n-way Hadamard products.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::{
    tree_elementwise_batched, Result, TreeAciNode, TreeAciOptions, TreeAciResult, TreeAciScalar,
};

/// Approximates the simultaneous pointwise product of every input TreeTN.
///
/// Unlike repeated pairwise multiplication, this performs one ACI run and
/// therefore avoids constructing intermediate products with avoidable ranks.
/// Inputs follow the topology, physical-index, and quantization rules of
/// [`tree_elementwise_batched`].
///
/// # Arguments
///
/// * `inputs` - Nonempty, topology-compatible native TreeTNs.
/// * `options` - Sweep, guard, rank, traversal, and allocation controls.
///
/// # Returns
///
/// The simultaneous product and its convergence diagnostics.
///
/// # Errors
///
/// Returns [`crate::TreeAciError`] for invalid input, resource, or numerical
/// failures.
///
/// # Panics
///
/// This function does not intentionally panic.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treeaci::{hadamard_many, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site.clone()], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let b = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![5.0_f64, 7.0])?],
///     vec![0usize],
/// )?;
/// let result = hadamard_many::<f64, _>(&[a, b], &TreeAciOptions::default())?;
/// assert_eq!(result.tree.to_dense()?.to_vec::<f64>()?, vec![10.0, 21.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hadamard_many<T, V>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    tree_elementwise_batched(
        |batch, output| {
            for (value, point_inputs) in output
                .iter_mut()
                .zip(batch.as_col_major_slice().chunks_exact(batch.n_inputs()))
            {
                *value = point_inputs.iter().copied().fold(
                    <T as tensor4all_core::Scalar>::from_f64(1.0),
                    |product, input| product * input,
                );
            }
            Ok(())
        },
        inputs,
        options,
    )
}
