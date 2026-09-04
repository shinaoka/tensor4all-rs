use super::*;
use crate::{DynIndex, IdxTensor};

// ============================================================================
// `DefaultOnlyTensor`: a minimal `TensorLike` implementor that never
// overrides `factorize_probe_columns_incremental`, `src_error_estimate`,
// `from_dense_any`, `from_dense`, `stack_along_new_index`,
// `concatenate_along_new_index`, or `select_indices`.
//
// `IdxTensor` (the only production `TensorLike`) overrides all seven, so those
// default bodies in `tensor_like.rs` are otherwise unreachable. This type
// exists solely to drive them; its arithmetic is a plain column-major dense
// `Vec<f64>` (real-only, no symmetry/complex support) and its error type
// reuses `TensorVectorSpaceError`, which already satisfies `TensorIndex`'s
// `Error` bound.
// ============================================================================

/// Column-major strides/coordinate helpers shared by every trait impl below.
fn dims_of(indices: &[DynIndex]) -> Vec<usize> {
    indices.iter().map(IndexLike::dim).collect()
}

fn linear_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut linear = 0;
    let mut stride = 1;
    for (&coord, &dim) in coords.iter().zip(dims) {
        linear += coord * stride;
        stride *= dim;
    }
    linear
}

fn coords_from_linear(mut linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(dims.len());
    for &dim in dims {
        coords.push(linear % dim);
        linear /= dim;
    }
    coords
}

/// Read `original`'s per-axis coordinates back out of a (shared, free)
/// coordinate split, used by pairwise contraction below.
fn full_coords(
    original: &[DynIndex],
    shared: &[DynIndex],
    free: &[DynIndex],
    shared_coords: &[usize],
    free_coords: &[usize],
) -> Vec<usize> {
    original
        .iter()
        .map(|index| match shared.iter().position(|s| s == index) {
            Some(position) => shared_coords[position],
            None => {
                let position = free
                    .iter()
                    .position(|f| f == index)
                    .expect("axis is shared or free");
                free_coords[position]
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
struct DefaultOnlyTensor {
    indices: Vec<DynIndex>,
    data: Vec<f64>,
}

impl DefaultOnlyTensor {
    /// Contract `self` against `other`, summing over every index `other`
    /// carries. `select_indices`'s default body (the only caller reached by
    /// these tests) always builds `other` as a one-hot tensor whose indices
    /// are a subset of `self`'s with none of its own left over, so that
    /// narrower case is all this needs to handle.
    fn contract_pairwise(&self, other: &Self) -> std::result::Result<Self, TensorVectorSpaceError> {
        let free: Vec<DynIndex> = self
            .indices
            .iter()
            .filter(|index| !other.indices.contains(index))
            .cloned()
            .collect();
        let self_dims = dims_of(&self.indices);
        let other_dims = dims_of(&other.indices);
        let free_dims = dims_of(&free);
        let free_count: usize = if free_dims.is_empty() {
            1
        } else {
            free_dims.iter().product()
        };
        let other_count: usize = if other_dims.is_empty() {
            1
        } else {
            other_dims.iter().product()
        };

        let mut data = Vec::with_capacity(free_count);
        for free_lin in 0..free_count {
            let free_coords = coords_from_linear(free_lin, &free_dims);
            let mut sum = 0.0;
            for other_lin in 0..other_count {
                let other_coords = coords_from_linear(other_lin, &other_dims);
                let self_coords = full_coords(
                    &self.indices,
                    &other.indices,
                    &free,
                    &other_coords,
                    &free_coords,
                );
                let self_lin = linear_index(&self_coords, &self_dims);
                sum += self.data[self_lin] * other.data[other_lin];
            }
            data.push(sum);
        }
        Ok(Self {
            indices: free,
            data,
        })
    }
}

impl TensorIndex for DefaultOnlyTensor {
    type Index = DynIndex;
    type Error = TensorVectorSpaceError;

    fn external_indices(&self) -> Vec<DynIndex> {
        self.indices.clone()
    }

    fn replaceind(
        &self,
        old_index: &DynIndex,
        new_index: &DynIndex,
    ) -> std::result::Result<Self, Self::Error> {
        let position = self
            .indices
            .iter()
            .position(|index| index == old_index)
            .ok_or_else(|| anyhow::anyhow!("replaceind: index not present"))?;
        let mut indices = self.indices.clone();
        indices[position] = new_index.clone();
        Ok(Self {
            indices,
            data: self.data.clone(),
        })
    }

    fn replace_indices(
        &self,
        old_indices: &[DynIndex],
        new_indices: &[DynIndex],
    ) -> std::result::Result<Self, Self::Error> {
        if old_indices.len() != new_indices.len() {
            return Err(anyhow::anyhow!("replace_indices: length mismatch").into());
        }
        let mut result = self.clone();
        for (old, new) in old_indices.iter().zip(new_indices) {
            result = result.replaceind(old, new)?;
        }
        Ok(result)
    }
}

impl TensorVectorSpace for DefaultOnlyTensor {
    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.data.iter().map(|value| value * value).sum())
    }

    fn axpby(
        &self,
        a: AnyScalar,
        other: &Self,
        b: AnyScalar,
    ) -> std::result::Result<Self, Self::Error> {
        if self.indices != other.indices {
            return Err(anyhow::anyhow!("axpby: index-space mismatch").into());
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| a.real() * x + b.real() * y)
            .collect();
        Ok(Self {
            indices: self.indices.clone(),
            data,
        })
    }

    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error> {
        let data = self
            .data
            .iter()
            .map(|value| value * scalar.real())
            .collect();
        Ok(Self {
            indices: self.indices.clone(),
            data,
        })
    }

    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error> {
        if self.indices != other.indices {
            return Err(anyhow::anyhow!("inner_product: index-space mismatch").into());
        }
        let value: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x * y)
            .sum();
        Ok(AnyScalar::new_real(value))
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self
            .data
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs())))
    }
}

impl TensorContractionLike for DefaultOnlyTensor {
    fn conj(&self) -> Self {
        self.clone()
    }

    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> std::result::Result<DirectSumResult<Self>, Self::Error> {
        let mut new_indices = Vec::with_capacity(pairs.len());
        for (left, right) in pairs {
            new_indices.push(
                <DynIndex as IndexLike>::new_link(left.dim() + right.dim())
                    .map_err(TensorVectorSpaceError::from)?,
            );
        }
        let tensor = self.outer_product(other)?;
        Ok(DirectSumResult {
            tensor,
            new_indices,
        })
    }

    fn outer_product(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        let mut indices = self.indices.clone();
        indices.extend(other.indices.iter().cloned());
        let mut data = Vec::with_capacity(self.data.len() * other.data.len());
        for &b in &other.data {
            for &a in &self.data {
                data.push(a * b);
            }
        }
        Ok(Self { indices, data })
    }

    fn permuteinds(&self, new_order: &[DynIndex]) -> std::result::Result<Self, Self::Error> {
        if new_order.len() != self.indices.len()
            || !new_order.iter().all(|index| self.indices.contains(index))
        {
            return Err(anyhow::anyhow!("permuteinds: index set mismatch").into());
        }
        let old_dims = dims_of(&self.indices);
        let new_dims = dims_of(new_order);
        let axis_map: Vec<usize> = new_order
            .iter()
            .map(|index| {
                self.indices
                    .iter()
                    .position(|i| i == index)
                    .expect("checked above")
            })
            .collect();
        let mut data = vec![0.0; self.data.len()];
        for linear in 0..self.data.len() {
            let old_coords = coords_from_linear(linear, &old_dims);
            let new_coords: Vec<usize> = axis_map.iter().map(|&axis| old_coords[axis]).collect();
            data[linear_index(&new_coords, &new_dims)] = self.data[linear];
        }
        Ok(Self {
            indices: new_order.to_vec(),
            data,
        })
    }

    fn fuse_indices(
        &self,
        old_indices: &[DynIndex],
        new_index: DynIndex,
        _order: LinearizationOrder,
    ) -> std::result::Result<Self, Self::Error> {
        // Not reached by any of the six default-method bodies under test;
        // kept minimal (no data reindexing) since nothing exercises it.
        if old_indices.is_empty() {
            return Err(anyhow::anyhow!("fuse_indices: no indices to fuse").into());
        }
        let expected: usize = old_indices.iter().map(|index| index.dim()).product();
        if new_index.dim() != expected {
            return Err(anyhow::anyhow!("fuse_indices: dimension mismatch").into());
        }
        let position = self
            .indices
            .iter()
            .position(|index| index == &old_indices[0])
            .ok_or_else(|| anyhow::anyhow!("fuse_indices: index not present"))?;
        let mut indices: Vec<DynIndex> = self
            .indices
            .iter()
            .filter(|index| !old_indices.contains(index))
            .cloned()
            .collect();
        indices.insert(position.min(indices.len()), new_index);
        Ok(Self {
            indices,
            data: self.data.clone(),
        })
    }

    fn contract(tensors: &[&Self]) -> std::result::Result<Self, Self::Error> {
        let mut iter = tensors.iter();
        let first: &&Self = iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("contract: requires at least one tensor"))?;
        let mut acc: Self = (*first).clone();
        for tensor in iter {
            acc = acc.contract_pairwise(tensor)?;
        }
        Ok(acc)
    }
}

impl TensorFactorizationLike for DefaultOnlyTensor {
    fn factorize(
        &self,
        left_inds: &[DynIndex],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        self.factorize_full_rank(left_inds, options.alg, options.canonical)
    }

    /// Degenerate rank-1 factorization: neither the coverage target
    /// (`factorize_probe_columns_incremental`'s default body) nor its test
    /// inspects factorization *correctness*, only that the default body
    /// exercises `stack_along_new_index` and then a `factorize_full_rank`
    /// call without panicking.
    fn factorize_full_rank(
        &self,
        left_inds: &[DynIndex],
        _alg: FactorizeAlg,
        _canonical: Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        let bond =
            <DynIndex as IndexLike>::new_link(1).map_err(FactorizeError::ComputationError)?;
        let right_rest: Vec<DynIndex> = self
            .indices
            .iter()
            .filter(|index| !left_inds.contains(index))
            .cloned()
            .collect();
        let mut left_indices = left_inds.to_vec();
        left_indices.push(bond.clone());
        let mut right_indices = vec![bond.clone()];
        right_indices.extend(right_rest);
        let left = Self {
            indices: left_indices,
            data: vec![1.0],
        };
        let right = Self {
            indices: right_indices,
            data: vec![1.0],
        };
        Ok(FactorizeResult::new(left, right, bond, None, 1))
    }
}

impl TensorConstructionLike for DefaultOnlyTensor {
    fn diagonal(
        input_index: &DynIndex,
        output_index: &DynIndex,
    ) -> std::result::Result<Self, Self::Error> {
        if input_index.dim() != output_index.dim() {
            return Err(anyhow::anyhow!("diagonal: dimension mismatch").into());
        }
        let dim = input_index.dim();
        let dims = [dim, dim];
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[linear_index(&[i, i], &dims)] = 1.0;
        }
        Ok(Self {
            indices: vec![input_index.clone(), output_index.clone()],
            data,
        })
    }

    fn scalar_one() -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            indices: Vec::new(),
            data: vec![1.0],
        })
    }

    fn ones(indices: &[DynIndex]) -> std::result::Result<Self, Self::Error> {
        let len: usize = {
            let dims = dims_of(indices);
            if dims.is_empty() {
                1
            } else {
                dims.iter().product()
            }
        };
        Ok(Self {
            indices: indices.to_vec(),
            data: vec![1.0; len],
        })
    }

    fn onehot(index_vals: &[(DynIndex, usize)]) -> std::result::Result<Self, Self::Error> {
        let indices: Vec<DynIndex> = index_vals.iter().map(|(index, _)| index.clone()).collect();
        let dims = dims_of(&indices);
        let coords: Vec<usize> = index_vals.iter().map(|(_, position)| *position).collect();
        for (&position, &dim) in coords.iter().zip(dims.iter()) {
            if position >= dim {
                return Err(anyhow::anyhow!("onehot: position out of range").into());
            }
        }
        let len: usize = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        let mut data = vec![0.0; len];
        data[linear_index(&coords, &dims)] = 1.0;
        Ok(Self { indices, data })
    }
}

#[test]
fn tensor_like_default_stack_along_new_index_batches_tensors() {
    let i = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(2);
    let first = DefaultOnlyTensor {
        indices: vec![i.clone()],
        data: vec![1.0, 2.0],
    };
    let second = DefaultOnlyTensor {
        indices: vec![i.clone()],
        data: vec![3.0, 4.0],
    };
    let stacked =
        DefaultOnlyTensor::stack_along_new_index(&[&first, &second], batch.clone(), -1).unwrap();
    assert_eq!(stacked.indices, vec![i, batch]);
    assert_eq!(stacked.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_like_default_concatenate_along_new_index_joins_batch_blocks() {
    let row = DynIndex::new_dyn(2);
    let first_batch = DynIndex::new_link(1).unwrap();
    let second_batch = DynIndex::new_link(2).unwrap();
    let combined = DynIndex::new_link(3).unwrap();
    let first = DefaultOnlyTensor {
        indices: vec![row.clone(), first_batch.clone()],
        data: vec![1.0, 2.0],
    };
    let second = DefaultOnlyTensor {
        indices: vec![row.clone(), second_batch.clone()],
        data: vec![3.0, 4.0, 5.0, 6.0],
    };
    let result = DefaultOnlyTensor::concatenate_along_new_index(
        &[&first, &second],
        &[first_batch, second_batch],
        combined.clone(),
    )
    .unwrap();
    assert_eq!(result.indices, vec![row, combined]);
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor_like_default_select_indices_fixes_a_coordinate() {
    let row = DynIndex::new_dyn(2);
    let column = DynIndex::new_dyn(2);
    let tensor = DefaultOnlyTensor {
        indices: vec![row.clone(), column.clone()],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let sliced = tensor.select_indices(&[column], &[1]).unwrap();
    assert_eq!(sliced.indices, vec![row]);
    assert_eq!(sliced.data, vec![3.0, 4.0]);
}

#[test]
fn tensor_like_default_from_dense_any_reconstructs_column_major_payload() {
    let index = DynIndex::new_dyn(2);
    let tensor = DefaultOnlyTensor::from_dense_any(
        vec![index.clone()],
        vec![AnyScalar::new_real(2.0), AnyScalar::new_real(3.0)],
    )
    .unwrap();
    assert_eq!(tensor.indices, vec![index]);
    assert_eq!(tensor.data, vec![2.0, 3.0]);
}

#[test]
fn tensor_like_typed_dense_uses_default_and_native_paths() {
    let index = DynIndex::new_dyn(2);
    let fallback = DefaultOnlyTensor::from_dense(vec![index.clone()], vec![2.0_f64, 3.0]).unwrap();
    assert_eq!(fallback.data, vec![2.0, 3.0]);

    let native =
        <IdxTensor as TensorConstructionLike>::from_dense(vec![index], vec![2.0_f32, 3.0]).unwrap();
    assert!(native.is_f32());
    assert_eq!(native.to_vec::<f32>().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn tensor_like_default_factorize_probe_columns_incremental_stacks_and_factorizes() {
    let row = DynIndex::new_dyn(2);
    let first = DefaultOnlyTensor {
        indices: vec![row.clone()],
        data: vec![1.0, 0.0],
    };
    let second = DefaultOnlyTensor {
        indices: vec![row.clone()],
        data: vec![0.0, 1.0],
    };
    let result =
        <DefaultOnlyTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            None,
            &[&first, &second],
            &[&first, &second],
            std::slice::from_ref(&row),
        )
        .unwrap();
    // `factorize_full_rank` above is a degenerate stub reporting rank 1
    // regardless of the sketch's numeric content; what this exercises is
    // that the default body successfully stacks the probe columns (via
    // `stack_along_new_index`) and hands the result to `factorize_full_rank`
    // without panicking.
    assert_eq!(result.rank, 1);
    assert_eq!(result.left.indices, vec![row, result.bond_index.clone()]);
}

#[test]
fn tensor_like_default_src_error_estimate_reports_unsupported_storage() {
    let tensor = DefaultOnlyTensor {
        indices: Vec::new(),
        data: vec![1.0],
    };
    let error = tensor.src_error_estimate().unwrap_err();
    assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
}

// Compile-time check that TensorLike requires Sized (no dyn TensorLike)
fn _assert_sized<T: TensorLike>() {
    // This confirms T: Sized is required
}

#[test]
fn factorize_options_builders_and_validation_accept_supported_fields() {
    let svd = FactorizeOptions::svd()
        .with_canonical(Canonical::Right)
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
        .with_max_bond_dim(4);
    assert_eq!(svd.alg, FactorizeAlg::SVD);
    assert_eq!(svd.canonical, Canonical::Right);
    assert_eq!(svd.max_bond_dim, Some(4));
    assert_eq!(svd.svd_policy, Some(SvdTruncationPolicy::new(1.0e-8)));
    svd.validate().unwrap();

    let qr = FactorizeOptions::qr()
        .with_qr_rtol(0.0)
        .with_max_bond_dim(3);
    assert_eq!(qr.alg, FactorizeAlg::QR);
    assert_eq!(qr.qr_rtol, Some(0.0));
    assert_eq!(qr.max_bond_dim, Some(3));
    qr.validate().unwrap();

    let lu = FactorizeOptions::lu();
    assert_eq!(lu.alg, FactorizeAlg::LU);
    lu.validate().unwrap();

    let ci = FactorizeOptions::ci();
    assert_eq!(ci.alg, FactorizeAlg::CI);
    ci.validate().unwrap();
}

#[test]
fn factorize_options_validation_rejects_algorithm_specific_mismatches() {
    assert!(matches!(
        FactorizeOptions::svd().with_qr_rtol(1.0e-8).validate(),
        Err(FactorizeError::InvalidOptions(
            "SVD factorization does not accept qr_rtol"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::qr()
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "QR factorization does not accept svd_policy"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::lu()
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "LU/CI factorization does not accept svd_policy"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::ci().with_qr_rtol(1.0e-8).validate(),
        Err(FactorizeError::InvalidOptions(
            "LU/CI factorization does not accept qr_rtol"
        ))
    ));
}

#[test]
fn factorize_options_validation_rejects_zero_cap_and_invalid_svd_thresholds() {
    // The shared `validate_svd_truncation_options` seam is delegated from
    // `FactorizeOptions::validate`; both typed error kinds it maps must be
    // exercised here (they are validation facts, not algorithm mismatches).
    assert!(matches!(
        FactorizeOptions::svd().with_max_bond_dim(0).validate(),
        Err(FactorizeError::InvalidOptions(
            "max_bond_dim must be at least 1"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::svd()
            .with_svd_policy(SvdTruncationPolicy::new(f64::NAN))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "SVD truncation threshold must be finite and non-negative"
        ))
    ));
}

#[test]
fn linearization_order_labels_are_stable() {
    assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
    assert_eq!(LinearizationOrder::RowMajor.as_str(), "row-major");
}

#[test]
fn tensor_like_default_neg_and_delta_helpers_work() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let l = DynIndex::new_dyn(3);

    let tensor = IdxTensor::from_dense(vec![i.clone()], vec![2.0, -3.0]).unwrap();
    let negated = tensor.neg().unwrap();
    assert_eq!(negated.to_vec::<f64>().unwrap(), vec![-2.0, 3.0]);

    let delta = IdxTensor::delta(&[i.clone(), j.clone()], &[k, l]).unwrap();
    assert_eq!(delta.dims(), vec![2, 2, 3, 3]);
    assert!((delta.sum().unwrap().real() - 6.0).abs() < 1.0e-12);

    let err = IdxTensor::delta(&[i], &[]).unwrap_err();
    assert!(err.to_string().contains("Number of input indices"));
}

#[test]
fn tensor_construction_supports_column_major_dense_payloads() {
    fn construct<T: TensorConstructionLike + TensorVectorSpace>(
        indices: Vec<T::Index>,
        data: Vec<AnyScalar>,
    ) -> std::result::Result<T, T::Error> {
        T::from_dense_any(indices, data)
    }

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = construct::<IdxTensor>(
        vec![i, j],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(AnyScalar::new_real)
            .collect(),
    )
    .unwrap();

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tensor_construction_supports_stacking_a_batch_axis() {
    fn stack<T: TensorConstructionLike + TensorVectorSpace>(
        tensors: &[&T],
        new_index: T::Index,
    ) -> std::result::Result<T, T::Error> {
        T::stack_along_new_index(tensors, new_index, -1)
    }

    let i = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(2);
    let first = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let second = IdxTensor::from_dense(vec![i], vec![3.0, 4.0]).unwrap();
    let stacked = stack(&[&first, &second], batch).unwrap();

    assert_eq!(stacked.external_indices().len(), 2);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_construction_concatenates_existing_batch_blocks() {
    fn concatenate<T: TensorConstructionLike + TensorVectorSpace>(
        tensors: &[&T],
        source_indices: &[T::Index],
        new_index: T::Index,
    ) -> std::result::Result<T, T::Error> {
        T::concatenate_along_new_index(tensors, source_indices, new_index)
    }

    let row = DynIndex::new_dyn(2);
    let first_batch = DynIndex::new_link(1).unwrap();
    let second_batch = DynIndex::new_link(2).unwrap();
    let combined = DynIndex::new_link(3).unwrap();
    let first =
        IdxTensor::from_dense(vec![row.clone(), first_batch.clone()], vec![1.0_f64, 2.0]).unwrap();
    let second = IdxTensor::from_dense(
        vec![row.clone(), second_batch.clone()],
        vec![3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let result = concatenate(
        &[&first, &second],
        &[first_batch, second_batch],
        combined.clone(),
    )
    .unwrap();

    assert_eq!(result.external_indices(), vec![row, combined]);
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tensor_factorization_supports_incremental_probe_prefixes() {
    let row = DynIndex::new_dyn(5);
    let first = IdxTensor::from_dense(vec![row.clone()], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let second = IdxTensor::from_dense(vec![row.clone()], vec![-2.0, 1.0, 0.0, 3.0, -1.0]).unwrap();
    let third = IdxTensor::from_dense(vec![row.clone()], vec![0.5, -1.0, 2.0, 1.0, 4.0]).unwrap();
    let fourth =
        IdxTensor::from_dense(vec![row.clone()], vec![3.0, -2.0, 1.5, 0.25, -3.0]).unwrap();

    let first_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            None,
            &[&first, &second],
            &[&first, &second],
            std::slice::from_ref(&row),
        )
        .unwrap();
    let second_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&first_factorization),
            &[&first, &second, &third],
            &[&third],
            std::slice::from_ref(&row),
        )
        .unwrap();
    let third_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&second_factorization),
            &[&first, &second, &third, &fourth],
            &[&fourth],
            std::slice::from_ref(&row),
        )
        .unwrap();

    assert_eq!(first_factorization.rank, 2);
    assert_eq!(second_factorization.rank, 3);
    assert_eq!(third_factorization.rank, 4);
    let reconstructed = third_factorization
        .left
        .contract_pair(&third_factorization.right)
        .unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.0, 3.0, -1.0, 0.5, -1.0, 2.0, 1.0, 4.0, 3.0, -2.0,
        1.5, 0.25, -3.0,
    ];
    let actual = reconstructed.to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12),
        "reconstructed columns differ: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn tensor_factorization_preserves_multi_axis_probe_row_order() {
    let first_row = DynIndex::new_dyn(2);
    let second_row = DynIndex::new_dyn(3);
    let rows = vec![first_row.clone(), second_row.clone()];
    let first = IdxTensor::from_dense(rows.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let second = IdxTensor::from_dense(rows.clone(), vec![-1.0, 0.5, 2.0, -2.0, 3.0, 4.0]).unwrap();
    let third =
        IdxTensor::from_dense(rows.clone(), vec![0.25, -3.0, 1.5, 2.5, -0.75, 5.0]).unwrap();
    let fourth =
        IdxTensor::from_dense(rows.clone(), vec![4.0, -1.0, 0.5, 3.5, 2.25, -2.5]).unwrap();

    let first_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            None,
            &[&first, &second],
            &[&first, &second],
            &rows,
        )
        .unwrap();
    let second_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&first_factorization),
            &[&first, &second, &third],
            &[&third],
            &rows,
        )
        .unwrap();
    let final_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&second_factorization),
            &[&first, &second, &third, &fourth],
            &[&fourth],
            &rows,
        )
        .unwrap();

    let reconstructed = final_factorization
        .left
        .contract_pair(&final_factorization.right)
        .unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.0, -2.0, 3.0, 4.0, 0.25, -3.0, 1.5, 2.5, -0.75,
        5.0, 4.0, -1.0, 0.5, 3.5, 2.25, -2.5,
    ];
    let actual = reconstructed.to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12),
        "reconstructed multi-axis sketch differs: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn tensor_like_default_context_seam_reports_unsupported() {
    use std::sync::Arc;

    use tenferro_cpu::CpuBackend;
    use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext};

    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )));
    let tensor = DefaultOnlyTensor {
        indices: Vec::new(),
        data: vec![1.0],
    };
    assert!(tensor.validate_context(&context).is_err());
    assert!(DefaultOnlyTensor::ones_in(&context, &[]).is_err());
    assert!(DefaultOnlyTensor::from_dense_in(&context, Vec::new(), Vec::<f64>::new()).is_err());
    assert!(tensor.scale_in(2.0, &context).is_err());
    assert!(tensor.norm_in(&context).is_err());
    assert!(tensor
        .factorize_in(&[], &FactorizeOptions::qr(), &context)
        .is_err());
    assert!(tensor
        .factorize_full_rank_in(&[], FactorizeAlg::QR, Canonical::Left, &context)
        .is_err());
    let error = tensor.src_error_estimate_in(&context).unwrap_err();
    assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
    let error = DefaultOnlyTensor::factorize_probe_batch_incremental_in(
        None,
        &tensor,
        &DynIndex::new_dyn(1),
        &[],
        &context,
    )
    .unwrap_err();
    assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
}
