//! Canonical-form sweeps for MPOs.
//!
//! Local SVD truncation of a tensor train only minimizes the global error when
//! the part of the train that is not being truncated is orthonormal. A
//! truncating left-to-right sweep makes the sites to the left of the active
//! bond left-orthogonal as it goes, but the sites to the right keep whatever
//! gauge they were built with. Bringing the MPO into right-canonical form
//! first, with rank-preserving QR factorizations, fixes that gauge so every
//! subsequent local truncation is optimal against an orthonormal environment.
//!
//! The sweep here is rank preserving: it never discards a bond dimension for
//! accuracy reasons, it only replaces each bond by the exact rank of the
//! corresponding matricization. That makes it safe to run before any
//! truncation policy is applied.

use crate::einsum_helper::{einsum_tensors, typed_tensor_reshape, EinsumScalar};

use super::error::{MPOError, Result};
use super::factorize::SVDScalar;
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use tenferro_tensor::TypedTensor;
use tensor4all_tensorbackend::qr_backend;

fn canonical_error(context: &str, err: impl std::fmt::Display) -> MPOError {
    MPOError::InvalidOperation {
        message: format!("{context}: {err}"),
    }
}

/// Bring `mpo` into right-canonical form without changing the operator it
/// represents.
///
/// After the sweep every site tensor `B` with index `i >= 1` satisfies
/// `sum_{s1, s2, r} B[l, s1, s2, r] * conj(B[l', s1, s2, r]) = delta_{l, l'}`,
/// and the whole norm sits on site `0`.
///
/// Each bond is replaced by the exact rank of the matricization at that bond,
/// so bond dimensions can shrink but no accuracy is lost.
pub(crate) fn right_canonicalize<T>(mpo: &mut MPO<T>) -> Result<()>
where
    T: SVDScalar + EinsumScalar,
{
    if mpo.len() <= 1 {
        return Ok(());
    }

    for i in (1..mpo.len()).rev() {
        let tensor = mpo.site_tensor(i);
        let left = tensor.left_dim();
        let s1 = tensor.site_dim_1();
        let s2 = tensor.site_dim_2();
        let right = tensor.right_dim();
        let rest = s1 * s2 * right;

        // Column-major data of `(left, s1, s2, right)` groups the trailing
        // three axes contiguously, so this reshape is a pure metadata change.
        let matrix = typed_tensor_reshape(tensor.as_inner(), &[left, rest])
            .map_err(|err| canonical_error("Failed to reshape MPO site tensor", err))?;

        // An LQ factorization `M = L * Q` with row-orthonormal `Q` is obtained
        // from the QR of the plain (unconjugated) transpose:
        // `M^T = Q_t R_t` gives `M = R_t^T Q_t^T`, and
        // `Q_t^T (Q_t^T)^H = conj(Q_t^H Q_t) = I`.
        let transposed = einsum_tensors("lx->xl", &[&matrix])
            .map_err(|err| canonical_error("Failed to transpose MPO site matrix", err))?;
        let (q, r) = qr_backend(&transposed)
            .map_err(|err| canonical_error("QR failed during MPO canonicalization", err))?;

        let rank = q.shape()[1];
        let q_rows: TypedTensor<T> = einsum_tensors("xk->kx", &[&q])
            .map_err(|err| canonical_error("Failed to transpose QR factor Q", err))?;
        let new_site = typed_tensor_reshape(&q_rows, &[rank, s1, s2, right])
            .map_err(|err| canonical_error("Failed to reshape QR factor Q", err))?;
        let l_factor: TypedTensor<T> = einsum_tensors("kl->lk", &[&r])
            .map_err(|err| canonical_error("Failed to transpose QR factor R", err))?;

        // Absorb `L` into the left neighbour so the represented operator is
        // unchanged.
        let prev = mpo.site_tensor(i - 1);
        let new_prev = einsum_tensors("ausl,lk->ausk", &[prev.as_inner(), &l_factor])
            .map_err(|err| canonical_error("Failed to absorb QR factor into MPO site", err))?;

        *mpo.site_tensor_mut(i) = Tensor4::from_tenferro(new_site)
            .map_err(|err| canonical_error("Canonicalized MPO site has invalid rank", err))?;
        *mpo.site_tensor_mut(i - 1) = Tensor4::from_tenferro(new_prev)
            .map_err(|err| canonical_error("Canonicalized MPO site has invalid rank", err))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests;
