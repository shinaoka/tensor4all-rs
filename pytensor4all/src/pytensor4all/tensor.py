"""Tensor class for tensor4all."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from ._capi import check_status, get_lib, T4AError
from ._ffi import ffi
from .index import Index

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StorageKind(IntEnum):
    """Storage type for tensor data."""
    DenseF64 = 0
    DenseC64 = 1
    DiagF64 = 2
    DiagC64 = 3


class Tensor:
    """A dense tensor with labeled indices.

    A Tensor stores multidimensional data with associated Index objects
    that label each dimension. The data is stored in row-major (C) order.

    Examples
    --------
    >>> i = Index(2)
    >>> j = Index(3)
    >>> data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    >>> t = Tensor([i, j], data)
    >>> t.rank
    2
    >>> t.dims
    (2, 3)
    """

    __slots__ = ("_ptr",)

    def __init__(self, indices: list[Index], data: NDArray):
        """Create a new Tensor.

        Parameters
        ----------
        indices : list[Index]
            List of indices, one for each dimension.
        data : np.ndarray
            NumPy array with shape matching the index dimensions.
            Supports float64 and complex128 dtypes.

        Raises
        ------
        ValueError
            If shapes don't match or dtype is unsupported.
        T4AError
            If creation fails.
        """
        if len(indices) != data.ndim:
            raise ValueError(
                f"Number of indices ({len(indices)}) must match data dimensions ({data.ndim})"
            )

        # Check dimensions match
        for i, idx in enumerate(indices):
            if idx.dim != data.shape[i]:
                raise ValueError(
                    f"Index {i} has dim {idx.dim} but data has shape {data.shape[i]} at that axis"
                )

        lib = get_lib()
        rank = len(indices)

        # Prepare index pointers
        index_ptrs = ffi.new(f"t4a_index*[{rank}]")
        for i, idx in enumerate(indices):
            index_ptrs[i] = idx._ptr

        # Prepare dimensions
        dims = ffi.new(f"size_t[{rank}]")
        for i, idx in enumerate(indices):
            dims[i] = idx.dim

        # Ensure contiguous C-order array
        if np.iscomplexobj(data):
            data = np.ascontiguousarray(data, dtype=np.complex128)
            data_re = np.ascontiguousarray(data.real)
            data_im = np.ascontiguousarray(data.imag)

            ptr = lib.t4a_tensor_new_dense_c64(
                rank,
                ffi.cast("const t4a_index**", index_ptrs),
                dims,
                ffi.cast("const double*", ffi.from_buffer(data_re)),
                ffi.cast("const double*", ffi.from_buffer(data_im)),
                data.size,
            )
        else:
            data = np.ascontiguousarray(data, dtype=np.float64)

            ptr = lib.t4a_tensor_new_dense_f64(
                rank,
                ffi.cast("const t4a_index**", index_ptrs),
                dims,
                ffi.cast("const double*", ffi.from_buffer(data)),
                data.size,
            )

        if ptr == ffi.NULL:
            raise T4AError("Failed to create Tensor")

        self._ptr = ptr

    @classmethod
    def _from_ptr(cls, ptr) -> Tensor:
        """Create a Tensor from an existing C pointer (internal use)."""
        if ptr == ffi.NULL:
            raise T4AError("Cannot create Tensor from NULL pointer")
        instance = object.__new__(cls)
        instance._ptr = ptr
        return instance

    def __del__(self):
        """Release the underlying C object."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            lib = get_lib()
            lib.t4a_tensor_release(self._ptr)
            self._ptr = ffi.NULL

    def __repr__(self) -> str:
        return f"Tensor(rank={self.rank}, dims={self.dims}, storage={self.storage_kind.name})"

    @property
    def rank(self) -> int:
        """Get the number of indices (dimensions)."""
        lib = get_lib()
        out = ffi.new("size_t*")
        status = lib.t4a_tensor_get_rank(self._ptr, out)
        check_status(status, "Failed to get tensor rank")
        return out[0]

    @property
    def dims(self) -> tuple[int, ...]:
        """Get the dimensions as a tuple."""
        lib = get_lib()
        rank = self.rank

        if rank == 0:
            return ()

        out_dims = ffi.new(f"size_t[{rank}]")
        status = lib.t4a_tensor_get_dims(self._ptr, out_dims, rank)
        check_status(status, "Failed to get tensor dimensions")

        return tuple(out_dims[i] for i in range(rank))

    @property
    def shape(self) -> tuple[int, ...]:
        """Alias for dims (NumPy-style)."""
        return self.dims

    @property
    def indices(self) -> list[Index]:
        """Get the list of indices (cloned)."""
        lib = get_lib()
        rank = self.rank

        if rank == 0:
            return []

        out_indices = ffi.new(f"t4a_index*[{rank}]")
        status = lib.t4a_tensor_get_indices(self._ptr, out_indices, rank)
        check_status(status, "Failed to get tensor indices")

        return [Index._from_ptr(out_indices[i]) for i in range(rank)]

    @property
    def storage_kind(self) -> StorageKind:
        """Get the storage type."""
        lib = get_lib()
        out = ffi.new("t4a_storage_kind*")
        status = lib.t4a_tensor_get_storage_kind(self._ptr, out)
        check_status(status, "Failed to get storage kind")
        return StorageKind(out[0])

    @property
    def dtype(self) -> np.dtype:
        """Get the NumPy dtype corresponding to the storage kind."""
        kind = self.storage_kind
        if kind in (StorageKind.DenseF64, StorageKind.DiagF64):
            return np.dtype(np.float64)
        else:
            return np.dtype(np.complex128)

    def to_numpy(self) -> NDArray:
        """Convert the tensor data to a NumPy array.

        Returns
        -------
        np.ndarray
            NumPy array with the tensor data in C order.
        """
        lib = get_lib()
        kind = self.storage_kind
        dims = self.dims

        # Query data length
        out_len = ffi.new("size_t*")

        if kind == StorageKind.DenseF64:
            status = lib.t4a_tensor_get_data_f64(self._ptr, ffi.NULL, 0, out_len)
            check_status(status, "Failed to get data length")

            buf = np.empty(out_len[0], dtype=np.float64)
            status = lib.t4a_tensor_get_data_f64(
                self._ptr,
                ffi.cast("double*", ffi.from_buffer(buf)),
                out_len[0],
                out_len,
            )
            check_status(status, "Failed to get f64 data")
            return buf.reshape(dims)

        elif kind == StorageKind.DenseC64:
            status = lib.t4a_tensor_get_data_c64(
                self._ptr, ffi.NULL, ffi.NULL, 0, out_len
            )
            check_status(status, "Failed to get data length")

            buf_re = np.empty(out_len[0], dtype=np.float64)
            buf_im = np.empty(out_len[0], dtype=np.float64)
            status = lib.t4a_tensor_get_data_c64(
                self._ptr,
                ffi.cast("double*", ffi.from_buffer(buf_re)),
                ffi.cast("double*", ffi.from_buffer(buf_im)),
                out_len[0],
                out_len,
            )
            check_status(status, "Failed to get c64 data")
            return (buf_re + 1j * buf_im).reshape(dims)

        else:
            raise NotImplementedError(f"Diagonal tensors not yet supported: {kind}")

    def clone(self) -> Tensor:
        """Create a copy of this tensor."""
        lib = get_lib()
        ptr = lib.t4a_tensor_clone(self._ptr)
        if ptr == ffi.NULL:
            raise T4AError("Failed to clone Tensor")
        return Tensor._from_ptr(ptr)
