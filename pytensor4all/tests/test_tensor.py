"""Tests for Tensor class."""

import numpy as np
import pytest

from pytensor4all import Index, Tensor, StorageKind


class TestTensorBasic:
    """Basic Tensor tests."""

    def test_create_1d(self):
        """Test creating a 1D tensor."""
        i = Index(5)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = Tensor([i], data)

        assert t.rank == 1
        assert t.dims == (5,)
        assert t.shape == (5,)
        assert t.storage_kind == StorageKind.DenseF64

    def test_create_2d(self):
        """Test creating a 2D tensor."""
        i = Index(2)
        j = Index(3)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        t = Tensor([i, j], data)

        assert t.rank == 2
        assert t.dims == (2, 3)
        assert t.storage_kind == StorageKind.DenseF64

    def test_create_complex(self):
        """Test creating a complex tensor."""
        i = Index(2)
        j = Index(2)
        data = np.array([[1 + 0.5j, 2 + 1.5j], [3 + 2.5j, 4 + 3.5j]])
        t = Tensor([i, j], data)

        assert t.rank == 2
        assert t.dims == (2, 2)
        assert t.storage_kind == StorageKind.DenseC64
        assert t.dtype == np.complex128

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        i = Index(3)
        data = np.array([1.0, 2.0])  # Wrong size
        with pytest.raises(ValueError):
            Tensor([i], data)

    def test_rank_mismatch(self):
        """Test that rank mismatch raises error."""
        i = Index(2)
        data = np.array([[1, 2], [3, 4]], dtype=np.float64)  # 2D
        with pytest.raises(ValueError):
            Tensor([i], data)  # Only 1 index


class TestTensorData:
    """Data access tests."""

    def test_to_numpy_f64(self):
        """Test converting f64 tensor to numpy."""
        i = Index(2)
        j = Index(3)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        t = Tensor([i, j], data)

        result = t.to_numpy()
        np.testing.assert_array_equal(result, data)

    def test_to_numpy_c64(self):
        """Test converting c64 tensor to numpy."""
        i = Index(2)
        j = Index(2)
        data = np.array([[1 + 0.5j, 2 + 1.5j], [3 + 2.5j, 4 + 3.5j]])
        t = Tensor([i, j], data)

        result = t.to_numpy()
        np.testing.assert_array_almost_equal(result, data)

    def test_data_order_c(self):
        """Test that data is in C (row-major) order."""
        i = Index(2)
        j = Index(3)
        # Row-major: [1,2,3] then [4,5,6]
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order="C")
        t = Tensor([i, j], data)

        result = t.to_numpy()
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, data)

    def test_non_contiguous_input(self):
        """Test that non-contiguous input is handled."""
        i = Index(2)
        j = Index(3)
        # Create non-contiguous array via transpose
        data = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64).T
        assert not data.flags["C_CONTIGUOUS"]

        t = Tensor([i, j], data)
        result = t.to_numpy()

        # Should still get correct values
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


class TestTensorIndices:
    """Index access tests."""

    def test_get_indices(self):
        """Test getting tensor indices."""
        i = Index(2, tags="i")
        j = Index(3, tags="j")
        data = np.zeros((2, 3))
        t = Tensor([i, j], data)

        indices = t.indices
        assert len(indices) == 2
        assert indices[0].dim == 2
        assert indices[1].dim == 3

    def test_indices_are_clones(self):
        """Test that returned indices are independent clones."""
        i = Index(2, tags="Original")
        data = np.zeros((2,))
        t = Tensor([i], data)

        indices = t.indices
        indices[0].add_tag("Modified")

        # Original should not be affected
        assert not i.has_tag("Modified")

        # Get indices again
        indices2 = t.indices
        # These should also not be affected (new clones each time)
        # Actually the returned indices from first call were modified,
        # but subsequent calls return fresh clones


class TestTensorClone:
    """Clone tests."""

    def test_clone(self):
        """Test cloning a tensor."""
        i = Index(2)
        j = Index(3)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        t = Tensor([i, j], data)

        cloned = t.clone()
        assert cloned.rank == t.rank
        assert cloned.dims == t.dims
        np.testing.assert_array_equal(cloned.to_numpy(), t.to_numpy())


class TestTensorRepr:
    """String representation tests."""

    def test_repr(self):
        """Test tensor repr."""
        i = Index(2)
        j = Index(3)
        data = np.zeros((2, 3))
        t = Tensor([i, j], data)

        r = repr(t)
        assert "Tensor" in r
        assert "rank=2" in r
        assert "dims=(2, 3)" in r
        assert "DenseF64" in r
