"""pytensor4all - Python bindings for tensor4all Rust library.

This package provides Python bindings to the tensor4all library,
which implements tensor operations for quantum physics applications.

Examples
--------
>>> from pytensor4all import Index, Tensor
>>> import numpy as np
>>>
>>> # Create indices
>>> i = Index(2, tags="Site")
>>> j = Index(3, tags="Link")
>>>
>>> # Create a tensor
>>> data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
>>> t = Tensor([i, j], data)
>>>
>>> # Access tensor properties
>>> print(t.rank)  # 2
>>> print(t.dims)  # (2, 3)
>>> print(t.to_numpy())
"""

from .index import Index
from .tensor import Tensor, StorageKind
from ._capi import (
    T4AError,
    NullPointerError,
    InvalidArgumentError,
    TagOverflowError,
    TagTooLongError,
    BufferTooSmallError,
    InternalError,
)

__version__ = "0.1.0"

__all__ = [
    # Classes
    "Index",
    "Tensor",
    "StorageKind",
    # Exceptions
    "T4AError",
    "NullPointerError",
    "InvalidArgumentError",
    "TagOverflowError",
    "TagTooLongError",
    "BufferTooSmallError",
    "InternalError",
]
