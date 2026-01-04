# pytensor4all

Python bindings for the [tensor4all](https://github.com/tensor4all/tensor4all-rs) Rust library.

## Installation

### Development Setup

1. Build the Rust shared library:

```bash
cd pytensor4all
python scripts/build_capi.py
```

2. Install in development mode with uv:

```bash
uv sync
uv pip install -e .
```

Or with pip:

```bash
pip install -e ".[dev]"
```

3. Run tests:

```bash
uv run pytest
# or
pytest
```

## Usage

```python
from pytensor4all import Index, Tensor
import numpy as np

# Create indices
i = Index(2, tags="Site")
j = Index(3, tags="Link")

# Create a tensor
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
t = Tensor([i, j], data)

# Access properties
print(t.rank)   # 2
print(t.dims)   # (2, 3)
print(t.shape)  # (2, 3)

# Get data back as numpy array
arr = t.to_numpy()

# Complex tensors are also supported
cdata = np.array([[1+0.5j, 2+1j], [3+1.5j, 4+2j]])
ct = Tensor([Index(2), Index(2)], cdata)
```

## API Reference

### Index

```python
Index(dim: int, *, tags: str = "", id: tuple[int, int] | None = None)
```

Properties:
- `dim`: Dimension (size) of the index
- `id`: 128-bit unique ID as `(high_64, low_64)` tuple
- `tags`: Comma-separated tags string

Methods:
- `has_tag(tag: str) -> bool`: Check if index has a tag
- `add_tag(tag: str)`: Add a tag
- `set_tags(tags: str)`: Replace all tags
- `clone() -> Index`: Create a copy

### Tensor

```python
Tensor(indices: list[Index], data: np.ndarray)
```

Properties:
- `rank`: Number of dimensions
- `dims` / `shape`: Tuple of dimensions
- `indices`: List of Index objects (cloned)
- `storage_kind`: `StorageKind.DenseF64` or `StorageKind.DenseC64`
- `dtype`: NumPy dtype

Methods:
- `to_numpy() -> np.ndarray`: Convert to NumPy array
- `clone() -> Tensor`: Create a copy

## Environment Variables

- `T4A_CAPI_LIB`: Path to the `libtensor4all_capi` shared library

## License

MIT
