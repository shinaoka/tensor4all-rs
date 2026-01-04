"""Index class for tensor4all."""

from __future__ import annotations

from ._capi import check_status, get_lib, T4AError
from ._ffi import ffi


class Index:
    """A tensor index with dimension, unique ID, and tags.

    An Index represents one dimension of a tensor and has:
    - A dimension (size)
    - A unique 128-bit ID
    - Optional tags (string labels like "Site", "n=1")

    Examples
    --------
    >>> i = Index(5)  # Create index with dimension 5
    >>> i.dim
    5
    >>> j = Index(3, tags="Site,n=1")  # Create with tags
    >>> j.has_tag("Site")
    True
    """

    __slots__ = ("_ptr",)

    def __init__(
        self,
        dim: int,
        *,
        tags: str = "",
        id: tuple[int, int] | None = None,
    ):
        """Create a new Index.

        Parameters
        ----------
        dim : int
            The dimension (size) of the index. Must be > 0.
        tags : str, optional
            Comma-separated tags, e.g., "Site,n=1". Default is no tags.
        id : tuple[int, int], optional
            128-bit ID as (high_64_bits, low_64_bits). If None, a random ID is generated.

        Raises
        ------
        ValueError
            If dim <= 0
        T4AError
            If creation fails (e.g., too many tags)
        """
        if dim <= 0:
            raise ValueError(f"Index dimension must be positive, got {dim}")

        lib = get_lib()

        if id is not None:
            id_hi, id_lo = id
            tags_bytes = tags.encode("utf-8") if tags else ffi.NULL
            ptr = lib.t4a_index_new_with_id(dim, id_hi, id_lo, tags_bytes)
        elif tags:
            tags_bytes = tags.encode("utf-8")
            ptr = lib.t4a_index_new_with_tags(dim, tags_bytes)
        else:
            ptr = lib.t4a_index_new(dim)

        if ptr == ffi.NULL:
            raise T4AError("Failed to create Index")

        self._ptr = ptr

    @classmethod
    def _from_ptr(cls, ptr) -> Index:
        """Create an Index from an existing C pointer (internal use)."""
        if ptr == ffi.NULL:
            raise T4AError("Cannot create Index from NULL pointer")
        instance = object.__new__(cls)
        instance._ptr = ptr
        return instance

    def __del__(self):
        """Release the underlying C object."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            lib = get_lib()
            lib.t4a_index_release(self._ptr)
            self._ptr = ffi.NULL

    def __repr__(self) -> str:
        tags_str = self.tags
        if tags_str:
            return f"Index(dim={self.dim}, tags='{tags_str}')"
        return f"Index(dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        """Two indices are equal if they have the same ID."""
        if not isinstance(other, Index):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)

    @property
    def dim(self) -> int:
        """Get the dimension (size) of the index."""
        lib = get_lib()
        out = ffi.new("size_t*")
        status = lib.t4a_index_dim(self._ptr, out)
        check_status(status, "Failed to get index dimension")
        return out[0]

    @property
    def id(self) -> tuple[int, int]:
        """Get the 128-bit ID as (high_64_bits, low_64_bits)."""
        lib = get_lib()
        out_hi = ffi.new("uint64_t*")
        out_lo = ffi.new("uint64_t*")
        status = lib.t4a_index_id_u128(self._ptr, out_hi, out_lo)
        check_status(status, "Failed to get index ID")
        return (out_hi[0], out_lo[0])

    @property
    def tags(self) -> str:
        """Get the comma-separated tags string."""
        lib = get_lib()

        # First, query required length
        out_len = ffi.new("size_t*")
        status = lib.t4a_index_get_tags(self._ptr, ffi.NULL, 0, out_len)
        check_status(status, "Failed to get tags length")

        if out_len[0] <= 1:
            return ""

        # Allocate buffer and get tags
        buf = ffi.new(f"uint8_t[{out_len[0]}]")
        status = lib.t4a_index_get_tags(self._ptr, buf, out_len[0], out_len)
        check_status(status, "Failed to get tags")

        return ffi.string(buf).decode("utf-8")

    def has_tag(self, tag: str) -> bool:
        """Check if the index has a specific tag.

        Parameters
        ----------
        tag : str
            The tag to check for.

        Returns
        -------
        bool
            True if the tag exists, False otherwise.
        """
        lib = get_lib()
        tag_bytes = tag.encode("utf-8")
        result = lib.t4a_index_has_tag(self._ptr, tag_bytes)
        if result < 0:
            raise T4AError("Failed to check tag")
        return result == 1

    def add_tag(self, tag: str) -> None:
        """Add a tag to the index.

        Parameters
        ----------
        tag : str
            The tag to add.

        Raises
        ------
        TagOverflowError
            If too many tags
        TagTooLongError
            If tag string is too long
        """
        lib = get_lib()
        tag_bytes = tag.encode("utf-8")
        status = lib.t4a_index_add_tag(self._ptr, tag_bytes)
        check_status(status, f"Failed to add tag '{tag}'")

    def set_tags(self, tags: str) -> None:
        """Set all tags from a comma-separated string (replaces existing).

        Parameters
        ----------
        tags : str
            Comma-separated tags, e.g., "Site,n=1"
        """
        lib = get_lib()
        tags_bytes = tags.encode("utf-8")
        status = lib.t4a_index_set_tags_csv(self._ptr, tags_bytes)
        check_status(status, "Failed to set tags")

    def clone(self) -> Index:
        """Create a copy of this index with the same ID and tags."""
        lib = get_lib()
        ptr = lib.t4a_index_clone(self._ptr)
        if ptr == ffi.NULL:
            raise T4AError("Failed to clone Index")
        return Index._from_ptr(ptr)
