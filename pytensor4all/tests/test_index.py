"""Tests for Index class."""

import pytest

from pytensor4all import Index, TagOverflowError


class TestIndexBasic:
    """Basic Index tests."""

    def test_create_simple(self):
        """Test creating a simple index."""
        idx = Index(5)
        assert idx.dim == 5

    def test_create_with_tags(self):
        """Test creating an index with tags."""
        idx = Index(3, tags="Site,n=1")
        assert idx.dim == 3
        assert idx.has_tag("Site")
        assert idx.has_tag("n=1")
        assert not idx.has_tag("Missing")

    def test_create_with_id(self):
        """Test creating an index with explicit ID."""
        id_hi = 0x12345678_9ABCDEF0
        id_lo = 0xFEDCBA98_76543210
        idx = Index(7, id=(id_hi, id_lo), tags="Custom")

        assert idx.dim == 7
        assert idx.id == (id_hi, id_lo)
        assert idx.has_tag("Custom")

    def test_invalid_dim(self):
        """Test that dimension must be positive."""
        with pytest.raises(ValueError):
            Index(0)
        with pytest.raises(ValueError):
            Index(-1)


class TestIndexTags:
    """Tag-related tests."""

    def test_get_tags(self):
        """Test getting tags as string."""
        idx = Index(2, tags="Site,Link")
        tags = idx.tags
        assert "Site" in tags
        assert "Link" in tags

    def test_add_tag(self):
        """Test adding a tag."""
        idx = Index(2)
        assert idx.tags == ""

        idx.add_tag("NewTag")
        assert idx.has_tag("NewTag")

    def test_set_tags(self):
        """Test setting tags (replaces existing)."""
        idx = Index(2, tags="Old")
        assert idx.has_tag("Old")

        idx.set_tags("New1,New2")
        assert not idx.has_tag("Old")
        assert idx.has_tag("New1")
        assert idx.has_tag("New2")


class TestIndexId:
    """ID-related tests."""

    def test_id_unique(self):
        """Test that IDs are unique by default."""
        idx1 = Index(3)
        idx2 = Index(3)
        assert idx1.id != idx2.id

    def test_id_preserved_in_clone(self):
        """Test that ID is preserved when cloning."""
        idx = Index(4)
        cloned = idx.clone()
        assert idx.id == cloned.id


class TestIndexEquality:
    """Equality tests."""

    def test_equality_same_id(self):
        """Test that indices with same ID are equal."""
        idx = Index(3, tags="Test")
        cloned = idx.clone()
        assert idx == cloned

    def test_inequality_different_id(self):
        """Test that indices with different IDs are not equal."""
        idx1 = Index(3)
        idx2 = Index(3)
        assert idx1 != idx2

    def test_hash_consistent(self):
        """Test that hash is consistent with equality."""
        idx = Index(3)
        cloned = idx.clone()
        assert hash(idx) == hash(cloned)


class TestIndexClone:
    """Clone tests."""

    def test_clone_independent(self):
        """Test that clone is independent."""
        idx = Index(3, tags="Original")
        cloned = idx.clone()

        # Modify original
        idx.add_tag("Modified")

        # Clone should not be affected
        assert idx.has_tag("Modified")
        assert not cloned.has_tag("Modified")


class TestIndexRepr:
    """String representation tests."""

    def test_repr_simple(self):
        """Test repr without tags."""
        idx = Index(5)
        assert "Index" in repr(idx)
        assert "dim=5" in repr(idx)

    def test_repr_with_tags(self):
        """Test repr with tags."""
        idx = Index(3, tags="Site")
        r = repr(idx)
        assert "Index" in r
        assert "dim=3" in r
        assert "Site" in r
