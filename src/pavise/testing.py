"""Test data generation utilities for Pavise."""

__all__ = ["ANY"]


class _AnySentinel:
    """Sentinel value for test data generation in DataFrame.for_test()."""

    def __repr__(self) -> str:
        return "ANY"

    def __str__(self) -> str:
        return "ANY"

    def __eq__(self, other: object) -> bool:
        """ANY is equal to any value (including itself)."""
        return True

    def __ne__(self, other: object) -> bool:
        """ANY is never not equal to any value."""
        return False

    def __hash__(self) -> int:
        """ANY has a consistent hash value."""
        return hash("ANY")


ANY = _AnySentinel()
