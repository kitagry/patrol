"""Type definitions for Pavise."""

from __future__ import annotations

from typing import Generic, TypeVar

__all__ = ["NotRequiredColumn"]

T = TypeVar("T")


class NotRequiredColumn(Generic[T]):
    """
    Marker type for optional columns in Protocol schemas.

    Use this to indicate that a column may not exist in the DataFrame.
    If the column is present, it will be validated according to its type.

    Key differences:
    - Optional[T]: Column must exist, but can contain None values
    - NotRequiredColumn[T]: Column can be missing, but if present must have type T
    - NotRequiredColumn[Optional[T]]: Column can be missing, and if present can contain None

    Example:
        >>> from typing import Protocol, Optional
        >>> from pavise.pandas import DataFrame
        >>> from pavise.types import NotRequiredColumn
        >>>
        >>> class UserSchema(Protocol):
        ...     user_id: int
        ...     name: str
        ...     age: NotRequiredColumn[int]  # Column can be missing
        ...     email: NotRequiredColumn[Optional[str]]  # Missing or nullable
        >>>
        >>> # Both are valid
        >>> df1 = DataFrame[UserSchema](pd.DataFrame({"user_id": [1], "name": ["Alice"]}))
        >>> df2 = DataFrame[UserSchema](
        ...     pd.DataFrame({"user_id": [1], "name": ["Alice"], "age": [25]})
        ... )
    """

    def __class_getitem__(cls, item: type) -> type:
        """Support NotRequiredColumn[T] syntax."""

        # This is a marker type, so we just return a new class with the inner type stored
        class _NotRequiredColumn(NotRequiredColumn):
            _inner_type = item

        return _NotRequiredColumn
