"""Schema column specification for Polars backend."""

from dataclasses import dataclass
from typing import Literal, get_args, get_origin, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from .validation import TYPE_TO_DTYPE


@dataclass
class ColumnSpec:
    """Schema column specification."""

    dtype: pl.DataType
    is_not_required: bool
    is_optional: bool


def get_dtype_for_type(base_type: type) -> pl.DataType:
    """
    Get polars dtype for a given Python type.

    Args:
        base_type: Python type (int, str, float, bool, datetime, date, timedelta)

    Returns:
        Polars DataType
    """

    if isinstance(base_type, type) and issubclass(base_type, pl.DataType):
        return base_type()

    return TYPE_TO_DTYPE.get(base_type, pl.Utf8())


def get_column_specs(schema: type) -> dict[str, ColumnSpec]:
    """
    Extract column specifications from schema.

    Args:
        schema: Schema Protocol class

    Returns:
        Dict mapping column_name -> ColumnSpec
    """
    from pavise._polars.validation import _extract_type_and_validators

    type_hints = get_type_hints(schema, include_extras=True)
    specs = {}

    for col_name, col_type in type_hints.items():
        base_type, _validators, is_optional, is_not_required = _extract_type_and_validators(
            col_type
        )

        # Resolve Literal types
        if get_origin(base_type) is Literal:
            literal_values = get_args(base_type)
            if literal_values:
                base_type = type(literal_values[0])

        dtype = get_dtype_for_type(base_type)
        specs[col_name] = ColumnSpec(
            dtype=dtype,
            is_not_required=is_not_required,
            is_optional=is_optional,
        )

    return specs
