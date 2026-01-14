"""Schema column specification for Pandas backend."""

from dataclasses import dataclass
from typing import Literal, Union, get_args, get_origin, get_type_hints

import pandas as pd

from .validation import INDEX_COLUMN_NAME, TYPE_TO_DTYPE, _extract_type_and_validators


@dataclass
class ColumnSpec:
    """Schema column specification."""

    dtype: Union[str, pd.api.extensions.ExtensionDtype]
    is_not_required: bool
    is_optional: bool


def get_dtype_for_type(base_type: type) -> Union[str, pd.api.extensions.ExtensionDtype]:
    """
    Get pandas dtype for a given Python type.

    Args:
        base_type: Python type (int, str, float, bool, datetime, date, timedelta)

    Returns:
        String representation of pandas dtype
    """

    if isinstance(base_type, type) and issubclass(base_type, pd.api.extensions.ExtensionDtype):
        return base_type()

    return TYPE_TO_DTYPE.get(base_type, "object")


def get_column_specs(schema: type) -> dict[str, ColumnSpec]:
    """
    Extract column specifications from schema.

    Args:
        schema: Schema Protocol class

    Returns:
        Dict mapping column_name -> ColumnSpec (excluding INDEX_COLUMN_NAME)
    """

    type_hints = get_type_hints(schema, include_extras=True)
    specs = {}

    for col_name, col_type in type_hints.items():
        # Skip the special index column
        if col_name == INDEX_COLUMN_NAME:
            continue

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
