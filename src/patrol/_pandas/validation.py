"""Common validation functions for DataFrame schema checking."""

from typing import Annotated, get_args, get_origin, get_type_hints

import pandas as pd

TYPE_CHECKERS = {
    int: pd.api.types.is_integer_dtype,
    float: pd.api.types.is_float_dtype,
    str: pd.api.types.is_string_dtype,
    bool: pd.api.types.is_bool_dtype,
}


def validate_dataframe(df: pd.DataFrame, schema: type) -> None:
    """
    Validate that a DataFrame conforms to a Protocol schema.

    Args:
        df: DataFrame to validate
        schema: Protocol type defining the expected schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        _check_column_exists(df, col_name)
        _check_column_type(df, col_name, col_type)


def _extract_type_and_validators(annotation: type) -> tuple[type, list]:
    """
    Extract base type and validators from a type annotation.

    Args:
        annotation: Type annotation (e.g., int or Annotated[int, Range(0, 100)])

    Returns:
        Tuple of (base_type, validators)
        - For Annotated[int, Range(0, 100)]: (int, [Range(0, 100)])
        - For int: (int, [])
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        validators = list(args[1:])
        return base_type, validators
    return annotation, []


def _check_column_exists(df: pd.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValueError(f"Missing column: {col_name}")


def _check_column_type(df: pd.DataFrame, col_name: str, expected_type: type) -> None:
    """Check if a column has the expected type and apply validators."""
    from patrol._pandas.validator_impl import apply_validator

    base_type, validators = _extract_type_and_validators(expected_type)

    if base_type not in TYPE_CHECKERS:
        raise ValueError(f"Unsupported type: {base_type}")

    type_checker = TYPE_CHECKERS[base_type]
    if not type_checker(df[col_name]):
        raise TypeError(
            f"Column '{col_name}' expected {base_type.__name__}, got {df[col_name].dtype}"
        )

    for validator in validators:
        apply_validator(df[col_name], validator, col_name)
