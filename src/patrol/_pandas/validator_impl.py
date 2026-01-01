"""Pandas-specific validator implementations."""

from typing import Any

import pandas as pd

from patrol.validators import In, MaxLen, MinLen, Range, Regex, Unique


def apply_validator(series: pd.Series, validator: Any, col_name: str) -> None:
    """
    Apply a validator to a pandas Series.

    Args:
        series: pandas Series to validate
        validator: Validator instance (e.g., Range, Unique, In, Regex, MinLen, MaxLen)
        col_name: column name for error messages

    Raises:
        ValueError: if validation fails
    """
    if isinstance(validator, Range):
        _validate_range(series, validator, col_name)
    elif isinstance(validator, Unique):
        _validate_unique(series, col_name)
    elif isinstance(validator, In):
        _validate_in(series, validator, col_name)
    elif isinstance(validator, Regex):
        _validate_regex(series, validator, col_name)
    elif isinstance(validator, MinLen):
        _validate_minlen(series, validator, col_name)
    elif isinstance(validator, MaxLen):
        _validate_maxlen(series, validator, col_name)
    else:
        raise ValueError(f"Unknown validator type: {type(validator)}")


def _validate_range(series: pd.Series, validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    if (series < validator.min).any() or (series > validator.max).any():
        raise ValueError(
            f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        )


def _validate_unique(series: pd.Series, col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    if series.duplicated().any():
        raise ValueError(f"Column '{col_name}': contains duplicate values")


def _validate_in(series: pd.Series, validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    if not series.isin(validator.allowed_values).all():
        raise ValueError(f"Column '{col_name}': contains values not in allowed values")


def _validate_regex(series: pd.Series, validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    if not series.str.match(validator.pattern).all():
        raise ValueError(f"Column '{col_name}': contains values that don't match the pattern")


def _validate_minlen(series: pd.Series, validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    if (series.str.len() < validator.min_length).any():
        raise ValueError(f"Column '{col_name}': contains strings shorter than minimum length")


def _validate_maxlen(series: pd.Series, validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    if (series.str.len() > validator.max_length).any():
        raise ValueError(f"Column '{col_name}': contains strings longer than maximum length")
