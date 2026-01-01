"""Polars-specific validator implementations."""

from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")

from patrol.validators import In, MaxLen, MinLen, Range, Regex, Unique


def apply_validator(series: "pl.Series", validator: Any, col_name: str) -> None:
    """
    Apply a validator to a polars Series.

    Args:
        series: polars Series to validate
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


def _validate_range(series: "pl.Series", validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    if (series < validator.min).any() or (series > validator.max).any():
        raise ValueError(
            f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        )


def _validate_unique(series: "pl.Series", col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    if series.is_duplicated().any():
        raise ValueError(f"Column '{col_name}': contains duplicate values")


def _validate_in(series: "pl.Series", validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    if not series.is_in(validator.allowed_values).all():
        raise ValueError(f"Column '{col_name}': contains values not in allowed values")


def _validate_regex(series: "pl.Series", validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    if not series.str.contains(f"^{validator.pattern}$").all():
        raise ValueError(f"Column '{col_name}': contains values that don't match the pattern")


def _validate_minlen(series: "pl.Series", validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    if (series.str.len_chars() < validator.min_length).any():
        raise ValueError(f"Column '{col_name}': contains strings shorter than minimum length")


def _validate_maxlen(series: "pl.Series", validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    if (series.str.len_chars() > validator.max_length).any():
        raise ValueError(f"Column '{col_name}': contains strings longer than maximum length")
