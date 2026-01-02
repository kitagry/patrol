"""Polars-specific validator implementations."""

from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")

from patrol.exceptions import ValidationError
from patrol.validators import In, MaxLen, MinLen, Range, Regex, Unique

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5


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
        raise ValidationError(f"Unknown validator type: {type(validator)}")


def _validate_range(series: "pl.Series", validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    invalid_mask = (series < validator.min) | (series > validator.max)
    if invalid_mask.any():
        invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
        samples = [
            (row["__row__"], row[series.name])
            for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
        ]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {val}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_unique(series: "pl.Series", col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    duplicated_mask = series.is_duplicated()
    if duplicated_mask.any():
        duplicated_values = series.filter(duplicated_mask).unique().head(MAX_SAMPLE_SIZE).to_list()

        msg = f"Column '{col_name}': contains duplicate values"
        msg += f"\n\nSample duplicate values (showing first {len(duplicated_values)}):"
        for val in duplicated_values:
            indices = series.to_frame().with_row_index("__row__").filter(pl.col(series.name) == val)
            row_nums = indices.head(2).select("__row__").to_series().to_list()
            msg += f"\n  Value {repr(val)} at rows: {row_nums}"

        raise ValidationError(msg, column_name=col_name)


def _validate_in(series: "pl.Series", validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    invalid_mask = ~series.is_in(validator.allowed_values)
    if invalid_mask.any():
        invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
        samples = [
            (row["__row__"], row[series.name])
            for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
        ]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains values not in allowed values"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_regex(series: "pl.Series", validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    invalid_mask = ~series.str.contains(f"^{validator.pattern}$")
    if invalid_mask.any():
        invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
        samples = [
            (row["__row__"], row[series.name])
            for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
        ]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains values that don't match the pattern"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_minlen(series: "pl.Series", validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    invalid_mask = series.str.len_chars() < validator.min_length
    if invalid_mask.any():
        invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
        samples = [
            (row["__row__"], row[series.name])
            for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
        ]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains strings shorter than minimum length"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)} (length: {len(val)})"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_maxlen(series: "pl.Series", validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    invalid_mask = series.str.len_chars() > validator.max_length
    if invalid_mask.any():
        invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
        samples = [
            (row["__row__"], row[series.name])
            for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
        ]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains strings longer than maximum length"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)} (length: {len(val)})"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)
