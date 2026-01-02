"""Pandas-specific validator implementations."""

from typing import Any

import pandas as pd

from patrol.exceptions import ValidationError
from patrol.validators import In, MaxLen, MinLen, Range, Regex, Unique

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5


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
        raise ValidationError(f"Unknown validator type: {type(validator)}")


def _validate_range(series: pd.Series, validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    invalid_mask = (series < validator.min) | (series > validator.max)
    if invalid_mask.any():
        invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
        samples = [(idx, series.loc[idx]) for idx in invalid_indices]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {val}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_unique(series: pd.Series, col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    duplicated_mask = series.duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_values = series[duplicated_mask].unique()[:MAX_SAMPLE_SIZE]

        msg = f"Column '{col_name}': contains duplicate values"
        msg += f"\n\nSample duplicate values (showing first {len(duplicated_values)}):"
        for val in duplicated_values:
            indices = series.index[series == val][:2]
            msg += f"\n  Value {repr(val)} at rows: {list(indices)}"

        raise ValidationError(msg, column_name=col_name)


def _validate_in(series: pd.Series, validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    invalid_mask = ~series.isin(validator.allowed_values)
    if invalid_mask.any():
        invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
        samples = [(idx, series.loc[idx]) for idx in invalid_indices]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains values not in allowed values"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_regex(series: pd.Series, validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    invalid_mask = ~series.str.match(validator.pattern)
    if invalid_mask.any():
        invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
        samples = [(idx, series.loc[idx]) for idx in invalid_indices]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains values that don't match the pattern"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)}"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_minlen(series: pd.Series, validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    invalid_mask = series.str.len() < validator.min_length
    if invalid_mask.any():
        invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
        samples = [(idx, series.loc[idx]) for idx in invalid_indices]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains strings shorter than minimum length"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)} (length: {len(val)})"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)


def _validate_maxlen(series: pd.Series, validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    invalid_mask = series.str.len() > validator.max_length
    if invalid_mask.any():
        invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
        samples = [(idx, series.loc[idx]) for idx in invalid_indices]

        total_invalid = invalid_mask.sum()
        msg = f"Column '{col_name}': contains strings longer than maximum length"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {repr(val)} (length: {len(val)})"

        raise ValidationError(msg, column_name=col_name, invalid_samples=samples)
