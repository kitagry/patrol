"""Custom exceptions for patrol."""

from typing import Any, Callable

from typing_extensions import Self


class PatrolError(Exception):
    """Base exception for all patrol errors."""

    pass


class ValidationError(PatrolError):
    """
    Raised when DataFrame validation fails.

    Attributes:
        message: Human-readable error message
        column_name: Name of the column that failed validation (if applicable)
        invalid_samples: List of (row_index, value) tuples showing sample invalid values
    """

    def __init__(
        self,
        message: str,
        column_name: str | None = None,
        invalid_samples: list[tuple] | None = None,
    ):
        super().__init__(message)
        self.column_name = column_name
        self.invalid_samples = invalid_samples or []

    @classmethod
    def from_column_and_samples(
        cls,
        col_name: str,
        base_message: str,
        samples: list[tuple],
        total_invalid: int,
        format_value: Callable[[Any], str] | None = None,
    ) -> Self:
        """
        Create a ValidationError with a formatted message including sample invalid values.
        Args:
            col_name: Column name
            base_message: Base error message (e.g., "values must be in range [0, 150]")
            samples: List of (index, value) tuples
            total_invalid: Total number of invalid values
            format_value: Optional function to format values (default: repr)

        Returns:
        Formatted error message
        """
        if format_value is None:
            format_value = repr

        msg = f"Column '{col_name}': {base_message}"
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        for idx, val in samples:
            msg += f"\n  Row {idx}: {format_value(val)}"
        return cls(msg, column_name=col_name, invalid_samples=samples)
