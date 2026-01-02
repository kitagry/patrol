"""Custom exceptions for patrol."""


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
