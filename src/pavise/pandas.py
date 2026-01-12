"""Pandas backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Any, Generic, Optional, TypeVar

import pandas as pd

from pavise._pandas.spec import get_column_specs
from pavise._pandas.testing import build_for_test_dataframe, convert_data_to_dict
from pavise._pandas.validation import validate_dataframe
from pavise.types import NotRequiredColumn

__all__ = ["DataFrame", "NotRequiredColumn"]

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


class DataFrame(pd.DataFrame, Generic[SchemaT_co]):
    """
    Type-parameterized DataFrame with runtime validation for pandas.

    Usage::

        # Static type checking only
        def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
            return df

        # Runtime validation
        validated = DataFrame[UserSchema](raw_df)

    The type parameter is covariant, allowing structural subtyping.
    DataFrame[ChildSchema] is compatible with DataFrame[ParentSchema]
    when ChildSchema has all columns of ParentSchema.
    """

    _schema: Optional[type] = None

    def __class_getitem__(cls, schema: type):
        """Create a new DataFrame class with schema validation."""

        class TypedDataFrame(DataFrame):
            _schema = schema

        return TypedDataFrame

    def __new__(cls, data: Any = None, *args: Any, strict: bool = False, **kwargs: Any):
        """Create a new DataFrame instance."""
        return super().__new__(cls)

    def __init__(self, data: Any = None, *args: Any, strict: bool = False, **kwargs: Any) -> None:
        """
        Initialize DataFrame with optional schema validation.

        Args:
            data: Data to create DataFrame from
            *args: Additional arguments passed to pd.DataFrame
            strict: If True, raise error on extra columns not in schema
            **kwargs: Additional keyword arguments passed to pd.DataFrame

        Raises:
            ValueError: If required column is missing
            TypeError: If column has wrong type
        """
        pd.DataFrame.__init__(self, data, *args, **kwargs)  # type: ignore[misc]
        if self._schema is not None:
            validate_dataframe(self, self._schema, strict=strict)

    @classmethod
    def make_empty(cls) -> "DataFrame[SchemaT_co]":
        """
        Create an empty DataFrame with columns from the schema.

        Returns:
            DataFrame: Empty DataFrame with correct column types
        """
        if cls._schema is None:
            return cls({})

        column_specs = get_column_specs(cls._schema)
        columns = {
            col_name: pd.Series([], dtype=spec.dtype) for col_name, spec in column_specs.items()
        }

        return cls(columns)

    @classmethod
    def for_test(cls, data) -> "DataFrame[SchemaT_co]":
        """
        Create a DataFrame for testing with partial data filled with sentinel values.

        Specified columns are used as-is, while unspecified columns are filled with ANY.

        Args:
            data: Partial data (dict, DataFrame, etc.)

        Returns:
            DataFrame: DataFrame with specified columns and ANY values for missing columns

        Raises:
            ValueError: If data is empty, schema is not defined, or contains unknown columns
        """
        if cls._schema is None:
            raise ValueError(
                "Cannot use for_test() without a schema. Use DataFrame[Schema].for_test(...)"
            )

        # Convert data to dict and validate
        data_dict, n_rows = convert_data_to_dict(data, cls._schema)

        # Build columns
        columns = build_for_test_dataframe(cls._schema, data_dict, n_rows)

        return cls(columns, strict=False)
