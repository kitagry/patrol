"""Polars backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Generic, Optional, TypeVar

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")

from patrol._polars.validation import validate_dataframe

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


class DataFrame(pl.DataFrame, Generic[SchemaT_co]):
    """
    Type-parameterized DataFrame with runtime validation for Polars.

    Usage:
        # Static type checking only
        def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
            return df

        # Runtime validation
        validated = DataFrame[UserSchema](raw_df)

    The type parameter is covariant, allowing structural subtyping:
        DataFrame[ChildSchema] is compatible with DataFrame[ParentSchema]
        when ChildSchema has all columns of ParentSchema.
    """

    _schema: Optional[type] = None

    def __class_getitem__(cls, schema: type):
        """Create a new DataFrame class with schema validation."""

        class TypedDataFrame(DataFrame):
            _schema = schema

        return TypedDataFrame

    def __new__(cls, data, *args, **kwargs):
        """
        Create DataFrame with optional schema validation.

        Args:
            data: Data to create DataFrame from (pl.DataFrame or dict/list)
            *args: Additional arguments passed to pl.DataFrame
            **kwargs: Additional keyword arguments passed to pl.DataFrame

        Raises:
            ValueError: If required column is missing
            TypeError: If column has wrong type
        """
        # Create DataFrame instance
        if isinstance(data, pl.DataFrame):
            instance = data
        else:
            instance = pl.DataFrame(data, *args, **kwargs)

        # Validate if schema is set
        if cls._schema is not None:
            validate_dataframe(instance, cls._schema)

        return instance
