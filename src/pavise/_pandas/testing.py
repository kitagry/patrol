"""Test data generation utilities for Pandas backend."""

from typing import Any

import pandas as pd

from pavise._pandas.spec import get_column_specs
from pavise.testing import ANY


def convert_data_to_dict(data, schema: type) -> tuple[dict[str, Any], int]:
    """
    Convert input data to dict format and validate against schema.

    Args:
        data: Input data (dict, DataFrame, etc.)
        schema: Schema Protocol class

    Returns:
        Tuple of (data_dict, n_rows)

    Raises:
        ValueError: If data is empty, invalid format, or contains unknown columns
    """
    temp_df = pd.DataFrame(data)

    if temp_df.empty:
        raise ValueError("data must not be empty. Use make_empty() to create an empty DataFrame.")

    data_dict = {col: temp_df[col].to_list() for col in temp_df.columns}
    n_rows = len(temp_df)

    # Validate: all data columns must exist in schema
    column_specs = get_column_specs(schema)
    unknown_cols = set(data_dict.keys()) - set(column_specs.keys())
    if unknown_cols:
        raise ValueError(
            f"Unknown columns in data: {sorted(unknown_cols)}. "
            f"Schema only defines: {sorted(column_specs.keys())}"
        )

    return data_dict, n_rows


def build_for_test_dataframe(
    schema: type,
    data_dict: dict[str, Any],
    n_rows: int,
) -> dict[str, pd.Series]:
    """
    Build columns for for_test DataFrame.

    Fills missing columns with ANY sentinel values.

    Args:
        schema: Schema Protocol class
        data_dict: Data as dict (column_name -> list of values)
        n_rows: Number of rows

    Returns:
        Dict of column_name -> Series
    """
    column_specs = get_column_specs(schema)

    # Build columns: use provided data or fill with ANY
    columns = {}
    for col_name, spec in column_specs.items():
        if col_name in data_dict:
            columns[col_name] = pd.Series(data_dict[col_name])
        else:
            # Fill all missing columns with ANY
            columns[col_name] = pd.Series([ANY] * n_rows, dtype="object")

    return columns
