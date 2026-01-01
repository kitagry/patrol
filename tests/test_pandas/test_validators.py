from typing import Annotated, Protocol

import pandas as pd
import pytest

from patrol.pandas import DataFrame
from patrol.validators import In, Range, Unique


class AgeSchema(Protocol):
    age: Annotated[int, Range(0, 150)]


class UserIdSchema(Protocol):
    user_id: Annotated[int, Unique()]


class StatusSchema(Protocol):
    status: Annotated[str, In(["pending", "approved", "rejected"])]


def test_range_validator_accepts_values_within_range():
    """Range validator accepts values within the specified range"""
    df = pd.DataFrame({"age": [25, 30, 45, 100]})
    result = DataFrame[AgeSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_range_validator_rejects_values_below_minimum():
    """Range validator rejects values below the minimum"""
    df = pd.DataFrame({"age": [25, -5, 30]})
    with pytest.raises(ValueError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_range_validator_rejects_values_above_maximum():
    """Range validator rejects values above the maximum"""
    df = pd.DataFrame({"age": [25, 200, 30]})
    with pytest.raises(ValueError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_unique_validator_accepts_unique_values():
    """Unique validator accepts columns with all unique values"""
    df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
    result = DataFrame[UserIdSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_unique_validator_rejects_duplicate_values():
    """Unique validator rejects columns with duplicate values"""
    df = pd.DataFrame({"user_id": [1, 2, 2, 3]})
    with pytest.raises(ValueError, match="user_id.*duplicate"):
        DataFrame[UserIdSchema](df)


def test_in_validator_accepts_allowed_values():
    """In validator accepts values within the allowed set"""
    df = pd.DataFrame({"status": ["pending", "approved", "rejected", "pending"]})
    result = DataFrame[StatusSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_in_validator_rejects_disallowed_values():
    """In validator rejects values not in the allowed set"""
    df = pd.DataFrame({"status": ["pending", "invalid", "approved"]})
    with pytest.raises(ValueError, match="status.*allowed values"):
        DataFrame[StatusSchema](df)
