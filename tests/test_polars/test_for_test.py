from datetime import date, datetime, timedelta
from typing import Literal, Optional, Protocol

import pytest

try:
    import polars as pl
    from polars.testing import assert_frame_equal

    from pavise.polars import DataFrame
    from pavise.testing import ANY
    from pavise.types import NotRequiredColumn

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


def assert_any_filled(result: "DataFrame", columns: list[str]) -> None:
    """Assert that specified columns are filled with ANY values."""
    for column in columns:
        for value in result[column]:
            assert value is ANY


class SimpleSchema(Protocol):
    a: int


class OptionalSchema(Protocol):
    user_id: int
    email: Optional[str]
    age: Optional[int]


class LiteralSchema(Protocol):
    status: Literal["pending", "approved", "rejected"]
    priority: Literal[1, 2, 3]


class NotRequiredSchema(Protocol):
    user_id: int
    name: str
    age: NotRequiredColumn[int]
    email: NotRequiredColumn[Optional[str]]


class DatetimeSchema(Protocol):
    created_at: datetime
    event_date: date
    duration: timedelta


def test_for_test_creates_dataframe_with_partial_data():
    """DataFrame.for_test() creates DataFrame with partial data"""
    result = DataFrame[SimpleSchema].for_test({"a": [1, 2, 3]})
    expected = pl.DataFrame({"a": [1, 2, 3]})

    assert_frame_equal(result, expected)
    assert_any_filled(result, [])


def test_for_test_with_optional_types():
    """DataFrame.for_test() works with Optional types"""
    result = DataFrame[OptionalSchema].for_test(
        {
            "user_id": [1, 2],
        }
    )
    expected = pl.DataFrame(
        {
            "user_id": [1, 2],
            "email": [ANY, ANY],
            "age": [ANY, ANY],
        }
    )

    assert_frame_equal(result, expected)
    assert_any_filled(result, ["email", "age"])


def test_for_test_with_literal_types():
    """DataFrame.for_test() works with Literal types"""
    result = DataFrame[LiteralSchema].for_test(
        {
            "status": ["pending", "approved"],
        }
    )
    expected = pl.DataFrame(
        {
            "status": ["pending", "approved"],
            "priority": [ANY, ANY],
        }
    )

    assert_frame_equal(result, expected)
    assert_any_filled(result, ["priority"])


def test_for_test_with_notrequired_not_specified():
    """DataFrame.for_test() fills NotRequiredColumn when not specified"""
    result = DataFrame[NotRequiredSchema].for_test(
        {
            "user_id": [1, 2],
            "name": ["Alice", "Bob"],
        }
    )
    expected = pl.DataFrame(
        {
            "user_id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [ANY, ANY],
            "email": [ANY, ANY],
        }
    )

    assert_frame_equal(result, expected)
    assert_any_filled(result, ["age", "email"])


def test_for_test_with_notrequired_specified():
    """DataFrame.for_test() uses provided NotRequiredColumn data"""
    result = DataFrame[NotRequiredSchema].for_test(
        {
            "user_id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [25, 30],
        }
    )
    expected = pl.DataFrame(
        {
            "user_id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "email": [ANY, ANY],
        }
    )

    assert_frame_equal(result, expected)
    assert_any_filled(result, ["email"])


def test_for_test_raises_on_empty_data():
    """DataFrame.for_test() raises on empty data"""
    with pytest.raises(ValueError, match="data must not be empty"):
        DataFrame[SimpleSchema].for_test({})


def test_for_test_raises_on_unknown_columns():
    """DataFrame.for_test() raises on unknown columns"""
    with pytest.raises(ValueError, match="Unknown columns in data"):
        DataFrame[SimpleSchema].for_test({"a": [1], "unknown": [2]})


def test_for_test_raises_on_no_schema():
    """DataFrame.for_test() raises when schema is not defined"""
    with pytest.raises(ValueError, match="Cannot use for_test\\(\\) without a schema"):
        DataFrame.for_test({"a": [1]})


def test_for_test_with_datetime_types():
    """DataFrame.for_test() works with datetime types"""
    now = datetime.now()

    result = DataFrame[DatetimeSchema].for_test(
        {
            "created_at": [now, now],
        }
    )
    expected = pl.DataFrame(
        {
            "created_at": [now, now],
            "event_date": [ANY, ANY],
            "duration": [ANY, ANY],
        }
    )

    assert_frame_equal(result, expected)
    assert_any_filled(result, ["event_date", "duration"])
