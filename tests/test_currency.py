import pandas as pd
import pytest

from backend.currency import detect_currency_metadata


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Currency", "USD"),
        ("Currency Code", "inr"),
        ("ISO Currency", "EUR"),
        ("ISO Currency Code", "GBP"),
    ],
)
def test_detects_supported_currency_columns_and_codes(column, value):
    dataframe = pd.DataFrame({column: [value, f" {value} "]})

    result = detect_currency_metadata(dataframe)

    assert result == {
        "status": "detected",
        "column": column,
        "detected_code": value.strip().upper(),
        "detected_codes": [value.strip().upper()],
        "unsupported_values": [],
    }


def test_missing_currency_column_remains_unspecified():
    result = detect_currency_metadata(pd.DataFrame({"Sales": [100]}))

    assert result["status"] == "unspecified"
    assert result["detected_code"] is None
    assert result["column"] is None


def test_multiple_currencies_are_not_collapsed_to_one_code():
    result = detect_currency_metadata(
        pd.DataFrame({"Currency": ["USD", "INR", "USD"], "Sales": [10, 20, 30]})
    )

    assert result["status"] == "multiple"
    assert result["detected_code"] is None
    assert result["detected_codes"] == ["INR", "USD"]


def test_unknown_values_are_reported_without_guessing():
    result = detect_currency_metadata(pd.DataFrame({"Currency Code": ["XYZ"]}))

    assert result["status"] == "unspecified"
    assert result["detected_codes"] == []
    assert result["unsupported_values"] == ["XYZ"]


def test_supported_and_unknown_values_are_still_treated_as_multiple_currencies():
    result = detect_currency_metadata(pd.DataFrame({"Currency": ["USD", "XYZ"]}))

    assert result["status"] == "multiple"
    assert result["detected_code"] is None
    assert result["detected_codes"] == ["USD"]
    assert result["unsupported_values"] == ["XYZ"]
