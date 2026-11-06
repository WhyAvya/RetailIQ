import pandas as pd

from backend.data_processing import preprocess_dataframe, profile_dataframe


def test_profile_and_preprocessing_report_changes_without_mutating_original():
    original = pd.DataFrame(
        {
            " order_date ": ["2024-01-01", "not-a-date", "2024-01-01", None],
            "sales": [10.0, None, 10.0, None],
            "category": ["A", "A", "A", None],
        }
    )
    snapshot = original.copy(deep=True)

    profile = profile_dataframe(original)
    cleaned, summary = preprocess_dataframe(original)

    pd.testing.assert_frame_equal(original, snapshot)
    assert profile["rows"] == 4
    assert profile["columns"] == 3
    assert profile["duplicate_rows"] == 1
    assert profile["column_profiles"][1]["missing_count"] == 2
    assert profile["column_profiles"][1]["missing_percentage"] == 50.0

    assert list(cleaned.columns) == ["order_date", "sales", "category"]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert summary["missing_values_before"] == 4
    assert summary["missing_values_after"] == 1
    assert summary["duplicate_rows_removed"] == 2
    assert summary["cleaned_rows"] == 2

    date_operation = next(
        operation
        for operation in summary["operations"]
        if operation["operation"] == "convert_dates"
    )
    assert date_operation["affected"] == 2
    assert "1 invalid values" in date_operation["description"]


def test_empty_dataset_with_columns_is_handled_gracefully():
    original = pd.DataFrame(columns=["value", "event_date"])

    profile = profile_dataframe(original)
    cleaned, summary = preprocess_dataframe(original)

    assert profile["rows"] == 0
    assert profile["columns"] == 2
    assert profile["duplicate_rows"] == 0
    assert all(column["missing_percentage"] == 0.0 for column in profile["column_profiles"])
    assert cleaned.empty
    assert summary["cleaned_rows"] == 0
    assert summary["duplicate_rows_removed"] == 0
