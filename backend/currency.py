"""Detect display currency metadata without changing dataset values."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


SUPPORTED_CURRENCY_CODES = frozenset(
    {
        "AUD",
        "CAD",
        "CHF",
        "CNY",
        "EUR",
        "GBP",
        "HKD",
        "INR",
        "JPY",
        "KRW",
        "NZD",
        "SGD",
        "USD",
        "ZAR",
    }
)
CURRENCY_COLUMN_NAMES = (
    "currency",
    "currency_code",
    "iso_currency",
    "iso_currency_code",
)


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def detect_currency_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return recognized ISO currency codes without mutating or aggregating rows."""
    columns = {_normalized_column(column): str(column) for column in dataframe.columns}
    column = next(
        (columns[name] for name in CURRENCY_COLUMN_NAMES if name in columns),
        None,
    )
    if column is None:
        return {
            "status": "unspecified",
            "column": None,
            "detected_code": None,
            "detected_codes": [],
            "unsupported_values": [],
        }

    values = {
        str(value).strip().upper()
        for value in dataframe[column].dropna().tolist()
        if str(value).strip()
    }
    detected = sorted(values & SUPPORTED_CURRENCY_CODES)
    unsupported = sorted(values - SUPPORTED_CURRENCY_CODES)
    if len(values) > 1:
        status = "multiple"
        detected_code = None
    elif len(detected) == 1:
        status = "detected"
        detected_code = detected[0]
    else:
        status = "unspecified"
        detected_code = None
    return {
        "status": status,
        "column": column,
        "detected_code": detected_code,
        "detected_codes": detected,
        "unsupported_values": unsupported,
    }
