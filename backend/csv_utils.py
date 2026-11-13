from typing import BinaryIO

import pandas as pd
from fastapi import UploadFile
from pandas.errors import EmptyDataError, ParserError
from backend.runtime_config import MAX_UPLOAD_BYTES

MAX_UPLOAD_SIZE = MAX_UPLOAD_BYTES
READ_CHUNK_SIZE = 1024 * 1024
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class CSVValidationError(ValueError):
    """Raised when an uploaded file cannot be accepted as CSV data."""


async def validate_csv_upload(file: UploadFile) -> int:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise CSVValidationError("Only files ending in .csv are accepted.")

    file_size = 0

    while chunk := await file.read(READ_CHUNK_SIZE):
        file_size += len(chunk)

        if file_size > MAX_UPLOAD_SIZE:
            raise CSVValidationError(f"The CSV file exceeds the {MAX_UPLOAD_SIZE // (1024 * 1024)} MB upload limit.")

        if b"\x00" in chunk:
            raise CSVValidationError("The uploaded file contains binary data and is not a valid CSV.")

    if file_size == 0:
        raise CSVValidationError("The uploaded CSV file is empty.")

    await file.seek(0)
    return file_size


def read_csv_dataframe(file_object: BinaryIO) -> pd.DataFrame:
    for encoding in CSV_ENCODINGS:
        file_object.seek(0)

        try:
            return pd.read_csv(file_object, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except EmptyDataError as exc:
            raise CSVValidationError(
                "The uploaded CSV does not contain readable columns."
            ) from exc
        except ParserError as exc:
            raise CSVValidationError(
                "The uploaded CSV is malformed and could not be parsed."
            ) from exc
        except (OSError, ValueError) as exc:
            raise CSVValidationError("The uploaded CSV could not be read.") from exc

    raise CSVValidationError("The CSV encoding is not supported.")
