"""Run one explicit live RAG verification with Windows-safe console output."""

import json
import sys
from typing import Any


UPLOAD_ID = "6377aeff-8a64-47ac-9748-28e0b0523d56"
QUESTION = "Which technology accessory orders had strong profits?"
TOP_K = 3


def configure_console_output() -> None:
    """Prefer UTF-8 while retaining escaped output as a console-safe fallback."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")


def serialize_response(payload: Any) -> str:
    """Preserve Unicode content as reversible ASCII JSON escape sequences."""
    return json.dumps(payload, ensure_ascii=True, indent=2)


def run_live_verification() -> int:
    """Call the existing endpoint once when this script is explicitly executed."""
    from fastapi.testclient import TestClient

    from backend.main import app

    configure_console_output()
    response = TestClient(app).post(
        "/api/rag/ask",
        json={
            "upload_id": UPLOAD_ID,
            "question": QUESTION,
            "top_k": TOP_K,
        },
    )
    payload = response.json()
    print(f"HTTP_STATUS={response.status_code}")
    print(serialize_response(payload))
    return 0 if response.is_success else 1


if __name__ == "__main__":
    raise SystemExit(run_live_verification())
