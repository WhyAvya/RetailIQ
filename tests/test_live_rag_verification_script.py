import json

from scripts.verify_live_rag import serialize_response


def test_response_serialization_round_trips_unicode_and_sources_on_windows_console():
    payload = {
        "answer": "Wireless Mouse → profit ₹300; café order preserved.",
        "sources": [
            {
                "source_row": 1,
                "document_excerpt": "Technology • Accessories",
                "similarity_score": 0.291889,
            }
        ],
    }

    rendered = serialize_response(payload)

    rendered.encode("cp1252", errors="strict")
    assert "\\u2192" in rendered
    assert "\\u20b9" in rendered
    assert json.loads(rendered) == payload
