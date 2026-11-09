from types import SimpleNamespace

import pytest

from backend.llm_service import LLMRateLimitError
from backend.rag_retrieval import RetrievalMatch
import json

from evaluation.live_nemotron_evaluation import (
    DEFAULT_RAW_PATH,
    DEFAULT_SCORED_PATH,
    run_live_sample,
    score_live_sample,
)


class FakeLLM:
    config = SimpleNamespace(model="fake-nemotron")

    def generate_grounded_answer(self, question, matches):
        return f"{question} Row {matches[0].source_row}."


def fake_retriever(upload_id, _question, _top_k):
    return [
        RetrievalMatch(
            1,
            "chunk-1",
            "Product Name: Example Organizer",
            1,
            {"upload_id": str(upload_id), "row_number": 1},
            0.9,
        )
    ]


def test_live_runner_limits_requests_and_records_real_outputs():
    report = run_live_sample(FakeLLM(), fake_retriever)

    assert report["requests_attempted"] == report["request_limit"] == 3
    assert report["answers_completed"] == 3
    assert report["provider_failure_count"] == 0
    assert all(item["answer"] for item in report["questions"])
    assert all(item["evidence"][0]["source_row"] == 1 for item in report["questions"])


def test_provider_failures_are_recorded_without_scores():
    class FailingLLM(FakeLLM):
        def generate_grounded_answer(self, *_args):
            raise LLMRateLimitError("limited")

    report = run_live_sample(FailingLLM(), fake_retriever)

    assert report["answers_completed"] == 0
    assert report["provider_failure_count"] == 3
    assert all(item["answer"] is None for item in report["questions"])
    scored = score_live_sample(report, [])
    assert scored["measured_scores"] is None
    assert scored["status"] == "completed_with_provider_failures"


def test_scoring_requires_annotations_to_match_live_answer_and_evidence():
    raw = run_live_sample(FakeLLM(), fake_retriever)
    first = raw["questions"][0]
    annotation = {
        "id": "REVIEW-1",
        "question_id": first["id"],
        "question": first["question"],
        "answer": first["answer"],
        "retrieved_evidence_rows": [1],
        "claims": [
            {
                "id": "C1",
                "text": "The answer cites row 1.",
                "factual": True,
                "supported": True,
                "evidence_rows": [1],
            }
        ],
        "relevance_review": {"score": 3, "notes": "Directly relevant."},
        "reviewer": "test-reviewer",
    }

    scored = score_live_sample(raw, [annotation])
    assert scored["measured_scores"]["faithfulness"] == 1.0
    changed = dict(annotation, answer="invented")
    with pytest.raises(ValueError, match="exactly match"):
        score_live_sample(raw, [changed])


def test_saved_live_report_contains_only_measured_real_answer_scores():
    raw = json.loads(DEFAULT_RAW_PATH.read_text(encoding="utf-8"))
    scored = json.loads(DEFAULT_SCORED_PATH.read_text(encoding="utf-8"))

    assert raw["result_kind"] == "real_live_llm"
    assert raw["requests_attempted"] == raw["answers_completed"] == 3
    assert raw["provider_failure_count"] == 0
    assert scored["status"] == "completed"
    assert scored["score_provenance"].startswith("Manual claim/evidence review")
    assert scored["measured_scores"]["answers_evaluated"] == 3
    assert scored["measured_scores"]["grounding"] == 0.75
    assert scored["measured_scores"]["faithfulness"] == 0.75
    assert scored["measured_scores"]["answer_relevance"] == pytest.approx(8 / 9)
    assert "fixture_validation" not in scored
