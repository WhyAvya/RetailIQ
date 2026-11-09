import copy
import json

import pytest

from evaluation.answer_evaluation import (
    DEFAULT_ANNOTATIONS_PATH,
    AnswerAnnotationError,
    build_answer_quality_report,
    evaluate_annotations,
    score_annotation,
    verify_numerical_answers,
)


def _fixture_annotations():
    return json.loads(DEFAULT_ANNOTATIONS_PATH.read_text(encoding="utf-8"))["annotations"]


def test_fixture_scores_have_mathematically_verified_claim_metrics():
    result = evaluate_annotations(_fixture_annotations())

    assert result["answers_evaluated"] == 3
    assert result["factual_claims"] == 5
    assert result["supported_claims"] == 4
    assert result["unsupported_claims"] == 1
    assert result["faithfulness"] == pytest.approx(4 / 5)
    assert result["unsupported_claim_rate"] == pytest.approx(1 / 5)
    assert result["answer_relevance"] == pytest.approx(2 / 3)


def test_non_factual_statements_are_excluded_from_claim_denominators():
    annotation = copy.deepcopy(_fixture_annotations()[2])
    result = score_annotation(annotation)

    assert result["factual_claims"] == 1
    assert result["supported_claims"] == 1
    assert result["faithfulness"] == 1.0
    assert result["unsupported_claim_rate"] == 0.0


def test_answer_with_no_factual_claims_uses_documented_neutral_convention():
    annotation = copy.deepcopy(_fixture_annotations()[0])
    annotation["claims"] = []
    result = score_annotation(annotation)

    assert result["factual_claims"] == 0
    assert result["faithfulness"] == 1.0
    assert result["unsupported_claim_rate"] == 0.0


def test_supported_claim_requires_retrieved_evidence_reference():
    annotation = copy.deepcopy(_fixture_annotations()[0])
    annotation["claims"][0]["evidence_rows"] = [12]

    with pytest.raises(AnswerAnnotationError, match="retrieved evidence rows"):
        score_annotation(annotation)


@pytest.mark.parametrize("score", [-1, 4, 1.5, True])
def test_relevance_review_uses_zero_to_three_integer_rubric(score):
    annotation = copy.deepcopy(_fixture_annotations()[0])
    annotation["relevance_review"]["score"] = score

    with pytest.raises(AnswerAnnotationError, match="integer from 0 to 3"):
        score_annotation(annotation)


def test_numerical_answers_match_existing_analytics_pipeline():
    result = verify_numerical_answers()

    assert result["status"] == "verified"
    assert result["result_kind"] == "actual_deterministic"
    assert result["questions_evaluated"] == result["correct_answers"] == 5
    assert result["numerical_correctness"] == 1.0
    assert all(item["correct"] for item in result["questions"])
    assert {item["metric"] for item in result["questions"]} == {
        "total_revenue",
        "total_profit",
        "number_of_orders",
        "average_order_value",
        "profit_margin",
    }


def test_report_keeps_live_scores_pending_and_fixture_scores_separate():
    report = build_answer_quality_report()

    live = report["live_llm_evaluation"]
    assert live["status"] == "pending"
    assert live["answers_evaluated"] == 0
    assert live["faithfulness"] is None
    assert live["answer_relevance"] is None
    assert live["unsupported_claim_rate"] is None
    assert "429" in live["reason"]
    assert report["fixture_validation"]["result_kind"] == "test_fixture_only"
    assert report["numerical_correctness"]["result_kind"] == "actual_deterministic"
