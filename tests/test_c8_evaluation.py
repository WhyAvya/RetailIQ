import json

from evaluation.c8_evaluation import (
    BENCHMARK_PATH,
    build_independent_benchmark,
    evaluate_independent,
    render_independent_markdown,
    verify_freeze_manifest,
)
from evaluation.benchmark_b1 import DEFAULT_BENCHMARK_PATH


def test_c8_benchmark_is_new_balanced_and_independently_reproducible():
    saved = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    rebuilt = build_independent_benchmark()
    rebuilt["created_at_utc"] = saved["created_at_utc"]
    assert rebuilt == saved
    assert len(saved["questions"]) == 25
    assert {
        kind: sum(question["type"] == kind for question in saved["questions"])
        for kind in ("filtering", "numerical", "ranking", "mixed", "semantic")
    } == {kind: 5 for kind in ("filtering", "numerical", "ranking", "mixed", "semantic")}
    previous = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert {question["question"] for question in saved["questions"]}.isdisjoint(
        question["question"] for question in previous["questions"]
    )


def test_c8_evaluator_scores_deterministic_families_and_generates_report():
    saved = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    relevant = {question["question"]: question.get("relevant_source_rows", []) for question in saved["questions"]}

    def retriever(upload_id, question, top_k):
        return [
            __import__("backend.rag_retrieval", fromlist=["RetrievalMatch"]).RetrievalMatch(
                rank, f"c8-{row}", f"row {row}", row,
                {"upload_id": str(upload_id), "row_number": row}, 1.0 - rank / 100,
            )
            for rank, row in enumerate(relevant[question][:top_k], 1)
        ]

    report = evaluate_independent(
        retriever=retriever,
        filtered_retriever=lambda *_args: [],
        require_frozen_backend=False,
    )
    assert report["question_count"] == 25
    assert report["llm_calls"] == 0
    assert report["metrics"]["filtering_accuracy"] == 1.0
    assert report["metrics"]["numerical_accuracy"] == 1.0
    assert report["metrics"]["ranking_accuracy"] == 1.0
    assert report["metrics"]["mixed_accuracy"] == 1.0
    assert "Deterministic failures" in render_independent_markdown(report)
    assert not verify_freeze_manifest()
