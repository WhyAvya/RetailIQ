"""Prepare Benchmark B1 artifacts without invoking the RAG pipeline or OpenRouter."""

import json

from evaluation.benchmark_b1 import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_COVERAGE_REPORT_PATH,
    build_benchmark,
    prepare_full_dataset_index,
    validate_benchmark,
)


def main() -> int:
    coverage = prepare_full_dataset_index()
    benchmark = build_benchmark()
    validate_benchmark(benchmark)
    DEFAULT_BENCHMARK_PATH.write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    DEFAULT_COVERAGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_COVERAGE_REPORT_PATH.write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"B1 prepared: {len(benchmark['questions'])} questions; "
        f"coverage complete={coverage['coverage_after']['complete']}; "
        f"indexed now={coverage['rows_indexed_during_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
