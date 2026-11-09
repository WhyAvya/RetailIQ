"""Limited repeated grounding probe for the C8 prompt changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.live_nemotron_evaluation import (
    render_markdown,
    run_live_sample,
    save_json,
    score_live_sample,
)


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "reports" / "c8_live_nemotron_raw.json"
SCORED_PATH = ROOT / "reports" / "c8_live_nemotron_evaluation.json"
MARKDOWN_PATH = ROOT / "reports" / "c8_live_nemotron_evaluation.md"
ANNOTATIONS_PATH = ROOT / "fixtures" / "c8_live_nemotron_annotations.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "score"))
    args = parser.parse_args()
    if args.stage == "run":
        report = run_live_sample()
        report["evaluation_id"] = "retailiq-c8-live-nemotron-repeated-probe-v1"
        report["comparison_note"] = (
            "The three C6 questions were intentionally repeated to isolate prompt-grounding changes; "
            "this is not a fresh held-out benchmark."
        )
        save_json(report, RAW_PATH)
        print(json.dumps({key: report[key] for key in ("model", "requests_attempted", "answers_completed", "provider_failure_count")}, indent=2))
        return
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))["annotations"]
    scored = score_live_sample(raw, annotations)
    save_json(scored, SCORED_PATH)
    MARKDOWN_PATH.write_text(
        render_markdown(scored)
        .replace("C6 Live", "C8 Live")
        .replace(
            "Scores above are from real live model outputs.",
            "This repeated three-question probe isolates the C8 grounding prompt; it is not fresh held-out evaluation. Scores above are from real live model outputs.",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
