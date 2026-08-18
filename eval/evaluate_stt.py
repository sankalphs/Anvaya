"""Score provider-produced speech transcripts without selecting an STT provider."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .metrics import (
        evaluate_stt_records,
        load_dataset,
        pair_cases_and_predictions,
        print_summary,
        read_jsonl,
        write_evaluation_run,
    )
except ImportError:
    from metrics import (  # type: ignore[no-redef]
        evaluate_stt_records,
        load_dataset,
        pair_cases_and_predictions,
        print_summary,
        read_jsonl,
        write_evaluation_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("eval/eval_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--allow-sealed-test", action="store_true")
    args = parser.parse_args()

    cases = load_dataset(
        args.dataset, split=args.split, allow_sealed_test=args.allow_sealed_test
    )
    pairs = pair_cases_and_predictions(cases, read_jsonl(args.predictions))
    metrics, details = evaluate_stt_records(pairs)
    summary_path, cases_path = write_evaluation_run(
        output_dir=args.output_dir,
        run_id=args.run_id,
        stage="stt",
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        split=args.split,
        metrics=metrics,
        details=details,
        system_id=args.system_id,
    )
    print_summary(summary_path, cases_path, metrics)


if __name__ == "__main__":
    main()
