"""Command line interface for phase-wise retrieval experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hh_goa_rag.config import load_config
from hh_goa_rag.dataset import prepare_dataset, read_manifest


def _dataset_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = prepare_dataset(config, force=args.force)
    manifest = read_manifest(output_dir / "manifest.json")
    table = Table(title="MSMARCO-XI prepared evaluation artifacts")
    table.add_column("Split")
    table.add_column("Source")
    table.add_column("Queries", justify="right")
    table.add_column("Parent passages", justify="right")
    table.add_column("Qrels", justify="right")
    for split, stats in manifest["splits"].items():
        table.add_row(
            split,
            stats["source_split"],
            str(stats["queries"]),
            str(stats["unique_parent_passages"]),
            str(stats["qrels"]),
        )
    console = Console()
    console.print(table)
    console.print(f"Artifacts: [bold]{output_dir}[/bold]")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    prepare_parser = dataset_subparsers.add_parser("prepare", help="Prepare fixed eval artifacts")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--json", action="store_true")
    prepare_parser.set_defaults(handler=_dataset_prepare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

