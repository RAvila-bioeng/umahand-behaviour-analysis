from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.config import default_output_dir
from umahand.dataset_summary import build_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the UMAHand dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the root UMAHand directory containing TRACES/ and metadata files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where summary tables, figures and report will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset_summary(data_root=args.data_root, output_dir=args.output_dir)
    print(f"Trial summary saved to: {args.output_dir / 'trial_summary.csv'}")
    print(f"Markdown report saved to: {result.report_path}")


if __name__ == "__main__":
    main()
