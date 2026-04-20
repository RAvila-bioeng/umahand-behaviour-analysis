from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.classification_interpretation import run_classification_interpretation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret previously generated UMAHand classification outputs.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to outputs/features/trial_features.csv",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        required=True,
        help="Path to outputs/classification/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where interpretation outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_classification_interpretation(
        features_csv=args.features_csv,
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
    )
    print(f"Report saved to: {result.report_path}")
    print(f"Figures generated: {len(result.generated_figures)}")
    print(f"Tables generated: {len(result.generated_tables)}")


if __name__ == "__main__":
    main()
