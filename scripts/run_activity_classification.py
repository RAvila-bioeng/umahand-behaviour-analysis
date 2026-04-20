from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.classification import run_activity_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline classification on UMAHand trial features.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to outputs/features/trial_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where classification outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_activity_classification(features_csv=args.features_csv, output_dir=args.output_dir)
    print(f"Best GroupKFold model: {result.best_model_name}")
    print(f"Best feature config: {result.best_feature_config}")
    print(f"Best GroupKFold macro F1: {result.best_groupkfold_macro_f1:.4f}")
    print(f"Report saved to: {result.report_path}")


if __name__ == "__main__":
    main()
