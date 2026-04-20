from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.eda import run_visual_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual exploratory analysis on UMAHand summary and feature CSVs.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to outputs/features/trial_features.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to outputs/dataset_summary/trial_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the EDA report and figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_visual_eda(
        features_csv=args.features_csv,
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
    )
    print(f"Trials analysed: {result.n_trials}")
    print(f"Subjects represented: {result.n_subjects}")
    print(f"Activities represented: {result.n_activities}")
    print(f"Figures generated: {len(result.generated_figures)}")
    print(f"Report saved to: {result.report_path}")
    if result.habit_summary_path is not None:
        print(f"Habit-vs-goal summary saved to: {result.habit_summary_path}")
    if result.warnings:
        print(f"Warnings recorded: {len(result.warnings)}")


if __name__ == "__main__":
    main()
