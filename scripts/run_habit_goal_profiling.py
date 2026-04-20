from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.habit_goal_profiling import run_habit_goal_profiling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provisional habit-like vs goal-directed profiling on UMAHand features.")
    parser.add_argument("--features-csv", type=Path, required=True, help="Path to outputs/features/trial_features.csv")
    parser.add_argument(
        "--families-csv",
        type=Path,
        required=True,
        help="Path to outputs/classification_interpretation/provisional_motor_families.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where profiling outputs will be written.")
    parser.add_argument(
        "--habit-summary-csv",
        type=Path,
        default=Path("outputs/eda/habit_vs_goal_summary_table.csv"),
        help="Optional earlier coarse habit-vs-goal summary table.",
    )
    parser.add_argument(
        "--activity-performance-csv",
        type=Path,
        default=Path("outputs/classification_interpretation/activity_performance_summary.csv"),
        help="Optional activity performance summary for contextual interpretation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_habit_goal_profiling(
        features_csv=args.features_csv,
        families_csv=args.families_csv,
        output_dir=args.output_dir,
        habit_summary_csv=args.habit_summary_csv,
        activity_performance_csv=args.activity_performance_csv,
    )
    print(f"Report saved to: {result.report_path}")
    print(f"Tables generated: {len(result.generated_tables)}")
    print(f"Figures generated: {len(result.generated_figures)}")
    print("Group composition:")
    for row in result.group_counts.itertuples(index=False):
        print(f"  {row.habit_goal_group}: {int(row.n_activities)} activities, {int(row.n_trials)} trials")


if __name__ == "__main__":
    main()
