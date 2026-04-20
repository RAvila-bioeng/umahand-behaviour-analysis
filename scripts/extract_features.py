from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from umahand.config import default_features_output_dir
from umahand.features import build_feature_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract trial-level features from the UMAHand dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the root UMAHand directory containing TRACES/ and metadata files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_features_output_dir(),
        help="Directory where feature tables and report will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_feature_dataset(data_root=args.data_root, output_dir=args.output_dir)
    print(f"Trials processed: {len(result.trial_features)}")
    print(f"Feature columns generated: {len(result.feature_summary)}")
    print(f"Trial feature table saved to: {args.output_dir / 'trial_features.csv'}")
    print(f"Feature summary saved to: {args.output_dir / 'feature_summary.csv'}")
    print(f"Markdown report saved to: {result.report_path}")


if __name__ == "__main__":
    main()
