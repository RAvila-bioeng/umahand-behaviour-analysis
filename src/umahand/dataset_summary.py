from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import (
    ACC_COLUMNS,
    EXPECTED_ACTIVITY_COUNT,
    EXPECTED_SAMPLING_HZ,
    EXPECTED_SUBJECT_COUNT,
    EXPECTED_TRIAL_COUNT,
    GYRO_COLUMNS,
    MAG_COLUMNS,
    SAMPLING_TOLERANCE_HZ,
    SENSOR_COLUMNS,
    TRACE_COLUMNS,
)
from .data_loading import TraceLoadError, find_trace_files, load_trace_csv, parse_trace_filename
from .metadata import load_activity_metadata, load_user_metadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetSummaryResult:
    trial_summary: pd.DataFrame
    activity_counts: pd.DataFrame
    subject_counts: pd.DataFrame
    activity_duration_summary: pd.DataFrame
    subject_duration_summary: pd.DataFrame
    data_quality_summary: pd.DataFrame
    report_path: Path


def build_dataset_summary(data_root: Path, output_dir: Path) -> DatasetSummaryResult:
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    user_metadata = load_user_metadata(data_root)
    activity_metadata = load_activity_metadata(data_root)
    trace_files = find_trace_files(data_root)

    trial_summary = build_trial_summary(
        data_root=data_root,
        trace_files=trace_files,
        activity_metadata=activity_metadata,
    )

    activity_counts = (
        trial_summary.groupby(["activity_id", "activity_name"], dropna=False)
        .size()
        .rename("n_trials")
        .reset_index()
        .sort_values(["activity_id", "activity_name"], ignore_index=True)
    )
    subject_counts = (
        trial_summary.groupby("user_id", dropna=False)
        .size()
        .rename("n_trials")
        .reset_index()
        .sort_values("user_id", ignore_index=True)
    )
    activity_duration_summary = _build_duration_summary(trial_summary, ["activity_id", "activity_name"])
    subject_duration_summary = _build_duration_summary(trial_summary, ["user_id"])
    data_quality_summary = _build_data_quality_summary(
        trial_summary=trial_summary,
        user_metadata=user_metadata,
        activity_metadata=activity_metadata,
    )

    trial_summary.to_csv(output_dir / "trial_summary.csv", index=False)
    activity_counts.to_csv(output_dir / "activity_counts.csv", index=False)
    subject_counts.to_csv(output_dir / "subject_counts.csv", index=False)
    activity_duration_summary.to_csv(output_dir / "activity_duration_summary.csv", index=False)
    subject_duration_summary.to_csv(output_dir / "subject_duration_summary.csv", index=False)
    data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)

    _generate_figures(
        trial_summary=trial_summary,
        activity_counts=activity_counts,
        subject_counts=subject_counts,
        figures_dir=figures_dir,
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        _build_report(
            trial_summary=trial_summary,
            user_metadata=user_metadata,
            activity_metadata=activity_metadata,
            activity_counts=activity_counts,
            subject_counts=subject_counts,
            data_quality_summary=data_quality_summary,
        ),
        encoding="utf-8",
    )

    return DatasetSummaryResult(
        trial_summary=trial_summary,
        activity_counts=activity_counts,
        subject_counts=subject_counts,
        activity_duration_summary=activity_duration_summary,
        subject_duration_summary=subject_duration_summary,
        data_quality_summary=data_quality_summary,
        report_path=report_path,
    )


def build_trial_summary(data_root: Path, trace_files: list[Path], activity_metadata: pd.DataFrame) -> pd.DataFrame:
    activity_lookup = dict(activity_metadata[["activity_id", "activity_name"]].itertuples(index=False, name=None))
    rows: list[dict[str, Any]] = []

    for path in trace_files:
        info = parse_trace_filename(path, data_root=data_root)
        base_row = {
            "relative_path": info.relative_path.as_posix(),
            "user_id": info.user_id,
            "activity_id": info.activity_id,
            "trial_id": info.trial_id,
            "activity_name": activity_lookup.get(info.activity_id, "unknown_activity"),
            "expected_sampling_hz": EXPECTED_SAMPLING_HZ,
        }
        try:
            trace_df = load_trace_csv(path)
        except TraceLoadError as exc:
            rows.append(_failed_trial_row(base_row=base_row, error=exc))
            continue

        rows.append(_summarize_trace(trace_df=trace_df, base_row=base_row))

    summary = pd.DataFrame(rows).sort_values(
        ["user_id", "activity_id", "trial_id", "relative_path"],
        ignore_index=True,
    )
    return summary


def _failed_trial_row(base_row: dict[str, Any], error: TraceLoadError) -> dict[str, Any]:
    row = dict(base_row)
    row.update(
        {
            "n_samples": np.nan,
            "duration_s": np.nan,
            "start_timestamp_ms": np.nan,
            "end_timestamp_ms": np.nan,
            "estimated_sampling_hz": np.nan,
            "median_dt_ms": np.nan,
            "min_dt_ms": np.nan,
            "max_dt_ms": np.nan,
            "has_missing_values": True,
            "n_missing_values": np.nan,
            "n_duplicate_timestamps": np.nan,
            "timestamps_monotonic": False,
            "timestamp_starts_at_zero": False,
            "n_columns": error.n_columns,
            "valid_column_count": False,
            "load_error": str(error),
            "quality_flags": "load_error",
        }
    )
    for column in SENSOR_COLUMNS:
        row[f"{column}_mean"] = np.nan
        row[f"{column}_std"] = np.nan
    row["acc_mag_mean"] = np.nan
    row["acc_mag_std"] = np.nan
    row["gyro_mag_mean"] = np.nan
    row["gyro_mag_std"] = np.nan
    row["mag_mag_mean"] = np.nan
    row["mag_mag_std"] = np.nan
    return row


def _summarize_trace(trace_df: pd.DataFrame, base_row: dict[str, Any]) -> dict[str, Any]:
    timestamps = trace_df["Timestamp"]
    dt = timestamps.diff().dropna()

    n_samples = int(len(trace_df))
    start_timestamp_ms = float(timestamps.iloc[0])
    end_timestamp_ms = float(timestamps.iloc[-1])
    duration_ms = end_timestamp_ms - start_timestamp_ms if n_samples else np.nan
    duration_s = duration_ms / 1000.0 if pd.notna(duration_ms) else np.nan

    median_dt_ms = float(dt.median()) if not dt.empty else np.nan
    min_dt_ms = float(dt.min()) if not dt.empty else np.nan
    max_dt_ms = float(dt.max()) if not dt.empty else np.nan
    estimated_sampling_hz = 1000.0 / median_dt_ms if median_dt_ms and not math.isclose(median_dt_ms, 0.0) else np.nan

    missing_values = int(trace_df.isna().sum().sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    monotonic = bool(timestamps.is_monotonic_increasing)
    starts_at_zero = bool(np.isclose(start_timestamp_ms, 0.0, atol=1e-6))

    row = dict(base_row)
    row.update(
        {
            "n_samples": n_samples,
            "duration_s": duration_s,
            "start_timestamp_ms": start_timestamp_ms,
            "end_timestamp_ms": end_timestamp_ms,
            "estimated_sampling_hz": estimated_sampling_hz,
            "median_dt_ms": median_dt_ms,
            "min_dt_ms": min_dt_ms,
            "max_dt_ms": max_dt_ms,
            "has_missing_values": missing_values > 0,
            "n_missing_values": missing_values,
            "n_duplicate_timestamps": duplicate_timestamps,
            "timestamps_monotonic": monotonic,
            "timestamp_starts_at_zero": starts_at_zero,
            "n_columns": trace_df.shape[1],
            "valid_column_count": trace_df.shape[1] == len(TRACE_COLUMNS),
            "load_error": "",
        }
    )

    for column in SENSOR_COLUMNS:
        row[f"{column}_mean"] = float(trace_df[column].mean())
        row[f"{column}_std"] = float(trace_df[column].std(ddof=1))

    row["acc_mag_mean"], row["acc_mag_std"] = _vector_magnitude_stats(trace_df, ACC_COLUMNS)
    row["gyro_mag_mean"], row["gyro_mag_std"] = _vector_magnitude_stats(trace_df, GYRO_COLUMNS)
    row["mag_mag_mean"], row["mag_mag_std"] = _vector_magnitude_stats(trace_df, MAG_COLUMNS)
    row["quality_flags"] = "|".join(_collect_quality_flags(row))
    return row


def _vector_magnitude_stats(trace_df: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    magnitudes = np.sqrt(np.square(trace_df[columns]).sum(axis=1))
    return float(magnitudes.mean()), float(magnitudes.std(ddof=1))


def _collect_quality_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if row["has_missing_values"]:
        flags.append("missing_values")
    if not row["timestamps_monotonic"]:
        flags.append("non_monotonic_timestamps")
    if not row["timestamp_starts_at_zero"]:
        flags.append("timestamp_not_zero")
    estimated_sampling_hz = row["estimated_sampling_hz"]
    if pd.notna(estimated_sampling_hz) and abs(estimated_sampling_hz - EXPECTED_SAMPLING_HZ) > SAMPLING_TOLERANCE_HZ:
        flags.append("sampling_rate_deviation")
    if not row["valid_column_count"]:
        flags.append("invalid_column_count")
    if row["n_duplicate_timestamps"] and row["n_duplicate_timestamps"] > 0:
        flags.append("duplicate_timestamps")
    return flags


def _build_duration_summary(trial_summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    valid = trial_summary.loc[trial_summary["load_error"] == ""].copy()
    grouped = valid.groupby(group_columns, dropna=False)["duration_s"]
    return (
        grouped.agg(duration_mean_s="mean", duration_min_s="min", duration_max_s="max", n_trials="size")
        .reset_index()
        .sort_values(group_columns, ignore_index=True)
    )


def _build_data_quality_summary(
    trial_summary: pd.DataFrame,
    user_metadata: pd.DataFrame,
    activity_metadata: pd.DataFrame,
) -> pd.DataFrame:
    valid_trials = trial_summary.loc[trial_summary["load_error"] == ""]
    metrics = [
        ("expected_subjects_readme", EXPECTED_SUBJECT_COUNT),
        ("expected_activities_readme", EXPECTED_ACTIVITY_COUNT),
        ("expected_trials_readme", EXPECTED_TRIAL_COUNT),
        ("expected_sampling_hz_readme", EXPECTED_SAMPLING_HZ),
        ("found_subjects_in_metadata", int(user_metadata["user_id"].nunique())),
        ("found_activities_in_metadata", int(activity_metadata["activity_id"].nunique())),
        ("found_subjects_in_trials", int(trial_summary["user_id"].nunique())),
        ("found_activities_in_trials", int(trial_summary["activity_id"].nunique())),
        ("found_trials", int(len(trial_summary))),
        ("load_errors", int((trial_summary["load_error"] != "").sum())),
        ("trials_with_missing_values", int(valid_trials["has_missing_values"].sum())),
        ("trials_with_non_monotonic_timestamps", int((~valid_trials["timestamps_monotonic"]).sum())),
        ("trials_not_starting_at_zero", int((~valid_trials["timestamp_starts_at_zero"]).sum())),
        (
            "trials_with_sampling_rate_deviation",
            int(
                (
                    valid_trials["estimated_sampling_hz"].sub(EXPECTED_SAMPLING_HZ).abs()
                    > SAMPLING_TOLERANCE_HZ
                ).sum()
            ),
        ),
        ("trials_with_duplicate_timestamps", int((valid_trials["n_duplicate_timestamps"] > 0).sum())),
        ("trials_with_invalid_column_count", int((~trial_summary["valid_column_count"]).sum())),
    ]
    summary = pd.DataFrame(metrics, columns=["metric", "value"])
    summary["value"] = pd.Series([value for _, value in metrics], dtype="object")
    return summary


def _generate_figures(
    trial_summary: pd.DataFrame,
    activity_counts: pd.DataFrame,
    subject_counts: pd.DataFrame,
    figures_dir: Path,
) -> None:
    valid = trial_summary.loc[trial_summary["load_error"] == ""].copy()
    if valid.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid["duration_s"], bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("Trial duration distribution")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "trial_duration_histogram.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(activity_counts["activity_id"].astype(str), activity_counts["n_trials"], color="#F58518")
    ax.set_title("Trials per activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Number of trials")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(figures_dir / "trials_per_activity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(subject_counts["user_id"].astype(str), subject_counts["n_trials"], color="#54A24B")
    ax.set_title("Trials per subject")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of trials")
    fig.tight_layout()
    fig.savefig(figures_dir / "trials_per_subject.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    valid.boxplot(column="duration_s", by="activity_id", ax=ax, grid=False, rot=90)
    ax.set_title("Trial duration by activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Duration (s)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(figures_dir / "duration_by_activity_boxplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(valid["duration_s"], valid["n_samples"], alpha=0.7, color="#E45756")
    ax.set_title("Trial duration versus number of samples")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(figures_dir / "duration_vs_samples.png", dpi=200)
    plt.close(fig)


def _build_report(
    trial_summary: pd.DataFrame,
    user_metadata: pd.DataFrame,
    activity_metadata: pd.DataFrame,
    activity_counts: pd.DataFrame,
    subject_counts: pd.DataFrame,
    data_quality_summary: pd.DataFrame,
) -> str:
    valid_trials = trial_summary.loc[trial_summary["load_error"] == ""].copy()
    load_errors = int((trial_summary["load_error"] != "").sum())
    sampling_deviation_mask = (
        valid_trials["estimated_sampling_hz"].sub(EXPECTED_SAMPLING_HZ).abs() > SAMPLING_TOLERANCE_HZ
    )

    duration_min = valid_trials["duration_s"].min()
    duration_max = valid_trials["duration_s"].max()
    duration_median = valid_trials["duration_s"].median()

    top_activity_lines = "\n".join(
        f"- Activity {int(row.activity_id):02d} ({row.activity_name}): {int(row.n_trials)} trials"
        for row in activity_counts.itertuples(index=False)
    )
    top_subject_lines = "\n".join(
        f"- Subject {int(row.user_id):02d}: {int(row.n_trials)} trials"
        for row in subject_counts.itertuples(index=False)
    )

    first_observations = [
        f"The discovered trial count is {len(trial_summary)}, compared with the README expectation of {EXPECTED_TRIAL_COUNT}.",
        (
            f"Observed trial durations range from {duration_min:.2f} s to {duration_max:.2f} s "
            f"(median {duration_median:.2f} s)."
            if not valid_trials.empty
            else "No valid trials were available to estimate trial durations."
        ),
        (
            f"The median estimated sampling rate across valid trials is "
            f"{valid_trials['estimated_sampling_hz'].median():.2f} Hz."
            if not valid_trials.empty
            else "No valid trials were available to estimate the sampling rate."
        ),
    ]

    if load_errors:
        first_observations.append(f"{load_errors} trial files could not be loaded cleanly and require inspection.")

    markdown = f"""# UMAHand dataset summary

## Dataset description

This report summarises the UMAHand dataset as a reproducible first-pass descriptive analysis focused on wrist-worn inertial traces from everyday manual activities.

## Dataset size

- Number of users found in metadata: {user_metadata['user_id'].nunique()}
- Number of activities found in metadata: {activity_metadata['activity_id'].nunique()}
- Number of trials found: {len(trial_summary)}

## README expectations versus observed data

| Item | Expected | Found |
| --- | ---: | ---: |
| Subjects | {EXPECTED_SUBJECT_COUNT} | {trial_summary['user_id'].nunique()} |
| Activities | {EXPECTED_ACTIVITY_COUNT} | {trial_summary['activity_id'].nunique()} |
| Trials | {EXPECTED_TRIAL_COUNT} | {len(trial_summary)} |
| Sampling rate (Hz) | {EXPECTED_SAMPLING_HZ:.0f} | {valid_trials['estimated_sampling_hz'].median():.2f} |

## Distribution by activity

{top_activity_lines}

## Distribution by subject

{top_subject_lines}

## Real duration range

- Minimum duration: {duration_min:.2f} s
- Maximum duration: {duration_max:.2f} s
- Median duration: {duration_median:.2f} s

## Data quality summary

- Files with missing values: {int(valid_trials['has_missing_values'].sum())}
- Files with non-monotonic timestamps: {int((~valid_trials['timestamps_monotonic']).sum())}
- Files not starting at 0 ms: {int((~valid_trials['timestamp_starts_at_zero']).sum())}
- Files with sampling frequency farther than {SAMPLING_TOLERANCE_HZ:.1f} Hz from 100 Hz: {int(sampling_deviation_mask.sum())}
- Files with incorrect column count: {int((~trial_summary['valid_column_count']).sum())}
- Files with load errors: {load_errors}

## First observations

{chr(10).join(f"- {item}" for item in first_observations)}

## Limitations

- This iteration does not analyse the example videos.
- The summary focuses on trial-level descriptives and acquisition quality, not on feature extraction or classification.
- Potential semantic grouping of activities into more habitual versus more goal-directed categories still needs a theory-driven annotation step.

## Recommended next steps

- Add reproducible feature extraction modules on top of the validated loader.
- Define an activity ontology or labels for habitual versus goal-directed task dimensions.
- Introduce train/validation/test protocols that respect subject-level separation for later classification work.
- Add lightweight tests for filename parsing, metadata loading and trace validation.
"""
    return markdown
