from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ACC_COLUMNS, EXPECTED_SAMPLING_HZ, GYRO_COLUMNS, MAG_COLUMNS, SENSOR_COLUMNS
from .data_loading import find_trace_files, load_trace_csv, parse_trace_filename
from .metadata import load_activity_metadata, load_user_metadata

DERIVED_SIGNAL_SPECS = {
    "acc_mag": ACC_COLUMNS,
    "gyro_mag": GYRO_COLUMNS,
    "mag_mag": MAG_COLUMNS,
}

FEATURE_SIGNALS = SENSOR_COLUMNS + list(DERIVED_SIGNAL_SPECS.keys())
STATIC_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "range",
    "rms",
    "energy",
    "skewness",
    "kurtosis",
]
KINEMATIC_SIGNALS = ["acc_mag", "gyro_mag"]
KINEMATIC_FEATURE_NAMES = [
    "mean_abs_derivative",
    "std_derivative",
    "max_abs_derivative",
    "mean_abs_jerk",
    "std_jerk",
    "n_peaks",
    "peak_rate_hz",
]
SPECTRAL_FEATURE_NAMES = [
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
    "total_spectral_power",
    "low_band_power",
    "mid_band_power",
    "high_band_power",
    "low_band_power_ratio",
    "mid_band_power_ratio",
    "high_band_power_ratio",
]


@dataclass(frozen=True)
class FeatureExtractionResult:
    trial_features: pd.DataFrame
    feature_summary: pd.DataFrame
    report_path: Path
    warnings: list[str]


def build_feature_dataset(data_root: Path, output_dir: Path) -> FeatureExtractionResult:
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metadata = load_user_metadata(data_root)
    activity_metadata = load_activity_metadata(data_root)
    trace_files = find_trace_files(data_root)

    activity_lookup = dict(activity_metadata[["activity_id", "activity_name"]].itertuples(index=False, name=None))
    user_lookup = user_metadata.set_index("user_id").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for path in trace_files:
        info = parse_trace_filename(path, data_root=data_root)
        trace_df = load_trace_csv(path)
        rows.append(
            extract_trial_features(
                trace_df=trace_df,
                relative_path=info.relative_path.as_posix(),
                user_id=info.user_id,
                activity_id=info.activity_id,
                trial_id=info.trial_id,
                activity_name=activity_lookup.get(info.activity_id, "unknown_activity"),
                user_metadata=user_lookup.get(info.user_id, {}),
            )
        )

    trial_features = pd.DataFrame(rows).sort_values(
        ["user_id", "activity_id", "trial_id", "relative_path"],
        ignore_index=True,
    )
    feature_summary = build_feature_summary(trial_features)

    trial_features.to_csv(output_dir / "trial_features.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_feature_report(
            trial_features=trial_features,
            feature_summary=feature_summary,
            warnings_list=warnings_list,
        ),
        encoding="utf-8",
    )

    return FeatureExtractionResult(
        trial_features=trial_features,
        feature_summary=feature_summary,
        report_path=report_path,
        warnings=warnings_list,
    )


def extract_trial_features(
    trace_df: pd.DataFrame,
    relative_path: str,
    user_id: int,
    activity_id: int,
    trial_id: int,
    activity_name: str,
    user_metadata: dict[str, Any],
) -> dict[str, Any]:
    derived_signals = compute_derived_signals(trace_df)
    timestamps = trace_df["Timestamp"].to_numpy(dtype=float)
    sampling_hz, duration_s = estimate_sampling_properties(timestamps)

    row: dict[str, Any] = {
        "relative_path": relative_path,
        "user_id": user_id,
        "activity_id": activity_id,
        "trial_id": trial_id,
        "activity_name": activity_name,
        "n_samples": int(len(trace_df)),
        "duration_s": duration_s,
        "estimated_sampling_hz": sampling_hz,
        "handedness_label": user_metadata.get("handedness_label"),
        "gender_label": user_metadata.get("gender_label"),
        "age_years": user_metadata.get("age_years"),
        "weight_kg": user_metadata.get("weight_kg"),
        "height_cm": user_metadata.get("height_cm"),
    }

    for signal_name in SENSOR_COLUMNS:
        row.update(_prefix_feature_names(signal_name, compute_basic_statistics(trace_df[signal_name].to_numpy(dtype=float))))

    for signal_name, values in derived_signals.items():
        row.update(_prefix_feature_names(signal_name, compute_basic_statistics(values)))

    effective_sampling_hz = sampling_hz if pd.notna(sampling_hz) and sampling_hz > 0 else EXPECTED_SAMPLING_HZ
    for signal_name in KINEMATIC_SIGNALS:
        signal = derived_signals[signal_name]
        row.update(_prefix_feature_names(signal_name, compute_dynamic_features(signal, effective_sampling_hz)))
        row.update(_prefix_feature_names(signal_name, compute_spectral_features(signal, effective_sampling_hz)))

    return row


def compute_derived_signals(trace_df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        signal_name: np.sqrt(np.square(trace_df[base_columns].to_numpy(dtype=float)).sum(axis=1))
        for signal_name, base_columns in DERIVED_SIGNAL_SPECS.items()
    }


def estimate_sampling_properties(timestamps: np.ndarray) -> tuple[float, float]:
    if timestamps.size < 2:
        return np.nan, np.nan

    dt_ms = np.diff(timestamps)
    finite_dt = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
    duration_ms = timestamps[-1] - timestamps[0]
    duration_s = float(duration_ms / 1000.0) if np.isfinite(duration_ms) else np.nan
    if finite_dt.size == 0:
        return np.nan, duration_s

    median_dt_ms = float(np.median(finite_dt))
    sampling_hz = 1000.0 / median_dt_ms if not math.isclose(median_dt_ms, 0.0) else np.nan
    return sampling_hz, duration_s


def compute_basic_statistics(values: np.ndarray) -> dict[str, float]:
    clean = _clean_signal(values)
    if clean.size == 0:
        return {name: np.nan for name in STATIC_FEATURE_NAMES}

    centered = clean - clean.mean()
    std = float(clean.std(ddof=1)) if clean.size > 1 else np.nan
    q75, q25 = np.percentile(clean, [75, 25])
    rms = float(np.sqrt(np.mean(np.square(clean))))
    energy = float(np.mean(np.square(clean)))

    if clean.size > 2:
        pop_std = clean.std(ddof=0)
        if pop_std > 0:
            skewness = float(np.mean(np.power(centered / pop_std, 3)))
        else:
            skewness = 0.0
    else:
        skewness = np.nan

    if clean.size > 3:
        pop_std = clean.std(ddof=0)
        if pop_std > 0:
            kurtosis = float(np.mean(np.power(centered / pop_std, 4)) - 3.0)
        else:
            kurtosis = 0.0
    else:
        kurtosis = np.nan

    return {
        "mean": float(clean.mean()),
        "std": std,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(np.median(clean)),
        "iqr": float(q75 - q25),
        "range": float(clean.max() - clean.min()),
        "rms": rms,
        "energy": energy,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def compute_dynamic_features(values: np.ndarray, sampling_hz: float) -> dict[str, float]:
    clean = _clean_signal(values)
    if clean.size < 2 or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        return {name: np.nan for name in KINEMATIC_FEATURE_NAMES}

    derivative = np.diff(clean) * sampling_hz
    jerk = np.diff(derivative) * sampling_hz if derivative.size >= 2 else np.array([], dtype=float)
    peaks = count_simple_peaks(clean)
    duration_s = clean.size / sampling_hz

    return {
        "mean_abs_derivative": float(np.mean(np.abs(derivative))),
        "std_derivative": float(np.std(derivative, ddof=1)) if derivative.size > 1 else np.nan,
        "max_abs_derivative": float(np.max(np.abs(derivative))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))) if jerk.size else np.nan,
        "std_jerk": float(np.std(jerk, ddof=1)) if jerk.size > 1 else np.nan,
        "n_peaks": float(peaks),
        "peak_rate_hz": float(peaks / duration_s) if duration_s > 0 else np.nan,
    }


def compute_spectral_features(values: np.ndarray, sampling_hz: float) -> dict[str, float]:
    clean = _clean_signal(values)
    if clean.size < 4 or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        return {name: np.nan for name in SPECTRAL_FEATURE_NAMES}

    centered = clean - clean.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sampling_hz)
    power = np.abs(spectrum) ** 2

    if power.size == 0:
        return {name: np.nan for name in SPECTRAL_FEATURE_NAMES}

    if power.size > 1:
        freqs_no_dc = freqs[1:]
        power_no_dc = power[1:]
    else:
        freqs_no_dc = freqs
        power_no_dc = power

    total_power = float(power_no_dc.sum()) if power_no_dc.size else 0.0
    if power_no_dc.size and total_power > 0:
        dominant_frequency_hz = float(freqs_no_dc[np.argmax(power_no_dc)])
        spectral_centroid_hz = float(np.sum(freqs_no_dc * power_no_dc) / total_power)
        normalized_power = power_no_dc / total_power
        spectral_entropy = float(
            -np.sum(normalized_power * np.log2(np.clip(normalized_power, 1e-12, None))) / np.log2(len(normalized_power))
        )
    else:
        dominant_frequency_hz = np.nan
        spectral_centroid_hz = np.nan
        spectral_entropy = np.nan

    low_power = _band_power(freqs_no_dc, power_no_dc, 0.1, 2.0)
    mid_power = _band_power(freqs_no_dc, power_no_dc, 2.0, 5.0)
    high_power = _band_power(freqs_no_dc, power_no_dc, 5.0, 15.0)

    return {
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_entropy": spectral_entropy,
        "total_spectral_power": total_power,
        "low_band_power": low_power,
        "mid_band_power": mid_power,
        "high_band_power": high_power,
        "low_band_power_ratio": float(low_power / total_power) if total_power > 0 else np.nan,
        "mid_band_power_ratio": float(mid_power / total_power) if total_power > 0 else np.nan,
        "high_band_power_ratio": float(high_power / total_power) if total_power > 0 else np.nan,
    }


def count_simple_peaks(values: np.ndarray) -> int:
    clean = _clean_signal(values)
    if clean.size < 3:
        return 0

    centered = clean - clean.mean()
    threshold = centered.std(ddof=0) * 0.5
    peaks = (centered[1:-1] > centered[:-2]) & (centered[1:-1] > centered[2:]) & (centered[1:-1] > threshold)
    return int(np.sum(peaks))


def build_feature_summary(trial_features: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = {
        "relative_path",
        "user_id",
        "activity_id",
        "trial_id",
        "activity_name",
        "n_samples",
        "duration_s",
        "estimated_sampling_hz",
        "handedness_label",
        "gender_label",
        "age_years",
        "weight_kg",
        "height_cm",
    }
    feature_columns = [column for column in trial_features.columns if column not in metadata_columns]

    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        series = pd.to_numeric(trial_features[column], errors="coerce")
        rows.append(
            {
                "feature_name": column,
                "n_missing_values": int(series.isna().sum()),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else np.nan,
                "min": float(series.min()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name", ignore_index=True)


def build_feature_report(
    trial_features: pd.DataFrame,
    feature_summary: pd.DataFrame,
    warnings_list: list[str],
) -> str:
    missing_features = feature_summary.loc[feature_summary["n_missing_values"] > 0, "feature_name"].tolist()
    total_features = len(feature_summary)
    feature_groups = [
        "Per-signal temporal statistics for raw axes, barometer and derived magnitudes.",
        "Dynamic features for acc_mag and gyro_mag based on discrete derivatives and jerk.",
        "Frequency-domain features for acc_mag and gyro_mag based on FFT power summaries.",
    ]

    warning_lines = "\n".join(f"- {item}" for item in warnings_list) if warnings_list else "- No extraction warnings were recorded."
    missing_lines = "\n".join(f"- {name}" for name in missing_features[:50]) if missing_features else "- None."

    return f"""# UMAHand trial feature extraction

## Summary

- Number of trials processed: {len(trial_features)}
- Number of generated feature columns: {total_features}

## Feature groups

{chr(10).join(f"- {item}" for item in feature_groups)}

## Features with missing values

{missing_lines}

## Warnings

{warning_lines}

## Recommended next steps

- Standardise or robust-scale features before training classification baselines.
- Add subject-wise split logic to avoid leakage in future modelling.
- Inspect highly correlated features and consider dimensionality reduction or grouped feature selection.
- Add automated tests for very short traces to lock in NaN behavior for derivative and spectral features.
"""


def _band_power(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    if freqs.size == 0 or power.size == 0:
        return 0.0
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(power[mask].sum()) if np.any(mask) else 0.0


def _clean_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _prefix_feature_names(prefix: str, feature_dict: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{feature_name}": value for feature_name, value in feature_dict.items()}
