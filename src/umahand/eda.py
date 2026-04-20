from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HABIT_LIKE_LABEL = "habit_like"
GOAL_DIRECTED_LABEL = "goal_directed"
UNMAPPED_LABEL = "unmapped"

PROVISIONAL_ACTIVITY_GROUPS = {
    HABIT_LIKE_LABEL: [
        "Brushing teeth with a manual toothbrush",
        "Brushing teeth with an electric toothbrush",
        "Washing hands",
        "Eat soup",
        "Aplauding",
        "combing hair",
        "Sweep with a broom",
        "Writing a sentence with a keyboard",
        "Write on a sheet of paper",
        "Send a message through the whatsap application",
        "Mark a phone number on a cell phone",
        "Drinking water from a glass",
        "Putting on a pair of glasses",
        "Remove a jacket/sweatshirt",
        "Waving goodbye",
        "Nose blowing",
        "Opening and closing a door",
        "Raising and lowering a zipper",
        "Putting on a jacket/sweatshirt",
        "Putting on a shoe and tying the laces",
        "Buttoning a shirt button",
    ],
    GOAL_DIRECTED_LABEL: [
        "Cutting food",
        "Peeling a fruit",
        "Cleaning (Wiping with a cloth",
        "Fold a piece of paper",
        "Picking up an object from the floor",
        "Opening a bottle with thread",
        "Pouring water into a glass",
        "Screwing a screw",
    ],
}


@dataclass(frozen=True)
class EDAResult:
    report_path: Path
    generated_figures: list[str]
    warnings: list[str]
    habit_summary_path: Path | None
    n_trials: int
    n_subjects: int
    n_activities: int


def run_visual_eda(features_csv: Path, output_dir: Path, summary_csv: Path | None = None) -> EDAResult:
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_csv)
    summary_df = pd.read_csv(summary_csv) if summary_csv else None
    merged_df = prepare_eda_dataframe(features_df, summary_df)
    merged_df["habit_goal_group"] = merged_df["activity_name"].apply(map_activity_to_provisional_group)

    warnings_list: list[str] = []
    generated_figures: list[str] = []

    generated_figures.append(_plot_activity_trial_counts(merged_df, figures_dir / "activity_trial_counts.png"))
    generated_figures.append(_plot_subject_trial_counts(merged_df, figures_dir / "subject_trial_counts.png"))
    generated_figures.append(_plot_duration_by_activity(merged_df, figures_dir / "duration_by_activity_boxplot.png"))
    generated_figures.append(_plot_duration_histogram(merged_df, figures_dir / "duration_histogram.png"))

    intensity_specs = detect_signal_columns(
        merged_df,
        {"acc": ["acc_mag_rms", "acc_mag_mean"], "gyro": ["gyro_mag_rms", "gyro_mag_mean"]},
    )
    _append_plot_result(
        _plot_intensity_by_activity(merged_df, figures_dir / "acc_gyro_intensity_by_activity.png", intensity_specs),
        generated_figures,
        warnings_list,
    )

    variability_specs = detect_signal_columns(
        merged_df,
        {"acc": ["acc_mag_std"], "gyro": ["gyro_mag_std"]},
    )
    _append_plot_result(
        _plot_variability_by_activity(merged_df, figures_dir / "movement_variability_by_activity.png", variability_specs),
        generated_figures,
        warnings_list,
    )

    dominant_freq_specs = detect_signal_columns(
        merged_df,
        {
            "acc": ["acc_mag_dominant_frequency_hz", "acc_mag_dominant_frequency"],
            "gyro": ["gyro_mag_dominant_frequency_hz", "gyro_mag_dominant_frequency"],
        },
    )
    _append_plot_result(
        _plot_dominant_frequency_by_activity(
            merged_df, figures_dir / "dominant_frequency_by_activity.png", dominant_freq_specs
        ),
        generated_figures,
        warnings_list,
    )

    heatmap_features = resolve_feature_candidates(
        merged_df,
        [
            ("duration_s", ["duration_s"]),
            ("acc_mag_mean", ["acc_mag_mean"]),
            ("acc_mag_std", ["acc_mag_std"]),
            ("acc_mag_rms", ["acc_mag_rms"]),
            ("gyro_mag_mean", ["gyro_mag_mean"]),
            ("gyro_mag_std", ["gyro_mag_std"]),
            ("gyro_mag_rms", ["gyro_mag_rms"]),
            ("acc_mag_dominant_frequency", ["acc_mag_dominant_frequency_hz", "acc_mag_dominant_frequency"]),
            ("gyro_mag_dominant_frequency", ["gyro_mag_dominant_frequency_hz", "gyro_mag_dominant_frequency"]),
            ("acc_mag_spectral_entropy", ["acc_mag_spectral_entropy"]),
            ("gyro_mag_spectral_entropy", ["gyro_mag_spectral_entropy"]),
            ("acc_mag_peak_rate", ["acc_mag_peak_rate_hz", "acc_mag_peak_rate"]),
            ("gyro_mag_peak_rate", ["gyro_mag_peak_rate_hz", "gyro_mag_peak_rate"]),
        ],
    )
    _append_plot_result(
        _plot_activity_feature_heatmap(merged_df, figures_dir / "activity_feature_heatmap.png", heatmap_features),
        generated_figures,
        warnings_list,
    )
    _append_plot_result(
        _plot_activity_similarity(merged_df, figures_dir / "activity_similarity_clustermap.png", heatmap_features),
        generated_figures,
        warnings_list,
    )

    pca_columns = select_numeric_feature_columns(merged_df)
    pca_result = compute_pca_projection(merged_df, pca_columns)
    if pca_result is None:
        warnings_list.append("PCA figures could not be generated because no suitable numeric features were available.")
    else:
        projection_df, explained_variance = pca_result
        generated_figures.append(
            _plot_pca_scatter(
                projection_df,
                figures_dir / "pca_activity_scatter.png",
                color_column="activity_id",
                title="PCA of trial features coloured by activity",
                explained_variance=explained_variance,
            )
        )
        generated_figures.append(
            _plot_pca_scatter(
                projection_df,
                figures_dir / "pca_subject_scatter.png",
                color_column="user_id",
                title="PCA of trial features coloured by subject",
                explained_variance=explained_variance,
            )
        )
        generated_figures.append(
            _plot_pca_activity_centroids(
                projection_df,
                figures_dir / "pca_activity_centroids.png",
                explained_variance=explained_variance,
            )
        )
        generated_figures.append(
            _plot_pca_scatter(
                projection_df,
                figures_dir / "pca_habit_vs_goal_directed.png",
                color_column="habit_goal_group",
                title="PCA of trial features coloured by provisional habit/goal grouping",
                explained_variance=explained_variance,
            )
        )

    habit_features = resolve_feature_candidates(
        merged_df,
        [
            ("duration_s", ["duration_s"]),
            ("acc_mag_std", ["acc_mag_std"]),
            ("gyro_mag_std", ["gyro_mag_std"]),
            ("acc_mag_peak_rate", ["acc_mag_peak_rate_hz", "acc_mag_peak_rate"]),
            ("gyro_mag_peak_rate", ["gyro_mag_peak_rate_hz", "gyro_mag_peak_rate"]),
            ("acc_mag_spectral_entropy", ["acc_mag_spectral_entropy"]),
            ("gyro_mag_spectral_entropy", ["gyro_mag_spectral_entropy"]),
        ],
    )
    habit_summary_path = output_dir / "habit_vs_goal_summary_table.csv"
    _append_plot_result(
        _plot_habit_vs_goal_boxplots(
            merged_df, figures_dir / "habit_vs_goal_feature_boxplots.png", habit_features
        ),
        generated_figures,
        warnings_list,
    )
    if habit_features:
        build_habit_goal_summary_table(merged_df, habit_features).to_csv(habit_summary_path, index=False)
    else:
        habit_summary_path = None
        warnings_list.append("Habit vs goal summary table was not generated because the selected features were unavailable.")

    _append_plot_result(
        _plot_top_variable_features(merged_df, figures_dir / "top_variable_features.png"),
        generated_figures,
        warnings_list,
    )

    correlation_features = resolve_feature_candidates(
        merged_df,
        [
            ("duration_s", ["duration_s"]),
            ("acc_mag_mean", ["acc_mag_mean"]),
            ("acc_mag_std", ["acc_mag_std"]),
            ("acc_mag_rms", ["acc_mag_rms"]),
            ("gyro_mag_mean", ["gyro_mag_mean"]),
            ("gyro_mag_std", ["gyro_mag_std"]),
            ("gyro_mag_rms", ["gyro_mag_rms"]),
            ("acc_mag_peak_rate", ["acc_mag_peak_rate_hz", "acc_mag_peak_rate"]),
            ("gyro_mag_peak_rate", ["gyro_mag_peak_rate_hz", "gyro_mag_peak_rate"]),
            ("acc_mag_spectral_entropy", ["acc_mag_spectral_entropy"]),
            ("gyro_mag_spectral_entropy", ["gyro_mag_spectral_entropy"]),
            ("acc_mag_dominant_frequency", ["acc_mag_dominant_frequency_hz", "acc_mag_dominant_frequency"]),
            ("gyro_mag_dominant_frequency", ["gyro_mag_dominant_frequency_hz", "gyro_mag_dominant_frequency"]),
        ],
    )
    _append_plot_result(
        _plot_correlation_heatmap(
            merged_df, figures_dir / "correlation_heatmap_selected_features.png", correlation_features
        ),
        generated_figures,
        warnings_list,
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_eda_report(
            merged_df=merged_df,
            summary_df=summary_df,
            generated_figures=generated_figures,
            warnings_list=warnings_list,
            habit_features=habit_features,
            habit_summary_path=habit_summary_path,
            pca_available=pca_result is not None,
        ),
        encoding="utf-8",
    )

    return EDAResult(
        report_path=report_path,
        generated_figures=generated_figures,
        warnings=warnings_list,
        habit_summary_path=habit_summary_path,
        n_trials=int(len(merged_df)),
        n_subjects=int(merged_df["user_id"].nunique()),
        n_activities=int(merged_df["activity_id"].nunique()),
    )


def prepare_eda_dataframe(features_df: pd.DataFrame, summary_df: pd.DataFrame | None) -> pd.DataFrame:
    merged_df = features_df.copy()
    if summary_df is not None:
        join_keys = ["relative_path", "user_id", "activity_id", "trial_id"]
        extra_columns = [
            column
            for column in ["duration_s", "estimated_sampling_hz", "activity_name", "n_samples"]
            if column in summary_df.columns and column not in merged_df.columns
        ]
        if extra_columns:
            merged_df = merged_df.merge(summary_df[join_keys + extra_columns], on=join_keys, how="left")

    if "activity_label" not in merged_df.columns:
        merged_df["activity_label"] = merged_df["activity_id"].apply(lambda value: f"{int(value):02d}")
    return merged_df


def map_activity_to_provisional_group(activity_name: str) -> str:
    text = str(activity_name).lower()
    for group_name, fragments in PROVISIONAL_ACTIVITY_GROUPS.items():
        for fragment in fragments:
            if fragment.lower() in text:
                return group_name
    return UNMAPPED_LABEL


def detect_signal_columns(df: pd.DataFrame, preferred_columns: dict[str, list[str]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, candidates in preferred_columns.items():
        for candidate in candidates:
            if candidate in df.columns:
                resolved[key] = candidate
                break
    return resolved


def resolve_feature_candidates(df: pd.DataFrame, named_candidates: list[tuple[str, list[str]]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for label, candidates in named_candidates:
        for candidate in candidates:
            if candidate in df.columns:
                resolved[label] = candidate
                break
    return resolved


def select_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "relative_path",
        "activity_name",
        "activity_label",
        "handedness_label",
        "gender_label",
        "habit_goal_group",
    }
    return [
        column
        for column in df.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(df[column])
    ]


def compute_pca_projection(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray] | None:
    usable = df[numeric_columns].select_dtypes(include=[np.number]).copy()
    usable = usable.loc[:, usable.notna().all(axis=0)]
    if usable.shape[0] < 2 or usable.shape[1] < 2:
        return None

    scaled = StandardScaler().fit_transform(usable)
    transformed = PCA(n_components=2, random_state=0).fit_transform(scaled)

    projection_df = df[
        ["relative_path", "user_id", "activity_id", "activity_name", "activity_label", "habit_goal_group"]
    ].copy()
    projection_df["PC1"] = transformed[:, 0]
    projection_df["PC2"] = transformed[:, 1]
    explained_variance = PCA(n_components=2, random_state=0).fit(scaled).explained_variance_ratio_
    return projection_df, explained_variance


def build_habit_goal_summary_table(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    filtered = df.loc[df["habit_goal_group"].isin([HABIT_LIKE_LABEL, GOAL_DIRECTED_LABEL])].copy()
    rows: list[dict[str, Any]] = []
    for label, column in feature_map.items():
        grouped = filtered.groupby("habit_goal_group")[column]
        for group_name, series in grouped:
            rows.append(
                {
                    "group": group_name,
                    "feature_label": label,
                    "feature_column": column,
                    "n": int(series.notna().sum()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_eda_report(
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame | None,
    generated_figures: list[str],
    warnings_list: list[str],
    habit_features: dict[str, str],
    habit_summary_path: Path | None,
    pca_available: bool,
) -> str:
    counts_by_activity = (
        merged_df.groupby(["activity_id", "activity_name"]).size().rename("n_trials").reset_index()
        .sort_values("n_trials", ascending=False, ignore_index=True)
    )
    duration_by_activity = (
        merged_df.groupby(["activity_id", "activity_name"])["duration_s"]
        .agg(["median", "std"])
        .reset_index()
        .sort_values("median", ascending=False, ignore_index=True)
    )
    duration_min = float(merged_df["duration_s"].min())
    duration_median = float(merged_df["duration_s"].median())
    duration_max = float(merged_df["duration_s"].max())

    top_trial_activity = counts_by_activity.iloc[0]
    bottom_trial_activity = counts_by_activity.iloc[-1]
    longest_activity = duration_by_activity.iloc[0]
    most_variable_activity = duration_by_activity.sort_values("std", ascending=False, ignore_index=True).iloc[0]

    input_lines = [
        "- `trial_features.csv` was used as the primary input.",
        "- `trial_summary.csv` was used to complement metadata and duration information."
        if summary_df is not None
        else "- No summary CSV was provided; the report relied only on `trial_features.csv`.",
    ]
    warning_lines = "\n".join(f"- {item}" for item in warnings_list) if warnings_list else "- No warnings were raised."
    figure_lines = "\n".join(f"- `{name}`" for name in generated_figures)
    habit_summary_line = (
        f"- A compact group summary table was saved to `{habit_summary_path.as_posix()}`."
        if habit_summary_path is not None
        else "- No habit-vs-goal summary table could be produced."
    )

    return f"""# UMAHand visual EDA report

## Objective

This exploratory analysis turns the previously generated CSVs into interpretable visual summaries so that the structure of the dataset, the variability of activities, and the readiness for downstream classification can be assessed without fitting a predictive model.

## Inputs used

{chr(10).join(input_lines)}

## Dataset summary

- Trials analysed: {len(merged_df)}
- Subjects represented: {merged_df['user_id'].nunique()}
- Activities represented: {merged_df['activity_id'].nunique()}
- Trial duration range: {duration_min:.2f} s / {duration_median:.2f} s / {duration_max:.2f} s (min / median / max)
- Activity with the most trials: Activity {int(top_trial_activity['activity_id']):02d} ({top_trial_activity['activity_name']}) with {int(top_trial_activity['n_trials'])} trials
- Activity with the fewest trials: Activity {int(bottom_trial_activity['activity_id']):02d} ({bottom_trial_activity['activity_name']}) with {int(bottom_trial_activity['n_trials'])} trials
- Longest typical activity by median duration: Activity {int(longest_activity['activity_id']):02d} ({longest_activity['activity_name']})
- Most duration-variable activity: Activity {int(most_variable_activity['activity_id']):02d} ({most_variable_activity['activity_name']})

## How to interpret the figures

### Dataset structure

The count plots and duration plots show whether the dataset is balanced enough for a first classification baseline and whether some activities are intrinsically more variable or longer than others. Strong imbalance or broad duration spread suggests we should evaluate models with per-class metrics, not only accuracy.

### Movement intensity and variability

{infer_intensity_notes(merged_df)}

### Activity comparison and similarity

The heatmap standardises feature means by activity, so warm and cool colours should be read as relative deviations from the across-activity average rather than raw physical units. The similarity matrix helps identify activities that may confuse a classifier because they occupy nearby regions in feature space at the activity-average level.

### PCA observations

{infer_pca_notes(merged_df, pca_available)}

### Provisional habit-like vs goal-directed grouping

This split is conceptual and editable, not ground truth. It is useful as a heuristic lens for exploration only.

{infer_habit_notes(merged_df, habit_features)}

{habit_summary_line}

## Figures generated

{figure_lines}

## Warnings and limitations

{warning_lines}

- PCA is descriptive only and compresses many features into two axes; overlap in PCA does not imply that classification is impossible.
- The provisional habit/goal grouping depends on substring matching of activity names and should be reviewed conceptually before any formal analysis.
- High feature correlation should be expected because many descriptors are built from the same underlying magnitudes.

## Recommended next phase

- Train a first baseline classifier using the tabular features, with `GroupKFold` or another subject-wise split to avoid leakage across participants.
- Inspect per-activity confusion matrices instead of only aggregate accuracy.
- Compare a simple linear baseline with a tree-based model to contrast interpretability and nonlinear capacity.
- Use feature importance or permutation importance only after validating a leak-free evaluation protocol.
- Revisit the conceptual habit-like versus goal-directed grouping before treating it as an analysis target.
"""


def infer_intensity_notes(df: pd.DataFrame) -> str:
    notes: list[str] = []
    if "acc_mag_rms" in df.columns:
        top_acc = df.groupby(["activity_id", "activity_name"])["acc_mag_rms"].median().sort_values(ascending=False).head(3)
        notes.append(
            "Acceleration magnitude appears highest in median RMS terms for "
            + ", ".join(f"Activity {int(idx[0]):02d}" for idx in top_acc.index)
            + "."
        )
    if "gyro_mag_rms" in df.columns:
        top_gyro = df.groupby(["activity_id", "activity_name"])["gyro_mag_rms"].median().sort_values(ascending=False).head(3)
        notes.append(
            "Angular-velocity magnitude appears highest in median RMS terms for "
            + ", ".join(f"Activity {int(idx[0]):02d}" for idx in top_gyro.index)
            + "."
        )
    if "acc_mag_std" in df.columns and "gyro_mag_std" in df.columns:
        notes.append(
            "Comparing `acc_mag_std` and `gyro_mag_std` by activity helps distinguish smoother actions from burstier or more articulated ones."
        )
    return " ".join(notes) if notes else "The available feature columns did not allow a detailed intensity summary."


def infer_pca_notes(df: pd.DataFrame, pca_available: bool) -> str:
    if not pca_available:
        return "PCA could not be generated from the available numeric features."
    subject_spread = df.groupby("user_id")["duration_s"].median().std()
    activity_spread = df.groupby("activity_id")["duration_s"].median().std()
    if activity_spread >= subject_spread:
        return "If the PCA coloured by activity looks more structured than the PCA coloured by subject, that supports the idea that the feature space captures task differences rather than only participant identity."
    return "If the PCA coloured by subject looks strongly structured, participant-specific style may be a major source of variance and future evaluation should be strictly subject-wise."


def infer_habit_notes(df: pd.DataFrame, habit_features: dict[str, str]) -> str:
    filtered = df.loc[df["habit_goal_group"].isin([HABIT_LIKE_LABEL, GOAL_DIRECTED_LABEL])].copy()
    if filtered.empty or not habit_features:
        return "The provisional grouping could not be summarised because the selected features were unavailable."

    notes: list[str] = []
    for label in ["duration_s", "acc_mag_std", "gyro_mag_spectral_entropy"]:
        column = habit_features.get(label)
        if column is None:
            continue
        medians = filtered.groupby("habit_goal_group")[column].median()
        if len(medians) == 2:
            notes.append(f"`{medians.idxmax()}` has the larger median `{label}`.")
    notes.append(
        "These differences are descriptive only and should not be interpreted as evidence for cognitive mechanisms without a stronger task ontology and formal modelling."
    )
    return " ".join(notes)


def _append_plot_result(result: tuple[str | None, str | None], figures: list[str], warnings_list: list[str]) -> None:
    figure_name, warning_message = result
    if figure_name:
        figures.append(figure_name)
    if warning_message:
        warnings_list.append(warning_message)


def _plot_activity_trial_counts(df: pd.DataFrame, output_path: Path) -> str:
    counts = (
        df.groupby(["activity_id", "activity_name"]).size().rename("n_trials").reset_index()
        .sort_values("activity_id", ignore_index=True)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts["activity_id"].astype(str), counts["n_trials"], color="#4C78A8")
    ax.set_title("Number of trials by activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def _plot_subject_trial_counts(df: pd.DataFrame, output_path: Path) -> str:
    counts = df.groupby("user_id").size().rename("n_trials").reset_index().sort_values("user_id", ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts["user_id"].astype(str), counts["n_trials"], color="#54A24B")
    ax.set_title("Number of trials by subject")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def _plot_duration_by_activity(df: pd.DataFrame, output_path: Path) -> str:
    ordered = df.groupby("activity_id")["duration_s"].median().sort_values().index.tolist()
    data = [df.loc[df["activity_id"] == activity_id, "duration_s"].dropna().to_numpy() for activity_id in ordered]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.boxplot(data, tick_labels=[f"{activity_id:02d}" for activity_id in ordered], showfliers=False)
    ax.set_title("Trial duration by activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Duration (s)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def _plot_duration_histogram(df: pd.DataFrame, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["duration_s"].dropna(), bins=30, color="#F58518", edgecolor="white")
    ax.set_title("Distribution of trial durations")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def _plot_intensity_by_activity(df: pd.DataFrame, output_path: Path, specs: dict[str, str]) -> tuple[str | None, str | None]:
    if len(specs) < 2:
        return None, "Could not generate `acc_gyro_intensity_by_activity.png` because suitable acceleration or gyroscope intensity columns were not found."

    grouped = df.groupby("activity_id")[[specs["acc"], specs["gyro"]]].median().sort_index()
    x = np.arange(len(grouped))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, grouped[specs["acc"]], width=width, label=specs["acc"], color="#4C78A8")
    ax.bar(x + width / 2, grouped[specs["gyro"]], width=width, label=specs["gyro"], color="#E45756")
    ax.set_title("Median movement intensity by activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Median feature value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{activity_id:02d}" for activity_id in grouped.index], rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_variability_by_activity(df: pd.DataFrame, output_path: Path, specs: dict[str, str]) -> tuple[str | None, str | None]:
    if len(specs) < 2:
        return None, "Could not generate `movement_variability_by_activity.png` because suitable variability columns were not found."

    ordered = df.groupby("activity_id")[specs["acc"]].median().sort_values().index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, key in zip(axes, ["acc", "gyro"]):
        data = [df.loc[df["activity_id"] == activity_id, specs[key]].dropna().to_numpy() for activity_id in ordered]
        ax.boxplot(data, tick_labels=[f"{activity_id:02d}" for activity_id in ordered], showfliers=False)
        ax.set_ylabel(specs[key])
        ax.set_title(f"{specs[key]} by activity")
    axes[-1].set_xlabel("Activity ID")
    axes[-1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_dominant_frequency_by_activity(df: pd.DataFrame, output_path: Path, specs: dict[str, str]) -> tuple[str | None, str | None]:
    if len(specs) < 2:
        return None, "Could not generate `dominant_frequency_by_activity.png` because dominant-frequency columns were not found."

    ordered = df.groupby("activity_id")[specs["acc"]].median().sort_values().index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, key in zip(axes, ["acc", "gyro"]):
        data = [df.loc[df["activity_id"] == activity_id, specs[key]].dropna().to_numpy() for activity_id in ordered]
        ax.boxplot(data, tick_labels=[f"{activity_id:02d}" for activity_id in ordered], showfliers=False)
        ax.set_ylabel(specs[key])
        ax.set_title(f"{specs[key]} by activity")
    axes[-1].set_xlabel("Activity ID")
    axes[-1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_activity_feature_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    feature_map: dict[str, str],
) -> tuple[str | None, str | None]:
    if len(feature_map) < 2:
        return None, "Could not generate `activity_feature_heatmap.png` because too few interpretable features were available."

    grouped = df.groupby("activity_id")[[column for column in feature_map.values()]].mean().sort_index()
    scaled_values = StandardScaler().fit_transform(grouped)
    scaled = pd.DataFrame(scaled_values, index=grouped.index, columns=list(feature_map.keys()))

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title("Standardised activity-level feature heatmap")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Activity ID")
    ax.set_xticks(np.arange(len(scaled.columns)))
    ax.set_xticklabels(scaled.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels([f"{activity_id:02d}" for activity_id in scaled.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_activity_similarity(
    df: pd.DataFrame,
    output_path: Path,
    feature_map: dict[str, str],
) -> tuple[str | None, str | None]:
    if len(feature_map) < 2:
        return None, "Could not generate `activity_similarity_clustermap.png` because too few interpretable features were available."

    grouped = df.groupby("activity_id")[[column for column in feature_map.values()]].mean().sort_index()
    scaled = StandardScaler().fit_transform(grouped)
    corr = np.corrcoef(scaled)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    labels = [f"{activity_id:02d}" for activity_id in grouped.index]
    ax.set_title("Activity similarity matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_pca_scatter(
    projection_df: pd.DataFrame,
    output_path: Path,
    color_column: str,
    title: str,
    explained_variance: np.ndarray,
) -> str:
    categories = projection_df[color_column].astype(str)
    unique_categories = sorted(categories.unique())
    cmap = plt.get_cmap("tab20", len(unique_categories))

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, category in enumerate(unique_categories):
        subset = projection_df.loc[categories == category]
        ax.scatter(subset["PC1"], subset["PC2"], s=28, alpha=0.75, color=cmap(idx), label=category)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% var)")
    if len(unique_categories) <= 15:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def _plot_pca_activity_centroids(projection_df: pd.DataFrame, output_path: Path, explained_variance: np.ndarray) -> str:
    centroids = (
        projection_df.groupby(["activity_id", "activity_name"])[["PC1", "PC2"]].mean().reset_index()
        .sort_values("activity_id", ignore_index=True)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(centroids["PC1"], centroids["PC2"], s=50, color="#4C78A8")
    for row in centroids.itertuples(index=False):
        ax.text(row.PC1, row.PC2, f"{int(row.activity_id):02d}", fontsize=8, ha="left", va="bottom")
    ax.set_title("Activity centroids in PCA space")
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% var)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def _plot_habit_vs_goal_boxplots(
    df: pd.DataFrame,
    output_path: Path,
    feature_map: dict[str, str],
) -> tuple[str | None, str | None]:
    filtered = df.loc[df["habit_goal_group"].isin([HABIT_LIKE_LABEL, GOAL_DIRECTED_LABEL])].copy()
    if filtered.empty or not feature_map:
        return None, "Could not generate `habit_vs_goal_feature_boxplots.png` because the provisional group labels or selected features were unavailable."

    labels = list(feature_map.keys())
    n_features = len(labels)
    fig, axes = plt.subplots(n_features, 1, figsize=(9, max(3 * n_features, 6)), sharex=True)
    if n_features == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        column = feature_map[label]
        data = [
            filtered.loc[filtered["habit_goal_group"] == HABIT_LIKE_LABEL, column].dropna().to_numpy(),
            filtered.loc[filtered["habit_goal_group"] == GOAL_DIRECTED_LABEL, column].dropna().to_numpy(),
        ]
        ax.boxplot(data, tick_labels=[HABIT_LIKE_LABEL, GOAL_DIRECTED_LABEL], showfliers=False)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by provisional group")
    axes[-1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_top_variable_features(df: pd.DataFrame, output_path: Path) -> tuple[str | None, str | None]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return None, "Could not generate `top_variable_features.png` because no numeric columns were available."

    variability = []
    for column in numeric.columns:
        mean_abs = abs(float(numeric[column].mean()))
        std = float(numeric[column].std(ddof=1))
        if np.isfinite(std):
            variability.append((column, std / (mean_abs + 1e-8)))

    top = sorted(variability, key=lambda item: item[1], reverse=True)[:20]
    if not top:
        return None, "Could not generate `top_variable_features.png` because feature variability scores could not be computed."

    labels, scores = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(labels)), scores, color="#E45756")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Top variable numeric features")
    ax.set_xlabel("Relative variability score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None


def _plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    feature_map: dict[str, str],
) -> tuple[str | None, str | None]:
    if len(feature_map) < 2:
        return None, "Could not generate `correlation_heatmap_selected_features.png` because too few selected features were available."

    selected = df[[column for column in feature_map.values()]].copy()
    selected.columns = list(feature_map.keys())
    corr = selected.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation heatmap for selected interpretable features")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name, None
