from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HABIT_LABEL = "habit_like"
GOAL_LABEL = "goal_directed"
AMBIGUOUS_LABEL = "ambiguous"

GROUPING_REVIEW_RULES: dict[int, dict[str, str]] = {
    1: {"group": HABIT_LABEL, "confidence": "high", "notes": "Overlearned hygiene routine with repetitive wrist motion."},
    2: {"group": HABIT_LABEL, "confidence": "high", "notes": "Same hygiene routine as activity 1, differing mainly by tool dynamics."},
    3: {"group": HABIT_LABEL, "confidence": "high", "notes": "Canonical everyday self-care sequence with repetitive sensorimotor structure."},
    4: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Eating with a spoon is everyday and routine, though still object-directed."},
    5: {"group": GOAL_LABEL, "confidence": "high", "notes": "Bimanual object manipulation with clear external goal constraints."},
    6: {"group": GOAL_LABEL, "confidence": "high", "notes": "Fine object manipulation with sustained tool use and explicit goal."},
    7: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Rhythmic stereotyped action, not strongly goal-directed in an object sense."},
    8: {"group": HABIT_LABEL, "confidence": "high", "notes": "Routine grooming action with stable repetitive kinematics."},
    9: {"group": GOAL_LABEL, "confidence": "medium", "notes": "Task is repetitive but externally focused on cleaning an object or surface."},
    10: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Sweeping can be highly practised and rhythmic, despite its external goal."},
    11: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Handwriting is practised and automatized for many adults, though symbolically goal-oriented."},
    12: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Typing is a strongly practised repetitive action in everyday behaviour."},
    13: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Texting is routine and overlearned, but still involves a communication goal."},
    14: {"group": GOAL_LABEL, "confidence": "high", "notes": "Multi-step object transformation task with explicit state-change target."},
    15: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Phone dialling plus lifting to the ear mixes practised routine and explicit action sequencing."},
    16: {"group": GOAL_LABEL, "confidence": "medium", "notes": "Object pickup involves clear external target and transport goal."},
    17: {"group": GOAL_LABEL, "confidence": "high", "notes": "Bottle opening is a constrained object-manipulation action with explicit outcome."},
    18: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Highly familiar consumption action, though still object-mediated."},
    19: {"group": GOAL_LABEL, "confidence": "medium", "notes": "Pouring requires controlled object interaction to reach a target state."},
    20: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Putting on glasses is everyday and stereotyped for frequent users."},
    21: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Putting on a jacket is habitual for many people but mechanically multi-step and context-sensitive."},
    22: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Removing a jacket mirrors dressing actions and is not cleanly separable conceptually."},
    23: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Shoe tying blends a routine context with explicit fine sequential control."},
    24: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Everyday social gesture with recognisable stereotyped dynamics."},
    25: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Well-practised self-care action, albeit brief and context-triggered."},
    26: {"group": HABIT_LABEL, "confidence": "medium", "notes": "Door opening is frequent and often highly automatized in natural behaviour."},
    27: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Buttoning is a dressing action with strong fine-control demands and possible strategy variability."},
    28: {"group": AMBIGUOUS_LABEL, "confidence": "low", "notes": "Zipper manipulation is routine but also strongly object-governed and fine-motor."},
    29: {"group": GOAL_LABEL, "confidence": "high", "notes": "Screwing is a constrained object-directed manipulation with explicit intended outcome."},
}

REQUESTED_FEATURE_CANDIDATES: list[tuple[str, list[str], str]] = [
    ("duration_s", ["duration_s"], "Primary duration descriptor."),
    ("n_samples", ["n_samples"], "Control for trial size and recording length."),
    ("acc_mag_mean_or_rms", ["acc_mag_rms", "acc_mag_mean"], "Acceleration magnitude level."),
    ("acc_mag_std", ["acc_mag_std"], "Acceleration variability."),
    ("gyro_mag_mean_or_rms", ["gyro_mag_rms", "gyro_mag_mean"], "Gyroscope magnitude level."),
    ("gyro_mag_std", ["gyro_mag_std"], "Gyroscope variability."),
    ("acc_mag_n_peaks", ["acc_mag_n_peaks"], "Count of acceleration peaks."),
    ("gyro_mag_n_peaks", ["gyro_mag_n_peaks"], "Count of gyroscope peaks."),
    ("acc_mag_peak_rate", ["acc_mag_peak_rate_hz", "acc_mag_peak_rate"], "Acceleration peak rate."),
    ("gyro_mag_peak_rate", ["gyro_mag_peak_rate_hz", "gyro_mag_peak_rate"], "Gyroscope peak rate."),
    ("acc_mag_mean_abs_derivative", ["acc_mag_mean_abs_derivative"], "Acceleration derivative magnitude."),
    ("gyro_mag_mean_abs_derivative", ["gyro_mag_mean_abs_derivative"], "Gyroscope derivative magnitude."),
    ("acc_mag_mean_abs_jerk", ["acc_mag_mean_abs_jerk"], "Acceleration jerk magnitude."),
    ("gyro_mag_mean_abs_jerk", ["gyro_mag_mean_abs_jerk"], "Gyroscope jerk magnitude."),
    ("acc_mag_dominant_frequency", ["acc_mag_dominant_frequency_hz", "acc_mag_dominant_frequency"], "Acceleration dominant tempo."),
    ("gyro_mag_dominant_frequency", ["gyro_mag_dominant_frequency_hz", "gyro_mag_dominant_frequency"], "Gyroscope dominant tempo."),
    ("acc_mag_spectral_entropy", ["acc_mag_spectral_entropy"], "Acceleration spectral complexity."),
    ("gyro_mag_spectral_entropy", ["gyro_mag_spectral_entropy"], "Gyroscope spectral complexity."),
    ("acc_mag_low_band_power", ["acc_mag_low_band_power"], "Low-frequency acceleration power."),
    ("acc_mag_mid_band_power", ["acc_mag_mid_band_power"], "Mid-frequency acceleration power."),
    ("acc_mag_high_band_power", ["acc_mag_high_band_power"], "High-frequency acceleration power."),
    ("gyro_mag_low_band_power", ["gyro_mag_low_band_power"], "Low-frequency gyroscope power."),
    ("gyro_mag_mid_band_power", ["gyro_mag_mid_band_power"], "Mid-frequency gyroscope power."),
    ("gyro_mag_high_band_power", ["gyro_mag_high_band_power"], "High-frequency gyroscope power."),
]


@dataclass(frozen=True)
class HabitGoalProfilingResult:
    report_path: Path
    generated_tables: list[str]
    generated_figures: list[str]
    group_counts: pd.DataFrame
    top_effects: pd.DataFrame


def run_habit_goal_profiling(
    features_csv: Path,
    families_csv: Path,
    output_dir: Path,
    habit_summary_csv: Path | None = None,
    activity_performance_csv: Path | None = None,
) -> HabitGoalProfilingResult:
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_csv)
    families_df = pd.read_csv(families_csv)
    habit_summary_df = pd.read_csv(habit_summary_csv) if habit_summary_csv and Path(habit_summary_csv).exists() else None
    activity_performance_df = pd.read_csv(activity_performance_csv) if activity_performance_csv and Path(activity_performance_csv).exists() else None

    grouping_review = build_activity_grouping_review(families_df)
    selected_features_used = build_selected_features_used(features_df)
    merged_df = features_df.merge(
        grouping_review[
            [
                "activity_id",
                "activity_name",
                "provisional_motor_family",
                "habit_goal_group",
                "grouping_confidence",
                "grouping_notes",
            ]
        ],
        on=["activity_id", "activity_name"],
        how="left",
    )
    selected_feature_map = {
        row.requested_feature_concept: row.actual_column_used
        for row in selected_features_used.itertuples(index=False)
        if row.status == "matched"
    }

    subset_a = merged_df.loc[merged_df["habit_goal_group"].isin([HABIT_LABEL, GOAL_LABEL])].copy()
    subset_b = merged_df.copy()

    group_feature_summary = build_group_feature_summary(subset_a, selected_feature_map)
    effect_sizes = build_effect_sizes(subset_a, selected_feature_map)
    statistical_tests = build_statistical_tests(subset_a, selected_feature_map)
    family_group_summary = build_family_group_summary(merged_df)
    family_feature_summary = build_family_feature_summary(merged_df, selected_feature_map)
    family_effect_sizes = build_family_effect_sizes(subset_a, selected_feature_map)

    generated_tables: list[str] = []
    save_csv(grouping_review, output_dir / "activity_grouping_review.csv", generated_tables)
    save_csv(selected_features_used, output_dir / "selected_features_used.csv", generated_tables)
    save_csv(group_feature_summary, output_dir / "group_feature_summary.csv", generated_tables)
    save_csv(effect_sizes, output_dir / "effect_sizes.csv", generated_tables)
    save_csv(statistical_tests, output_dir / "statistical_tests.csv", generated_tables)
    save_csv(family_group_summary, output_dir / "family_group_summary.csv", generated_tables)
    save_csv(family_feature_summary, output_dir / "family_feature_summary.csv", generated_tables)
    save_csv(family_effect_sizes, output_dir / "family_effect_sizes.csv", generated_tables)

    generated_figures = [
        plot_habit_goal_group_sizes(grouping_review, merged_df, figures_dir / "habit_goal_group_sizes.png"),
        plot_habit_goal_feature_boxplots(subset_a, selected_feature_map, figures_dir / "habit_goal_feature_boxplots.png"),
        plot_effect_sizes_barplot(effect_sizes, figures_dir / "effect_sizes_barplot.png"),
        plot_group_means_heatmap(subset_a, selected_feature_map, figures_dir / "habit_goal_heatmap_group_means.png"),
        plot_pca(subset_b, selected_feature_map, figures_dir / "pca_habit_goal_with_ambiguous.png", include_ambiguous=True),
        plot_pca(subset_a, selected_feature_map, figures_dir / "pca_habit_goal_without_ambiguous.png", include_ambiguous=False),
        plot_family_stratified_effects(family_effect_sizes, figures_dir / "family_stratified_effects.png"),
        plot_family_group_counts(family_group_summary, figures_dir / "family_group_counts.png"),
    ]

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_report(
            grouping_review=grouping_review,
            selected_features_used=selected_features_used,
            group_feature_summary=group_feature_summary,
            effect_sizes=effect_sizes,
            statistical_tests=statistical_tests,
            family_group_summary=family_group_summary,
            family_effect_sizes=family_effect_sizes,
            generated_tables=generated_tables,
            generated_figures=generated_figures,
            habit_summary_df=habit_summary_df,
            activity_performance_df=activity_performance_df,
        ),
        encoding="utf-8",
    )

    group_counts = summarize_group_counts(grouping_review, merged_df)
    return HabitGoalProfilingResult(
        report_path=report_path,
        generated_tables=generated_tables,
        generated_figures=generated_figures,
        group_counts=group_counts,
        top_effects=effect_sizes.head(10),
    )


def build_activity_grouping_review(families_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in families_df.sort_values("activity_id").itertuples(index=False):
        rule = GROUPING_REVIEW_RULES[int(row.activity_id)]
        rows.append(
            {
                "activity_id": int(row.activity_id),
                "activity_name": row.activity_name,
                "provisional_motor_family": row.provisional_family,
                "habit_goal_group": rule["group"],
                "grouping_confidence": rule["confidence"],
                "grouping_notes": rule["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_selected_features_used(features_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for concept, candidates, notes in REQUESTED_FEATURE_CANDIDATES:
        matched = next((column for column in candidates if column in features_df.columns), "")
        rows.append(
            {
                "requested_feature_concept": concept,
                "actual_column_used": matched,
                "status": "matched" if matched else "missing",
                "notes": notes if matched else f"No candidate found among: {', '.join(candidates)}",
            }
        )
    return pd.DataFrame(rows)


def build_group_feature_summary(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, column in feature_map.items():
        for group, group_df in df.groupby("habit_goal_group"):
            values = pd.to_numeric(group_df[column], errors="coerce").dropna()
            rows.append(
                {
                    "feature_name": feature_name,
                    "group": group,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan,
                    "median": float(values.median()),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_effect_sizes(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    habit_df = df.loc[df["habit_goal_group"] == HABIT_LABEL]
    goal_df = df.loc[df["habit_goal_group"] == GOAL_LABEL]
    rows: list[dict[str, Any]] = []
    for feature_name, column in feature_map.items():
        habit_values = pd.to_numeric(habit_df[column], errors="coerce").dropna().to_numpy()
        goal_values = pd.to_numeric(goal_df[column], errors="coerce").dropna().to_numpy()
        pooled_std = pooled_standard_deviation(habit_values, goal_values)
        mean_difference = float(np.mean(habit_values) - np.mean(goal_values))
        cohens_d = mean_difference / pooled_std if pooled_std and np.isfinite(pooled_std) and pooled_std > 0 else np.nan
        rows.append(
            {
                "feature_name": feature_name,
                "n_habit_like": int(habit_values.size),
                "n_goal_directed": int(goal_values.size),
                "mean_habit_like": float(np.mean(habit_values)),
                "mean_goal_directed": float(np.mean(goal_values)),
                "median_habit_like": float(np.median(habit_values)),
                "median_goal_directed": float(np.median(goal_values)),
                "mean_difference": mean_difference,
                "pooled_std": pooled_std,
                "cohens_d": cohens_d,
            }
        )
    result = pd.DataFrame(rows)
    result["rank_by_abs_d"] = result["cohens_d"].abs().rank(method="min", ascending=False).astype(int)
    return result.sort_values("rank_by_abs_d", ignore_index=True)


def build_statistical_tests(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    habit_df = df.loc[df["habit_goal_group"] == HABIT_LABEL]
    goal_df = df.loc[df["habit_goal_group"] == GOAL_LABEL]
    rows: list[dict[str, Any]] = []
    for feature_name, column in feature_map.items():
        habit_values = pd.to_numeric(habit_df[column], errors="coerce").dropna().to_numpy()
        goal_values = pd.to_numeric(goal_df[column], errors="coerce").dropna().to_numpy()
        rows.append(
            {
                "feature_name": feature_name,
                "mannwhitney_u_pvalue": float(mannwhitneyu(habit_values, goal_values, alternative="two-sided").pvalue),
                "ttest_pvalue": float(ttest_ind(habit_values, goal_values, equal_var=False, nan_policy="omit").pvalue),
            }
        )
    result = pd.DataFrame(rows)
    result["mannwhitney_fdr_bh"] = benjamini_hochberg(result["mannwhitney_u_pvalue"].to_numpy())
    result["ttest_fdr_bh"] = benjamini_hochberg(result["ttest_pvalue"].to_numpy())
    return result.sort_values("mannwhitney_u_pvalue", ignore_index=True)


def build_family_group_summary(grouped_df: pd.DataFrame) -> pd.DataFrame:
    activity_counts = (
        grouped_df[["activity_id", "provisional_motor_family", "habit_goal_group"]]
        .drop_duplicates()
        .groupby(["provisional_motor_family", "habit_goal_group"])
        .size()
        .rename("n_activities")
        .reset_index()
    )
    trial_counts = (
        grouped_df.groupby(["provisional_motor_family", "habit_goal_group"])
        .size()
        .rename("n_trials")
        .reset_index()
    )
    return activity_counts.merge(trial_counts, on=["provisional_motor_family", "habit_goal_group"], how="outer").fillna(0)


def build_family_feature_summary(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, group), group_df in df.groupby(["provisional_motor_family", "habit_goal_group"]):
        for feature_name, column in feature_map.items():
            values = pd.to_numeric(group_df[column], errors="coerce").dropna()
            rows.append(
                {
                    "provisional_motor_family": family,
                    "habit_goal_group": group,
                    "feature_name": feature_name,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()) if values.shape[0] else np.nan,
                    "std": float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan,
                    "median": float(values.median()) if values.shape[0] else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_family_effect_sizes(df: pd.DataFrame, feature_map: dict[str, str], min_trials_per_group: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, family_df in df.groupby("provisional_motor_family"):
        habit_df = family_df.loc[family_df["habit_goal_group"] == HABIT_LABEL]
        goal_df = family_df.loc[family_df["habit_goal_group"] == GOAL_LABEL]
        if habit_df.shape[0] < min_trials_per_group or goal_df.shape[0] < min_trials_per_group:
            for feature_name in feature_map:
                rows.append(
                    {
                        "provisional_motor_family": family,
                        "feature_name": feature_name,
                        "n_habit_like": int(habit_df.shape[0]),
                        "n_goal_directed": int(goal_df.shape[0]),
                        "cohens_d": np.nan,
                        "notes": "Insufficient support from both groups for stable family-level comparison.",
                    }
                )
            continue
        for feature_name, column in feature_map.items():
            habit_values = pd.to_numeric(habit_df[column], errors="coerce").dropna().to_numpy()
            goal_values = pd.to_numeric(goal_df[column], errors="coerce").dropna().to_numpy()
            pooled_std = pooled_standard_deviation(habit_values, goal_values)
            mean_difference = float(np.mean(habit_values) - np.mean(goal_values))
            cohens_d = mean_difference / pooled_std if pooled_std and np.isfinite(pooled_std) and pooled_std > 0 else np.nan
            rows.append(
                {
                    "provisional_motor_family": family,
                    "feature_name": feature_name,
                    "n_habit_like": int(habit_values.size),
                    "n_goal_directed": int(goal_values.size),
                    "cohens_d": cohens_d,
                    "notes": "Family-level exploratory comparison.",
                }
            )
    return pd.DataFrame(rows)


def summarize_group_counts(grouping_review: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    activity_counts = grouping_review.groupby("habit_goal_group").size().rename("n_activities").reset_index()
    trial_counts = merged_df.groupby("habit_goal_group").size().rename("n_trials").reset_index()
    return activity_counts.merge(trial_counts, on="habit_goal_group", how="outer").fillna(0)


def pooled_standard_deviation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    pooled = ((x.size - 1) * x_var + (y.size - 1) * y_var) / (x.size + y.size - 2)
    return float(np.sqrt(pooled))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        cumulative_min = min(cumulative_min, ranked[i] * n / rank)
        adjusted[i] = cumulative_min
    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def save_csv(df: pd.DataFrame, path: Path, generated_tables: list[str]) -> None:
    df.to_csv(path, index=False)
    generated_tables.append(path.name)


def plot_habit_goal_group_sizes(grouping_review: pd.DataFrame, merged_df: pd.DataFrame, output_path: Path) -> str:
    activity_counts = grouping_review.groupby("habit_goal_group").size().reindex([HABIT_LABEL, GOAL_LABEL, AMBIGUOUS_LABEL], fill_value=0)
    trial_counts = merged_df.groupby("habit_goal_group").size().reindex([HABIT_LABEL, GOAL_LABEL, AMBIGUOUS_LABEL], fill_value=0)
    x = np.arange(len(activity_counts))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, activity_counts.values, width=width, label="activities", color="#4C78A8")
    ax.bar(x + width / 2, trial_counts.values, width=width, label="trials", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(activity_counts.index, rotation=20)
    ax.set_ylabel("Count")
    ax.set_title("Composition of provisional conceptual groups")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_habit_goal_feature_boxplots(df: pd.DataFrame, feature_map: dict[str, str], output_path: Path) -> str:
    selected = [
        concept
        for concept in [
            "duration_s",
            "acc_mag_std",
            "gyro_mag_std",
            "acc_mag_peak_rate",
            "gyro_mag_peak_rate",
            "acc_mag_spectral_entropy",
            "gyro_mag_spectral_entropy",
        ]
        if concept in feature_map
    ]
    fig, axes = plt.subplots(len(selected), 1, figsize=(9, max(3 * len(selected), 8)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, concept in zip(axes, selected):
        column = feature_map[concept]
        data = [
            pd.to_numeric(df.loc[df["habit_goal_group"] == HABIT_LABEL, column], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(df.loc[df["habit_goal_group"] == GOAL_LABEL, column], errors="coerce").dropna().to_numpy(),
        ]
        ax.boxplot(data, tick_labels=[HABIT_LABEL, GOAL_LABEL], showfliers=False)
        ax.set_title(concept)
        ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_effect_sizes_barplot(effect_sizes: pd.DataFrame, output_path: Path, top_n: int = 15) -> str:
    top = effect_sizes.assign(abs_d=effect_sizes["cohens_d"].abs()).sort_values("abs_d", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature_name"][::-1], top["cohens_d"][::-1], color="#E45756")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top exploratory effect sizes by |Cohen's d|")
    ax.set_xlabel("Cohen's d (habit_like minus goal_directed)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_group_means_heatmap(df: pd.DataFrame, feature_map: dict[str, str], output_path: Path) -> str:
    grouped = df.groupby("habit_goal_group")[[column for column in feature_map.values()]].mean()
    renamed = grouped.rename(columns={value: key for key, value in feature_map.items()})
    scaled = pd.DataFrame(StandardScaler().fit_transform(renamed.T).T, index=renamed.index, columns=renamed.columns)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title("Standardized group means across selected features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Conceptual group")
    ax.set_xticks(np.arange(len(scaled.columns)))
    ax.set_xticklabels(scaled.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels(scaled.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_pca(df: pd.DataFrame, feature_map: dict[str, str], output_path: Path, include_ambiguous: bool) -> str:
    numeric = df[[column for column in feature_map.values()]].apply(pd.to_numeric, errors="coerce").dropna()
    aligned = df.loc[numeric.index].copy()
    scaled = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(scaled)
    aligned["PC1"] = transformed[:, 0]
    aligned["PC2"] = transformed[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {HABIT_LABEL: "#4C78A8", GOAL_LABEL: "#E45756", AMBIGUOUS_LABEL: "#9D9D9D"}
    groups = [HABIT_LABEL, GOAL_LABEL, AMBIGUOUS_LABEL] if include_ambiguous else [HABIT_LABEL, GOAL_LABEL]
    for group in groups:
        subset = aligned.loc[aligned["habit_goal_group"] == group]
        if subset.empty:
            continue
        ax.scatter(subset["PC1"], subset["PC2"], s=28, alpha=0.75, color=palette[group], label=group)
    ax.set_title("PCA of selected profiling features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_family_stratified_effects(family_effect_sizes: pd.DataFrame, output_path: Path) -> str:
    valid = family_effect_sizes.dropna(subset=["cohens_d"]).copy()
    if valid.empty:
        valid = family_effect_sizes.copy()
        valid["cohens_d"] = 0.0
    top_features = valid.groupby("feature_name")["cohens_d"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).head(6).index.tolist()
    pivot = (
        valid.loc[valid["feature_name"].isin(top_features)]
        .pivot(index="provisional_motor_family", columns="feature_name", values="cohens_d")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_title("Family-stratified exploratory effect sizes")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Motor family")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's d")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_family_group_counts(family_group_summary: pd.DataFrame, output_path: Path) -> str:
    pivot = family_group_summary.pivot(index="provisional_motor_family", columns="habit_goal_group", values="n_trials").fillna(0)
    groups = [column for column in [HABIT_LABEL, GOAL_LABEL, AMBIGUOUS_LABEL] if column in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, group in enumerate(groups):
        ax.bar(x + (idx - (len(groups) - 1) / 2) * width, pivot[group], width=width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Trials")
    ax.set_title("Trial counts by family and conceptual group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def build_report(
    grouping_review: pd.DataFrame,
    selected_features_used: pd.DataFrame,
    group_feature_summary: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    family_group_summary: pd.DataFrame,
    family_effect_sizes: pd.DataFrame,
    generated_tables: list[str],
    generated_figures: list[str],
    habit_summary_df: pd.DataFrame | None,
    activity_performance_df: pd.DataFrame | None,
) -> str:
    counts = grouping_review.groupby("habit_goal_group").size()
    trial_counts = family_group_summary.groupby("habit_goal_group")["n_trials"].sum()
    top_effects = effect_sizes.assign(abs_d=effect_sizes["cohens_d"].abs()).sort_values("abs_d", ascending=False).head(8)
    matched_features = selected_features_used.loc[selected_features_used["status"] == "matched"]
    strongest_family_effects = (
        family_effect_sizes.dropna(subset=["cohens_d"])
        .assign(abs_d=lambda df: df["cohens_d"].abs())
        .sort_values("abs_d", ascending=False)
        .head(8)
    )

    group_lines = "\n".join(
        f"- `{group}`: {int(counts.get(group, 0))} activities, {int(trial_counts.get(group, 0))} trials"
        for group in [HABIT_LABEL, GOAL_LABEL, AMBIGUOUS_LABEL]
    )
    effect_lines = "\n".join(
        f"- `{row.feature_name}`: Cohen's d = {row.cohens_d:.3f}"
        for row in top_effects.itertuples(index=False)
    )
    family_lines = "\n".join(
        f"- `{row.provisional_motor_family}` / `{row.feature_name}`: Cohen's d = {row.cohens_d:.3f}"
        for row in strongest_family_effects.itertuples(index=False)
    ) if not strongest_family_effects.empty else "- No family-level comparisons had sufficient support."

    extra_context = []
    if habit_summary_df is not None:
        extra_context.append("An earlier coarse habit-vs-goal summary table was available and is superseded here by a revised grouping with an explicit `ambiguous` category.")
    if activity_performance_df is not None:
        extra_context.append("Activity-level classification performance was available and used qualitatively when thinking about ambiguous families and likely overlap.")
    extra_context_text = " ".join(extra_context) if extra_context else "No optional auxiliary tables were required."

    return f"""# Habit-like vs goal-directed profiling report

## Objective

This phase performs an exploratory comparison between a provisional conceptual grouping of activities into `habit_like`, `goal_directed`, and `ambiguous`, using descriptive statistics, effect sizes, and family-aware profiling. The goal is hypothesis generation rather than validation of a finished theory of habit.

## Inputs used

- `outputs/features/trial_features.csv`
- `outputs/classification_interpretation/provisional_motor_families.csv`
- Optional context: {extra_context_text}

## How the grouping was reviewed

The grouping was revised explicitly at the activity level using four ingredients:

- task semantics from the activity names;
- the provisional motor families from the earlier interpretation phase;
- known behaviourally plausible confusions from the multi-class baseline;
- basic biomechanical and behavioural plausibility.

Importantly, the analysis does **not** force every activity into a binary contrast. The `ambiguous` category is treated as an analytical virtue because it captures tasks that plausibly mix routine everyday execution with object-governed sequential control.

## Final group composition

{group_lines}

Subset usage in this phase:

- `Subset A`: only `habit_like` vs `goal_directed`, excluding `ambiguous`, used for effect sizes and inferential support statistics.
- `Subset B`: all three groups, used for descriptive visualisations and PCA.
- `Subset C`: family-stratified summaries, only when both conceptual groups had enough support.

## Features included

- Matched feature concepts: {matched_features.shape[0]} of {selected_features_used.shape[0]}
- The selected feature map was kept intentionally interpretable, focusing on duration, intensity, variability, rhythmic structure, derivative/jerk, and spectral summaries.

## Main global differences

Features with the largest absolute exploratory effect sizes:

{effect_lines}

These differences should be read as feature-based differences between provisional conceptual groupings. They may reflect both the habit-like / goal-directed framing and the fact that different groups draw more heavily on different families of actions.

## Family-aware interpretation

Strongest family-level exploratory effects:

{family_lines}

This family-level view is important because it helps distinguish two possibilities:

- a broadly consistent global contrast across many families;
- or apparent global effects driven mainly by certain motor families such as dressing or fine object manipulation.

The current output should therefore be interpreted as a mixed picture rather than a single clean latent axis.

## Statistical support

The `statistical_tests.csv` table provides Mann-Whitney and Welch-style t-test p-values with FDR-adjusted columns as exploratory support only. These values should not be used as standalone evidence, especially given correlated features and the provisional nature of the grouping.

## Limitations

- The conceptual grouping is provisional and theory-guided, not ground truth.
- Some activities are better described as mixed or context-dependent, which is why `ambiguous` was retained explicitly.
- Feature correlations mean that multiple high-ranked effect sizes may partly reflect the same underlying movement dimension.
- Family-level comparisons can become unstable when a family has few trials or only weak representation from one conceptual group.
- Observed differences cannot be interpreted as a definitive behavioural signature of habit.

## Implications for the next phase

- The current results are strong enough to justify a cautious Phase 4B binary classification attempt, but only if `ambiguous` is handled explicitly rather than silently forced into one side.
- A sensible next step would be to compare two protocols:
  - binary classification on `habit_like` vs `goal_directed` after excluding `ambiguous`;
  - sensitivity analyses where ambiguous activities are reintroduced or reassigned.
- It would also be worthwhile to test whether family-aware analyses explain more variance than the global dichotomy alone.

## Outputs generated

Tables:
{chr(10).join(f"- `{name}`" for name in generated_tables)}

Figures:
{chr(10).join(f"- `{name}`" for name in generated_figures)}
"""
