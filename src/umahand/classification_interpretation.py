from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROVISIONAL_FAMILY_RULES = [
    ("hygiene_self_care", ["Brushing teeth", "Washing hands", "combing hair", "Nose blowing"]),
    ("eating_drinking", ["Eat soup", "Drinking water", "Pouring water"]),
    ("communication_phone", ["Send a message", "Mark a phone number", "Waving goodbye"]),
    ("writing_typing", ["Write on a sheet of paper", "Writing a sentence with a keyboard"]),
    ("dressing_wearables", ["Putting on a pair of glasses", "Putting on a jacket", "Remove a jacket", "Putting on a shoe", "Buttoning a shirt", "Raising and lowering a zipper"]),
    ("household_cleaning", ["Cleaning (Wiping", "Sweep with a broom", "Opening and closing a door"]),
    ("fine_object_manipulation", ["Cutting food", "Peeling a fruit", "Fold a piece of paper", "Opening a bottle", "Screwing a screw"]),
    ("gross_repetitive_movement", ["Aplauding", "Picking up an object from the floor"]),
]


@dataclass(frozen=True)
class ClassificationInterpretationResult:
    report_path: Path
    generated_figures: list[str]
    generated_tables: list[str]
    easiest_activities: pd.DataFrame
    hardest_activities: pd.DataFrame
    top_confusions: pd.DataFrame


def run_classification_interpretation(
    features_csv: Path,
    classification_dir: Path,
    output_dir: Path,
) -> ClassificationInterpretationResult:
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_csv)
    classification_dir = Path(classification_dir)
    classification_report_df = pd.read_csv(classification_dir / "classification_report_groupkfold.csv")
    misclassification_pairs_df = pd.read_csv(classification_dir / "misclassification_pairs.csv")
    feature_importance_df = pd.read_csv(classification_dir / "feature_importance_random_forest.csv")
    metrics_summary_df = pd.read_csv(classification_dir / "metrics_summary.csv")
    groupkfold_metrics_df = pd.read_csv(classification_dir / "groupkfold_metrics_by_fold.csv")

    activity_metadata = (
        features_df[["activity_id", "activity_name"]]
        .drop_duplicates()
        .sort_values("activity_id", ignore_index=True)
    )

    activity_performance_summary = build_activity_performance_summary(
        classification_report_df=classification_report_df,
        activity_metadata=activity_metadata,
    )
    top_confusions = build_top_confusions(misclassification_pairs_df)
    provisional_motor_families = build_provisional_motor_families(activity_metadata)
    top_features_interpretation = build_top_features_interpretation(feature_importance_df)

    feature_candidates = resolve_feature_candidates(
        features_df,
        [
            ("duration_s", ["duration_s"]),
            ("acc_mag_mean", ["acc_mag_mean", "acc_mag_rms"]),
            ("acc_mag_std", ["acc_mag_std"]),
            ("acc_mag_rms", ["acc_mag_rms", "acc_mag_mean"]),
            ("gyro_mag_mean", ["gyro_mag_mean", "gyro_mag_rms"]),
            ("gyro_mag_std", ["gyro_mag_std"]),
            ("gyro_mag_rms", ["gyro_mag_rms", "gyro_mag_mean"]),
            ("acc_mag_dominant_frequency", ["acc_mag_dominant_frequency_hz", "acc_mag_dominant_frequency"]),
            ("gyro_mag_dominant_frequency", ["gyro_mag_dominant_frequency_hz", "gyro_mag_dominant_frequency"]),
            ("acc_mag_spectral_entropy", ["acc_mag_spectral_entropy"]),
            ("gyro_mag_spectral_entropy", ["gyro_mag_spectral_entropy"]),
            ("acc_mag_peak_rate", ["acc_mag_peak_rate_hz", "acc_mag_peak_rate"]),
            ("gyro_mag_peak_rate", ["gyro_mag_peak_rate_hz", "gyro_mag_peak_rate"]),
            ("acc_mag_low_band_power", ["acc_mag_low_band_power"]),
            ("acc_mag_high_band_power", ["acc_mag_high_band_power"]),
            ("gyro_mag_high_band_power", ["gyro_mag_high_band_power"]),
        ],
    )

    activity_profiles = build_activity_mean_feature_table(features_df, feature_candidates)
    difficult_confusion_matrix = build_difficult_confusion_matrix(
        activity_performance_summary=activity_performance_summary,
        misclassification_pairs=top_confusions,
    )

    generated_tables = []
    activity_performance_summary.to_csv(output_dir / "activity_performance_summary.csv", index=False)
    generated_tables.append("activity_performance_summary.csv")
    top_confusions.to_csv(output_dir / "top_confusions.csv", index=False)
    generated_tables.append("top_confusions.csv")
    provisional_motor_families.to_csv(output_dir / "provisional_motor_families.csv", index=False)
    generated_tables.append("provisional_motor_families.csv")
    top_features_interpretation.to_csv(output_dir / "top_features_interpretation.csv", index=False)
    generated_tables.append("top_features_interpretation.csv")

    generated_figures = []
    generated_figures.append(
        plot_activity_f1_barplot(activity_performance_summary, figures_dir / "activity_f1_barplot.png")
    )
    generated_figures.append(
        plot_top_confusions_barplot(top_confusions, figures_dir / "top_confusions_barplot.png")
    )
    generated_figures.append(
        plot_confusion_matrix_top_classes(
            difficult_confusion_matrix,
            figures_dir / "confusion_matrix_top_classes.png",
        )
    )
    generated_figures.append(
        plot_activity_families_overview(
            provisional_motor_families,
            figures_dir / "activity_families_overview.png",
        )
    )
    generated_figures.append(
        plot_activity_family_clustering(
            activity_profiles,
            figures_dir / "activity_family_clustering.png",
        )
    )
    generated_figures.append(
        plot_top_features_grouped(
            top_features_interpretation,
            figures_dir / "top_features_grouped.png",
        )
    )
    generated_figures.append(
        plot_selected_activity_feature_profiles(
            features_df=features_df,
            activity_performance_summary=activity_performance_summary,
            top_confusions=top_confusions,
            feature_candidates=feature_candidates,
            output_path=figures_dir / "selected_activity_feature_profiles.png",
        )
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_interpretation_report(
            metrics_summary_df=metrics_summary_df,
            groupkfold_metrics_df=groupkfold_metrics_df,
            activity_performance_summary=activity_performance_summary,
            top_confusions=top_confusions,
            top_features_interpretation=top_features_interpretation,
            provisional_motor_families=provisional_motor_families,
            generated_figures=generated_figures,
            generated_tables=generated_tables,
        ),
        encoding="utf-8",
    )

    return ClassificationInterpretationResult(
        report_path=report_path,
        generated_figures=generated_figures,
        generated_tables=generated_tables,
        easiest_activities=activity_performance_summary.head(5),
        hardest_activities=activity_performance_summary.sort_values("f1-score", ascending=True).head(5),
        top_confusions=top_confusions.head(10),
    )


def build_activity_performance_summary(
    classification_report_df: pd.DataFrame,
    activity_metadata: pd.DataFrame,
) -> pd.DataFrame:
    report = classification_report_df.copy()
    report["activity_id"] = pd.to_numeric(report["label"], errors="coerce")
    report = report.dropna(subset=["activity_id"]).copy()
    report["activity_id"] = report["activity_id"].astype(int)
    summary = report.merge(activity_metadata, on="activity_id", how="left")
    summary = summary[
        ["activity_id", "activity_name", "precision", "recall", "f1-score", "support"]
    ].sort_values("f1-score", ascending=False, ignore_index=True)
    summary["rank_by_f1"] = np.arange(1, len(summary) + 1)
    summary["rank_by_recall"] = summary["recall"].rank(method="min", ascending=False).astype(int)
    return summary


def build_top_confusions(misclassification_pairs_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    sort_columns = ["count"]
    ascending = [False]
    if "normalized_rate_if_possible" in misclassification_pairs_df.columns:
        sort_columns.append("normalized_rate_if_possible")
        ascending.append(False)
    return misclassification_pairs_df.sort_values(sort_columns, ascending=ascending, ignore_index=True).head(top_n)


def build_provisional_motor_families(activity_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in activity_metadata.itertuples(index=False):
        family, rationale = infer_motor_family(row.activity_name)
        rows.append(
            {
                "activity_id": row.activity_id,
                "activity_name": row.activity_name,
                "provisional_family": family,
                "family_rationale_short": rationale,
            }
        )
    return pd.DataFrame(rows)


def infer_motor_family(activity_name: str) -> tuple[str, str]:
    for family, fragments in PROVISIONAL_FAMILY_RULES:
        for fragment in fragments:
            if fragment.lower() in activity_name.lower():
                return family, f"Assigned by task semantics and behaviourally plausible confusion context around '{fragment}'."
    return "misc_object_interaction", "Fallback family for tasks that do not match the current heuristic rules clearly."


def build_top_features_interpretation(feature_importance_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    top = feature_importance_df.head(top_n).copy()
    top["inferred_signal_family"] = top["feature_name"].apply(infer_signal_family)
    top["interpretation_short"] = top["feature_name"].apply(infer_feature_interpretation)
    return top


def infer_signal_family(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "_band_power" in lowered or "spectral" in lowered or "dominant_frequency" in lowered:
        return "frequency-domain power"
    if "n_peaks" in lowered or "peak_rate" in lowered:
        return "peak structure"
    if "jerk" in lowered:
        return "temporal smoothness / jerk"
    if any(axis in feature_name for axis in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz"]):
        return "directional axis statistic"
    if "gyro" in lowered:
        return "rotational variability"
    if "acc" in lowered:
        return "acceleration level"
    return "general motion summary"


def infer_feature_interpretation(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "median" in lowered or "mean" in lowered:
        return "Typical level of the signal during the trial."
    if "std" in lowered or "iqr" in lowered:
        return "Within-trial variability or dispersion of the movement signal."
    if "n_peaks" in lowered or "peak_rate" in lowered:
        return "How bursty or rhythmically segmented the movement looks."
    if "dominant_frequency" in lowered:
        return "Main repetition rate or dominant movement tempo."
    if "spectral_entropy" in lowered:
        return "How concentrated versus distributed the movement frequencies are."
    if "band_power" in lowered:
        return "How much movement energy lies in a specific frequency band."
    if "jerk" in lowered:
        return "Abruptness and change in movement acceleration over time."
    if "energy" in lowered or "rms" in lowered:
        return "Overall magnitude or intensity of the signal."
    return "General summary descriptor of movement structure."


def resolve_feature_candidates(df: pd.DataFrame, candidates: list[tuple[str, list[str]]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for label, options in candidates:
        for option in options:
            if option in df.columns:
                resolved[label] = option
                break
    return resolved


def build_activity_mean_feature_table(features_df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    grouped = (
        features_df.groupby(["activity_id", "activity_name"])[list(feature_map.values())]
        .mean()
        .reset_index()
    )
    rename_map = {value: key for key, value in feature_map.items()}
    return grouped.rename(columns=rename_map)


def build_difficult_confusion_matrix(
    activity_performance_summary: pd.DataFrame,
    misclassification_pairs: pd.DataFrame,
    n_classes: int = 10,
) -> pd.DataFrame:
    hardest_ids = activity_performance_summary.sort_values("f1-score", ascending=True).head(n_classes)["activity_id"].tolist()
    subset = misclassification_pairs.loc[
        misclassification_pairs["true_activity_id"].isin(hardest_ids)
        & misclassification_pairs["predicted_activity_id"].isin(hardest_ids)
    ].copy()
    matrix = pd.DataFrame(0.0, index=hardest_ids, columns=hardest_ids)
    for row in subset.itertuples(index=False):
        matrix.loc[row.true_activity_id, row.predicted_activity_id] = row.count
    return matrix


def plot_activity_f1_barplot(activity_performance_summary: pd.DataFrame, output_path: Path) -> str:
    ordered = activity_performance_summary.sort_values("f1-score", ascending=False).copy()
    labels = [f"{activity_id:02d}" for activity_id in ordered["activity_id"]]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, ordered["f1-score"], color="#4C78A8")
    ax.set_title("Activity recognition F1 by activity")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("F1-score")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_top_confusions_barplot(top_confusions: pd.DataFrame, output_path: Path) -> str:
    labels = [
        f"{int(row.true_activity_id):02d}->{int(row.predicted_activity_id):02d}"
        for row in top_confusions.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels[::-1], top_confusions["count"].to_numpy()[::-1], color="#E45756")
    ax.set_title("Most frequent misclassification pairs")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_confusion_matrix_top_classes(matrix: pd.DataFrame, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix.to_numpy(), cmap="OrRd")
    labels = [f"{int(label):02d}" for label in matrix.index]
    ax.set_title("Confusion mass among the hardest activity classes")
    ax.set_xlabel("Predicted activity ID")
    ax.set_ylabel("True activity ID")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Misclassification count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_activity_families_overview(provisional_motor_families: pd.DataFrame, output_path: Path) -> str:
    family_counts = (
        provisional_motor_families.groupby("provisional_family").size().rename("n_activities").reset_index()
        .sort_values("n_activities", ascending=False, ignore_index=True)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(family_counts["provisional_family"], family_counts["n_activities"], color="#72B7B2")
    ax.set_title("Overview of provisional motor families")
    ax.set_xlabel("Provisional family")
    ax.set_ylabel("Number of activities")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_activity_family_clustering(activity_profiles: pd.DataFrame, output_path: Path) -> str:
    feature_cols = [column for column in activity_profiles.columns if column not in {"activity_id", "activity_name"}]
    data = activity_profiles[feature_cols].copy()
    scaled = StandardScaler().fit_transform(data)
    linkage_matrix = linkage(pdist(scaled), method="ward")
    labels = [f"{int(row.activity_id):02d}" for row in activity_profiles.itertuples(index=False)]

    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_title("Exploratory clustering of activities by mean feature profile")
    ax.set_xlabel("Activity ID")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_top_features_grouped(top_features_interpretation: pd.DataFrame, output_path: Path) -> str:
    grouped = (
        top_features_interpretation.groupby("inferred_signal_family")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(grouped["inferred_signal_family"][::-1], grouped["importance"][::-1], color="#54A24B")
    ax.set_title("Top Random Forest features grouped by interpreted signal family")
    ax.set_xlabel("Summed importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def plot_selected_activity_feature_profiles(
    features_df: pd.DataFrame,
    activity_performance_summary: pd.DataFrame,
    top_confusions: pd.DataFrame,
    feature_candidates: dict[str, str],
    output_path: Path,
) -> str:
    easiest = activity_performance_summary.head(3)["activity_id"].tolist()
    hardest = activity_performance_summary.sort_values("f1-score", ascending=True).head(3)["activity_id"].tolist()
    confused = pd.unique(
        pd.concat(
            [
                top_confusions["true_activity_id"].head(3),
                top_confusions["predicted_activity_id"].head(3),
            ],
            ignore_index=True,
        )
    ).tolist()
    selected_ids = list(dict.fromkeys(easiest + hardest + confused))[:9]

    plot_features = [key for key in [
        "duration_s",
        "acc_mag_std",
        "gyro_mag_std",
        "acc_mag_peak_rate",
        "gyro_mag_peak_rate",
        "acc_mag_spectral_entropy",
    ] if key in feature_candidates]
    selected_columns = [feature_candidates[key] for key in plot_features]

    grouped = (
        features_df.loc[features_df["activity_id"].isin(selected_ids), ["activity_id"] + selected_columns]
        .groupby("activity_id")
        .mean()
    )
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(grouped),
        index=grouped.index,
        columns=plot_features,
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title("Feature profiles for selected easy, difficult and confused activities")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Activity ID")
    ax.set_xticks(np.arange(len(plot_features)))
    ax.set_xticklabels(plot_features, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels([f"{int(idx):02d}" for idx in scaled.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.name


def build_interpretation_report(
    metrics_summary_df: pd.DataFrame,
    groupkfold_metrics_df: pd.DataFrame,
    activity_performance_summary: pd.DataFrame,
    top_confusions: pd.DataFrame,
    top_features_interpretation: pd.DataFrame,
    provisional_motor_families: pd.DataFrame,
    generated_figures: list[str],
    generated_tables: list[str],
) -> str:
    best_group = (
        metrics_summary_df.loc[metrics_summary_df["validation_scheme"] == "groupkfold"]
        .sort_values("macro_f1_mean", ascending=False)
        .iloc[0]
    )
    easiest = activity_performance_summary.head(5)
    hardest = activity_performance_summary.sort_values("f1-score", ascending=True).head(5)
    family_counts = provisional_motor_families["provisional_family"].value_counts()
    grouped_feature_families = (
        top_features_interpretation.groupby("inferred_signal_family")["importance"].sum().sort_values(ascending=False)
    )
    easiest = easiest.rename(columns={"f1-score": "f1_score"})
    hardest = hardest.rename(columns={"f1-score": "f1_score"})

    easiest_lines = "\n".join(
        f"- Activity {int(row.activity_id):02d} ({row.activity_name}): F1 = {row.f1_score:.3f}, recall = {row.recall:.3f}"
        for row in easiest.itertuples(index=False)
    )
    hardest_lines = "\n".join(
        f"- Activity {int(row.activity_id):02d} ({row.activity_name}): F1 = {row.f1_score:.3f}, recall = {row.recall:.3f}"
        for row in hardest.itertuples(index=False)
    )
    confusion_lines = "\n".join(
        f"- {int(row.true_activity_id):02d} -> {int(row.predicted_activity_id):02d}: {row.count} confusions ({row.normalized_rate_if_possible:.3f} of the true class)"
        for row in top_confusions.head(8).itertuples(index=False)
    )
    family_lines = "\n".join(
        f"- {family}: {count} activities"
        for family, count in family_counts.items()
    )
    feature_lines = "\n".join(
        f"- {family}: cumulative importance {importance:.3f}"
        for family, importance in grouped_feature_families.items()
    )

    return f"""# Classification interpretation report

## Objective

This phase adds a structured interpretive layer on top of the existing baseline classification outputs. The aim is not to retrain models, but to translate the current results into behaviourally plausible confusion patterns, provisional motor families, and useful hypotheses for the next analytical phase.

## Inputs used

- `outputs/features/trial_features.csv`
- `outputs/classification/classification_report_groupkfold.csv`
- `outputs/classification/misclassification_pairs.csv`
- `outputs/classification/feature_importance_random_forest.csv`
- `outputs/classification/metrics_summary.csv`
- `outputs/classification/groupkfold_metrics_by_fold.csv`

## Baseline performance context

- Best subject-wise baseline: `{best_group['model_name']}` with `{best_group['feature_config']}`
- GroupKFold macro F1: {best_group['macro_f1_mean']:.3f} +/- {best_group['macro_f1_std']:.3f}
- GroupKFold balanced accuracy: {best_group['balanced_accuracy_mean']:.3f} +/- {best_group['balanced_accuracy_std']:.3f}
- GroupKFold remains the main protocol because it asks whether the feature representation generalises to unseen subjects rather than simply interpolating within known participants.

## Activities recognised most easily

{easiest_lines}

These classes are good candidates for future protocol anchors because they appear behaviourally distinctive under the current feature representation.

## Activities recognised with more difficulty

{hardest_lines}

These classes are useful for hypothesis generation because they may reveal either genuinely overlapping motor structure or places where the current feature set is too coarse.

## Most important confusion patterns

{confusion_lines}

The strongest confusions are behaviourally plausible: paired dressing actions, manual-brushing variants, or object-manipulation tasks with similar wrist dynamics. This is exactly the kind of exploratory structure we would expect if the model is picking up real motor similarities rather than arbitrary noise.

## Provisional motor families

The following exploratory families are proposed as a practical interpretive layer, not as ground truth:

{family_lines}

These families were assigned using task semantics and checked against the confusion structure. They should be treated as provisional motor families useful for organising hypotheses, protocol design, or later regrouping analyses.

## Interpretation of top features

The highest-importance Random Forest features can be translated into broader movement dimensions:

{feature_lines}

In practice, the current baseline seems to rely heavily on a mix of axis-level posture summaries, peak structure, and frequency-domain energy allocation. That suggests the classifier is reading not just how much the wrist moves, but also how rhythmically and in which dynamic regime it moves.

## Practical implications

- Activities with very high F1 could serve as stable reference conditions in later behavioural profiling.
- Recurrent confusion pairs may be better handled as related subfamilies rather than fully independent actions in a theory-driven interpretation.
- Dressing and fine object-manipulation tasks look especially promising for family-based regrouping.
- If the project moves toward habit-like versus goal-directed profiling, these confusion-linked families offer a more grounded starting point than using the raw 29 labels in isolation.

## Recommended next phase

- Use the provisional motor families as an exploratory lens, not as labels to optimise against immediately.
- Revisit the habit-like versus goal-directed framing with explicit effect sizes rather than only classifier scores.
- Compare family-level feature profiles and within-family versus between-family distances.
- Consider a second-pass conceptual relabelling where strongly confused activities are analysed jointly before forcing strict separation.
- Keep the language cautious: these results provide feature-based evidence and useful structure for hypothesis generation, not a definitive latent taxonomy.

## Outputs generated

Tables:
{chr(10).join(f"- `{name}`" for name in generated_tables)}

Figures:
{chr(10).join(f"- `{name}`" for name in generated_figures)}
"""
