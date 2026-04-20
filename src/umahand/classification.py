from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42

TARGET_COLUMN = "activity_id"
GROUP_COLUMN = "user_id"
ACTIVITY_NAME_COLUMN = "activity_name"

ALWAYS_EXCLUDED_COLUMNS = {
    "activity_id",
    "activity_name",
    "relative_path",
    "user_id",
    "trial_id",
    "handedness_label",
    "gender_label",
}

SUBJECT_METADATA_COLUMNS = {
    "age_years",
    "weight_kg",
    "height_cm",
}

DURATION_COLUMNS = {
    "duration_s",
    "n_samples",
    "estimated_sampling_hz",
}


@dataclass(frozen=True)
class ClassificationResult:
    report_path: Path
    metrics_summary: pd.DataFrame
    random_split_metrics: pd.DataFrame
    groupkfold_metrics_by_fold: pd.DataFrame
    classification_report_groupkfold: pd.DataFrame
    misclassification_pairs: pd.DataFrame
    feature_importance_random_forest: pd.DataFrame
    best_model_name: str
    best_feature_config: str
    best_groupkfold_macro_f1: float


def run_activity_classification(features_csv: Path, output_dir: Path) -> ClassificationResult:
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    feature_configs = build_feature_configurations(df)
    models = build_models()

    random_split_rows: list[dict[str, Any]] = []
    groupkfold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    groupkfold_predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for feature_config_name, feature_columns in feature_configs.items():
        X = df[feature_columns].copy()
        y = df[TARGET_COLUMN].astype(int).copy()
        groups = df[GROUP_COLUMN].astype(int).copy()
        activity_names = df[[TARGET_COLUMN, ACTIVITY_NAME_COLUMN]].drop_duplicates().sort_values(TARGET_COLUMN)

        for model_name, model in models.items():
            random_metrics = evaluate_random_split(X, y, model)
            random_split_rows.append(
                {
                    "feature_config": feature_config_name,
                    "validation_scheme": "random_split",
                    "model_name": model_name,
                    **random_metrics,
                }
            )
            summary_rows.append(
                {
                    "feature_config": feature_config_name,
                    "validation_scheme": "random_split",
                    "model_name": model_name,
                    "accuracy_mean": random_metrics["accuracy"],
                    "accuracy_std": 0.0,
                    "balanced_accuracy_mean": random_metrics["balanced_accuracy"],
                    "balanced_accuracy_std": 0.0,
                    "macro_f1_mean": random_metrics["macro_f1"],
                    "macro_f1_std": 0.0,
                    "weighted_f1_mean": random_metrics["weighted_f1"],
                    "weighted_f1_std": 0.0,
                }
            )

            fold_metrics, oof_predictions = evaluate_groupkfold(X, y, groups, model)
            for row in fold_metrics:
                row.update(
                    {
                        "feature_config": feature_config_name,
                        "validation_scheme": "groupkfold",
                        "model_name": model_name,
                    }
                )
                groupkfold_rows.append(row)

            aggregated = summarize_groupkfold_metrics(fold_metrics)
            summary_rows.append(
                {
                    "feature_config": feature_config_name,
                    "validation_scheme": "groupkfold",
                    "model_name": model_name,
                    **aggregated,
                }
            )

            prediction_df = pd.DataFrame(
                {
                    "true_activity_id": y.to_numpy(),
                    "predicted_activity_id": oof_predictions,
                }
            ).merge(
                activity_names.rename(
                    columns={TARGET_COLUMN: "true_activity_id", ACTIVITY_NAME_COLUMN: "true_activity_name"}
                ),
                on="true_activity_id",
                how="left",
            ).merge(
                activity_names.rename(
                    columns={TARGET_COLUMN: "predicted_activity_id", ACTIVITY_NAME_COLUMN: "predicted_activity_name"}
                ),
                on="predicted_activity_id",
                how="left",
            )
            groupkfold_predictions[(feature_config_name, model_name)] = prediction_df

    metrics_summary = pd.DataFrame(summary_rows).sort_values(
        ["validation_scheme", "feature_config", "macro_f1_mean"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    random_split_metrics = pd.DataFrame(random_split_rows).sort_values(
        ["feature_config", "macro_f1"],
        ascending=[True, False],
        ignore_index=True,
    )
    groupkfold_metrics_by_fold = pd.DataFrame(groupkfold_rows).sort_values(
        ["feature_config", "model_name", "fold"],
        ignore_index=True,
    )

    best_row = (
        metrics_summary.loc[metrics_summary["validation_scheme"] == "groupkfold"]
        .sort_values(["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"], ascending=False)
        .iloc[0]
    )
    best_feature_config = str(best_row["feature_config"])
    best_model_name = str(best_row["model_name"])
    best_predictions = groupkfold_predictions[(best_feature_config, best_model_name)]

    classification_report_groupkfold = build_classification_report(best_predictions)
    misclassification_pairs = build_misclassification_pairs(best_predictions)

    best_feature_columns = feature_configs[best_feature_config]
    rf_importance = train_random_forest_importance(df, best_feature_columns)

    plot_confusion_matrices(
        best_predictions=best_predictions,
        output_dir=output_dir,
    )
    plot_random_forest_importance(
        rf_importance=rf_importance,
        output_path=figures_dir / "feature_importance_random_forest.png",
    )

    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    random_split_metrics.to_csv(output_dir / "random_split_metrics.csv", index=False)
    groupkfold_metrics_by_fold.to_csv(output_dir / "groupkfold_metrics_by_fold.csv", index=False)
    classification_report_groupkfold.to_csv(output_dir / "classification_report_groupkfold.csv", index=False)
    misclassification_pairs.to_csv(output_dir / "misclassification_pairs.csv", index=False)
    rf_importance.to_csv(output_dir / "feature_importance_random_forest.csv", index=False)

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_report(
            df=df,
            metrics_summary=metrics_summary,
            random_split_metrics=random_split_metrics,
            groupkfold_metrics_by_fold=groupkfold_metrics_by_fold,
            classification_report_groupkfold=classification_report_groupkfold,
            misclassification_pairs=misclassification_pairs,
            rf_importance=rf_importance,
            best_feature_config=best_feature_config,
            best_model_name=best_model_name,
        ),
        encoding="utf-8",
    )

    return ClassificationResult(
        report_path=report_path,
        metrics_summary=metrics_summary,
        random_split_metrics=random_split_metrics,
        groupkfold_metrics_by_fold=groupkfold_metrics_by_fold,
        classification_report_groupkfold=classification_report_groupkfold,
        misclassification_pairs=misclassification_pairs,
        feature_importance_random_forest=rf_importance,
        best_model_name=best_model_name,
        best_feature_config=best_feature_config,
        best_groupkfold_macro_f1=float(best_row["macro_f1_mean"]),
    )


def build_feature_configurations(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    base_columns = [
        column
        for column in numeric_columns
        if column not in ALWAYS_EXCLUDED_COLUMNS
        and column not in SUBJECT_METADATA_COLUMNS
    ]
    with_duration = sorted(base_columns)
    without_duration = sorted([column for column in base_columns if column not in DURATION_COLUMNS])
    return {
        "with_duration": with_duration,
        "without_duration": without_duration,
    }


def build_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                )
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_random_split(X: pd.DataFrame, y: pd.Series, model: Any) -> dict[str, float]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))
    fitted_model = clone(model)
    fitted_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = fitted_model.predict(X.iloc[test_idx])
    return compute_metrics(y.iloc[test_idx], predictions)


def evaluate_groupkfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: Any,
    n_splits: int = 5,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    oof_predictions = np.empty(len(y), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        fitted_model = clone(model)
        fitted_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fitted_model.predict(X.iloc[test_idx])
        oof_predictions[test_idx] = predictions
        metrics = compute_metrics(y.iloc[test_idx], predictions)
        fold_rows.append({"fold": fold_idx, **metrics})

    return fold_rows, oof_predictions


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def summarize_groupkfold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    return {
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=1)),
        "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(df["balanced_accuracy"].std(ddof=1)),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std(ddof=1)),
        "weighted_f1_mean": float(df["weighted_f1"].mean()),
        "weighted_f1_std": float(df["weighted_f1"].std(ddof=1)),
    }


def build_classification_report(predictions_df: pd.DataFrame) -> pd.DataFrame:
    report = classification_report(
        predictions_df["true_activity_id"],
        predictions_df["predicted_activity_id"],
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({"label": label, **metrics})
    return pd.DataFrame(rows)


def build_misclassification_pairs(predictions_df: pd.DataFrame) -> pd.DataFrame:
    misclassified = predictions_df.loc[
        predictions_df["true_activity_id"] != predictions_df["predicted_activity_id"]
    ].copy()
    if misclassified.empty:
        return pd.DataFrame(
            columns=[
                "true_activity_id",
                "true_activity_name",
                "predicted_activity_id",
                "predicted_activity_name",
                "count",
                "normalized_rate_if_possible",
            ]
        )

    pair_counts = (
        misclassified.groupby(
            [
                "true_activity_id",
                "true_activity_name",
                "predicted_activity_id",
                "predicted_activity_name",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )
    true_counts = predictions_df.groupby("true_activity_id").size().rename("true_total").reset_index()
    pair_counts = pair_counts.merge(true_counts, on="true_activity_id", how="left")
    pair_counts["normalized_rate_if_possible"] = pair_counts["count"] / pair_counts["true_total"]
    return pair_counts.drop(columns="true_total").sort_values(
        ["count", "normalized_rate_if_possible"],
        ascending=[False, False],
        ignore_index=True,
    )


def train_random_forest_importance(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)
    return (
        pd.DataFrame({"feature_name": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def plot_confusion_matrices(best_predictions: pd.DataFrame, output_dir: Path) -> None:
    labels = sorted(best_predictions["true_activity_id"].unique())
    cm = confusion_matrix(
        best_predictions["true_activity_id"],
        best_predictions["predicted_activity_id"],
        labels=labels,
    )
    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        output_path=output_dir / "confusion_matrix_groupkfold_best_model.png",
        title="GroupKFold confusion matrix",
        normalize=False,
    )
    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        output_path=output_dir / "confusion_matrix_groupkfold_best_model_normalized.png",
        title="GroupKFold confusion matrix (row-normalized)",
        normalize=True,
    )


def plot_confusion_matrix(cm: np.ndarray, labels: list[int], output_path: Path, title: str, normalize: bool) -> None:
    plot_values = cm.astype(float)
    if normalize:
        row_sums = plot_values.sum(axis=1, keepdims=True)
        plot_values = np.divide(plot_values, row_sums, out=np.zeros_like(plot_values), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(plot_values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted activity ID")
    ax.set_ylabel("True activity ID")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([f"{label:02d}" for label in labels], rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([f"{label:02d}" for label in labels])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_random_forest_importance(rf_importance: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top = rf_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature_name"], top["importance"], color="#4C78A8")
    ax.set_title(f"Top {top_n} Random Forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_report(
    df: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    random_split_metrics: pd.DataFrame,
    groupkfold_metrics_by_fold: pd.DataFrame,
    classification_report_groupkfold: pd.DataFrame,
    misclassification_pairs: pd.DataFrame,
    rf_importance: pd.DataFrame,
    best_feature_config: str,
    best_model_name: str,
) -> str:
    random_best = (
        metrics_summary.loc[metrics_summary["validation_scheme"] == "random_split"]
        .sort_values("macro_f1_mean", ascending=False)
        .iloc[0]
    )
    group_best = (
        metrics_summary.loc[metrics_summary["validation_scheme"] == "groupkfold"]
        .sort_values("macro_f1_mean", ascending=False)
        .iloc[0]
    )
    duration_comparison = (
        metrics_summary.loc[metrics_summary["validation_scheme"] == "groupkfold", ["feature_config", "model_name", "macro_f1_mean"]]
        .pivot(index="model_name", columns="feature_config", values="macro_f1_mean")
        .reset_index()
    )
    top_misclassifications = misclassification_pairs.head(10)
    top_features = rf_importance.head(10)

    easiest_classes = (
        classification_report_groupkfold.loc[
            ~classification_report_groupkfold["label"].isin(["accuracy", "macro avg", "weighted avg"])
        ]
        .copy()
    )
    easiest_classes["label_numeric"] = pd.to_numeric(easiest_classes["label"], errors="coerce")
    easiest_classes = easiest_classes.dropna(subset=["label_numeric"])
    easiest_classes["label_numeric"] = easiest_classes["label_numeric"].astype(int)
    easiest_classes = easiest_classes.rename(columns={"f1-score": "f1_score"})
    easiest_sorted = easiest_classes.sort_values("f1_score", ascending=False)
    hardest_sorted = easiest_classes.sort_values("f1_score", ascending=True)

    excluded_columns_text = ", ".join(sorted(ALWAYS_EXCLUDED_COLUMNS | SUBJECT_METADATA_COLUMNS))
    generalization_gap = float(random_best["macro_f1_mean"] - group_best["macro_f1_mean"])
    if generalization_gap >= 0:
        gap_text = (
            f"The best random split is higher by {generalization_gap:.3f} macro-F1 points, which is the expected optimistic pattern when the same subjects can appear in both train and test partitions."
        )
    else:
        gap_text = (
            f"In this run the best GroupKFold result is higher by {abs(generalization_gap):.3f} macro-F1 points. Because the random split here is only a single split, this should be treated as split variability rather than as evidence that subject-wise evaluation is easier."
        )

    easiest_lines = "\n".join(
        f"- Activity {int(row.label_numeric):02d}: F1 = {getattr(row, 'f1_score'):.3f}"
        for row in easiest_sorted.head(5).itertuples(index=False)
    )
    hardest_lines = "\n".join(
        f"- Activity {int(row.label_numeric):02d}: F1 = {getattr(row, 'f1_score'):.3f}"
        for row in hardest_sorted.head(5).itertuples(index=False)
    )

    return f"""# UMAHand baseline classification report

## Objective

This phase evaluates a baseline classification setup for feature-based activity recognition across the 29 UMAHand activities, with a special focus on generalization to unseen subjects.

## Input used

- Source file: `outputs/features/trial_features.csv`
- Number of trials: {len(df)}
- Number of subjects: {df[GROUP_COLUMN].nunique()}
- Number of activities: {df[TARGET_COLUMN].nunique()}

## Feature inclusion and exclusions

- The target was `activity_id`, while `activity_name` was kept only for reporting.
- Non-numeric columns were excluded automatically.
- The following columns were always excluded to avoid leakage or identifier effects: `{excluded_columns_text}`.
- Subject metadata such as `age_years`, `weight_kg`, and `height_cm` were excluded by default because the goal is to classify motor activity from the inertial-derived features, not from participant descriptors.
- Two feature configurations were compared:
  - `with_duration`: includes `duration_s`, `n_samples`, and `estimated_sampling_hz` when available.
  - `without_duration`: removes those columns to test how much performance depends on trial duration.

## Validation schemes

- `random_split` is a stratified 80/20 split by activity. It is useful as a reference but is optimistic because train and test sets can contain the same subjects.
- `groupkfold` uses subject-wise folds through `GroupKFold`, which is the main evaluation because it measures generalization to unseen participants.

## Main results

- Best random-split result: `{random_best['model_name']}` with `{random_best['feature_config']}` and macro F1 = {random_best['macro_f1_mean']:.3f}
- Best GroupKFold result: `{group_best['model_name']}` with `{group_best['feature_config']}` and macro F1 = {group_best['macro_f1_mean']:.3f} +/- {group_best['macro_f1_std']:.3f}
- GroupKFold balanced accuracy for the best model: {group_best['balanced_accuracy_mean']:.3f} +/- {group_best['balanced_accuracy_std']:.3f}
- {gap_text}

## With-duration versus without-duration

{dataframe_to_markdown(duration_comparison)}

This comparison should be read as a sensitivity analysis: if performance drops noticeably without duration features, then duration is carrying substantial information and could be acting as a shortcut for some activities.

## Activity-level difficulty

- Easiest activities by GroupKFold F1:
{easiest_lines}
- Hardest activities by GroupKFold F1:
{hardest_lines}

## Most frequent confusions

{dataframe_to_markdown(top_misclassifications) if not top_misclassifications.empty else "No misclassification pairs were found."}

## Random Forest descriptive feature importances

{dataframe_to_markdown(top_features)}

These importances are descriptive and should not be interpreted as causal evidence. They are useful for spotting which summary descriptors the baseline tree ensemble relies on most heavily.

## Limitations

- This is a baseline classification analysis, not a final model.
- The dataset is relatively small for a 29-class problem.
- Class frequencies are uneven across activities.
- Trials have different durations, which may help but can also inflate apparent discriminability.
- Subject-specific style may still influence the learned decision boundaries.
- Random Forest importances reflect this fitted model only and do not imply mechanistic relevance.

## Recommended next phase

- Keep subject-wise validation as the default protocol.
- Add confusion-matrix-driven error analysis by grouping related activities.
- Compare simple feature-selection or dimensionality-reduction strategies to reduce redundancy.
- Test calibration, per-class recall, and perhaps a smaller activity subset if some classes remain severely confused.
- If the baseline is promising, move to a more systematic comparison of models and feature sets before considering sequence models or deep learning.
"""


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join([header, separator, *rows])


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
