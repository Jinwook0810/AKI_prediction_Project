from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import split_tabular_features
from aki_prediction.paths import build_paths


RANDOM_SEED = 42
FEATURE_VERSION = "v3"
MODELS = ["logistic", "random_forest", "catboost", "lightgbm", "xgboost"]
TOPK_FRACTIONS = [0.05, 0.10, 0.20]
THRESHOLDS = [0.20, 0.30, 0.35, 0.40, 0.50]


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)
    if positive <= 0:
        return 1.0
    return negative / positive


def build_model(model_name: str, y_train: pd.Series):
    scale_pos_weight = compute_scale_pos_weight(y_train)

    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=False,
        )

    if model_name == "lightgbm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=20,
                        objective="binary",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        min_child_weight=1,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=RANDOM_SEED,
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def topk_metrics(
    model_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    fractions: list[float],
) -> list[dict[str, float]]:
    n = len(y_true)
    positive_total = int(y_true.sum())
    base_rate = float(y_true.mean())
    order = np.argsort(-y_prob)
    y_sorted = y_true.to_numpy()[order]

    rows = []
    for fraction in fractions:
        k = max(1, int(math.ceil(n * fraction)))
        selected = y_sorted[:k]
        tp = int(selected.sum())
        precision_at_k = float(tp / k)
        recall_at_k = float(tp / positive_total) if positive_total > 0 else 0.0
        lift = float(precision_at_k / base_rate) if base_rate > 0 else 0.0
        rows.append(
            {
                "model": model_name,
                "coverage_fraction": fraction,
                "k": k,
                "captured_positives": tp,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_vs_base_rate": lift,
                "base_positive_rate": base_rate,
            }
        )
    return rows


def threshold_metrics(
    model_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "review_rate": float(pred.mean()),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return rows


def save_markdown_summary(
    output_path: Path,
    cohort_n: int,
    positive_n: int,
    topk_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    lightgbm_top10 = topk_df[
        (topk_df["model"] == "lightgbm")
        & np.isclose(topk_df["coverage_fraction"], 0.10)
    ].iloc[0]
    catboost_top10 = topk_df[
        (topk_df["model"] == "catboost")
        & np.isclose(topk_df["coverage_fraction"], 0.10)
    ].iloc[0]

    lines = [
        "# Decision Analysis Summary",
        "",
        f"- feature_version: `{FEATURE_VERSION}`",
        f"- evaluated_split: `test`",
        f"- test_n: `{cohort_n}`",
        f"- test_positive: `{positive_n}`",
        "",
        "## Interpretation",
        "",
        (
            "If operations can review only the top 10% highest-risk stays, "
            f"`LightGBM v3` captures `{lightgbm_top10['recall_at_k']:.4f}` of positives "
            f"with precision `{lightgbm_top10['precision_at_k']:.4f}`."
        ),
        (
            "Under the same 10% review budget, "
            f"`CatBoost v3` captures `{catboost_top10['recall_at_k']:.4f}` of positives "
            f"with precision `{catboost_top10['precision_at_k']:.4f}`."
        ),
        "",
        "## Top-k Recall",
        "",
        "| model | coverage_fraction | k | captured_positives | precision_at_k | recall_at_k | lift_vs_base_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in topk_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['coverage_fraction']:.2f} | {int(row['k'])} | "
            f"{int(row['captured_positives'])} | {row['precision_at_k']:.4f} | "
            f"{row['recall_at_k']:.4f} | {row['lift_vs_base_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Trade-off",
            "",
            "| model | threshold | review_rate | precision | recall | f1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in threshold_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['threshold']:.2f} | {row['review_rate']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_topk_figure(topk_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model_name in MODELS:
        subset = topk_df[topk_df["model"] == model_name]
        plt.plot(
            subset["coverage_fraction"] * 100,
            subset["recall_at_k"],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Reviewed top-risk fraction (%)")
    plt.ylabel("Recall within reviewed set")
    plt.title("Top-k Capture on Test Cohort")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_threshold_figure(threshold_df: pd.DataFrame, output_path: Path) -> None:
    focus_models = ["lightgbm", "catboost"]
    plt.figure(figsize=(8, 5))
    for model_name in focus_models:
        subset = threshold_df[threshold_df["model"] == model_name]
        plt.plot(subset["threshold"], subset["precision"], marker="o", label=f"{model_name} precision")
        plt.plot(subset["threshold"], subset["recall"], marker="s", linestyle="--", label=f"{model_name} recall")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold Trade-off")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    paths = build_paths(ROOT_DIR)
    decision_dir = paths.reports_dir / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)

    tabular_path = paths.processed_dir / f"tabular_features_{FEATURE_VERSION}.csv"
    model_table = pd.read_csv(tabular_path)
    x_train, y_train, _, _, x_test, y_test = split_tabular_features(
        model_table,
        paths.split_json,
    )

    topk_rows: list[dict[str, float]] = []
    threshold_rows: list[dict[str, float]] = []

    for model_name in MODELS:
        model = build_model(model_name, y_train)
        model.fit(x_train, y_train)
        test_prob = predict_probability(model, x_test)
        topk_rows.extend(topk_metrics(model_name, y_test, test_prob, TOPK_FRACTIONS))
        threshold_rows.extend(
            threshold_metrics(model_name, y_test, test_prob, THRESHOLDS)
        )

    topk_df = pd.DataFrame(topk_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    topk_csv = decision_dir / "tabular_v3_topk_summary.csv"
    threshold_csv = decision_dir / "tabular_v3_threshold_summary.csv"
    summary_md = decision_dir / "tabular_v3_decision_summary.md"
    topk_fig = paths.figures_dir / "tabular_v3_topk_capture.png"
    threshold_fig = paths.figures_dir / "tabular_v3_threshold_tradeoff.png"

    topk_df.to_csv(topk_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)
    save_markdown_summary(summary_md, len(y_test), int(y_test.sum()), topk_df, threshold_df)
    make_topk_figure(topk_df, topk_fig)
    make_threshold_figure(threshold_df, threshold_fig)

    print(
        json.dumps(
            {
                "topk_csv": str(topk_csv),
                "threshold_csv": str(threshold_csv),
                "summary_md": str(summary_md),
                "topk_fig": str(topk_fig),
                "threshold_fig": str(threshold_fig),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
