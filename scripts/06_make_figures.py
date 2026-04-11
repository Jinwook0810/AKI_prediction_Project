from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import split_tabular_features
from aki_prediction.paths import build_paths


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_tabular_probabilities(paths) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_module = load_script_module(ROOT_DIR / "scripts" / "02_train_tabular.py", "train_tabular")

    model_table = pd.read_csv(paths.processed_dir / "tabular_features_v2.csv")
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_tabular_features(
        model_table,
        paths.split_json,
    )
    x_full_train = pd.concat([x_train, x_valid], axis=0)
    y_full_train = pd.concat([y_train, y_valid], axis=0)

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in ["logistic", "random_forest", "catboost"]:
        model = train_module.build_model(model_name)
        model.fit(x_full_train, y_full_train)
        y_prob = train_module.predict_probability(model, x_test)
        results[model_name] = (np.asarray(y_test), np.asarray(y_prob))

    return results


def make_model_comparison_figure(paths) -> Path:
    metrics_dir = paths.metrics_dir
    entries = [
        ("LogReg v2", metrics_dir / "logistic_tabular_v2.json"),
        ("RF v2", metrics_dir / "random_forest_tabular_v2.json"),
        ("CatBoost v2", metrics_dir / "catboost_tabular_v2.json"),
        ("LSTM", metrics_dir / "lstm_sequence_v1.json"),
        ("LSTM + Mask", metrics_dir / "lstm_sequence_masked_v1.json"),
        ("Transformer + Mask", metrics_dir / "transformer_sequence_masked_v1.json"),
    ]

    rows = []
    for label, path in entries:
        metrics = load_metrics(path)
        test = metrics["test"]
        rows.append(
            {
                "label": label,
                "AUROC": test["auroc"],
                "AUPRC": test["auprc"],
                "F1": test["f1"],
            }
        )

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    for ax, metric in zip(axes, ["AUROC", "AUPRC", "F1"]):
        ax.bar(df["label"], df[metric], color=colors[: len(df)])
        ax.set_title(metric)
        ax.set_ylim(0.3, 0.85)
        ax.tick_params(axis="x", rotation=30)
        for idx, value in enumerate(df[metric]):
            ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("AKI Prediction Model Comparison on Cleaned Test Cohort", fontsize=14)
    out_path = paths.figures_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_importance_figure(csv_path: Path, title: str, output_path: Path) -> Path:
    df = pd.read_csv(csv_path).sort_values("importance", ascending=False).head(10).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(df["feature"], df["importance"], color="#4E79A7")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    for y, value in enumerate(df["importance"]):
        ax.text(value, y, f" {value:.3f}", va="center", fontsize=9)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_pr_curve_figure(paths) -> Path:
    results = get_tabular_probabilities(paths)
    labels = {
        "logistic": "LogReg v2",
        "random_forest": "RF v2",
        "catboost": "CatBoost v2",
    }
    colors = {
        "logistic": "#4E79A7",
        "random_forest": "#F28E2B",
        "catboost": "#E15759",
    }

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for key, (y_true, y_prob) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = load_metrics(paths.metrics_dir / f"{key}_tabular_v2.json")["test"]["auprc"]
        ax.plot(recall, precision, label=f"{labels[key]} (AUPRC={auprc:.3f})", color=colors[key], linewidth=2)

    prevalence = next(iter(results.values()))[0].mean()
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Tabular Models")
    ax.legend()
    out_path = paths.figures_dir / "tabular_pr_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_calibration_figure(paths) -> Path:
    results = get_tabular_probabilities(paths)
    labels = {
        "logistic": "LogReg v2",
        "random_forest": "RF v2",
        "catboost": "CatBoost v2",
    }
    colors = {
        "logistic": "#4E79A7",
        "random_forest": "#F28E2B",
        "catboost": "#E15759",
    }

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    for key, (y_true, y_prob) in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color=colors[key], label=f"{labels[key]} (Brier={brier:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Plot: Tabular Models")
    ax.legend()
    out_path = paths.figures_dir / "tabular_calibration.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_top_feature_distribution_figure(paths) -> Path:
    df = pd.read_csv(paths.processed_dir / "tabular_features_v2.csv")
    features = [
        "Urine_ml_per_kg_hr_last6_mean",
        "Creatinine_relative_delta",
        "Creatinine_delta",
        "Age_first",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, feature in zip(axes.flatten(), features):
        data0 = df.loc[df["aki_future"] == 0, feature].dropna()
        data1 = df.loc[df["aki_future"] == 1, feature].dropna()
        if not data0.empty:
            upper = np.nanpercentile(pd.concat([data0, data1], axis=0), 99)
            data0 = data0.clip(upper=upper)
            data1 = data1.clip(upper=upper)
        ax.boxplot(
            [data0, data1],
            tick_labels=["No AKI", "AKI"],
            patch_artist=True,
            boxprops=dict(facecolor="#4E79A7"),
        )
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Top Feature Distributions by Target Label", fontsize=14)
    out_path = paths.figures_dir / "top_feature_distributions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_missingness_heatmap(paths) -> Path:
    df = pd.read_csv(paths.filtered_csv)
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)].copy()
    variables = [
        "Urine",
        "Creatinine",
        "BUN",
        "Weight",
        "HR",
        "MAP",
        "SysABP",
        "RespRate",
        "SaO2",
        "Temp",
        "pH",
        "PaO2",
    ]
    observed = (
        df.groupby("hour")[variables]
        .agg(lambda col: float(col.notna().mean()))
        .T
    )

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    im = ax.imshow(observed.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_xticks(np.arange(observed.shape[1]))
    ax.set_xticklabels(observed.columns.tolist())
    ax.set_xlabel("Hour")
    ax.set_title("Observed Rate by Hour for Key Variables (Input Window)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Observed rate")
    out_path = paths.figures_dir / "key_variable_missingness_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    paths = build_paths(ROOT_DIR)
    outputs = []
    outputs.append(make_model_comparison_figure(paths))
    outputs.append(make_importance_figure(
        ROOT_DIR / "reports" / "importance" / "random_forest_tabular_v2_feature_importance.csv",
        "RandomForest v2 Top Features",
        paths.figures_dir / "random_forest_top_features.png",
    ))
    outputs.append(make_importance_figure(
        ROOT_DIR / "reports" / "importance" / "catboost_tabular_v2_feature_importance.csv",
        "CatBoost v2 Top Features",
        paths.figures_dir / "catboost_top_features.png",
    ))
    outputs.append(make_pr_curve_figure(paths))
    outputs.append(make_calibration_figure(paths))
    outputs.append(make_top_feature_distribution_figure(paths))
    outputs.append(make_missingness_heatmap(paths))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
