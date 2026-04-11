from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import (
    add_split_column,
    build_tabular_features,
    split_tabular_features,
)
from aki_prediction.paths import build_paths


RANDOM_SEED = 42


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def get_tuning_grid(model_name: str) -> list[dict]:
    if model_name == "random_forest":
        return expand_grid(
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 10, 16],
                "min_samples_leaf": [1, 5],
                "max_features": ["sqrt", 0.5],
            }
        )

    if model_name == "catboost":
        return expand_grid(
            {
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05],
                "l2_leaf_reg": [3, 10],
            }
        )

    raise ValueError(f"Tuning grid is not defined for model: {model_name}")


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def build_model(model_name: str, params: dict | None = None):
    params = params or {}

    if model_name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params.get("C", 1.0),
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
                        n_estimators=params.get("n_estimators", 500),
                        max_depth=params.get("max_depth"),
                        min_samples_leaf=params.get("min_samples_leaf", 1),
                        max_features=params.get("max_features", "sqrt"),
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=params.get("iterations", 500),
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", 6),
            l2_leaf_reg=params.get("l2_leaf_reg", 3),
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raise TypeError("Model does not support predict_proba")


def evaluate(
    model_name: str,
    y_valid: pd.Series,
    valid_prob: np.ndarray,
    y_test: pd.Series,
    test_prob: np.ndarray,
) -> dict:
    threshold, valid_best_f1 = find_best_threshold(y_valid, valid_prob)
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    return {
        "model": model_name,
        "threshold": threshold,
        "valid_best_f1": valid_best_f1,
        "test": {
            "n": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "auroc": float(roc_auc_score(y_test, test_prob)),
            "auprc": float(average_precision_score(y_test, test_prob)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        },
    }


def tune_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[object, dict, float, list[dict]]:
    best_model = None
    best_params = None
    best_threshold = 0.5
    best_valid_f1 = -1.0
    best_valid_auprc = -1.0
    trials: list[dict] = []

    for params in get_tuning_grid(model_name):
        model = build_model(model_name, params=params)
        model.fit(x_train, y_train)
        valid_prob = predict_probability(model, x_valid)
        threshold, valid_best_f1 = find_best_threshold(y_valid, valid_prob)
        valid_auprc = float(average_precision_score(y_valid, valid_prob))
        valid_auroc = float(roc_auc_score(y_valid, valid_prob))

        trial = {
            "params": params,
            "threshold": threshold,
            "valid_best_f1": valid_best_f1,
            "valid_auprc": valid_auprc,
            "valid_auroc": valid_auroc,
        }
        trials.append(trial)

        is_better = (
            (valid_best_f1 > best_valid_f1)
            or (
                np.isclose(valid_best_f1, best_valid_f1)
                and valid_auprc > best_valid_auprc
            )
        )
        if is_better:
            best_model = model
            best_params = params
            best_threshold = threshold
            best_valid_f1 = valid_best_f1
            best_valid_auprc = valid_auprc

    if best_model is None or best_params is None:
        raise RuntimeError(f"Failed to tune model: {model_name}")

    return best_model, best_params, best_threshold, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["dummy", "logistic", "random_forest", "catboost"],
        default="logistic",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild patient-level tabular features from filtered_cohort.csv.",
    )
    parser.add_argument(
        "--feature-version",
        choices=["v1", "v2", "compact"],
        default="v1",
        help="Patient-level tabular feature set version.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Save tabular features and exit without fitting a model.",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Search a small fixed hyperparameter grid using the valid split.",
    )
    args = parser.parse_args()

    paths = build_paths(ROOT_DIR)
    tabular_csv = paths.processed_dir / f"tabular_features_{args.feature_version}.csv"

    if args.rebuild_features or not tabular_csv.exists():
        filtered_df = pd.read_csv(paths.filtered_csv)
        model_table = build_tabular_features(
            filtered_df,
            feature_version=args.feature_version,
        )
        model_table = add_split_column(model_table, paths.split_json)
        model_table.to_csv(tabular_csv, index=False)
        print(f"Saved tabular features: {tabular_csv}")
    else:
        model_table = pd.read_csv(tabular_csv)
        if "split" not in model_table.columns:
            model_table = add_split_column(model_table, paths.split_json)
            model_table.to_csv(tabular_csv, index=False)
        print(f"Loaded tabular features: {tabular_csv}")

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_tabular_features(
        model_table,
        paths.split_json,
    )
    if args.features_only:
        summary = {
            "feature_version": args.feature_version,
            "n_features": int(x_train.shape[1]),
            "train_n": int(x_train.shape[0]),
            "valid_n": int(x_valid.shape[0]),
            "test_n": int(x_test.shape[0]),
            "train_positive": int(y_train.sum()),
            "valid_positive": int(y_valid.sum()),
            "test_positive": int(y_test.sum()),
        }
        print(json.dumps(summary, indent=2))
        print("Features-only mode: skipped model fitting and metric JSON writing.")
        return

    if args.tune_hyperparameters:
        model, best_params, threshold, trials = tune_model(
            args.model,
            x_train,
            y_train,
            x_valid,
            y_valid,
        )
        test_prob = predict_probability(model, x_test)
        test_pred = (test_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        metrics = {
            "model": args.model,
            "threshold": threshold,
            "valid_best_f1": max(trial["valid_best_f1"] for trial in trials),
            "test": {
                "n": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "auroc": float(roc_auc_score(y_test, test_prob)),
                "auprc": float(average_precision_score(y_test, test_prob)),
                "f1": float(f1_score(y_test, test_pred, zero_division=0)),
                "precision": float(precision_score(y_test, test_pred, zero_division=0)),
                "recall": float(recall_score(y_test, test_pred, zero_division=0)),
                "confusion_matrix": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                },
            },
            "tuning": {
                "enabled": True,
                "selection_metric": "valid_best_f1",
                "best_params": best_params,
                "n_trials": len(trials),
                "trials": trials,
            },
        }
        metrics_path = (
            paths.metrics_dir
            / f"{args.model}_tuned_tabular_{args.feature_version}.json"
        )
    else:
        model = build_model(args.model)
        model.fit(x_train, y_train)

        valid_prob = predict_probability(model, x_valid)
        test_prob = predict_probability(model, x_test)

        metrics = evaluate(args.model, y_valid, valid_prob, y_test, test_prob)
        metrics_path = (
            paths.metrics_dir / f"{args.model}_tabular_{args.feature_version}.json"
        )

    metrics["feature_version"] = args.feature_version
    metrics["features"] = {
        "n_features": int(x_train.shape[1]),
        "train_n": int(x_train.shape[0]),
        "valid_n": int(x_valid.shape[0]),
        "test_n": int(x_test.shape[0]),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
