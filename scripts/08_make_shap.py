from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import split_tabular_features
from aki_prediction.paths import build_paths


RANDOM_SEED = 42
FEATURE_VERSION = "v3"
MAX_DISPLAY = 20


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)
    if positive <= 0:
        return 1.0
    return negative / positive


def build_model(model_name: str, y_train: pd.Series):
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

    raise ValueError(f"Unsupported model: {model_name}")


def fit_and_prepare_features(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    model = build_model(model_name, y_train)
    model.fit(x_train, y_train)

    if model_name == "lightgbm":
        imputer = model.named_steps["imputer"]
        estimator = model.named_steps["model"]
        x_train_ready = pd.DataFrame(
            imputer.transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )
        x_test_ready = pd.DataFrame(
            imputer.transform(x_test),
            columns=x_test.columns,
            index=x_test.index,
        )
        return model, estimator, x_train_ready, x_test_ready

    return model, model, x_train.copy(), x_test.copy()


def extract_binary_shap(explainer, shap_values_raw) -> tuple[np.ndarray, float]:
    expected = explainer.expected_value

    if isinstance(shap_values_raw, list):
        if len(shap_values_raw) == 2:
            base = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
            return np.asarray(shap_values_raw[1]), float(base)
        base = expected[0] if isinstance(expected, (list, np.ndarray)) else expected
        return np.asarray(shap_values_raw[0]), float(base)

    shap_values = np.asarray(shap_values_raw)
    if shap_values.ndim == 3:
        base = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        return shap_values[:, :, 1], float(base)

    base = expected[1] if isinstance(expected, (list, np.ndarray)) and np.ndim(expected) > 0 and len(np.atleast_1d(expected)) > 1 else expected
    return shap_values, float(base)


def save_summary_plot(shap_values: np.ndarray, x_test: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, x_test, show=False, max_display=MAX_DISPLAY)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_dependence_plot(
    feature_name: str,
    shap_values: np.ndarray,
    x_test: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        x_test,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_global_csv(shap_values: np.ndarray, x_test: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = range(1, len(importance) + 1)
    importance.to_csv(output_path, index=False)
    return importance


def save_local_markdown(
    model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_prob: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
    output_path: Path,
) -> None:
    positive_idx = y_test[y_test == 1].index
    if len(positive_idx) > 0:
        sample_index = test_prob[y_test.to_numpy() == 1].argmax()
        row_index = positive_idx[sample_index]
    else:
        row_index = y_test.index[int(np.argmax(test_prob))]

    row_position = x_test.index.get_loc(row_index)
    contrib = pd.DataFrame(
        {
            "feature": x_test.columns,
            "value": x_test.loc[row_index].to_numpy(),
            "shap_value": shap_values[row_position],
        }
    )
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(15)

    lines = [
        f"# {model_name} SHAP Local Explanation",
        "",
        f"- feature_version: `{FEATURE_VERSION}`",
        f"- selected_test_index: `{int(row_index)}`",
        f"- true_label: `{int(y_test.loc[row_index])}`",
        f"- predicted_probability: `{float(test_prob[row_position]):.6f}`",
        f"- base_value: `{base_value:.6f}`",
        "",
        "## Top Contributions",
        "",
        "| feature | value | shap_value |",
        "| --- | ---: | ---: |",
    ]
    for _, row in contrib.iterrows():
        value = row["value"]
        value_text = "NaN" if pd.isna(value) else f"{float(value):.6f}"
        lines.append(
            f"| {row['feature']} | {value_text} | {float(row['shap_value']):.6f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model(model_name: str) -> dict[str, str]:
    paths = build_paths(ROOT_DIR)
    tabular_path = paths.processed_dir / f"tabular_features_{FEATURE_VERSION}.csv"
    model_table = pd.read_csv(tabular_path)
    x_train, y_train, _, _, x_test, y_test = split_tabular_features(
        model_table,
        paths.split_json,
    )

    model, estimator, x_train_ready, x_test_ready = fit_and_prepare_features(
        model_name,
        x_train,
        y_train,
        x_test,
    )

    explainer = shap.TreeExplainer(estimator)
    shap_values_raw = explainer.shap_values(x_test_ready)
    shap_values, base_value = extract_binary_shap(explainer, shap_values_raw)

    if hasattr(model, "predict_proba"):
        test_prob = model.predict_proba(x_test)[:, 1]
    else:
        test_prob = model.predict_proba(x_test_ready)[:, 1]

    figures_dir = paths.figures_dir
    importance_dir = paths.reports_dir / "importance"
    figures_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)

    summary_path = figures_dir / f"{model_name}_tabular_{FEATURE_VERSION}_shap_summary.png"
    global_csv_path = importance_dir / f"{model_name}_tabular_{FEATURE_VERSION}_shap_global.csv"
    local_md_path = importance_dir / f"{model_name}_tabular_{FEATURE_VERSION}_shap_local.md"

    importance = save_global_csv(shap_values, x_test_ready, global_csv_path)
    top_features = importance["feature"].head(2).tolist()
    save_summary_plot(shap_values, x_test_ready, summary_path)

    dependence_paths = []
    for feature_name in top_features:
        dep_path = figures_dir / f"{model_name}_tabular_{FEATURE_VERSION}_shap_dependence_{feature_name}.png"
        save_dependence_plot(feature_name, shap_values, x_test_ready, dep_path)
        dependence_paths.append(str(dep_path))

    save_local_markdown(
        model_name,
        x_test_ready,
        y_test,
        test_prob,
        shap_values,
        base_value,
        local_md_path,
    )

    return {
        "model": model_name,
        "summary": str(summary_path),
        "global_csv": str(global_csv_path),
        "local_md": str(local_md_path),
        "dependence": dependence_paths,
    }


def main() -> None:
    results = [run_model("lightgbm"), run_model("catboost")]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
