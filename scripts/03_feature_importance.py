from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import split_tabular_features
from aki_prediction.paths import build_paths


RANDOM_SEED = 42


def load_best_params(paths, model_name: str, feature_version: str) -> dict:
    tuned_path = paths.metrics_dir / f"{model_name}_tuned_tabular_{feature_version}.json"
    if tuned_path.exists():
        with tuned_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        tuning = payload.get("tuning", {})
        if tuning.get("enabled") and tuning.get("best_params"):
            return tuning["best_params"]
    return {}


def build_importance_model(model_name: str, params: dict):
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


def extract_importance(model, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    if model_name == "random_forest":
        estimator = model.named_steps["model"]
        raw = estimator.feature_importances_
    elif model_name == "catboost":
        raw = model.get_feature_importance()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": raw,
        }
    ).sort_values("importance", ascending=False)

    total = importance["importance"].sum()
    importance["importance_share"] = (
        importance["importance"] / total if total > 0 else 0.0
    )
    importance["rank"] = range(1, len(importance) + 1)
    return importance


def save_markdown_summary(
    output_path: Path,
    model_name: str,
    feature_version: str,
    params: dict,
    importance: pd.DataFrame,
) -> None:
    top20 = importance.head(20).copy()
    top20["importance"] = top20["importance"].map(lambda v: f"{v:.6f}")
    top20["importance_share"] = top20["importance_share"].map(lambda v: f"{v:.6f}")

    lines = [
        f"# {model_name} Feature Importance",
        "",
        f"- feature_version: `{feature_version}`",
        f"- tuned_params_used: `{json.dumps(params, ensure_ascii=True)}`",
        "",
        "## Top 20",
        "",
        "| rank | feature | importance | importance_share |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in top20.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {row['importance']} | {row['importance_share']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model(paths, model_name: str, feature_version: str) -> tuple[Path, Path]:
    tabular_path = paths.processed_dir / f"tabular_features_{feature_version}.csv"
    model_table = pd.read_csv(tabular_path)
    x_train, y_train, _, _, _, _ = split_tabular_features(model_table, paths.split_json)
    params = load_best_params(paths, model_name, feature_version)
    model = build_importance_model(model_name, params)
    model.fit(x_train, y_train)

    importance = extract_importance(model, model_name, list(x_train.columns))

    importance_dir = paths.reports_dir / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    csv_path = importance_dir / f"{model_name}_tabular_{feature_version}_feature_importance.csv"
    md_path = importance_dir / f"{model_name}_tabular_{feature_version}_feature_importance.md"

    importance.to_csv(csv_path, index=False)
    save_markdown_summary(md_path, model_name, feature_version, params, importance)
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["random_forest", "catboost", "all"],
        default="all",
    )
    parser.add_argument(
        "--feature-version",
        choices=["v3", "v2", "compact", "v1"],
        default="v2",
    )
    args = parser.parse_args()

    paths = build_paths(ROOT_DIR)
    models = ["random_forest", "catboost"] if args.model == "all" else [args.model]

    results = []
    for model_name in models:
        csv_path, md_path = run_model(paths, model_name, args.feature_version)
        results.append((model_name, csv_path, md_path))

    for model_name, csv_path, md_path in results:
        print(f"{model_name}:")
        print(f"- csv: {csv_path}")
        print(f"- md: {md_path}")


if __name__ == "__main__":
    main()
