from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.paths import build_paths
from aki_prediction.sequence import (
    build_sequence_dataset,
    combine_value_and_mask,
    fit_sequence_scaler,
    split_sequence_dataset,
    transform_sequence_array,
)


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_best_transformer(
    transformer_module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict,
) -> torch.nn.Module:
    device = torch.device("cpu")
    transformer_module.set_seed(transformer_module.RANDOM_SEED)

    train_loader = transformer_module.make_loader(
        x_train, y_train, int(params["batch_size"]), shuffle=True
    )
    valid_loader = transformer_module.make_loader(
        x_valid, y_valid, int(params["batch_size"]), shuffle=False
    )

    model = transformer_module.TransformerClassifier(
        input_size=x_train.shape[2],
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        max_len=x_train.shape[1],
    ).to(device)

    pos_count = float(y_train.sum())
    neg_count = float(y_train.shape[0] - pos_count)
    pos_weight = torch.tensor(
        [neg_count / max(pos_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_valid_auroc = -1.0
    patience_counter = 0
    epochs = int(params.get("epochs", 30))
    patience = int(params.get("patience", 5))

    for _epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        valid_prob = transformer_module.predict_probs(model, valid_loader, device)
        valid_auroc = float(roc_auc_score(y_valid, valid_prob))
        if valid_auroc > best_valid_auroc:
            best_valid_auroc = valid_auroc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("Failed to train transformer checkpoint for permutation importance.")

    model.load_state_dict(best_state)
    return model


def main() -> None:
    paths = build_paths(ROOT_DIR)
    importance_dir = ROOT_DIR / "reports" / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    transformer_module = load_script_module(
        ROOT_DIR / "scripts" / "05_train_transformer.py",
        "train_transformer_module",
    )

    metrics_path = paths.metrics_dir / "transformer_sequence_masked_v1.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    training = metrics["training"]
    params = {
        "d_model": training["d_model"],
        "nhead": training["nhead"],
        "num_layers": training["num_layers"],
        "dropout": training["dropout"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "patience": training["patience"],
        "epochs": 30,
    }

    filtered_df = pd.read_csv(paths.filtered_csv)
    observed_mask_df = pd.read_csv(paths.filtered_observed_mask_csv)
    x_value, x_mask, y, feature_cols, stay_ids = build_sequence_dataset(
        filtered_df,
        observed_mask_df=observed_mask_df,
    )
    (
        x_train_value,
        y_train,
        x_valid_value,
        y_valid,
        x_test_value,
        y_test,
    ) = split_sequence_dataset(
        x_value,
        y,
        stay_ids,
        paths.split_json,
    )
    x_train_mask, _, x_valid_mask, _, x_test_mask, _ = split_sequence_dataset(
        x_mask,
        y,
        stay_ids,
        paths.split_json,
    )

    mean, std = fit_sequence_scaler(x_train_value)
    x_train = transform_sequence_array(x_train_value, mean, std)
    x_valid = transform_sequence_array(x_valid_value, mean, std)
    x_test = transform_sequence_array(x_test_value, mean, std)

    x_train = combine_value_and_mask(x_train, x_train_mask)
    x_valid = combine_value_and_mask(x_valid, x_valid_mask)
    x_test = combine_value_and_mask(x_test, x_test_mask)

    model = train_best_transformer(
        transformer_module,
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
    )

    device = torch.device("cpu")
    valid_loader = transformer_module.make_loader(
        x_valid, y_valid, int(params["batch_size"]), shuffle=False
    )
    test_loader = transformer_module.make_loader(
        x_test, y_test, int(params["batch_size"]), shuffle=False
    )

    valid_prob = transformer_module.predict_probs(model, valid_loader, device)
    test_prob = transformer_module.predict_probs(model, test_loader, device)
    threshold, _ = transformer_module.find_best_threshold(y_valid, valid_prob)
    baseline_auroc = float(roc_auc_score(y_test, test_prob))
    baseline_auprc = float(average_precision_score(y_test, test_prob))

    rng = np.random.default_rng(transformer_module.RANDOM_SEED)
    n_value_features = x_test_value.shape[2]
    rows = []
    for idx, feature in enumerate(feature_cols):
        perm_indices = rng.permutation(x_test.shape[0])
        x_perm = x_test.copy()
        x_perm[:, :, idx] = x_test[perm_indices, :, idx]
        x_perm[:, :, idx + n_value_features] = x_test[perm_indices, :, idx + n_value_features]

        perm_loader = transformer_module.make_loader(
            x_perm,
            y_test,
            int(params["batch_size"]),
            shuffle=False,
        )
        perm_prob = transformer_module.predict_probs(model, perm_loader, device)
        perm_auroc = float(roc_auc_score(y_test, perm_prob))
        perm_auprc = float(average_precision_score(y_test, perm_prob))
        rows.append(
            {
                "feature": feature,
                "baseline_auroc": baseline_auroc,
                "permuted_auroc": perm_auroc,
                "auroc_drop": baseline_auroc - perm_auroc,
                "baseline_auprc": baseline_auprc,
                "permuted_auprc": perm_auprc,
                "auprc_drop": baseline_auprc - perm_auprc,
                "threshold": threshold,
            }
        )

    importance_df = pd.DataFrame(rows).sort_values(
        ["auprc_drop", "auroc_drop"],
        ascending=False,
    ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    csv_path = importance_dir / "transformer_sequence_masked_permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    md_path = importance_dir / "transformer_sequence_masked_permutation_importance.md"
    top20 = importance_df.head(20)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Transformer Masked Sequence Permutation Importance\n\n")
        f.write("- baseline: `transformer_sequence_masked_v1`\n")
        f.write(f"- baseline_auroc: `{baseline_auroc:.6f}`\n")
        f.write(f"- baseline_auprc: `{baseline_auprc:.6f}`\n")
        f.write("- importance metric: feature-wise drop after shuffling both value and mask channels together across stays\n\n")
        f.write("## Top 20 by AUPRC Drop\n\n")
        f.write("| rank | feature | auprc_drop | auroc_drop |\n")
        f.write("| --- | --- | --- | --- |\n")
        for _, row in top20.iterrows():
            f.write(
                f"| {int(row['rank'])} | {row['feature']} | {row['auprc_drop']:.6f} | {row['auroc_drop']:.6f} |\n"
            )

    fig_path = paths.figures_dir / "transformer_sequence_permutation_importance.png"
    top10 = importance_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(top10["feature"], top10["auprc_drop"], color="#59A14F")
    ax.set_title("Transformer Masked Sequence Top Features\nPermutation Importance (AUPRC Drop)")
    ax.set_xlabel("AUPRC drop after permutation")
    for y_pos, value in enumerate(top10["auprc_drop"]):
        ax.text(value, y_pos, f" {value:.3f}", va="center", fontsize=9)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(csv_path)
    print(md_path)
    print(fig_path)


if __name__ == "__main__":
    main()
