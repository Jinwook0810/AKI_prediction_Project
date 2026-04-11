from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
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


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 24,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        hidden = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        encoded = self.encoder(hidden)
        pooled = self.norm(encoded.mean(dim=1))
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        return logits


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_and_evaluate_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    device: torch.device,
    *,
    batch_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    patience: int,
    epochs: int,
    use_mask: bool,
) -> dict:
    train_loader = make_loader(x_train, y_train, batch_size, shuffle=True)
    valid_loader = make_loader(x_valid, y_valid, batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)

    model = TransformerClassifier(
        input_size=x_train.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        max_len=x_train.shape[1],
    ).to(device)

    pos_count = float(y_train.sum())
    neg_count = float(y_train.shape[0] - pos_count)
    pos_weight = torch.tensor(
        [neg_count / max(pos_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_valid_auroc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        valid_prob = predict_probs(model, valid_loader, device)
        valid_auroc = float(roc_auc_score(y_valid, valid_prob))
        valid_auprc = float(average_precision_score(y_valid, valid_prob))
        threshold, valid_best_f1 = find_best_threshold(y_valid, valid_prob)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "valid_auroc": valid_auroc,
            "valid_auprc": valid_auprc,
            "valid_best_f1": valid_best_f1,
            "valid_best_threshold": threshold,
        }
        history.append(epoch_summary)

        if valid_auroc > best_valid_auroc:
            best_valid_auroc = valid_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("Transformer training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    valid_prob = predict_probs(model, valid_loader, device)
    test_prob = predict_probs(model, test_loader, device)
    threshold, valid_best_f1 = find_best_threshold(y_valid, valid_prob)
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    return {
        "model": "transformer",
        "input_variant": "masked" if use_mask else "value_only",
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
        "features": {
            "n_features": int(x_train.shape[2]),
            "sequence_length": int(x_train.shape[1]),
            "train_n": int(x_train.shape[0]),
            "valid_n": int(x_valid.shape[0]),
            "test_n": int(x_test.shape[0]),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_valid_auroc": best_valid_auroc,
            "epochs_ran": len(history),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "use_mask": use_mask,
            "feature_columns": feature_cols,
            "mask_feature_columns": [f"{col}_observed" for col in feature_cols]
            if use_mask
            else [],
            "history": history,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--tune-hyperparameters", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cpu")
    paths = build_paths(ROOT_DIR)

    filtered_df = pd.read_csv(paths.filtered_csv)
    observed_mask_df = None
    if args.use_mask:
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

    if args.use_mask:
        x_train = combine_value_and_mask(x_train, x_train_mask)
        x_valid = combine_value_and_mask(x_valid, x_valid_mask)
        x_test = combine_value_and_mask(x_test, x_test_mask)

    if args.tune_hyperparameters:
        grid = list(
            itertools.product(
                [(64, 4), (128, 8)],
                [0.2, 0.3],
                [1e-3, 5e-4],
                [2, 3],
            )
        )
        best_metrics = None
        trials = []

        for (d_model, nhead), dropout, learning_rate, num_layers in grid:
            set_seed(RANDOM_SEED)
            trial_metrics = train_and_evaluate_transformer(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
                feature_cols,
                device,
                batch_size=args.batch_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                patience=args.patience,
                epochs=args.epochs,
                use_mask=args.use_mask,
            )
            trials.append(
                {
                    "params": {
                        "d_model": d_model,
                        "nhead": nhead,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "num_layers": num_layers,
                    },
                    "valid_best_f1": trial_metrics["valid_best_f1"],
                    "best_valid_auroc": trial_metrics["training"]["best_valid_auroc"],
                    "best_epoch": trial_metrics["training"]["best_epoch"],
                    "test_auroc": trial_metrics["test"]["auroc"],
                    "test_auprc": trial_metrics["test"]["auprc"],
                    "test_f1": trial_metrics["test"]["f1"],
                }
            )

            if best_metrics is None:
                best_metrics = trial_metrics
                continue

            candidate_key = (
                trial_metrics["valid_best_f1"],
                trial_metrics["training"]["best_valid_auroc"],
            )
            best_key = (
                best_metrics["valid_best_f1"],
                best_metrics["training"]["best_valid_auroc"],
            )
            if candidate_key > best_key:
                best_metrics = trial_metrics

        if best_metrics is None:
            raise RuntimeError("No Transformer tuning trial completed.")

        metrics = best_metrics
        metrics["tuning"] = {
            "enabled": True,
            "n_trials": len(trials),
            "best_params": {
                "d_model": metrics["training"]["d_model"],
                "nhead": metrics["training"]["nhead"],
                "dropout": metrics["training"]["dropout"],
                "learning_rate": metrics["training"]["learning_rate"],
                "num_layers": metrics["training"]["num_layers"],
            },
            "trials": trials,
        }
    else:
        metrics = train_and_evaluate_transformer(
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            feature_cols,
            device,
            batch_size=args.batch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            patience=args.patience,
            epochs=args.epochs,
            use_mask=args.use_mask,
        )

    metrics["features"]["n_value_features"] = int(x_train_value.shape[2])
    metrics["features"]["n_mask_features"] = int(x_train_mask.shape[2]) if args.use_mask else 0

    if args.tune_hyperparameters:
        metrics_filename = (
            "transformer_sequence_masked_tuned_v1.json"
            if args.use_mask
            else "transformer_sequence_tuned_v1.json"
        )
    else:
        metrics_filename = (
            "transformer_sequence_masked_v1.json"
            if args.use_mask
            else "transformer_sequence_v1.json"
        )
    metrics_path = paths.metrics_dir / metrics_filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
