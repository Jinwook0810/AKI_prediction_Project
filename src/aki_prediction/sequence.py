from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import HOUR_COL, ID_COL, INPUT_END_HOUR, INPUT_START_HOUR
from .value_cleaning import EDA_DERIVED_PLAUSIBILITY_RANGES


LABEL_COLS = {"aki", "aki_creat", "aki_urine"}
TARGET_COL = "aki_future"
TARGET_START_HOUR = 24
TARGET_END_HOUR = 47


def _build_target(filtered_df: pd.DataFrame) -> pd.DataFrame:
    target_window = filtered_df[
        (filtered_df[HOUR_COL] >= TARGET_START_HOUR)
        & (filtered_df[HOUR_COL] <= TARGET_END_HOUR)
    ].copy()
    target_window["_aki_event"] = (
        (target_window["aki_creat"] == 1) | (target_window["aki_urine"] == 1)
    ).astype(int)
    return (
        target_window.groupby(ID_COL)["_aki_event"]
        .max()
        .rename(TARGET_COL)
        .reset_index()
    )


def add_sequence_derived_features(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    if {"Urine", "Weight"}.issubset(df.columns):
        weight = df["Weight"].where(df["Weight"] > 0)
        urine_rate = df["Urine"] / weight
        _, upper = EDA_DERIVED_PLAUSIBILITY_RANGES["Urine_ml_per_kg_hr"]
        urine_rate = urine_rate.mask(urine_rate > upper)
        df["Urine_ml_per_kg_hr"] = urine_rate

    if {"BUN", "Creatinine"}.issubset(df.columns):
        df["BUN_Creatinine_ratio"] = df["BUN"] / df["Creatinine"].where(
            df["Creatinine"] > 0
        )

    return df


def _build_sequence_mask_frame(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    observed_mask_df: pd.DataFrame | None,
) -> pd.DataFrame:
    mask_frame = pd.concat(
        [
            input_df[[ID_COL, HOUR_COL]].reset_index(drop=True),
            input_df[feature_cols].notna().astype(np.float32).reset_index(drop=True),
        ],
        axis=1,
    )

    if observed_mask_df is None:
        return mask_frame

    mask_input = observed_mask_df[
        (observed_mask_df[HOUR_COL] >= INPUT_START_HOUR)
        & (observed_mask_df[HOUR_COL] <= INPUT_END_HOUR)
    ].copy()
    mask_input = mask_input.sort_values([ID_COL, HOUR_COL]).reset_index(drop=True)

    overlap_cols = [col for col in feature_cols if col in mask_input.columns]
    if overlap_cols:
        merged = mask_frame.merge(
            mask_input[[ID_COL, HOUR_COL, *overlap_cols]],
            on=[ID_COL, HOUR_COL],
            how="left",
            validate="one_to_one",
            suffixes=("", "_observed"),
        )
        for col in overlap_cols:
            observed_col = f"{col}_observed"
            merged[col] = merged[observed_col].fillna(merged[col]).astype(np.float32)
        keep_cols = [ID_COL, HOUR_COL, *feature_cols]
        mask_frame = merged[keep_cols].copy()

    return mask_frame


def build_sequence_dataset(
    filtered_df: pd.DataFrame,
    observed_mask_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str], np.ndarray]:
    input_df = filtered_df[
        (filtered_df[HOUR_COL] >= INPUT_START_HOUR)
        & (filtered_df[HOUR_COL] <= INPUT_END_HOUR)
    ].copy()
    input_df = add_sequence_derived_features(input_df)
    input_df = input_df.sort_values([ID_COL, HOUR_COL])

    feature_cols = [
        col
        for col in input_df.columns
        if col not in {ID_COL, HOUR_COL, *LABEL_COLS}
    ]
    feature_df = input_df[[ID_COL, HOUR_COL, *feature_cols]].copy()
    mask_df = _build_sequence_mask_frame(input_df, feature_cols, observed_mask_df)

    sequence_lengths = feature_df.groupby(ID_COL).size()
    if not (sequence_lengths == (INPUT_END_HOUR - INPUT_START_HOUR + 1)).all():
        raise ValueError("Every stay must have exactly 24 input hours for sequence modeling.")

    stay_ids = sequence_lengths.index.to_numpy()
    n_stays = stay_ids.shape[0]
    n_features = len(feature_cols)
    x = feature_df[feature_cols].to_numpy(dtype=np.float32).reshape(n_stays, 24, n_features)
    x_mask = mask_df[feature_cols].to_numpy(dtype=np.float32).reshape(n_stays, 24, n_features)

    target = _build_target(filtered_df)
    target_by_stay = target.set_index(ID_COL).reindex(stay_ids)[TARGET_COL].fillna(0).astype(int)
    y = target_by_stay.to_numpy(dtype=np.float32)
    return x, x_mask, y, feature_cols, stay_ids


def split_sequence_dataset(
    x: np.ndarray,
    y: np.ndarray,
    stay_ids: np.ndarray,
    split_json_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with split_json_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    index_by_stay = {int(stay_id): idx for idx, stay_id in enumerate(stay_ids)}

    def subset(split_name: str) -> tuple[np.ndarray, np.ndarray]:
        indices = [
            index_by_stay[int(stay_id)]
            for stay_id in split_ids[split_name]
            if int(stay_id) in index_by_stay
        ]
        return x[indices], y[indices]

    x_train, y_train = subset("train")
    x_valid, y_valid = subset("valid")
    x_test, y_test = subset("test")
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def fit_sequence_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=(0, 1))
    mean = np.where(np.isnan(mean), 0.0, mean)

    std = np.nanstd(x_train, axis=(0, 1))
    std = np.where(np.isnan(std) | (std == 0), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_sequence_array(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    filled = np.where(np.isnan(x), mean.reshape(1, 1, -1), x)
    return ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def combine_value_and_mask(x_value: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
    return np.concatenate([x_value, x_mask.astype(np.float32)], axis=2)
