from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .value_cleaning import EDA_DERIVED_PLAUSIBILITY_RANGES


INPUT_START_HOUR = 0
INPUT_END_HOUR = 23
TARGET_START_HOUR = 24
TARGET_END_HOUR = 47
INPUT_WINDOW_HOURS = INPUT_END_HOUR - INPUT_START_HOUR + 1

ID_COL = "stay_id"
HOUR_COL = "hour"
TARGET_COL = "aki_future"
SPLIT_COL = "split"
LABEL_COLS = {"aki", "aki_creat", "aki_urine"}
STATIC_COLS = {"Age", "Gender", "Height", "ICUType"}
COMPACT_DROP_COLS = {"MechVent"}


def _first_valid(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.empty else values.iloc[0]


def _last_valid(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.empty else values.iloc[-1]


def _delta(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.shape[0] < 2 else values.iloc[-1] - values.iloc[0]


def _sum_or_nan(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.empty else values.sum()


def _mean_or_nan(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.empty else values.mean()


def _min_or_nan(series: pd.Series) -> float:
    values = series.dropna()
    return np.nan if values.empty else values.min()


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def _missing_rate(series: pd.Series, denominator: int = INPUT_WINDOW_HOURS) -> float:
    return 1.0 - float(series.notna().sum()) / float(denominator)


def _build_target(filtered_df: pd.DataFrame) -> pd.DataFrame:
    target_window = filtered_df[
        (filtered_df[HOUR_COL] >= TARGET_START_HOUR)
        & (filtered_df[HOUR_COL] <= TARGET_END_HOUR)
    ].copy()

    target_window["_aki_event"] = (
        (target_window["aki_creat"] == 1) | (target_window["aki_urine"] == 1)
    ).astype(int)

    target = (
        target_window.groupby(ID_COL)["_aki_event"]
        .max()
        .rename(TARGET_COL)
        .reset_index()
    )
    return target


def _summarize_static(group: pd.DataFrame, col: str) -> dict[str, float]:
    return {f"{col}_first": _first_valid(group[col])}


def _summarize_urine(group: pd.DataFrame, compact: bool = False) -> dict[str, float]:
    series = group["Urine"]
    last6 = group[group[HOUR_COL] >= INPUT_END_HOUR - 5]["Urine"]
    last12 = group[group[HOUR_COL] >= INPUT_END_HOUR - 11]["Urine"]

    summary = {
        "Urine_sum": _sum_or_nan(series),
        "Urine_last6_sum": _sum_or_nan(last6),
        "Urine_last12_sum": _sum_or_nan(last12),
        "Urine_count": float(series.notna().sum()),
    }
    if not compact:
        summary["Urine_missing_rate"] = _missing_rate(series)
    return summary


def _summarize_creatinine(group: pd.DataFrame, compact: bool = False) -> dict[str, float]:
    series = group["Creatinine"]
    summary = {
        "Creatinine_mean": series.mean(),
        "Creatinine_max": series.max(),
        "Creatinine_last": _last_valid(series),
        "Creatinine_delta": _delta(series),
        "Creatinine_count": float(series.notna().sum()),
    }
    if not compact:
        summary["Creatinine_missing_rate"] = _missing_rate(series)
    return summary


def _summarize_renal_v2(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values(HOUR_COL).copy()

    weight = group["Weight"].ffill().bfill()
    urine_per_kg_hr = group["Urine"] / weight.where(weight > 0)
    _, urine_rate_upper = EDA_DERIVED_PLAUSIBILITY_RANGES["Urine_ml_per_kg_hr"]
    urine_per_kg_hr = urine_per_kg_hr.mask(urine_per_kg_hr > urine_rate_upper)
    last6 = urine_per_kg_hr[group[HOUR_COL] >= INPUT_END_HOUR - 5]
    last12 = urine_per_kg_hr[group[HOUR_COL] >= INPUT_END_HOUR - 11]

    creatinine = group["Creatinine"]
    creat_first = _first_valid(creatinine)
    creat_last = _last_valid(creatinine)
    creat_min = creatinine.min()
    creat_max = creatinine.max()

    bun = group["BUN"] if "BUN" in group else pd.Series(dtype=float)
    bun_last = _last_valid(bun)
    bun_mean = bun.mean()
    creat_mean = creatinine.mean()

    return {
        "Urine_ml_per_kg_hr_mean": _mean_or_nan(urine_per_kg_hr),
        "Urine_ml_per_kg_hr_min": _min_or_nan(urine_per_kg_hr),
        "Urine_ml_per_kg_hr_last6_mean": _mean_or_nan(last6),
        "Urine_ml_per_kg_hr_last12_mean": _mean_or_nan(last12),
        "Creatinine_relative_delta": _safe_divide(creat_last - creat_first, creat_first),
        "Creatinine_max_minus_min": creat_max - creat_min,
        "BUN_Creatinine_ratio_last": _safe_divide(bun_last, creat_last),
        "BUN_Creatinine_ratio_mean": _safe_divide(bun_mean, creat_mean),
    }


def _summarize_dynamic(
    group: pd.DataFrame,
    col: str,
    compact: bool = False,
) -> dict[str, float]:
    series = group[col]
    if compact:
        return {
            f"{col}_mean": series.mean(),
            f"{col}_last": _last_valid(series),
            f"{col}_delta": _delta(series),
            f"{col}_count": float(series.notna().sum()),
        }

    return {
        f"{col}_mean": series.mean(),
        f"{col}_min": series.min(),
        f"{col}_max": series.max(),
        f"{col}_last": _last_valid(series),
        f"{col}_std": series.std(),
        f"{col}_delta": _delta(series),
        f"{col}_count": float(series.notna().sum()),
        f"{col}_missing_rate": _missing_rate(series),
    }


def _summarize_stay(
    group: pd.DataFrame,
    feature_cols: list[str],
    compact: bool = False,
) -> dict[str, float]:
    group = group.sort_values(HOUR_COL)
    row: dict[str, float] = {ID_COL: group[ID_COL].iloc[0]}

    for col in feature_cols:
        if col in STATIC_COLS:
            row.update(_summarize_static(group, col))
        elif col == "Urine":
            row.update(_summarize_urine(group, compact=compact))
        elif col == "Creatinine":
            row.update(_summarize_creatinine(group, compact=compact))
        else:
            row.update(_summarize_dynamic(group, col, compact=compact))

    return row


def build_tabular_features(
    filtered_df: pd.DataFrame,
    feature_version: str = "v1",
) -> pd.DataFrame:
    if feature_version not in {"v1", "v2", "compact"}:
        raise ValueError(f"Unsupported feature_version: {feature_version}")

    input_df = filtered_df[
        (filtered_df[HOUR_COL] >= INPUT_START_HOUR)
        & (filtered_df[HOUR_COL] <= INPUT_END_HOUR)
    ].copy()

    feature_cols = [
        col
        for col in filtered_df.columns
        if col not in {ID_COL, HOUR_COL, *LABEL_COLS}
    ]
    if feature_version == "compact":
        feature_cols = [col for col in feature_cols if col not in COMPACT_DROP_COLS]

    rows = [
        _summarize_stay(
            group,
            feature_cols,
            compact=feature_version == "compact",
        )
        for _, group in input_df.groupby(ID_COL, sort=False)
    ]
    features = pd.DataFrame(rows)

    if feature_version in {"v2", "compact"}:
        renal_rows = [
            {ID_COL: group[ID_COL].iloc[0], **_summarize_renal_v2(group)}
            for _, group in input_df.groupby(ID_COL, sort=False)
        ]
        renal_features = pd.DataFrame(renal_rows)
        features = features.merge(renal_features, on=ID_COL, how="left")

    target = _build_target(filtered_df)
    model_table = features.merge(target, on=ID_COL, how="left")
    model_table[TARGET_COL] = model_table[TARGET_COL].fillna(0).astype(int)
    return model_table


def add_split_column(
    model_table: pd.DataFrame,
    split_json_path: Path,
) -> pd.DataFrame:
    with split_json_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    split_by_stay_id = {
        stay_id: split_name
        for split_name, stay_ids in split_ids.items()
        for stay_id in stay_ids
    }

    table = model_table.copy()
    table[SPLIT_COL] = table[ID_COL].map(split_by_stay_id)
    return table


def split_tabular_features(
    model_table: pd.DataFrame,
    split_json_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    with split_json_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    def subset(split_name: str) -> pd.DataFrame:
        if SPLIT_COL in model_table.columns:
            return model_table[model_table[SPLIT_COL] == split_name].copy()
        return model_table[model_table[ID_COL].isin(split_ids[split_name])].copy()

    train = subset("train")
    valid = subset("valid")
    test = subset("test")

    x_cols = [
        col for col in model_table.columns if col not in {ID_COL, TARGET_COL, SPLIT_COL}
    ]

    return (
        train[x_cols],
        train[TARGET_COL],
        valid[x_cols],
        valid[TARGET_COL],
        test[x_cols],
        test[TARGET_COL],
    )
