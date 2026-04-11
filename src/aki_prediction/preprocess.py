from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ProjectPaths
from .urine_cleaning import clean_urine_raw
from .value_cleaning import clean_values_by_range


@dataclass(frozen=True)
class PreprocessArtifacts:
    hourly_csv: Path
    filtered_csv: Path
    filtered_observed_mask_csv: Path
    cohort_summary_json: Path


def read_raw_df(paths: ProjectPaths) -> pd.DataFrame:
    if paths.raw_csv_gz.exists():
        return pd.read_csv(paths.raw_csv_gz, compression="gzip")
    if paths.raw_csv.exists():
        return pd.read_csv(paths.raw_csv)
    raise FileNotFoundError(
        "Missing released_df.csv.gz or released_df.csv in data directory: "
        f"{paths.data_dir}"
    )


def read_split_ids(paths: ProjectPaths) -> dict:
    if not paths.split_json.exists():
        raise FileNotFoundError(f"Missing split file: {paths.split_json}")

    with paths.split_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_raw_events(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    urine = df[df["label"] == "Urine"].copy()
    non_urine = df[df["label"] != "Urine"].copy()

    urine_clean, urine_stats = clean_urine_raw(urine)
    combined = pd.concat([non_urine, urine_clean], ignore_index=True)

    stats = {
        "urine_cleaning": urine_stats.to_dict(),
    }
    return combined, stats


def build_hourly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["charttime_td"] = pd.to_timedelta(df["charttime"] + ":00")
    df = df.sort_values(["stay_id", "label", "charttime_td"])
    df["hour"] = (df["charttime_td"].dt.total_seconds() // 3600).astype(int)

    urine = df[df["label"] == "Urine"].copy()
    urine_hourly = urine.groupby(
        ["stay_id", "hour", "label"], as_index=False
    )["value"].sum()

    creat = df[df["label"] == "Creatinine"].copy()
    creat_hourly = creat.groupby(
        ["stay_id", "hour", "label"], as_index=False
    )["value"].min()

    others = df[~df["label"].isin(["Urine", "Creatinine"])].copy()
    others_hourly = others.groupby(
        ["stay_id", "hour", "label"], as_index=False
    )["value"].last()

    hourly_long = pd.concat(
        [urine_hourly, creat_hourly, others_hourly],
        ignore_index=True,
    )

    hourly = hourly_long.pivot_table(
        index=["stay_id", "hour"],
        columns="label",
        values="value",
    ).sort_index()

    def add_missing_hours(group: pd.DataFrame) -> pd.DataFrame:
        stay_id = group.index.get_level_values("stay_id")[0]
        hours = group.index.get_level_values("hour")
        full_hours = range(int(hours.min()), int(hours.max()) + 1)
        new_index = pd.MultiIndex.from_product(
            [[stay_id], full_hours],
            names=["stay_id", "hour"],
        )
        return group.reindex(new_index)

    hourly_full = hourly.groupby(level="stay_id", group_keys=False).apply(
        add_missing_hours
    )
    hourly_clean, value_cleaning_stats = clean_values_by_range(hourly_full)
    observed_mask = hourly_clean.notna().astype("int8").reset_index()

    cols_no_ffill = ["Urine", "Creatinine"]
    cols_ffill = [col for col in hourly_clean.columns if col not in cols_no_ffill]

    hourly_ffill = hourly_clean.copy()
    hourly_ffill[cols_ffill] = hourly_clean[cols_ffill].groupby(level="stay_id").ffill()

    return hourly_ffill.reset_index(), observed_mask, value_cleaning_stats


def detect_creatinine_aki(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("hour")
    creat = group["Creatinine"]
    baseline_48h = creat.rolling(window=48, min_periods=2).min()
    delta = creat - baseline_48h
    group["aki_creat"] = (delta >= 0.3).astype(int)
    return group


def detect_urine_aki(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("hour")
    group["Weight"] = group["Weight"].ffill().bfill()
    urine_rate = group["Urine"] / group["Weight"]
    low = (urine_rate <= 0.5).astype(int)
    low_6h = low.rolling(window=6, min_periods=6).sum() == 6
    group["aki_urine"] = low_6h.astype(int)
    return group


def _apply_per_stay(df: pd.DataFrame, func) -> pd.DataFrame:
    def wrapper(group: pd.DataFrame) -> pd.DataFrame:
        stay_id = group.name
        work = group.copy()
        work = func(work)
        work["stay_id"] = stay_id
        return work

    result = df.groupby("stay_id", group_keys=False).apply(
        wrapper,
        include_groups=False,
    )
    columns = ["stay_id"] + [col for col in result.columns if col != "stay_id"]
    return result[columns].reset_index(drop=True)


def add_aki_flags(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = _apply_per_stay(hourly_df, detect_creatinine_aki)
    df = _apply_per_stay(df, detect_urine_aki)
    df["aki"] = ((df["aki_creat"] == 1) | (df["aki_urine"] == 1)).astype(int)
    df["aki"] = df.groupby("stay_id")["aki"].transform("max")
    return df


def remove_early_aki(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    early_mask = (
        (df["hour"] >= 0)
        & (df["hour"] <= 23)
        & ((df["aki_creat"] == 1) | (df["aki_urine"] == 1))
    )
    early_ids = pd.Index(df.loc[early_mask, "stay_id"].unique(), name="stay_id")
    filtered = df[~df["stay_id"].isin(early_ids)].copy()
    return filtered, early_ids


def build_cohort_summary(
    filtered_df: pd.DataFrame,
    split_ids: dict,
    early_ids: pd.Index,
    raw_cleaning_stats: dict | None = None,
) -> dict:
    target_mask = (filtered_df["hour"] >= 24) & (filtered_df["hour"] <= 47)
    target_df = filtered_df[target_mask].copy()
    target_df["_aki_future"] = (
        (target_df["aki_creat"] == 1) | (target_df["aki_urine"] == 1)
    ).astype(int)
    stay_target = target_df.groupby("stay_id")["_aki_future"].max()

    summary = {
        "split_source": "split_stay_id.json",
        "rules": {
            "input_hours": [0, 23],
            "target_hours": [24, 47],
            "remove_early_aki": True,
        },
        "raw_cleaning": raw_cleaning_stats or {},
        "counts": {
            "filtered_rows": int(filtered_df.shape[0]),
            "filtered_stays": int(filtered_df["stay_id"].nunique()),
            "removed_early_aki_stays": int(len(early_ids)),
        },
        "splits": {},
    }

    for split_name in ("train", "valid", "test"):
        ids = set(split_ids[split_name])
        subset = filtered_df[filtered_df["stay_id"].isin(ids)].copy()
        split_stays = pd.Index(subset["stay_id"].unique())
        stay_level = stay_target.reindex(split_stays, fill_value=0)
        summary["splits"][split_name] = {
            "rows": int(subset.shape[0]),
            "stays": int(subset["stay_id"].nunique()),
            "aki_positive_stays": int((stay_level == 1).sum()),
            "aki_negative_stays": int((stay_level == 0).sum()),
        }

    return summary


def run_preprocessing(paths: ProjectPaths) -> PreprocessArtifacts:
    raw_df = read_raw_df(paths)
    cleaned_raw_df, raw_cleaning_stats = clean_raw_events(raw_df)
    split_ids = read_split_ids(paths)

    hourly_df, observed_mask_df, value_cleaning_stats = build_hourly_features(cleaned_raw_df)
    hourly_df.to_csv(paths.hourly_csv, index=False)

    with_aki = add_aki_flags(hourly_df)
    filtered_df, early_ids = remove_early_aki(with_aki)
    filtered_observed_mask_df = observed_mask_df[
        ~observed_mask_df["stay_id"].isin(early_ids)
    ].copy()
    filtered_observed_mask_df.to_csv(paths.filtered_observed_mask_csv, index=False)
    filtered_df.to_csv(paths.filtered_csv, index=False)

    summary = build_cohort_summary(
        filtered_df,
        split_ids,
        early_ids,
        raw_cleaning_stats={
            **raw_cleaning_stats,
            "value_cleaning": value_cleaning_stats,
        },
    )
    with paths.cohort_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return PreprocessArtifacts(
        hourly_csv=paths.hourly_csv,
        filtered_csv=paths.filtered_csv,
        filtered_observed_mask_csv=paths.filtered_observed_mask_csv,
        cohort_summary_json=paths.cohort_summary_json,
    )
