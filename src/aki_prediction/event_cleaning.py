from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STATIC_FIRST = {"Age", "Gender", "Height", "ICUType"}
STATIC_LAST = {"Weight"}
VITALS_MEDIAN = {
    "HR",
    "SysABP",
    "DiasABP",
    "MAP",
    "NISysABP",
    "NIDiasABP",
    "NIMAP",
    "RespRate",
    "Temp",
    "SaO2",
}
RENAL_SUM = {"Urine"}
RENAL_LAST = {"Creatinine", "BUN"}
CHEMISTRY_LAST = {
    "ALP",
    "ALT",
    "AST",
    "Albumin",
    "Bilirubin",
    "Cholesterol",
    "Glucose",
    "HCO3",
    "HCT",
    "K",
    "Lactate",
    "Mg",
    "Na",
    "PaCO2",
    "PaO2",
    "Platelets",
    "TroponinI",
    "TroponinT",
    "WBC",
    "pH",
}
SUPPORT_LAST = {"FiO2", "GCS"}
SUPPORT_MAX = {"MechVent"}


LABEL_GROUPS = {
    "static_first": STATIC_FIRST,
    "static_last": STATIC_LAST,
    "vitals_median": VITALS_MEDIAN,
    "renal_sum": RENAL_SUM,
    "renal_last": RENAL_LAST,
    "chemistry_last": CHEMISTRY_LAST,
    "support_last": SUPPORT_LAST,
    "support_max": SUPPORT_MAX,
}


def labels_with_ffill() -> set[str]:
    return STATIC_FIRST | STATIC_LAST | VITALS_MEDIAN | SUPPORT_LAST | SUPPORT_MAX


def all_known_labels() -> set[str]:
    known: set[str] = set()
    for values in LABEL_GROUPS.values():
        known |= values
    return known


@dataclass(frozen=True)
class RawEventCleaningStats:
    total_rows: int
    rows_after_cleaning: int
    duplicate_groups: int
    duplicate_rows: int
    conflicting_duplicate_groups: int

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "rows_after_cleaning": self.rows_after_cleaning,
            "duplicate_groups": self.duplicate_groups,
            "duplicate_rows": self.duplicate_rows,
            "conflicting_duplicate_groups": self.conflicting_duplicate_groups,
        }


def _resolve_min_positive(values: pd.Series) -> float:
    uniq = sorted({float(v) for v in values.dropna().tolist()})
    positive = [v for v in uniq if v > 0]
    if positive:
        return min(positive)
    return min(uniq)


def _resolve_last(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[-1])


def _resolve_first(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[0])


def _resolve_median(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.median())


def _resolve_max(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.max())


def resolve_same_timestamp_value(label: str, values: pd.Series) -> float:
    if label in RENAL_SUM:
        return _resolve_min_positive(values)
    if label in VITALS_MEDIAN or label in STATIC_LAST:
        return _resolve_median(values)
    if label in SUPPORT_MAX:
        return _resolve_max(values)
    if label in STATIC_FIRST:
        return _resolve_first(values)
    return _resolve_last(values)


def clean_raw_events_by_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df.copy(), {"overall": RawEventCleaningStats(0, 0, 0, 0, 0).to_dict(), "by_label": {}}

    work = df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work[work["value"].notna()].copy()

    cleaned_frames: list[pd.DataFrame] = []
    label_stats: dict[str, dict] = {}
    total_rows = 0
    total_cleaned = 0
    total_duplicate_groups = 0
    total_duplicate_rows = 0
    total_conflicting_groups = 0

    for label, label_df in work.groupby("label", sort=True):
        label_df = label_df.copy()
        total_rows += int(len(label_df))

        group_cols = ["stay_id", "charttime"]
        group_sizes = label_df.groupby(group_cols).size()
        duplicate_groups = group_sizes[group_sizes > 1]

        rows = []
        conflicting_groups = 0
        for _, group in label_df.groupby(group_cols, sort=False):
            base_row = group.iloc[-1].copy()
            uniq_values = pd.unique(group["value"].dropna())
            if len(uniq_values) > 1:
                conflicting_groups += 1
            base_row["value"] = resolve_same_timestamp_value(label, group["value"])
            rows.append(base_row)

        cleaned_label_df = pd.DataFrame(rows).reset_index(drop=True)
        cleaned_frames.append(cleaned_label_df)

        stats = RawEventCleaningStats(
            total_rows=int(len(label_df)),
            rows_after_cleaning=int(len(cleaned_label_df)),
            duplicate_groups=int(len(duplicate_groups)),
            duplicate_rows=int(duplicate_groups.sum()) if len(duplicate_groups) else 0,
            conflicting_duplicate_groups=int(conflicting_groups),
        )
        label_stats[label] = stats.to_dict()
        total_cleaned += stats.rows_after_cleaning
        total_duplicate_groups += stats.duplicate_groups
        total_duplicate_rows += stats.duplicate_rows
        total_conflicting_groups += stats.conflicting_duplicate_groups

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    overall = RawEventCleaningStats(
        total_rows=total_rows,
        rows_after_cleaning=total_cleaned,
        duplicate_groups=total_duplicate_groups,
        duplicate_rows=total_duplicate_rows,
        conflicting_duplicate_groups=total_conflicting_groups,
    )
    return cleaned, {"overall": overall.to_dict(), "by_label": label_stats}


def aggregate_hourly_by_label(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def _group_values(subset: pd.DataFrame, how: str) -> pd.DataFrame:
        grouped = subset.groupby(["stay_id", "hour", "label"], as_index=False)["value"]
        if how == "sum":
            return grouped.sum()
        if how == "median":
            return grouped.median()
        if how == "max":
            return grouped.max()
        if how == "first":
            return grouped.first()
        return grouped.last()

    for label_set, how in (
        (STATIC_FIRST, "first"),
        (STATIC_LAST, "last"),
        (VITALS_MEDIAN, "median"),
        (RENAL_SUM, "sum"),
        (RENAL_LAST, "last"),
        (CHEMISTRY_LAST, "last"),
        (SUPPORT_LAST, "last"),
        (SUPPORT_MAX, "max"),
    ):
        subset = df[df["label"].isin(label_set)].copy()
        if subset.empty:
            continue
        frames.append(_group_values(subset, how))

    remaining = df[~df["label"].isin(all_known_labels())].copy()
    if not remaining.empty:
        frames.append(_group_values(remaining, "last"))

    return pd.concat(frames, ignore_index=True)
