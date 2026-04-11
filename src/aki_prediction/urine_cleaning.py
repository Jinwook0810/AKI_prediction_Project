from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UrineCleaningStats:
    total_urine_rows: int
    rows_after_cleaning: int
    exact_duplicate_groups: int
    exact_duplicate_rows: int
    identical_duplicate_groups: int
    conflicting_duplicate_groups: int

    def to_dict(self) -> dict:
        return {
            "total_urine_rows": self.total_urine_rows,
            "rows_after_cleaning": self.rows_after_cleaning,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "identical_duplicate_groups": self.identical_duplicate_groups,
            "conflicting_duplicate_groups": self.conflicting_duplicate_groups,
        }


def _resolve_conflicting_values(values: pd.Series) -> float:
    uniq = sorted({float(v) for v in values.dropna().tolist()})
    positive_values = [v for v in uniq if v > 0]
    if positive_values:
        return min(positive_values)
    return min(uniq)


def clean_urine_raw(urine_df: pd.DataFrame) -> tuple[pd.DataFrame, UrineCleaningStats]:
    if urine_df.empty:
        stats = UrineCleaningStats(
            total_urine_rows=0,
            rows_after_cleaning=0,
            exact_duplicate_groups=0,
            exact_duplicate_rows=0,
            identical_duplicate_groups=0,
            conflicting_duplicate_groups=0,
        )
        return urine_df.copy(), stats

    df = urine_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    df = df[df["value"] >= 0].copy()

    group_cols = ["stay_id", "charttime"]
    exact_group_sizes = df.groupby(group_cols).size()
    duplicate_groups = exact_group_sizes[exact_group_sizes > 1]

    cleaned_rows: list[pd.Series] = []
    identical_duplicate_groups = 0
    conflicting_duplicate_groups = 0

    for _, group in df.groupby(group_cols, sort=False):
        base_row = group.iloc[0].copy()
        uniq_values = sorted({float(v) for v in group["value"].tolist()})

        if len(group) == 1:
            base_row["value"] = uniq_values[0]
            cleaned_rows.append(base_row)
            continue

        if len(uniq_values) == 1:
            identical_duplicate_groups += 1
            base_row["value"] = uniq_values[0]
            cleaned_rows.append(base_row)
            continue

        conflicting_duplicate_groups += 1
        base_row["value"] = _resolve_conflicting_values(group["value"])
        cleaned_rows.append(base_row)

    cleaned_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)

    stats = UrineCleaningStats(
        total_urine_rows=int(len(df)),
        rows_after_cleaning=int(len(cleaned_df)),
        exact_duplicate_groups=int(len(duplicate_groups)),
        exact_duplicate_rows=int(duplicate_groups.sum()) if len(duplicate_groups) else 0,
        identical_duplicate_groups=int(identical_duplicate_groups),
        conflicting_duplicate_groups=int(conflicting_duplicate_groups),
    )

    return cleaned_df, stats
