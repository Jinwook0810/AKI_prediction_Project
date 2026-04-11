from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.features import (
    ID_COL,
    INPUT_END_HOUR,
    INPUT_START_HOUR,
    TARGET_COL,
    add_split_column,
    build_tabular_features,
)
from aki_prediction.paths import build_paths
from aki_prediction.value_cleaning import (
    BROAD_PLAUSIBILITY_RANGES,
    EDA_DERIVED_PLAUSIBILITY_RANGES,
)


LABEL_COLS = {"aki", "aki_creat", "aki_urine", TARGET_COL}
META_COLS = {ID_COL, "hour", "split", *LABEL_COLS}

KEY_VARIABLES = [
    "Creatinine",
    "Urine",
    "Weight",
    "BUN",
    "HR",
    "SysABP",
    "DiasABP",
    "MAP",
    "NISysABP",
    "NIDiasABP",
    "NIMAP",
    "RespRate",
    "SaO2",
    "Temp",
    "GCS",
    "WBC",
    "HCT",
    "Platelets",
    "Na",
    "K",
    "HCO3",
    "Glucose",
    "pH",
]

NON_NEGATIVE_VARIABLES = [
    "ALP",
    "ALT",
    "AST",
    "Age",
    "Albumin",
    "BUN",
    "Bilirubin",
    "Cholesterol",
    "Creatinine",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "Height",
    "K",
    "Lactate",
    "MAP",
    "MechVent",
    "Mg",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Na",
    "PaCO2",
    "PaO2",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
]

STATIC_COLS = ["Age", "Gender", "Height", "ICUType"]

EDA_PLAUSIBILITY_RANGES = {
    **BROAD_PLAUSIBILITY_RANGES,
    **EDA_DERIVED_PLAUSIBILITY_RANGES,
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_map(split_json_path: Path) -> dict[int, str]:
    split_ids = read_json(split_json_path)
    return {
        int(stay_id): split_name
        for split_name, stay_ids in split_ids.items()
        for stay_id in stay_ids
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.4g}"
            )
        else:
            display_df[col] = display_df[col].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    headers = [str(col) for col in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display_df.columns) + " |")

    return "\n".join(lines)


def summarize_cohort(tabular: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, group in tabular.groupby("split", dropna=False):
        rows.append(
            {
                "split": split_name,
                "stays": int(group[ID_COL].nunique()),
                "aki_positive_stays": int(group[TARGET_COL].sum()),
                "aki_negative_stays": int((group[TARGET_COL] == 0).sum()),
                "aki_positive_rate": float(group[TARGET_COL].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("split")


def summarize_missingness(input_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        series = input_df[col]
        pos = input_df[input_df[TARGET_COL] == 1][col]
        neg = input_df[input_df[TARGET_COL] == 0][col]
        rows.append(
            {
                "variable": col,
                "missing_rate": float(series.isna().mean()),
                "observed_rate": float(series.notna().mean()),
                "positive_missing_rate": float(pos.isna().mean()),
                "negative_missing_rate": float(neg.isna().mean()),
                "pos_neg_missing_rate_diff": float(pos.isna().mean() - neg.isna().mean()),
                "observed_rows": int(series.notna().sum()),
                "total_rows": int(len(series)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "variable"], ascending=[False, True]
    )


def summarize_missingness_by_split(
    input_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for split_name, split_df in input_df.groupby("split"):
        for col in feature_cols:
            rows.append(
                {
                    "split": split_name,
                    "variable": col,
                    "missing_rate": float(split_df[col].isna().mean()),
                    "observed_rows": int(split_df[col].notna().sum()),
                    "total_rows": int(len(split_df)),
                }
            )

    return pd.DataFrame(rows).sort_values(["variable", "split"])


def summarize_hourly_missingness(
    input_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for hour, hour_df in input_df.groupby("hour"):
        for col in feature_cols:
            rows.append(
                {
                    "hour": int(hour),
                    "variable": col,
                    "missing_rate": float(hour_df[col].isna().mean()),
                    "observed_rows": int(hour_df[col].notna().sum()),
                    "total_rows": int(len(hour_df)),
                }
            )

    return pd.DataFrame(rows).sort_values(["variable", "hour"])


def add_derived_eda_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    if {"Urine", "Weight"}.issubset(df.columns):
        weight = df.groupby(ID_COL)["Weight"].ffill().bfill()
        urine_rate = df["Urine"] / weight.where(weight > 0)
        lower, upper = EDA_DERIVED_PLAUSIBILITY_RANGES["Urine_ml_per_kg_hr"]
        urine_rate = urine_rate.mask((urine_rate < lower) | (urine_rate > upper))
        df["Urine_ml_per_kg_hr"] = urine_rate
    if {"BUN", "Creatinine"}.issubset(df.columns):
        df["BUN_Creatinine_ratio"] = df["BUN"] / df["Creatinine"].where(
            df["Creatinine"] > 0
        )
    return df


def summarize_distributions(
    input_df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    rows = []
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]

    for variable in variables:
        if variable not in input_df.columns:
            continue

        for label_name, group in [
            ("all", input_df),
            ("aki_positive", input_df[input_df[TARGET_COL] == 1]),
            ("aki_negative", input_df[input_df[TARGET_COL] == 0]),
        ]:
            series = group[variable].dropna()
            row = {
                "variable": variable,
                "group": label_name,
                "n": int(series.shape[0]),
                "missing_rate": float(group[variable].isna().mean()),
                "mean": float(series.mean()) if not series.empty else np.nan,
                "std": float(series.std()) if series.shape[0] > 1 else np.nan,
            }
            for q in quantiles:
                label = f"p{int(q * 100):02d}"
                row[label] = float(series.quantile(q)) if not series.empty else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_tabular_missingness(tabular: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col for col in tabular.columns if col not in {ID_COL, TARGET_COL, "split"}
    ]
    rows = []
    for col in feature_cols:
        rows.append(
            {
                "feature": col,
                "missing_rate": float(tabular[col].isna().mean()),
                "observed_rows": int(tabular[col].notna().sum()),
                "total_rows": int(len(tabular)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "feature"], ascending=[False, True]
    )


def summarize_feature_group_differences(tabular: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for col in tabular.select_dtypes(include="number").columns
        if col not in {ID_COL, TARGET_COL}
    ]
    pos = tabular[tabular[TARGET_COL] == 1]
    neg = tabular[tabular[TARGET_COL] == 0]

    rows = []
    for col in feature_cols:
        pos_series = pos[col].dropna()
        neg_series = neg[col].dropna()
        pooled_std = tabular[col].std()
        rows.append(
            {
                "feature": col,
                "positive_mean": float(pos_series.mean()) if not pos_series.empty else np.nan,
                "negative_mean": float(neg_series.mean()) if not neg_series.empty else np.nan,
                "mean_diff_positive_minus_negative": float(
                    pos_series.mean() - neg_series.mean()
                )
                if not pos_series.empty and not neg_series.empty
                else np.nan,
                "standardized_mean_diff": float(
                    (pos_series.mean() - neg_series.mean()) / pooled_std
                )
                if pd.notna(pooled_std) and pooled_std > 0
                else np.nan,
                "positive_missing_rate": float(pos[col].isna().mean()),
                "negative_missing_rate": float(neg[col].isna().mean()),
            }
        )

    diff = pd.DataFrame(rows)
    diff["abs_standardized_mean_diff"] = diff["standardized_mean_diff"].abs()
    return diff.sort_values("abs_standardized_mean_diff", ascending=False)


def summarize_high_correlations(tabular: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    feature_cols = [
        col
        for col in tabular.select_dtypes(include="number").columns
        if col not in {ID_COL, TARGET_COL}
    ]
    feature_df = tabular[feature_cols]
    corr = feature_df.corr(numeric_only=True).abs()
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    stacked = corr.where(upper_mask).stack().reset_index()
    stacked.columns = ["feature_a", "feature_b", "abs_correlation"]
    return stacked.sort_values("abs_correlation", ascending=False).head(top_n)


def summarize_nontrivial_high_correlations(
    high_correlations: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    def is_trivial_pair(row: pd.Series) -> bool:
        feature_a = str(row["feature_a"])
        feature_b = str(row["feature_b"])
        if feature_a.endswith("_count") and feature_b.endswith("_missing_rate"):
            return feature_a.removesuffix("_count") == feature_b.removesuffix(
                "_missing_rate"
            )
        if feature_b.endswith("_count") and feature_a.endswith("_missing_rate"):
            return feature_b.removesuffix("_count") == feature_a.removesuffix(
                "_missing_rate"
            )
        return False

    filtered = high_correlations[~high_correlations.apply(is_trivial_pair, axis=1)]
    return filtered.head(top_n)


def summarize_quality_flags(input_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        series = input_df[col]
        observed = series.dropna()
        if observed.empty:
            rows.append(
                {
                    "check": "all_missing",
                    "variable": col,
                    "count": int(len(series)),
                    "rate": 1.0,
                    "note": "No observed values in the 0-23h input window.",
                }
            )
            continue

        if col in NON_NEGATIVE_VARIABLES:
            count = int((observed < 0).sum())
            if count > 0:
                rows.append(
                    {
                        "check": "negative_value",
                        "variable": col,
                        "count": count,
                        "rate": float(count / len(observed)),
                        "note": "Screening flag only; review before deleting or clipping.",
                    }
                )

        if observed.nunique(dropna=True) <= 1:
            rows.append(
                {
                    "check": "near_zero_variance",
                    "variable": col,
                    "count": int(len(observed)),
                    "rate": 1.0,
                    "note": "Only one observed value in the input window.",
                }
            )

        if col in EDA_PLAUSIBILITY_RANGES:
            lower, upper = EDA_PLAUSIBILITY_RANGES[col]
            implausible = observed[(observed < lower) | (observed > upper)]
            if not implausible.empty:
                rows.append(
                    {
                        "check": "broad_plausibility_range",
                        "variable": col,
                        "count": int(implausible.shape[0]),
                        "rate": float(implausible.shape[0] / observed.shape[0]),
                        "note": f"Allowed screening range [{lower}, {upper}], observed min={observed.min():.4g}, max={observed.max():.4g}. Review before cleaning.",
                    }
                )

        p99 = observed.quantile(0.99)
        p999 = observed.quantile(0.999)
        max_value = observed.max()
        if pd.notna(p99) and p99 > 0 and max_value > 10 * p99:
            rows.append(
                {
                    "check": "extreme_tail",
                    "variable": col,
                    "count": int((observed > 10 * p99).sum()),
                    "rate": float((observed > 10 * p99).mean()),
                    "note": f"max={max_value:.4g}, p99={p99:.4g}, p99.9={p999:.4g}. Review for possible unit or charting errors.",
                }
            )

    for col in STATIC_COLS:
        if col not in input_df.columns:
            continue
        nunique_by_stay = input_df.groupby(ID_COL)[col].nunique(dropna=True)
        inconsistent_count = int((nunique_by_stay > 1).sum())
        if inconsistent_count > 0:
            rows.append(
                {
                    "check": "static_variable_changes_within_stay",
                    "variable": col,
                    "count": inconsistent_count,
                    "rate": float(inconsistent_count / nunique_by_stay.shape[0]),
                    "note": "Static variable has multiple observed values within at least one stay.",
                }
            )

    return pd.DataFrame(rows).sort_values(["check", "variable"])


def write_summary(
    eda_dir: Path,
    cohort: pd.DataFrame,
    missingness: pd.DataFrame,
    tabular_missingness: pd.DataFrame,
    quality_flags: pd.DataFrame,
    high_correlations: pd.DataFrame,
    nontrivial_high_correlations: pd.DataFrame,
) -> None:
    high_missing = missingness[missingness["missing_rate"] >= 0.8]
    high_tabular_missing = tabular_missingness[
        tabular_missingness["missing_rate"] >= 0.8
    ]
    strongest_corr = nontrivial_high_correlations.head(10)

    lines = [
        "# EDA Summary",
        "",
        "This report is generated from the fixed cohort after early AKI exclusion.",
        "",
        "## Cohort",
        "",
        to_markdown_table(cohort),
        "",
        "## Main Checks",
        "",
        f"- Variables with >=80% hourly missingness in the 0-23h input window: {len(high_missing)}",
        f"- Tabular features with >=80% missingness: {len(high_tabular_missing)}",
        f"- Data quality flags generated: {len(quality_flags)}",
        f"- High-correlation feature pairs exported: {len(high_correlations)}",
        f"- Non-trivial high-correlation feature pairs after excluding count/missing-rate duplicates: {len(nontrivial_high_correlations)}",
        "",
        "## Highest Hourly Missingness",
        "",
        to_markdown_table(high_missing.head(20)),
        "",
        "## Top Quality Flags",
        "",
        to_markdown_table(quality_flags.head(30))
        if not quality_flags.empty
        else "No quality flags generated.",
        "",
        "## Strongest Non-Trivial Tabular Feature Correlations",
        "",
        to_markdown_table(strongest_corr)
        if not strongest_corr.empty
        else "No correlation pairs generated.",
        "",
        "## Interpretation Rule",
        "",
        "Do not automatically remove rows or variables from this report alone. Use these files to decide whether preprocessing rules should be changed, then rerun preprocessing and modeling.",
        "",
    ]

    (eda_dir / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = build_paths(ROOT_DIR)
    eda_dir = paths.reports_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    if not paths.filtered_csv.exists():
        raise FileNotFoundError(
            f"Missing filtered cohort file: {paths.filtered_csv}. Run scripts/01_preprocess.py first."
        )

    filtered = pd.read_csv(paths.filtered_csv)
    tabular_v1 = add_split_column(
        build_tabular_features(filtered, feature_version="v1"),
        paths.split_json,
    )
    tabular_v2 = add_split_column(
        build_tabular_features(filtered, feature_version="v2"),
        paths.split_json,
    )

    split_map = load_split_map(paths.split_json)
    target_lookup = tabular_v2[[ID_COL, TARGET_COL, "split"]].copy()

    input_df = filtered[
        (filtered["hour"] >= INPUT_START_HOUR) & (filtered["hour"] <= INPUT_END_HOUR)
    ].merge(target_lookup, on=ID_COL, how="inner")
    input_df = add_derived_eda_columns(input_df)

    feature_cols = [
        col
        for col in input_df.columns
        if col not in META_COLS and pd.api.types.is_numeric_dtype(input_df[col])
    ]
    key_variables = [col for col in KEY_VARIABLES if col in input_df.columns]
    key_variables.extend(["Urine_ml_per_kg_hr", "BUN_Creatinine_ratio"])

    cohort = summarize_cohort(tabular_v2)
    missingness = summarize_missingness(input_df, feature_cols)
    missingness_by_split = summarize_missingness_by_split(input_df, feature_cols)
    hourly_missingness = summarize_hourly_missingness(input_df, key_variables)
    distributions = summarize_distributions(input_df, key_variables)
    tabular_v1_missing = summarize_tabular_missingness(tabular_v1)
    tabular_v2_missing = summarize_tabular_missingness(tabular_v2)
    group_differences = summarize_feature_group_differences(tabular_v2)
    high_correlations = summarize_high_correlations(tabular_v2, top_n=1000)
    nontrivial_high_correlations = summarize_nontrivial_high_correlations(
        high_correlations
    )
    quality_flags = summarize_quality_flags(input_df, feature_cols)

    save_csv(cohort, eda_dir / "cohort_by_split.csv")
    save_csv(missingness, eda_dir / "missingness_by_variable.csv")
    save_csv(missingness_by_split, eda_dir / "missingness_by_split.csv")
    save_csv(hourly_missingness, eda_dir / "missingness_by_hour_key_variables.csv")
    save_csv(distributions, eda_dir / "key_variable_distributions.csv")
    save_csv(tabular_v1_missing, eda_dir / "tabular_v1_missingness.csv")
    save_csv(tabular_v2_missing, eda_dir / "tabular_v2_missingness.csv")
    save_csv(group_differences, eda_dir / "tabular_v2_group_differences.csv")
    save_csv(high_correlations, eda_dir / "tabular_v2_high_correlation_pairs.csv")
    save_csv(
        nontrivial_high_correlations,
        eda_dir / "tabular_v2_high_correlation_nontrivial_pairs.csv",
    )
    save_csv(quality_flags, eda_dir / "data_quality_flags.csv")

    write_summary(
        eda_dir,
        cohort,
        missingness,
        tabular_v2_missing,
        quality_flags,
        high_correlations,
        nontrivial_high_correlations,
    )

    print(f"Saved EDA outputs to: {eda_dir}")
    print(cohort.to_string(index=False))
    print()
    print("Top hourly missingness:")
    print(missingness.head(10).to_string(index=False))
    print()
    print("Top quality flags:")
    if quality_flags.empty:
        print("No quality flags generated.")
    else:
        print(quality_flags.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
