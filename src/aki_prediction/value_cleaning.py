from __future__ import annotations

import numpy as np
import pandas as pd


BROAD_PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "Age": (0, 120),
    "Height": (50, 250),
    "Weight": (20, 350),
    "HR": (20, 250),
    "SysABP": (30, 300),
    "DiasABP": (20, 200),
    "MAP": (20, 250),
    "NISysABP": (30, 300),
    "NIDiasABP": (20, 200),
    "NIMAP": (20, 250),
    "RespRate": (1, 80),
    "SaO2": (50, 100),
    "Temp": (25, 45),
    "GCS": (3, 15),
    "pH": (6.5, 8.0),
    "Na": (80, 200),
    "K": (1, 10),
    "Mg": (0, 15),
    "HCO3": (0, 80),
    "HCT": (0, 80),
    "Glucose": (0, 1000),
    "BUN": (0, 250),
    "Creatinine": (0, 30),
    "PaCO2": (0, 200),
    "PaO2": (0, 700),
    "Platelets": (0, 1500),
    "WBC": (0, 300),
    "Urine": (0, 5000),
}

EDA_DERIVED_PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "Urine_ml_per_kg_hr": (0, 50),
}


def clean_values_by_range(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    ranges = ranges or BROAD_PLAUSIBILITY_RANGES
    cleaned = df.copy()
    stats = {
        "total_values_set_to_nan": 0,
        "by_variable": {},
    }

    for col, (lower, upper) in ranges.items():
        if col not in cleaned.columns:
            continue

        values = pd.to_numeric(cleaned[col], errors="coerce")
        invalid = values.notna() & ((values < lower) | (values > upper))
        invalid_count = int(invalid.sum())
        if invalid_count:
            cleaned.loc[invalid, col] = np.nan

        stats["by_variable"][col] = {
            "lower": lower,
            "upper": upper,
            "values_set_to_nan": invalid_count,
        }
        stats["total_values_set_to_nan"] += invalid_count

    return cleaned, stats
