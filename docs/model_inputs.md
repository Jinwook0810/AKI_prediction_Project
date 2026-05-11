# Model Inputs

This document summarizes what each model family uses as input.

## Common Cohort Definition

All current models use the same cleaned cohort:

- input hours: `0-23`
- target hours: `24-47`
- provided split from `split_stay_id.json`
- early AKI stays removed before modeling

## Tabular Models

Tabular models convert each stay into one row.

### Logistic Regression v1

Input file:

- `reports/processed/tabular_features_v1.csv`

Feature structure:

- static variables: first observed value
  - `Age_first`, `Gender_first`, `Height_first`, `ICUType_first`
- general dynamic variables:
  - `mean`, `min`, `max`, `last`, `std`, `delta`, `count`, `missing_rate`
- urine-specific summaries:
  - `Urine_sum`
  - `Urine_last6_sum`
  - `Urine_last12_sum`
  - `Urine_count`
  - `Urine_missing_rate`
- creatinine-specific summaries:
  - `Creatinine_mean`
  - `Creatinine_max`
  - `Creatinine_last`
  - `Creatinine_delta`
  - `Creatinine_count`
  - `Creatinine_missing_rate`

Feature count:

- `295`

### Logistic Regression v2

Input file:

- `reports/processed/tabular_features_v2.csv`

Uses all `v1` features plus renal-focused engineered features:

- `Urine_ml_per_kg_hr_mean`
- `Urine_ml_per_kg_hr_min`
- `Urine_ml_per_kg_hr_last6_mean`
- `Urine_ml_per_kg_hr_last12_mean`
- `Creatinine_relative_delta`
- `Creatinine_max_minus_min`
- `BUN_Creatinine_ratio_last`
- `BUN_Creatinine_ratio_mean`

Feature count:

- `303`

### Logistic Regression v3

Input file:

- `reports/processed/tabular_features_v3.csv`

Uses all `v2` features plus additional burden and trend features:

- urine burden:
  - `Urine_low_output_hours_24h`
  - `Urine_low_output_hours_last12`
  - `Urine_longest_low_output_streak`
  - `Urine_any_6h_oliguria`
- recent creatinine trend:
  - `Creatinine_slope_last6`
  - `Creatinine_slope_last12`
- hemodynamic burden:
  - `MAP_below_65_hours`
  - `SysABP_below_90_hours`
  - `ShockIndex_mean`
  - `ShockIndex_max`

Feature count:

- `313`

### Logistic Regression compact

Input file:

- `reports/processed/tabular_features_compact.csv`

Compact design choices:

- start from `v2`
- remove near-zero variance `MechVent`
- for general dynamic variables keep only:
  - `mean`, `last`, `delta`, `count`
- keep urine, creatinine, and renal-focused engineered features

Feature count:

- `157`

### RandomForest v2

Input file:

- `reports/processed/tabular_features_v2.csv`

Feature set:

- same `303` tabular `v2` features as Logistic Regression v2

### CatBoost v2

Input file:

- `reports/processed/tabular_features_v2.csv`

Feature set:

- same `303` tabular `v2` features as Logistic Regression v2

## Sequence Models

Sequence models keep the hourly structure.

Each stay is represented as `24 x F`.

### Base hourly value features

Used by all current sequence models:

- `ALP`
- `ALT`
- `AST`
- `Age`
- `Albumin`
- `BUN`
- `Bilirubin`
- `Cholesterol`
- `Creatinine`
- `DiasABP`
- `FiO2`
- `GCS`
- `Gender`
- `Glucose`
- `HCO3`
- `HCT`
- `HR`
- `Height`
- `ICUType`
- `K`
- `Lactate`
- `MAP`
- `MechVent`
- `Mg`
- `NIDiasABP`
- `NIMAP`
- `NISysABP`
- `Na`
- `PaCO2`
- `PaO2`
- `Platelets`
- `RespRate`
- `SaO2`
- `SysABP`
- `Temp`
- `TroponinI`
- `TroponinT`
- `Urine`
- `WBC`
- `Weight`
- `pH`

Derived hourly sequence features:

- `Urine_ml_per_kg_hr`
- `BUN_Creatinine_ratio`

Total value features:

- `43`

### LSTM value-only

Input:

- hourly values only
- shape: `24 x 43`

### LSTM masked

Input:

- hourly values
- hourly observed mask for each feature
- shape: `24 x 86`

Mask design:

- first 43 channels: scaled value features
- next 43 channels: observed indicators

### Transformer masked

Input:

- same masked sequence tensor as masked LSTM
- shape: `24 x 86`

## Important Difference Between Tabular and Sequence Inputs

Tabular models use stay-level summaries.

Sequence models use the hourly order directly.

That means:

- tabular models use explicit feature engineering to summarize time
- sequence models rely more on model structure and mask information to capture temporal patterns
