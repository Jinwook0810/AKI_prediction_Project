# AKI Prediction Project Scope

## Goal

This project predicts whether AKI will occur during hours 24-47 after ICU admission by using clinical observations from hours 0-23.

## Fixed Rules

These rules are the canonical experiment definition for the cleaned GitHub version of the project.

1. Raw data source
   - `released_df.csv.gz` or `released_df.csv`
2. Split source
   - `split_stay_id.json`
   - No random split is allowed in the final pipeline.
3. Input window
   - ICU admission hours `0-23`
4. Prediction target
   - Whether AKI occurs during hours `24-47`
5. Cohort exclusion
   - Patients with AKI already occurring during hours `0-23` are excluded from prediction experiments.
6. Evaluation consistency
   - All models must use the same cohort, same split, same feature definition, and same evaluation metrics.

## AKI Definition

The current project definition follows the existing coursework logic:

- Creatinine-based AKI:
  - creatinine increase of at least `0.3` from the rolling 48-hour minimum
- Urine-based AKI:
  - urine output per body weight less than or equal to `0.5` for 6 consecutive hours

Any stay meeting either rule is labeled as AKI-positive at the hourly level.

## Modeling Table Definition

For tabular baseline models:

- Use only hours `0-23`
- Aggregate by `stay_id`
- Default aggregation rule:
  - `Urine`: `sum`
  - `Creatinine`: `max`
  - static variables such as `Age`, `Gender`, `Height`, `ICUType`: `first`
  - other variables: `mean`

## Required Outputs

The cleaned pipeline should produce at least:

1. Preprocessed hourly table
2. Filtered cohort table after removing early AKI stays
3. Train/valid/test cohort summary
4. Model metrics saved as structured files

## Non-Goals For The First Milestone

The first milestone is not model improvement.

The first milestone is:

> reproduce one consistent CatBoost baseline under the fixed rules above

After that:

1. LightGBM
2. MLP / Bagging MLP
3. LSTM / Transformer

## Notes

- Existing notebooks remain as historical material only.
- The final GitHub project should run from scripts and source files, not from one monolithic notebook.
