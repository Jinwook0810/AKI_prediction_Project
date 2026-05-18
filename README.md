# AKI Prediction from ICU EMR

This repository organizes an ICU acute kidney injury prediction project around a reproducible preprocessing and modeling pipeline.

The current emphasis is:

1. make preprocessing rules explicit,
2. preserve the provided `train/valid/test` split,
3. keep cohort definition consistent,
4. compare tabular and sequence baselines on the same cleaned cohort.

## Prediction Task

- input window: hours `0-23`
- target window: hours `24-47`
- split source: `split_stay_id.json`
- early AKI stays: removed if AKI already appears during hours `0-23`

AKI labeling uses:

- creatinine increase of at least `0.3` within a rolling 48 hour window
- urine output criterion of `<= 0.5 mL/kg/hr` sustained for 6 hours

The raw data does not provide a reliable pre-ICU baseline creatinine, so the KDIGO baseline-ratio criterion was not used.

## Data Inputs

Expected local files:

- `IMEN383_Team_Project_Files/released_df.csv.gz` or `released_df.csv`
- `IMEN383_Team_Project_Files/split_stay_id.json`

## Repository Layout

- `src/`: preprocessing, feature engineering, and sequence utilities
- `scripts/`: runnable entry points
- `reports/`: processed artifacts, metrics, EDA outputs, feature importance, figures
- `docs/`: project notes and results summaries
- `notebooks/`: archived and exploratory notebooks
- `legacy/`: old notebook-era scripts and course materials

## Current Cohort

Current cohort summary from [cohort_summary.json](reports/processed/cohort_summary.json):

- filtered stays: `8962`
- removed early AKI stays: `2552`
- train: `7193` stays, `1377` positives
- valid: `878` stays, `168` positives
- test: `891` stays, `160` positives

## Preprocessing Decisions

### 1. Raw duplicate cleaning is variable-aware

The raw event table is audited for same-timestamp duplicates for every label, not only `Urine`.

Representative-value rules:

- `Urine`: smallest positive value
- vitals such as `HR`, `MAP`, `Temp`: `median`
- static variables such as `Age`, `Height`, `ICUType`: `first`
- `Weight`: `median`
- labs such as `Creatinine`, `BUN`, `WBC`: `last`
- `MechVent`: `max`

### 2. Hourly aggregation is variable-family specific

The raw event table is converted to an hourly wide table.

Aggregation rules:

- `Urine`: hourly `sum`
- static variables: `first` or `last` depending on meaning
- vitals: hourly `median`
- `Creatinine` and most labs: hourly `last`
- `MechVent`: hourly `max`

### 3. Broad plausibility cleaning

Clearly invalid values are set to `NaN`, not dropped row-wise.

Examples:

- impossible blood pressure values
- `Weight = 0`
- impossible `pH`
- extreme `Urine`

Cleaning stats are recorded in [cohort_summary.json](reports/processed/cohort_summary.json).

### 4. Forward fill policy

After value cleaning:

- static variables, vitals, and support variables are forward-filled within stay
- `Urine`, `Creatinine`, and most labs are not forward-filled

### 5. Sequence observed mask

For sequence models, the project also saves:

- [filtered_observed_mask.csv](reports/processed/filtered_observed_mask.csv)

This preserves whether each hourly value was originally observed before forward fill.

## Processed Artifacts

Core processed outputs:

- [hourly_features.csv](reports/processed/hourly_features.csv)
- [filtered_cohort.csv](reports/processed/filtered_cohort.csv)
- [filtered_observed_mask.csv](reports/processed/filtered_observed_mask.csv)
- [tabular_features_v1.csv](reports/processed/tabular_features_v1.csv)
- [tabular_features_v2.csv](reports/processed/tabular_features_v2.csv)
- [tabular_features_v3.csv](reports/processed/tabular_features_v3.csv)
- [tabular_features_compact.csv](reports/processed/tabular_features_compact.csv)

## EDA Outputs

Useful files:

- [eda_summary.md](reports/eda/eda_summary.md)
- [missingness_by_variable.csv](reports/eda/missingness_by_variable.csv)
- [data_quality_flags.csv](reports/eda/data_quality_flags.csv)

Current prominent EDA finding:

- `MechVent` remains near-zero variance in the current input window

## Models Implemented

### Tabular models

- Logistic Regression
- Random Forest
- CatBoost

### Sequence models

- value-only LSTM
- masked LSTM
- masked Transformer

For masked sequence models, each timestep contains:

- 43 value features
- 43 observed-mask features

for a total of `86` features per timestep.

Detailed input definitions:

- [model_inputs.md](docs/model_inputs.md)
- [limitations_and_future_work.md](docs/limitations_and_future_work.md)

## Current Results

Full comparison:

- [results_summary.md](docs/results_summary.md)

Current practical takeaways:

- best ranking-oriented tabular baseline: `RandomForest v3`
- best thresholded tabular baseline by F1: `CatBoost v3`
- best interpretable baseline: `Logistic Regression v3`
- best sequence baseline so far: `masked LSTM v1`
- mask information materially helps sequence models
- `v3` feature engineering improved tabular performance by adding oliguria burden and recent trend features

## Key Figures

Pipeline overview:

- [pipeline_overview.md](docs/pipeline_overview.md)

Model comparison:

![Model Comparison](reports/figures/model_comparison.png)

RandomForest top features:

![RandomForest Top Features](reports/figures/random_forest_top_features.png)

CatBoost top features:

![CatBoost Top Features](reports/figures/catboost_top_features.png)

Tabular precision-recall curves:

![Tabular PR Curves](reports/figures/tabular_pr_curves.png)

Tabular calibration:

![Tabular Calibration](reports/figures/tabular_calibration.png)

Top feature distributions:

![Top Feature Distributions](reports/figures/top_feature_distributions.png)

Key variable missingness heatmap:

![Missingness Heatmap](reports/figures/key_variable_missingness_heatmap.png)

Masked Transformer permutation importance:

![Transformer Sequence Permutation Importance](reports/figures/transformer_sequence_permutation_importance.png)

## How To Run

Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

Run preprocessing:

```bash
py -3 scripts/01_preprocess.py
```

Run EDA:

```bash
py -3 scripts/00_eda.py
```

Build tabular features only:

```bash
py -3 scripts/02_train_tabular.py --feature-version v1 --rebuild-features --features-only
py -3 scripts/02_train_tabular.py --feature-version v2 --rebuild-features --features-only
py -3 scripts/02_train_tabular.py --feature-version v3 --rebuild-features --features-only
```

Run tabular models:

```bash
py -3 scripts/02_train_tabular.py --model logistic --feature-version v3
py -3 scripts/02_train_tabular.py --model random_forest --feature-version v3
py -3 scripts/02_train_tabular.py --model catboost --feature-version v3
```

Run sequence models:

```bash
py -3 scripts/04_train_sequence.py
py -3 scripts/04_train_sequence.py --use-mask
py -3 scripts/05_train_transformer.py --use-mask
```

Run tuning:

```bash
py -3 scripts/02_train_tabular.py --model random_forest --feature-version v3 --tune-hyperparameters
py -3 scripts/02_train_tabular.py --model catboost --feature-version v3 --tune-hyperparameters
py -3 scripts/04_train_sequence.py --use-mask --tune-hyperparameters
py -3 scripts/05_train_transformer.py --use-mask --tune-hyperparameters
```

## Repository Notes

- The script-based pipeline is the primary path; notebooks are reference only.
- If `filtered_cohort.csv` is open in Excel or another viewer, preprocessing may fail when trying to overwrite it.
- The current repository is reproducible for local experimentation, but raw patient-level data is intentionally excluded from version control.
