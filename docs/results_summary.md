# Results Summary

This document summarizes the current cleaned-cohort experiments.

## Cohort Used For All Results

All results below use the same cleaned cohort and provided split:

- train: `7194` stays
- valid: `878` stays
- test: `891` stays

Target:

- use hours `0-23` as input
- predict AKI during hours `24-47`

## Tabular Feature Sets

- `v1`: basic summary statistics
- `v2`: `v1` plus renal-focused features
- `compact`: reduced feature subset for a simpler Logistic baseline

Feature counts:

- `v1`: `295`
- `v2`: `303`
- `compact`: `157`

## Model Comparison

| Model | Input | Features | AUROC | AUPRC | F1 | Precision | Recall | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression v1 | Tabular | 295 | 0.7350 | 0.3849 | 0.4371 | 0.3262 | 0.6625 | Basic summary baseline |
| Logistic Regression v2 | Tabular | 303 | 0.7415 | 0.3923 | 0.4668 | 0.3846 | 0.5938 | Interpretable baseline |
| Logistic Regression compact | Tabular | 157 | 0.7394 | 0.3843 | 0.4338 | 0.3322 | 0.6250 | Simpler feature set |
| RandomForest v2 | Tabular | 303 | 0.7854 | 0.4477 | 0.4869 | 0.4189 | 0.5813 | Best overall tabular baseline |
| RandomForest tuned v2 | Tabular | 303 | 0.7844 | 0.4507 | 0.4787 | 0.4167 | 0.5625 | Better AUPRC, lower F1 |
| CatBoost v2 | Tabular | 303 | 0.7759 | 0.4440 | 0.4661 | 0.4413 | 0.4938 | Strong precision-focused tabular model |
| CatBoost tuned v2 | Tabular | 303 | 0.7723 | 0.4434 | 0.4689 | 0.4278 | 0.5188 | Slight F1 gain vs untuned CatBoost |
| LSTM value-only | Sequence | 43 per timestep | 0.7135 | 0.3765 | 0.4174 | 0.3297 | 0.5688 | No mask features |
| LSTM masked | Sequence | 86 per timestep | 0.7498 | 0.4441 | 0.4337 | 0.3664 | 0.5313 | 43 values + 43 masks |
| LSTM masked tuned | Sequence | 86 per timestep | 0.7279 | 0.4410 | 0.4481 | 0.3981 | 0.5125 | Higher F1, lower AUROC |
| Transformer masked | Sequence | 86 per timestep | 0.7488 | 0.3900 | 0.4607 | 0.3964 | 0.5500 | Best untuned sequence F1 |
| Transformer masked tuned | Sequence | 86 per timestep | 0.7487 | 0.4026 | 0.4513 | 0.3493 | 0.6375 | Higher recall, lower F1 |

## Main Takeaways

### 1. Best current model

The strongest current overall baseline is:

- `RandomForest v2`

It gives the best combination of AUROC, AUPRC, and F1 on the cleaned cohort.

### 2. Best interpretable baseline

For a simpler and easier-to-explain baseline:

- `Logistic Regression v2`

This remains useful as the main linear reference model.

### 3. Masking matters for sequence models

Moving from value-only LSTM to masked LSTM changed:

- AUROC from `0.7135` to `0.7498`
- AUPRC from `0.3765` to `0.4441`

This supports the assumption that observation pattern itself carries signal in ICU data.

### 4. Sequence models are competitive but not yet better than tabular tree models

The best sequence models are in the same broad range as the tabular linear baseline, but still below the best tabular tree ensemble.

This suggests the next sequence gains are more likely to come from:

- better model selection criteria
- improved mask/time-gap handling
- more deliberate tuning

not from adding more model names alone.

### 5. Sequence permutation importance is clinically plausible

For the masked Transformer baseline, permutation importance ranked the following variables highly:

- `Urine_ml_per_kg_hr`
- `PaO2`
- `Temp`
- `Na`
- `Creatinine`

This is useful because the sequence model is no longer just a black box. The ranking suggests it is using a mix of:

- renal-specific signal
- respiratory / oxygenation signal
- general physiologic instability

Detailed file:

- [transformer_sequence_masked_permutation_importance.md](C:/Users/USER/Desktop/대학교 자료/3-2학기/수업/헬스시스템엔지니어링/reports/importance/transformer_sequence_masked_permutation_importance.md)

## Feature Importance Notes

From the current tree models:

- recent urine-per-kg summaries are consistently important
- creatinine delta and relative creatinine change are consistently important
- age and some general severity-related signals also appear in top-ranked features

Detailed files:

- [random_forest_tabular_v2_feature_importance.md](C:/Users/USER/Desktop/대학교 자료/3-2학기/수업/헬스시스템엔지니어링/reports/importance/random_forest_tabular_v2_feature_importance.md)
- [catboost_tabular_v2_feature_importance.md](C:/Users/USER/Desktop/대학교 자료/3-2학기/수업/헬스시스템엔지니어링/reports/importance/catboost_tabular_v2_feature_importance.md)

## Recommended Baselines To Report

If you want a compact set of headline baselines for GitHub or a report, use:

1. Logistic Regression v2
2. RandomForest v2
3. CatBoost v2
4. masked Transformer v1

This set is broad enough to show:

- linear tabular baseline
- non-linear tree ensemble
- stronger boosted tabular model
- sequence model with missingness information

## Important Caveat

Tuned models here were selected by validation-set F1. That means:

- they are valid experiments
- but the tuned model with the best validation F1 is not always the model with the best test F1 or best AUROC

That is expected. The selection rule and the final test metric are not identical.
