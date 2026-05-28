# Results Summary

This document summarizes the current experiments after the preprocessing redesign and `v3` feature engineering update.

## Cohort Used For Current Results

All current `v3` tabular and sequence results use the same cleaned cohort and provided split:

- train: `7193` stays
- valid: `878` stays
- test: `891` stays

Target:

- use hours `0-23` as input
- predict AKI during hours `24-47`

## Tabular Feature Sets

- `v1`: basic summary statistics
- `v2`: `v1` plus renal-focused features
- `v3`: `v2` plus oliguria burden, recent creatinine trend, and hemodynamic burden
- `compact`: reduced subset for a simpler baseline

Feature counts:

- `v1`: `295`
- `v2`: `303`
- `v3`: `313`
- `compact`: `157`

## Current Tabular Comparison

| Model | Input | Features | AUROC | AUPRC | F1 | Precision | Recall | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression v3 | Tabular | 313 | 0.7550 | 0.4118 | 0.4631 | 0.3445 | 0.7063 | Best interpretable baseline after redesign |
| RandomForest v3 | Tabular | 313 | 0.7917 | 0.4680 | 0.4862 | 0.4059 | 0.6063 | Best AUROC on current tabular cohort |
| CatBoost v3 | Tabular | 313 | 0.7785 | 0.4681 | 0.4936 | 0.4192 | 0.6000 | Best AUPRC by a small margin |
| LightGBM v3 | Tabular | 313 | 0.7895 | 0.4540 | 0.5223 | 0.4972 | 0.5500 | Best F1 on current tabular cohort |
| XGBoost v3 | Tabular | 313 | 0.7842 | 0.4549 | 0.4958 | 0.4513 | 0.5500 | Strong additional boosting baseline |
| TabNet v3 | Tabular | 313 | 0.7237 | 0.3824 | 0.4403 | 0.3312 | 0.6563 | Tabular deep learning reference baseline |

## Earlier Reference Results

These older results are still useful as historical baselines, but they were produced before the latest preprocessing redesign.

| Model | Input | Features | AUROC | AUPRC | F1 | Precision | Recall | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression v2 | Tabular | 303 | 0.7415 | 0.3923 | 0.4668 | 0.3846 | 0.5938 | Earlier interpretable baseline |
| RandomForest v2 | Tabular | 303 | 0.7854 | 0.4477 | 0.4869 | 0.4189 | 0.5813 | Earlier strongest overall tabular baseline |
| CatBoost v2 | Tabular | 303 | 0.7759 | 0.4440 | 0.4661 | 0.4413 | 0.4938 | Earlier precision-focused model |
| Logistic Regression compact | Tabular | 157 | 0.7394 | 0.3843 | 0.4338 | 0.3322 | 0.6250 | Simpler feature set |

## Sequence Reference Results

The sequence baselines below were rerun on the redesigned cohort.

| Model | Input | Features | AUROC | AUPRC | F1 | Precision | Recall | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LSTM value-only | Sequence | 43 per timestep | 0.7376 | 0.4406 | 0.4191 | 0.3641 | 0.4938 | No mask features |
| LSTM masked | Sequence | 86 per timestep | 0.7608 | 0.4488 | 0.4720 | 0.3769 | 0.6313 | Current best sequence baseline |
| Transformer masked | Sequence | 86 per timestep | 0.7533 | 0.4180 | 0.4683 | 0.4187 | 0.5313 | Better precision than masked LSTM |

## Main Takeaways

### 1. The preprocessing redesign materially changed the benchmark

The cohort changed from the earlier version:

- train `7194 -> 7193`
- early AKI removals `2551 -> 2552`

That confirms the preprocessing rules affect not only features, but also label assignment and cohort inclusion.

### 2. `v3` feature engineering improved the strongest tabular baselines

Compared with `v2`, the new `v3` features improved the most competitive models and introduced stronger burden-oriented signal:

- oliguria burden
- recent creatinine trend
- hemodynamic burden

Those additions helped:

- Logistic Regression AUROC and AUPRC
- RandomForest AUROC and AUPRC
- CatBoost F1 and AUPRC
- and provided a stronger feature set for LightGBM, XGBoost, and TabNet

### 3. The best tabular model depends on the metric

- best AUROC: `RandomForest v3`
- best AUPRC: `CatBoost v3`
- best F1: `LightGBM v3`
- best interpretable baseline: `Logistic Regression v3`

So the current conclusion is not that one model wins everything.

Instead:

- `RandomForest v3` is the strongest ranking-oriented baseline
- `LightGBM v3` is the strongest thresholded baseline by F1
- `CatBoost v3` remains a strong high-AUPRC baseline
- `TabNet v3` is a useful tabular deep-learning reference, but it does not beat the best boosting models on this cohort

### 4. Sequence models still matter, but masked LSTM is the current best sequence baseline

Sequence masking still matters on the redesigned cohort.

Compared with value-only LSTM:

- AUROC improved from `0.7376` to `0.7608`
- AUPRC improved from `0.4406` to `0.4488`
- F1 improved from `0.4191` to `0.4720`

The current best sequence baseline is `masked LSTM v1`.

It is still below the best tabular tree models, but it remains a meaningful sequence reference.

### 5. `v3` feature importance supports the new engineering choices

The new burden features were not just decorative. They moved into top-ranked positions.

Examples:

- `Urine_low_output_hours_last12`
- `Urine_low_output_hours_24h`
- `Urine_longest_low_output_streak`
- `Creatinine_slope_last6`
- `Creatinine_slope_last12`

Detailed files:

- [random_forest_tabular_v3_feature_importance.md](../reports/importance/random_forest_tabular_v3_feature_importance.md)
- [catboost_tabular_v3_feature_importance.md](../reports/importance/catboost_tabular_v3_feature_importance.md)
- [transformer_sequence_masked_permutation_importance.md](../reports/importance/transformer_sequence_masked_permutation_importance.md)

### 6. SHAP supports explainability for the boosting baselines

SHAP explanations were added for the strongest boosting models.

Current SHAP global rankings show the same core pattern across `LightGBM v3` and `CatBoost v3`:

- `Urine_low_output_hours_last12`
- `Creatinine_delta`
- `Urine_ml_per_kg_hr_last6_mean` or `Urine_last6_sum`
- `Age_first`

Artifacts:

- [lightgbm_tabular_v3_shap_global.csv](../reports/importance/lightgbm_tabular_v3_shap_global.csv)
- [catboost_tabular_v3_shap_global.csv](../reports/importance/catboost_tabular_v3_shap_global.csv)
- [lightgbm_tabular_v3_shap_local.md](../reports/importance/lightgbm_tabular_v3_shap_local.md)
- [catboost_tabular_v3_shap_local.md](../reports/importance/catboost_tabular_v3_shap_local.md)
- [lightgbm_tabular_v3_shap_summary.png](../reports/figures/lightgbm_tabular_v3_shap_summary.png)
- [catboost_tabular_v3_shap_summary.png](../reports/figures/catboost_tabular_v3_shap_summary.png)

### 7. Decision-oriented analysis makes the risk models easier to operationalize

The tabular baselines were also compared under review-budget and threshold scenarios.

On the current test cohort:

- at a **top 10% review budget**, `RandomForest v3` and `LightGBM v3` each capture `33.1%` of positives with `58.9%` precision
- at a **top 20% review budget**, `LightGBM v3` captures `55.6%` of positives with `49.7%` precision
- `CatBoost v3` remains attractive when slightly stronger AUPRC matters

Artifacts:

- [tabular_v3_topk_summary.csv](../reports/decision/tabular_v3_topk_summary.csv)
- [tabular_v3_threshold_summary.csv](../reports/decision/tabular_v3_threshold_summary.csv)
- [tabular_v3_decision_summary.md](../reports/decision/tabular_v3_decision_summary.md)
- [tabular_v3_topk_capture.png](../reports/figures/tabular_v3_topk_capture.png)
- [tabular_v3_threshold_tradeoff.png](../reports/figures/tabular_v3_threshold_tradeoff.png)

## Recommended Results To Report Now

For the current repository state, the most defensible headline set is:

1. `Logistic Regression v3`
2. `RandomForest v3`
3. `LightGBM v3`
4. `CatBoost v3`
5. `TabNet v3` as a tabular deep-learning reference
6. `masked LSTM v1` as a sequence reference baseline

That set best reflects the current preprocessing, feature-engineering, explainability, and decision-analysis story.
