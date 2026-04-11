# Limitations and Future Work

This project reached a reproducible baseline stage, but it still has clear limitations.

The point of this document is to make those limits explicit rather than hide them.

## Current Limitations

### 1. AKI definition is only partially aligned with full KDIGO logic

The original KDIGO concept includes:

- creatinine increase within a time window
- baseline-relative creatinine increase
- urine output criterion

In this project, the raw data did not provide a reliable pre-ICU baseline creatinine for every stay.
As a result, the implemented labeling used:

- creatinine increase of at least `0.3`
- urine-output-based AKI logic

This is clinically meaningful, but it is not a complete KDIGO implementation.

### 2. Aggregation and cleaning rules still involve heuristic choices

Several preprocessing choices are defensible, but still heuristic:

- exact-timestamp urine conflict resolution
- hourly aggregation rules
- broad physiological plausibility ranges
- not forward-filling `Urine` and `Creatinine`

These rules improved consistency, but another reasonable clinical-data pipeline could make different choices and get different results.

### 3. Missingness is modeled, but not exhaustively exploited

The project now uses:

- tabular `count` and `missing_rate`
- sequence observed-mask features

That is a strong baseline treatment of irregular ICU measurement patterns, but it is still not the most expressive possible design.

What is not yet modeled explicitly:

- time since last observation
- measurement interval structure
- different missingness handling by variable family

### 4. Sequence models are still relatively simple

The current sequence models are:

- value-only LSTM
- masked LSTM
- masked Transformer

These are useful baselines, but they do not yet include:

- separate static and dynamic branches
- time-gap encoding
- richer temporal feature engineering
- more systematic sequence model selection

### 5. Hyperparameter search is intentionally small

The tuning experiments were limited to small fixed grids.

This was appropriate for a baseline project, but it means:

- the tuned model is not necessarily close to the best attainable model
- comparisons should be interpreted as controlled baseline comparisons, not exhaustive optimization

### 6. Validation strategy is still limited to the provided split

The project correctly preserves the provided `train/valid/test` split, which is important for reproducibility.

However, it does not yet include:

- temporal validation
- external validation
- hospital-to-hospital generalization checks

So the current results should be interpreted as internal benchmark results, not deployment-ready evidence.

### 7. Business utility is argued, not operationally tested

The project connects naturally to ICU risk stratification and earlier AKI monitoring.

But it does not yet evaluate:

- alert burden
- false positive cost
- intervention capacity limits
- clinician workflow integration
- net benefit under specific hospital operating assumptions

So the current output is best understood as a modeling and risk-stratification prototype.

## What Worked Well

Despite those limits, several parts of the project are meaningful:

- the messy notebook workflow was turned into a script-based pipeline
- the provided split was preserved consistently
- early AKI exclusion and cohort definition were made explicit
- data cleaning decisions were documented
- tabular and sequence baselines were compared on the same cleaned cohort
- missingness was treated as signal rather than ignored

That makes the project more valuable as a portfolio piece than a single high score would.

## Recommended Next Improvements

### 1. Improve sequence input design

Most useful next upgrades:

- add `time_since_last_observed`
- separate static variables from dynamic sequence variables
- compare value-only, mask-only, and value-plus-mask variants more systematically

### 2. Add clinically motivated temporal features

Examples:

- oliguria burden features
- recent creatinine slope
- recent hemodynamic burden such as low-MAP duration
- shock-related features

### 3. Use broader model selection criteria

Right now, tuned models were selected mainly by validation F1.

It would be better to compare selection rules using:

- F1
- AUPRC
- calibration quality
- threshold-dependent utility

### 4. Evaluate calibration and decision thresholds more seriously

For a hospital-facing decision support use case, ranking performance alone is not enough.

The next step would be to decide:

- what recall level is operationally acceptable
- how many alerts per day the ICU can absorb
- whether the model should prioritize precision or recall in deployment

### 5. Add external or temporal robustness checks

Even a simple follow-up such as:

- earlier-period vs later-period split
- unit-level subgroup comparison

would strengthen the credibility of the project substantially.

## Bottom Line

This repository is best viewed as:

- a cleaned and reproducible AKI prediction baseline project
- a portfolio-quality ICU machine learning workflow
- not yet a clinically validated production model

That is an acceptable stopping point for this stage of the project, as long as the limitations are stated clearly.
