# EDA Summary

This report is generated from the fixed cohort after early AKI exclusion.

## Cohort

| split | stays | aki_positive_stays | aki_negative_stays | aki_positive_rate |
| --- | --- | --- | --- | --- |
| test | 891 | 160 | 731 | 0.1796 |
| train | 7194 | 1376 | 5818 | 0.1913 |
| valid | 878 | 168 | 710 | 0.1913 |

## Main Checks

- Variables with >=80% hourly missingness in the 0-23h input window: 5
- Tabular features with >=80% missingness: 18
- Data quality flags generated: 1
- High-correlation feature pairs exported: 1000
- Non-trivial high-correlation feature pairs after excluding count/missing-rate duplicates: 100

## Highest Hourly Missingness

| variable | missing_rate | observed_rate | positive_missing_rate | negative_missing_rate | pos_neg_missing_rate_diff | observed_rows | total_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TroponinI | 0.9683 | 0.03167 | 0.9664 | 0.9688 | -0.002353 | 6812 | 215112 |
| Cholesterol | 0.9492 | 0.05075 | 0.9584 | 0.9471 | 0.01131 | 10918 | 215112 |
| BUN_Creatinine_ratio | 0.9171 | 0.08286 | 0.9118 | 0.9184 | -0.006654 | 17824 | 215112 |
| Creatinine | 0.917 | 0.08304 | 0.9116 | 0.9182 | -0.006587 | 17862 | 215112 |
| TroponinT | 0.8498 | 0.1502 | 0.8356 | 0.8532 | -0.01758 | 32305 | 215112 |

## Top Quality Flags

| check | variable | count | rate | note |
| --- | --- | --- | --- | --- |
| near_zero_variance | MechVent | 104982 | 1 | Only one observed value in the input window. |

## Strongest Non-Trivial Tabular Feature Correlations

| feature_a | feature_b | abs_correlation |
| --- | --- | --- |
| DiasABP_missing_rate | SysABP_count | 1 |
| DiasABP_count | SysABP_missing_rate | 1 |
| DiasABP_count | SysABP_count | 1 |
| DiasABP_missing_rate | SysABP_missing_rate | 1 |
| Cholesterol_min | Cholesterol_last | 0.9998 |
| Cholesterol_mean | Cholesterol_last | 0.9996 |
| Cholesterol_mean | Cholesterol_min | 0.9996 |
| Cholesterol_mean | Cholesterol_max | 0.9992 |
| ALP_min | ALP_last | 0.9986 |
| Cholesterol_max | Cholesterol_last | 0.9982 |

## Interpretation Rule

Do not automatically remove rows or variables from this report alone. Use these files to decide whether preprocessing rules should be changed, then rerun preprocessing and modeling.
