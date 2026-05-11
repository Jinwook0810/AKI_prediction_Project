# EDA Summary

This report is generated from the fixed cohort after early AKI exclusion.

## Cohort

| split | stays | aki_positive_stays | aki_negative_stays | aki_positive_rate |
| --- | --- | --- | --- | --- |
| test | 891 | 160 | 731 | 0.1796 |
| train | 7193 | 1377 | 5816 | 0.1914 |
| valid | 878 | 168 | 710 | 0.1913 |

## Main Checks

- Variables with >=80% hourly missingness in the 0-23h input window: 23
- Tabular features with >=80% missingness: 28
- Data quality flags generated: 1
- High-correlation feature pairs exported: 1000
- Non-trivial high-correlation feature pairs after excluding count/missing-rate duplicates: 100

## Highest Hourly Missingness

| variable | missing_rate | observed_rate | positive_missing_rate | negative_missing_rate | pos_neg_missing_rate_diff | observed_rows | total_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TroponinI | 0.9971 | 0.002948 | 0.9969 | 0.9971 | -0.0002228 | 634 | 215088 |
| Cholesterol | 0.9968 | 0.003236 | 0.9974 | 0.9966 | 0.0007368 | 696 | 215088 |
| TroponinT | 0.9849 | 0.01514 | 0.9837 | 0.9851 | -0.001435 | 3256 | 215088 |
| Albumin | 0.9838 | 0.01625 | 0.9834 | 0.9838 | -0.0003949 | 3495 | 215088 |
| ALP | 0.9798 | 0.02021 | 0.9783 | 0.9801 | -0.001871 | 4347 | 215088 |
| Bilirubin | 0.9793 | 0.02071 | 0.9776 | 0.9797 | -0.002036 | 4455 | 215088 |
| AST | 0.9792 | 0.02081 | 0.9777 | 0.9795 | -0.001788 | 4477 | 215088 |
| ALT | 0.9792 | 0.02082 | 0.9778 | 0.9795 | -0.001722 | 4478 | 215088 |
| Lactate | 0.9486 | 0.05141 | 0.9339 | 0.952 | -0.01815 | 11057 | 215088 |
| Glucose | 0.9229 | 0.07712 | 0.9254 | 0.9223 | 0.003128 | 16587 | 215088 |
| Mg | 0.9219 | 0.07813 | 0.9188 | 0.9226 | -0.003829 | 16805 | 215088 |
| WBC | 0.9215 | 0.07854 | 0.9122 | 0.9236 | -0.01139 | 16892 | 215088 |
| Na | 0.9201 | 0.0799 | 0.922 | 0.9197 | 0.002312 | 17186 | 215088 |
| HCO3 | 0.9189 | 0.08111 | 0.9147 | 0.9199 | -0.005159 | 17446 | 215088 |
| BUN_Creatinine_ratio | 0.9175 | 0.08255 | 0.912 | 0.9187 | -0.006674 | 17755 | 215088 |
| BUN | 0.9173 | 0.08271 | 0.912 | 0.9185 | -0.006588 | 17791 | 215088 |
| Creatinine | 0.917 | 0.08304 | 0.9116 | 0.9182 | -0.006609 | 17861 | 215088 |
| K | 0.9148 | 0.08519 | 0.9175 | 0.9142 | 0.003323 | 18324 | 215088 |
| Platelets | 0.9128 | 0.08721 | 0.8975 | 0.9164 | -0.01893 | 18758 | 215088 |
| HCT | 0.8902 | 0.1098 | 0.8734 | 0.8941 | -0.02072 | 23625 | 215088 |

## Top Quality Flags

| check | variable | count | rate | note |
| --- | --- | --- | --- | --- |
| near_zero_variance | MechVent | 104962 | 1 | Only one observed value in the input window. |

## Strongest Non-Trivial Tabular Feature Correlations

| feature_a | feature_b | abs_correlation |
| --- | --- | --- |
| Cholesterol_delta | TroponinI_std | 1 |
| Cholesterol_std | TroponinI_std | 1 |
| Cholesterol_std | TroponinI_delta | 1 |
| Cholesterol_delta | TroponinI_delta | 1 |
| DiasABP_missing_rate | SysABP_missing_rate | 1 |
| DiasABP_count | SysABP_missing_rate | 1 |
| DiasABP_missing_rate | SysABP_count | 1 |
| DiasABP_count | SysABP_count | 1 |
| Cholesterol_min | Cholesterol_last | 0.9998 |
| Cholesterol_mean | Cholesterol_min | 0.9995 |

## Interpretation Rule

Do not automatically remove rows or variables from this report alone. Use these files to decide whether preprocessing rules should be changed, then rerun preprocessing and modeling.
