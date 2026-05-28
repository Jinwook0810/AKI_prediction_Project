# Decision Analysis Summary

- feature_version: `v3`
- evaluated_split: `test`
- test_n: `891`
- test_positive: `160`

## Interpretation

If operations can review only the top 10% highest-risk stays, `LightGBM v3` captures `0.3312` of positives with precision `0.5889`.
Under the same 10% review budget, `CatBoost v3` captures `0.3063` of positives with precision `0.5444`.

## Top-k Recall

| model | coverage_fraction | k | captured_positives | precision_at_k | recall_at_k | lift_vs_base_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.05 | 45 | 22 | 0.4889 | 0.1375 | 2.7225 |
| logistic | 0.10 | 90 | 46 | 0.5111 | 0.2875 | 2.8462 |
| logistic | 0.20 | 179 | 81 | 0.4525 | 0.5062 | 2.5199 |
| random_forest | 0.05 | 45 | 25 | 0.5556 | 0.1562 | 3.0938 |
| random_forest | 0.10 | 90 | 53 | 0.5889 | 0.3312 | 3.2794 |
| random_forest | 0.20 | 179 | 82 | 0.4581 | 0.5125 | 2.5510 |
| catboost | 0.05 | 45 | 26 | 0.5778 | 0.1625 | 3.2175 |
| catboost | 0.10 | 90 | 49 | 0.5444 | 0.3063 | 3.0319 |
| catboost | 0.20 | 179 | 85 | 0.4749 | 0.5312 | 2.6444 |
| lightgbm | 0.05 | 45 | 25 | 0.5556 | 0.1562 | 3.0938 |
| lightgbm | 0.10 | 90 | 53 | 0.5889 | 0.3312 | 3.2794 |
| lightgbm | 0.20 | 179 | 89 | 0.4972 | 0.5563 | 2.7688 |
| xgboost | 0.05 | 45 | 25 | 0.5556 | 0.1562 | 3.0938 |
| xgboost | 0.10 | 90 | 48 | 0.5333 | 0.3000 | 2.9700 |
| xgboost | 0.20 | 179 | 82 | 0.4581 | 0.5125 | 2.5510 |

## Threshold Trade-off

| model | threshold | review_rate | precision | recall | f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.20 | 0.6251 | 0.2478 | 0.8625 | 0.3849 |
| logistic | 0.30 | 0.4882 | 0.2920 | 0.7937 | 0.4269 |
| logistic | 0.35 | 0.4310 | 0.3125 | 0.7500 | 0.4412 |
| logistic | 0.40 | 0.3895 | 0.3343 | 0.7250 | 0.4576 |
| logistic | 0.50 | 0.3120 | 0.3777 | 0.6562 | 0.4795 |
| random_forest | 0.20 | 0.3378 | 0.3787 | 0.7125 | 0.4946 |
| random_forest | 0.30 | 0.1706 | 0.4868 | 0.4625 | 0.4744 |
| random_forest | 0.35 | 0.1212 | 0.5648 | 0.3812 | 0.4552 |
| random_forest | 0.40 | 0.0875 | 0.5897 | 0.2875 | 0.3866 |
| random_forest | 0.50 | 0.0449 | 0.5750 | 0.1437 | 0.2300 |
| catboost | 0.20 | 0.5118 | 0.2895 | 0.8250 | 0.4286 |
| catboost | 0.30 | 0.3603 | 0.3458 | 0.6937 | 0.4615 |
| catboost | 0.35 | 0.3098 | 0.3841 | 0.6625 | 0.4862 |
| catboost | 0.40 | 0.2772 | 0.4008 | 0.6188 | 0.4865 |
| catboost | 0.50 | 0.2110 | 0.4574 | 0.5375 | 0.4943 |
| lightgbm | 0.20 | 0.2963 | 0.3902 | 0.6438 | 0.4858 |
| lightgbm | 0.30 | 0.2245 | 0.4700 | 0.5875 | 0.5222 |
| lightgbm | 0.35 | 0.1987 | 0.4972 | 0.5500 | 0.5223 |
| lightgbm | 0.40 | 0.1728 | 0.5195 | 0.5000 | 0.5096 |
| lightgbm | 0.50 | 0.1459 | 0.5231 | 0.4250 | 0.4690 |
| xgboost | 0.20 | 0.3322 | 0.3649 | 0.6750 | 0.4737 |
| xgboost | 0.30 | 0.2581 | 0.4087 | 0.5875 | 0.4821 |
| xgboost | 0.35 | 0.2189 | 0.4513 | 0.5500 | 0.4958 |
| xgboost | 0.40 | 0.1908 | 0.4588 | 0.4875 | 0.4727 |
| xgboost | 0.50 | 0.1403 | 0.5200 | 0.4062 | 0.4561 |