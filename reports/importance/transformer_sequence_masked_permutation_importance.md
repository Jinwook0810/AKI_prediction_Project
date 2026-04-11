# Transformer Masked Sequence Permutation Importance

- baseline: `transformer_sequence_masked_v1`
- baseline_auroc: `0.748803`
- baseline_auprc: `0.390005`
- importance metric: feature-wise drop after shuffling both value and mask channels together across stays

## Top 20 by AUPRC Drop

| rank | feature | auprc_drop | auroc_drop |
| --- | --- | --- | --- |
| 1 | Urine_ml_per_kg_hr | 0.034831 | 0.011226 |
| 2 | PaO2 | 0.019190 | 0.009867 |
| 3 | Temp | 0.018929 | 0.018434 |
| 4 | Na | 0.018652 | 0.017408 |
| 5 | Creatinine | 0.017746 | 0.034037 |
| 6 | HCT | 0.016272 | 0.015048 |
| 7 | Weight | 0.014868 | 0.017237 |
| 8 | pH | 0.013873 | 0.004685 |
| 9 | HR | 0.013065 | 0.009217 |
| 10 | Urine | 0.012133 | 0.008806 |
| 11 | GCS | 0.011531 | -0.001659 |
| 12 | Height | 0.011403 | 0.005215 |
| 13 | Gender | 0.010144 | -0.000009 |
| 14 | RespRate | 0.009852 | 0.004848 |
| 15 | NISysABP | 0.007766 | 0.002266 |
| 16 | HCO3 | 0.007401 | 0.005780 |
| 17 | NIMAP | 0.006766 | 0.005275 |
| 18 | Lactate | 0.006244 | 0.002779 |
| 19 | Mg | 0.005570 | 0.003942 |
| 20 | K | 0.004568 | 0.003591 |
