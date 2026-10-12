# AquaGuard ML Laboratory — demo run

**Run date:** 2026-10-06  
**Dataset:** `frontend/public/demo/demo_water_quality.csv` — synthetic sample data, not real readings.  
**Purpose:** integration verification of the user-authorized complete-case policy and ML Laboratory APIs; results are illustrative only, not water-safety evidence.

## Processing-copy check

| Check | Result |
|---|---:|
| Original upload | 73 rows × 10 columns |
| Rows containing at least one missing cell removed | 6 |
| Processed copy | 67 rows × 10 columns |
| Missing cells left in processed copy | 0 |
| Exact duplicate rows retained | 1 |
| Original download byte-identical to bundled upload | Yes |

The only preprocessing operation was complete-case row removal on a separate copy. No imputation, duplicate removal, scaling, encoding, or other transformation was applied. The new rule is **not** claimed to reproduce the unavailable original pipeline.

## Classification — target `Potability`

Processed target counts: class 0: 41, class 1: 26 (synthetic labels). Nine numeric inputs; 5-fold stratified evaluation; K=5; 100 estimators; AdaBoost learning rate 0.8. Metrics below are actual API results from this run. Accuracy, weighted precision/recall/F1 use the deterministic stratified holdout; CV is mean 5-fold accuracy.

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | 5-fold mean accuracy |
|---|---:|---:|---:|---:|---:|
| Decision Tree / CART | 0.4286 | 0.4694 | 0.4286 | 0.4405 | 0.5703 |
| C4.5-style (gain ratio) | 0.4286 | 0.4694 | 0.4286 | 0.4405 | 0.6407 |
| Gaussian Naive Bayes | 0.5714 | 0.6122 | 0.5714 | 0.5804 | 0.6725 |
| K-Nearest Neighbors | 0.4286 | 0.4286 | 0.4286 | 0.4286 | 0.5692 |
| Bagging | 0.5000 | 0.5625 | 0.5000 | 0.5077 | 0.6736 |
| AdaBoost | 0.5000 | 0.5625 | 0.5000 | 0.5077 | 0.6571 |
| Random Forest | 0.5000 | 0.5625 | 0.5000 | 0.5077 | 0.6725 |

All seven classifiers completed without API errors; each used 67 processed rows. The classes in this synthetic demo are illustrative and unvalidated.

## Regression — `Hardness` from `ph`

Linear Regression ran on 67 processed rows with a deterministic holdout and 5-fold CV.

| Metric | Value |
|---|---:|
| MAE | 63.446234 |
| MSE | 5821.437877 |
| RMSE | 76.298348 |
| R² | -0.071367 |
| 5-fold mean R² | -0.212605 |

## PCA and clustering checks

- PCA completed on 67 rows for `ph`, `Hardness`, and `Solids` with 2 components. Explained variance ratios: 0.9999, 0.0001 (cumulative 1.0000).
- K-Means and Ward hierarchical clustering each completed on 67 rows for the same 3 numeric features with 3 clusters.
- K-Means: silhouette 0.6161, Davies–Bouldin 0.4697; sizes {'0': 21, '1': 23, '2': 23}.
- Hierarchical: silhouette 0.5861, Davies–Bouldin 0.4786; sizes {'0': 27, '1': 23, '2': 17}.
- PCA/clustering used native processed values with no scaler. These analyses are exploratory; differences in feature units can dominate their results.

## Production boundary

**No production model was installed, no experiment was promoted, and production eligibility remained false.** The newly authorized complete-case rule is limited to ML Laboratory experiments and does not support raw production inference. `DATABASE_URL` was not configured during this verification, so these test runs were not persisted to PostgreSQL.

The processed CSV is included alongside this report as `demo_water_quality_complete_cases.csv` (SHA-256 `7b9ef6c74afa4b2d2a46928c0af9f75383966a4019cadf39fc9ba5d785a1147b`). It is a separate output; the bundled original remains unchanged.
