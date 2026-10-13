# AquaGuard — Water Quality ML + IoT Analytics

AquaGuard is a React/JavaScript + FastAPI + PostgreSQL platform for water-quality dataset inspection, task-appropriate ML experiments, a configurable production inference path, and IoT history. It is intentionally not a single pipeline that forces every algorithm to solve the same problem.

> **Model status:** Production inference uses only the artifact and exact feature contract selected in `backend/models/feature_config.json`. A standalone estimator without its fitted preprocessing artifact cannot safely accept missing values; AquaGuard reports that incompatibility instead of fitting transforms on prediction data or substituting another model. The bundled `Demo Dataset — Sample Data` is synthetic and is not live water data or a validated safety dataset.

## Architecture

```text
React / Vite (JavaScript + JSX only)
          │ same-origin /api proxy
          ▼
FastAPI ── pandas / NumPy / scikit-learn / joblib
   │                  │
   │                  └── backend/models/ (filesystem artifacts)
   └── PostgreSQL (devices, readings, datasets, predictions, experiment history)
```

- **Production prediction:** the explicitly selected production classifier + its saved fitted preprocessing/scaler (when required) and exact ordered feature list. No automatic PCA, newly fitted prediction transform, or substitute lab estimator is inserted.
- **Regression:** continuous targets only; Linear, Multiple Linear, Polynomial and Locally Weighted Linear Regression. K-Fold reports actual fold scores.
- **Classification:** CART Decision Tree, numeric-threshold C4.5-style gain-ratio tree, Gaussian Naive Bayes, KNN, Bagging, AdaBoost and Random Forest. Model comparison uses actual holdout/cross-validation results.
- **PCA:** exploratory dimensionality reduction over selected numeric features, with numeric conversion and missing-value handling reported in the result.
- **Clustering:** K-Means and Ward agglomerative clustering over selected dataset-aware numeric/encoded features, with Silhouette, Davies–Bouldin and Calinski–Harabasz where defined. Cluster IDs are not Safe/Unsafe labels.
- **IoT:** registered devices and readings are persisted to PostgreSQL. The optional simulation submits deterministic rows from the demo CSV and is explicitly marked `simulation`.

## Project structure

```text
AquaGuard/
├── frontend/
│   ├── public/demo/demo_water_quality.csv
│   ├── src/components/       # UI, layout, icons and charts
│   ├── src/hooks/            # platform health/dashboard hook
│   ├── src/pages/            # dashboard, datasets, labs, IoT, history, about
│   ├── src/services/         # all browser API calls
│   ├── src/utils/            # CSV result export
│   ├── App.jsx
│   └── vite.config.js
├── backend/
│   ├── routes/               # FastAPI route modules
│   ├── services/             # dataset, preprocessing, ML and device logic
│   ├── schemas/               # Pydantic request/response models
│   ├── utils/                 # validators, artifact loader, preprocessing helpers
│   ├── database/              # PostgreSQL ORM and migration notes
│   ├── models/                # user artifacts (none fabricated or included)
│   ├── main.py
│   └── requirements.txt
├── ml/                        # project ML workspace and training/evaluation entry points
├── .env.example
└── README.md
```

## Requirements

- Node.js 20+ and npm
- Python 3.10+ (the dependency set also supports current Python releases)
- PostgreSQL 14+ (required for persistent IoT and history)

## PostgreSQL setup

Create a database and a least-privileged application user using your own local credentials. Example commands for a local PostgreSQL shell (replace the password securely):

```sql
CREATE ROLE aquaguard LOGIN PASSWORD 'replace-with-a-local-secret';
CREATE DATABASE aquaguard OWNER aquaguard;
```

Copy the environment template and set a **PostgreSQL** URL:

```bash
cp .env.example .env
# Edit DATABASE_URL in .env; never commit .env
```

Example URL shape:

```text
postgresql+psycopg://aquaguard:<your-secret>@localhost:5432/aquaguard
```

The API creates missing development tables from `backend/database/models.py` at startup. Schema changes in production should use reviewed Alembic migrations. There is deliberately **no SQLite, Supabase, MongoDB, Firebase or MySQL fallback**. If PostgreSQL is not reachable, analysis endpoints can still calculate in memory and dataset files can be inspected, but persistent endpoints report that PostgreSQL is unavailable.

## Backend setup

From the project root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Default CSV limit: 25 MB; configure `AQUAGUARD_MAX_UPLOAD_MB` to change it.
- Uploads are validated as CSV, stored as their original bytes under `backend/data/uploads/`, and never overwritten. Each operation creates a separate in-memory or exported processing copy.

## Frontend setup

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). `vite.config.js` proxies same-origin `/api/*` requests to FastAPI at port 8000. React components call the API only through `src/services/` modules. For a production deployment, build with `npm run build` and serve the static assets behind an HTTPS reverse proxy that forwards `/api` to FastAPI.

## User model / preprocessing setup

AquaGuard does not replace or silently retrain the user's existing ML work. ML experiments use a dataset-aware feature pipeline: selected columns are normalized by detected type, feature imputers/encoders are fitted on the training partition only, and that fitted transformer is reused for validation, evaluation, and prediction. Missing or invalid supervised targets are not imputed; excluded target rows and feature transformations are reported. No source upload is changed. The separate optional preprocessing-copy endpoint below continues to run only an explicitly installed user pipeline.

### Install the exact Python preprocessing implementation

Install the user's reviewed pipeline locally at `backend/models/user_preprocessing_pipeline.joblib`. It must be a trusted artifact (joblib/pickle artifacts can execute code when loaded; never accept one directly from an untrusted upload). The artifact must either be the pipeline object itself or a dictionary of this shape:

```text
{
  "pipeline": <trusted Python object>,
  "metadata": {
    "name": "the user's pipeline name",
    "steps": ["optional exact step descriptions"],
    "input_features": ["raw input columns in exact order"]
  }
}
```

The object must implement `preprocess_frame(frame: pandas.DataFrame) -> pandas.DataFrame`. Implement this method using **only the same steps and order as the user's Python ML development code**. AquaGuard passes it a deep copy of the uploaded dataframe. The method's result must retain the original dataframe index labels for all retained rows and any target columns needed by the optional processing copy. This preserves row-to-label alignment when the user's own code excludes rows. This optional endpoint does not alter the independent dataset-aware ML experiment pipeline.

If the original user pipeline excludes rows or explicitly imputes, that behavior belongs inside `preprocess_frame`; only the separate processing copy is affected. The ML experiment path independently reports and excludes rows with unusable targets and fits selected-feature imputation/encoding on its training partition. There is no universal minimum of 12 rows: requirements depend on the task, target distribution, requested evaluation method, and algorithm. For example, holdout regression needs at least two valid targets, while classification also needs enough observations per class to put every class in both partitions.

Lab promotion saves the estimator together with its fitted dataset-aware preprocessing pipeline and the exact ordered raw feature list. Promotion is explicit and remains disabled for synthetic demo data; it does not reuse or refit transforms from the optional processing-copy artifact.

### Production model artifacts

Preferred production layout:

```text
backend/models/
├── production_model.pkl       # fitted estimator, if transforms are separate
├── scaler.pkl                 # optional only if it is the exact fitted scaler
├── preprocessing.pkl          # optional only if it is the exact fitted preprocessor
├── feature_config.json        # exact ordered raw input feature names/model name
└── aquaguard_pipeline.pkl     # optional integrated fitted preprocessing + estimator
```

`feature_config.example.json` is a schema example only. Do not copy its placeholder feature names into a real deployment. Add an actual `feature_config.json` only when names, order, target and preprocessing match the validated artifact. Prefer an integrated model pipeline saved from the user's Python code. Model artifacts remain in filesystem/artifact storage, not PostgreSQL.

Production prediction uses only the installed model's actual preprocessing/model artifact. Missing feature columns or values the artifact cannot process return a compatibility error; the API never inserts a guessed transform. `predict_proba` is used only if the artifact exposes it; a returned probability is labeled model-reported and is not claimed to be calibrated.

The Classification Lab saves evaluation artifacts under `backend/models/experiments/`. Synthetic demo data cannot be promoted. A non-demo candidate is promotable only when the installed user preprocessing artifact also supports the exact raw-inference contract above. Promotion is explicit, non-destructive, and versioned; it does not certify drinking-water safety.

## Demo dataset and upload workflow

`frontend/public/demo/demo_water_quality.csv` uses a familiar water-quality feature layout (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`, `Potability`) and includes missing cells plus a duplicate row for data-quality demonstrations. It is deterministic **synthetic sample data** with illustrative labels—not a lab result, live sensor reading, safety threshold or validated dataset.

Use **Load demo dataset** to test CSV validation and read-only inspection. The demo is synthetic and cannot be promoted. Demo and uploaded datasets follow the same dataset-aware ML handling. AquaGuard reports rows, columns, inferred types, missing values, and duplicates; unsupported selections fail clearly. The separately installed user pipeline is used only when explicitly creating a processing copy.

The upload is stored byte-for-byte and never overwritten. `POST /datasets/{dataset_id}/preprocess` applies only the locally installed user's pipeline to a separate copy and saves that output separately. The processing report shows the pipeline and row counts. An incomplete processed feature/target set is rejected; the app does not silently modify the user's optional processing-copy pipeline.

## ML Laboratory guide

| Concept | Purpose | Evaluation / output |
|---|---|---|
| Linear Regression | One numeric feature → continuous target | MAE, MSE, RMSE, R², coefficient, intercept |
| Multiple Linear Regression | Multiple numeric features → continuous target | Same regression metrics and coefficients |
| Polynomial Regression | Nonlinear feature expansion, degree 2–5 | Same metrics, generated terms and actual-vs-predicted chart |
| LWLR | Local weighted linear fit for continuous targets | Configurable bandwidth, query estimate, holdout metrics |
| K-Fold | Validation technique, not a predictor | Individual fold scores, mean and standard deviation |
| Decision Tree / CART | Classification via CART tree splits | Weighted metrics, confusion matrix |
| C4.5-style tree | Numeric threshold splits ranked by entropy-based gain ratio, with mean-gain filter | Explicitly distinct from CART; tree summary and metrics |
| Gaussian Naive Bayes | Uses training-fitted numeric imputation and categorical encoding; numeric scales are unchanged | Weighted metrics and confusion matrix |
| KNN | Fits feature imputation/encoding on the training split; numeric scales are otherwise unchanged | Configurable K, weighted metrics |
| Bagging | Bootstrap aggregation of decision-tree estimators on processed inputs | Actual holdout and CV metrics |
| AdaBoost | Sequentially reweighted weak tree estimators | Estimator count, learning rate and actual metrics |
| Random Forest | Bagged randomized decision trees | Metrics and impurity-based feature importance |
| PCA | Selected numerical features only; numeric conversion and analysis-fitted missing-value handling; no automatic standardization | Explained and cumulative variance, loadings and PC scores |
| K-Means | Dataset-aware numeric conversion, imputation and categorical encoding | Sizes, centers and internal scores |
| Hierarchical | Dataset-aware numeric conversion, imputation and categorical encoding | Sizes, dendrogram and internal scores |

ML analyses use the selected original columns and task-aware conversion for numeric, categorical, Boolean, and date/time data. Missing/invalid target rows are counted and excluded; missing feature values are handled by transformations fitted only on training data and reused for the holdout/folds. Categorical features are encoded by the same fitted pipeline. Classification comparison uses a shared split and task-compatible estimators. K-Fold uses stratified folds for classification and shuffled folds for regression when available class counts/sample count support the requested fold count. PCA and clustering report the feature handling they apply. Sample and feature requirements are validated per algorithm/evaluation method; no universal 12-row or fixed feature-count rule is imposed. Metrics are calculated from actual runs; there are no preset performance values or auto-promoted models.

## IoT simulation and real sensor integration

Register a device under **IoT Monitoring** after PostgreSQL is online. Real sensors can send either nested or flat numeric JSON:

```http
POST /devices/AquaSensor-01/readings
Content-Type: application/json
```

```json
{
  "device_id": "AquaSensor-01",
  "timestamp": "2026-10-05T15:30:00+05:30",
  "ph": 7.2,
  "turbidity": 3.8,
  "conductivity": 420
}
```

The reading is validated and persisted. If a production artifact exists and the reading feature keys match, the configured preprocessing/model is then used and the output is stored. If not, telemetry remains saved and the response says prediction is unavailable. The UI's optional simulator posts up to three existing demo preview rows with `source: "simulation"`; it never calls the values live or invents random readings.

## PostgreSQL tables

`devices`, `sensor_readings`, `datasets`, `predictions`, `model_runs`, `clustering_runs`, `regression_runs`, and `pca_runs` are declared in `backend/database/models.py`. JSON columns retain feature payloads/metrics; model artifacts remain on the backend filesystem. Device status is based on a reading received in the last 10 minutes.

## API endpoints

- `GET /health`, `GET /dashboard`
- `POST /predict`, `POST /predict-file`, `GET /predictions`
- `GET /datasets`, `POST /datasets/upload`, `GET /datasets/{dataset_id}`, `GET /datasets/{dataset_id}/original`, `POST /datasets/{dataset_id}/preprocess` (applies the installed user pipeline only), `GET /preprocessing/status`
- `POST /regression/linear`, `/regression/multiple`, `/regression/polynomial`, `/regression/lwlr`
- `POST /classification/decision-tree`, `/c45`, `/naive-bayes`, `/knn`, `/bagging`, `/adaboost`, `/random-forest`, `/compare`, `/production`
- `POST /pca/analyze`
- `POST /clustering/kmeans`, `/hierarchical`, `/compare`
- `GET /devices`, `POST /devices`, `GET /devices/{device_id}`, `GET /devices/{device_id}/readings`, `POST /devices/{device_id}/readings`
- Processed-copy export: `GET /datasets/processed/{process_id}`

OpenAPI/Swagger is served at `/docs` when FastAPI is running.

## Security and operational notes

- CSV-only uploads, a configurable size limit, bounded row/column counts, supervised and pairwise-analysis resource guards, polynomial-term limits and duplicate-column validation.
- Uploaded files are stored as data, never executed.
- SQLAlchemy ORM uses parameterized statements; no user-built SQL strings.
- Secrets are environment variables; `.env` is ignored by Git.
- CORS defaults to the local Vite origin and can be changed with `AQUAGUARD_CORS_ORIGINS`.
- The API has no authentication layer in this starter. Put it behind authenticated HTTPS before exposing it beyond a trusted network.
- Do not use synthetic demo/simulated data or an unvalidated model to make drinking-water safety decisions.

## Verification

```bash
# Frontend production build
cd frontend && npm run build

# Backend compile and preprocessing-contract tests
cd .. && python -m compileall backend
python -m unittest backend.tests.test_preprocessing_contract -v
# With dependencies installed:
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

This starter prioritizes transparent behavior: incompatible datasets fail clearly, original data remains unchanged, actual experiments calculate their own values, and absent user artifacts remain absent.
