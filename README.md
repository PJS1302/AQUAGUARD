# AquaGuard — Water Quality ML + IoT Analytics

AquaGuard is a React/JavaScript + FastAPI + PostgreSQL platform for water-quality dataset inspection, task-appropriate ML experiments, a configurable production inference path, and IoT history. It is intentionally not a single pipeline that forces every algorithm to solve the same problem.

> **Model status for this project snapshot:** The user's original training code, model artifacts, feature contract, preprocessing objects and evaluation report were not supplied. AquaGuard therefore ships with **no production model** and makes no production prediction until a real artifact is installed or a real dataset experiment is explicitly promoted. The bundled `Demo Dataset — Sample Data` is synthetic and is not live water data or a validated safety dataset.

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

- **Production prediction:** one selected and validated classifier + the user's saved preprocessing/scaler, exact ordered feature list and model artifact. No automatic PCA or lab estimator is inserted.
- **Regression:** continuous targets only; Linear, Multiple Linear, Polynomial and Locally Weighted Linear Regression. K-Fold reports actual fold scores.
- **Classification:** CART Decision Tree, numeric-threshold C4.5-style gain-ratio tree, Gaussian Naive Bayes, KNN, Bagging, AdaBoost and Random Forest. Model comparison uses actual holdout/cross-validation results.
- **PCA:** exploratory dimensionality reduction over the user's processed numeric values; AquaGuard adds no scaler.
- **Clustering:** K-Means and Ward agglomerative clustering over user-processed numeric values as-is, with Silhouette, Davies–Bouldin and Calinski–Harabasz where defined. Cluster IDs are not Safe/Unsafe labels.
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

AquaGuard does **not** replace or silently retrain the user's existing ML work. No original preprocessing code or artifact was supplied with this project, so AquaGuard ships with **no default preprocessing pipeline**. Until the real pipeline is installed, datasets can still be uploaded, preserved, and inspected, but preprocessing-copy creation and ML experiments fail closed rather than substituting app-defined transforms.

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

The object must implement `preprocess_frame(frame: pandas.DataFrame) -> pandas.DataFrame`. Implement this method using **only the same steps and order as the user's Python ML development code**. AquaGuard passes it a deep copy of the uploaded dataframe. The method's result must retain the original dataframe index labels for all retained rows, and include the processed target column for supervised experiments. This preserves row-to-label alignment when the user's own code excludes rows. Do not add an app-side `dropna`, imputer, duplicate remover, scaler, encoder, or any other step.

If the original pipeline excludes missing-value rows, it should do so inside `preprocess_frame`; only the separate processing copy is affected. If the pipeline explicitly imputes, the imputation must be implemented there because it was part of the user's original workflow. Any missing selected feature/target values left by the user's pipeline block an experiment; AquaGuard will not drop or impute them. Regression and classification train/test experiments require at least **12 complete processed rows** across the selected inputs and target. With fewer than 12, no split, fit, or metrics are produced.

For explicit production promotion of a lab classifier, the same object must additionally implement `transform_features(raw_frame: pandas.DataFrame) -> pandas.DataFrame` and declare the exact raw `input_features`. `transform_features` must preserve every input row and its index. The processed feature columns must match those used in the lab fit. If this inference contract is absent or incompatible, promotion is disabled; install the validated production model/pipeline directly instead.

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

Use **Load demo dataset** to test CSV validation and read-only inspection. The demo is synthetic and cannot be promoted. Demo and uploaded datasets follow the same installed user pipeline before any ML experiment; if no actual pipeline is installed, AquaGuard blocks processing and experiments rather than falling back to sample transforms. For real datasets, choose **Datasets & Upload** and upload a CSV. AquaGuard reports rows, columns, types, missing values and duplicates; incompatible processed features fail clearly.

The upload is stored byte-for-byte and never overwritten. `POST /datasets/{dataset_id}/preprocess` applies only the locally installed user's pipeline to a separate copy and saves that output separately. The processing report shows the pipeline and row counts. An incomplete processed feature/target set is rejected; the application never applies its own missing-row deletion or imputation.

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
| Gaussian Naive Bayes | Numeric processed features; categorical inputs must be handled by the user's pipeline | Weighted metrics and confusion matrix |
| KNN | Uses the user's processed feature values exactly; no AquaGuard scaler or encoder | Configurable K, weighted metrics |
| Bagging | Bootstrap aggregation of decision-tree estimators on processed inputs | Actual holdout and CV metrics |
| AdaBoost | Sequentially reweighted weak tree estimators | Estimator count, learning rate and actual metrics |
| Random Forest | Bagged randomized decision trees | Metrics and impurity-based feature importance |
| PCA | Explicit exploratory reduction of the user's processed numeric values; no automatic scaler | Explained and cumulative variance, loadings and PC scores |
| K-Means | Unsupervised partitioning of user-processed numeric values as-is | Sizes, centers and internal scores |
| Hierarchical | Ward agglomerative grouping of user-processed numeric values as-is | Sizes, dendrogram and internal scores |

The exact installed user preprocessing pipeline runs on a separate copy before each ML analysis. Classification comparison uses the same processed feature set and stratified split across all methods. K-Fold uses stratified folds for classification and shuffled folds for regression. Regression/classification require at least 12 complete processed selected rows; any remaining missing inputs block the run rather than being dropped or imputed. Metrics are calculated from each actual run; there are no preset performance values or auto-promoted model.

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
