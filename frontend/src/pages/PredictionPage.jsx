import React, { useEffect, useState } from 'react';
import Icon from '../components/Icons';
import { Button, Card, LoadingButton, Notice, Tag } from '../components/UI';
import { ShapContributionBars, ShapWaterfallChart, ShapGlobalSummary } from '../components/Charts';
import { predictDataset, predictOne, predictUploadFile, getGlobalShap } from '../services/predictionService';
import { getDevices, getReadings } from '../services/iotService';
import { getHealth } from '../services/dashboardService';
import { downloadCsv } from '../utils/export';

const DEFAULT_UNITS = {
  ph: 'pH',
  Hardness: 'mg/L',
  Solids: 'ppm',
  Chloramines: 'ppm',
  Sulfate: 'mg/L',
  Conductivity: 'μS/cm',
  Organic_carbon: 'ppm',
  Trihalomethanes: 'μg/L',
  Turbidity: 'NTU',
};

const PRESETS = [
  {
    name: 'Clean Tap Water',
    values: {
      ph: 7.25,
      Hardness: 185.0,
      Solids: 14200.0,
      Chloramines: 6.8,
      Sulfate: 295.0,
      Conductivity: 410.0,
      Organic_carbon: 10.5,
      Trihalomethanes: 52.0,
      Turbidity: 2.8,
    },
  },
  {
    name: 'High Hardness Well',
    values: {
      ph: 7.95,
      Hardness: 340.0,
      Solids: 22000.0,
      Chloramines: 4.2,
      Sulfate: 380.0,
      Conductivity: 650.0,
      Organic_carbon: 14.8,
      Trihalomethanes: 78.0,
      Turbidity: 4.9,
    },
  },
  {
    name: 'Contaminated Runoff',
    values: {
      ph: 5.85,
      Hardness: 360.0,
      Solids: 31000.0,
      Chloramines: 2.1,
      Sulfate: 410.0,
      Conductivity: 820.0,
      Organic_carbon: 22.5,
      Trihalomethanes: 105.0,
      Turbidity: 7.4,
    },
  },
  {
    name: 'Mountain Spring',
    values: {
      ph: 7.1,
      Hardness: 145.0,
      Solids: 9500.0,
      Chloramines: 5.5,
      Sulfate: 220.0,
      Conductivity: 320.0,
      Organic_carbon: 8.2,
      Trihalomethanes: 35.0,
      Turbidity: 1.6,
    },
  },
];

export default function PredictionPage({ health, activeDataset, onRefresh }) {
  const [directModel, setDirectModel] = useState(health?.production_model || null);
  const [loadingModel, setLoadingModel] = useState(!health?.production_model);

  useEffect(() => {
    if (health?.production_model?.available) {
      setDirectModel(health.production_model);
      setLoadingModel(false);
    } else {
      getHealth()
        .then((res) => {
          if (res?.production_model?.available) {
            setDirectModel(res.production_model);
          }
        })
        .catch(() => {})
        .finally(() => setLoadingModel(false));
    }
  }, [health?.production_model]);

  const model = (directModel?.available ? directModel : null) || health?.production_model;
  const required = model?.features || [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
  ];
  const units = model?.feature_units || DEFAULT_UNITS;

  const [inputSource, setInputSource] = useState('manual'); // 'manual' | 'csv' | 'iot'
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [fileResult, setFileResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [fileBusy, setFileBusy] = useState(false);
  const [error, setError] = useState('');
  const [fileError, setFileError] = useState('');
  const [shapTab, setShapTab] = useState('local'); // 'local' | 'waterfall' | 'global'
  const [globalShap, setGlobalShap] = useState(null);
  const [loadingGlobalShap, setLoadingGlobalShap] = useState(false);

  // IoT sensor integration state
  const [iotDevices, setIotDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [latestIotReading, setLatestIotReading] = useState(null);
  const [loadingIot, setLoadingIot] = useState(false);

  // Initialize input fields
  useEffect(() => {
    const initial = {};
    required.forEach((f) => {
      initial[f] = '';
    });
    setValues(initial);
    setResult(null);
  }, [required.join('|')]);

  // Load IoT devices when IoT tab selected
  useEffect(() => {
    if (inputSource === 'iot') {
      setLoadingIot(true);
      getDevices()
        .then((res) => {
          const list = res.items || [];
          setIotDevices(list);
          if (list.length > 0) {
            setSelectedDevice(list[0].device_id);
          }
        })
        .catch(() => {})
        .finally(() => setLoadingIot(false));
    }
  }, [inputSource]);

  // Fetch latest reading for selected IoT device
  useEffect(() => {
    if (selectedDevice && inputSource === 'iot') {
      getReadings(selectedDevice, 1)
        .then((res) => {
          const items = res.items || [];
          if (items.length > 0) {
            setLatestIotReading(items[0]);
          } else {
            setLatestIotReading(null);
          }
        })
        .catch(() => setLatestIotReading(null));
    }
  }, [selectedDevice, inputSource]);

  // Load Global SHAP importance when tab selected
  useEffect(() => {
    if (shapTab === 'global' && !globalShap && model?.available) {
      setLoadingGlobalShap(true);
      getGlobalShap()
        .then((res) => setGlobalShap(res))
        .catch(() => {})
        .finally(() => setLoadingGlobalShap(false));
    }
  }, [shapTab, model?.available]);

  const applyPreset = (presetValues) => {
    setValues((prev) => ({ ...prev, ...presetValues }));
    setError('');
  };

  const applyIotReading = () => {
    if (!latestIotReading?.features) return;
    const readingFeats = latestIotReading.features;
    const newVals = { ...values };
    required.forEach((f) => {
      if (readingFeats[f] != null) {
        newVals[f] = readingFeats[f];
      }
    });
    setValues(newVals);
    setInputSource('manual');
  };

  // Submit manual prediction
  async function submit(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    setBusy(true);

    try {
      const features = {};
      for (const feature of required) {
        const val = values[feature];
        if (val === '' || val === undefined || !Number.isFinite(Number(val))) {
          throw new Error(`Enter a valid numeric value for ${feature}.`);
        }
        features[feature] = Number(val);
      }

      const response = await predictOne(features, 'manual');
      setResult(response);
      onRefresh?.();
    } catch (e) {
      setError(e.message || 'Prediction could not be completed.');
    } finally {
      setBusy(false);
    }
  }

  // Submit active dataset prediction
  async function predictCurrentDataset() {
    if (!activeDataset) return;
    setFileError('');
    setFileResult(null);
    setFileBusy(true);
    try {
      const response = await predictDataset(activeDataset.id);
      setFileResult(response);
      onRefresh?.();
    } catch (e) {
      setFileError(e.message || 'Dataset prediction could not be completed.');
    } finally {
      setFileBusy(false);
    }
  }

  // Upload custom CSV file for immediate prediction
  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileError('');
    setFileResult(null);
    setFileBusy(true);
    try {
      const response = await predictUploadFile(file);
      setFileResult(response);
      onRefresh?.();
    } catch (e) {
      setFileError(e.message || 'File prediction failed. Ensure the CSV contains required headers.');
    } finally {
      setFileBusy(false);
    }
  }

  const isModelReady = Boolean(model?.available);
  const isPotable = result?.predicted_label === 'POTABLE';
  const hasProbabilities = result?.class_probabilities != null;

  return (
    <div className="page-stack">
      {/* Model Status Alert */}
      {loadingModel && !isModelReady ? (
        <Notice
          tone="neutral"
          title="Connecting to Machine Learning Model…"
        >
          Connecting to backend and loading trained model specifications…
        </Notice>
      ) : !isModelReady ? (
        <Notice
          tone="warning"
          title="AI prediction unavailable — trained model not configured"
        >
          {model?.reason && `${model.reason} `}
          The application requires a compatible trained water-potability model and matching preprocessing pipeline installed in <code>backend/models/production_model.pkl</code>. You can train and save a genuine classifier using the ML Laboratory or the training workflow in <code>ml/train.py</code>.
        </Notice>
      ) : (
        <Notice
          tone="success"
          title={`Active Model: ${model.model_name}`}
        >
          Trained ML classifier loaded with {required.length} input features and verified potability target mapping (0: NON-POTABLE, 1: POTABLE). Preprocessing and SHAP explainability pipelines are connected.
        </Notice>
      )}

      {/* Main Prediction Workspace Layout */}
      <div className="prediction-layout">
        <Card className="prediction-form-card">
          <div className="card-head">
            <div>
              <div className="eyebrow">INPUT SOURCE & PARAMETERS</div>
              <h3>Water Quality Readings</h3>
              <p>Enter chemical and physical sensor measurements to analyze potability.</p>
            </div>
            <div className="prediction-shield">
              <Icon name="droplet" size={20} />
            </div>
          </div>

          {/* Three Input Sources Selector */}
          <div className="input-source-nav">
            <button
              type="button"
              className={`source-tab-btn ${inputSource === 'manual' ? 'active' : ''}`}
              onClick={() => setInputSource('manual')}
            >
              <Icon name="settings" size={14} />
              Manual Readings
            </button>
            <button
              type="button"
              className={`source-tab-btn ${inputSource === 'csv' ? 'active' : ''}`}
              onClick={() => setInputSource('csv')}
            >
              <Icon name="database" size={14} />
              CSV Batch File
            </button>
            <button
              type="button"
              className={`source-tab-btn ${inputSource === 'iot' ? 'active' : ''}`}
              onClick={() => setInputSource('iot')}
            >
              <Icon name="radio" size={14} />
              Live IoT Sensors
            </button>
          </div>

          {/* Source 1: Manual Input */}
          {inputSource === 'manual' && isModelReady && (
            <>
              {/* Quick Presets */}
              <div className="preset-strip">
                <span className="preset-label">Quick Samples:</span>
                {PRESETS.map((p) => (
                  <button
                    key={p.name}
                    type="button"
                    className="preset-btn"
                    onClick={() => applyPreset(p.values)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>

              <form onSubmit={submit}>
                <div className="prediction-fields">
                  {required.map((feature) => (
                    <label className="field" key={feature}>
                      <span className="field-label">
                        {feature.replace('_', ' ')}
                        {units[feature] && <small className="muted">({units[feature]})</small>}
                      </span>
                      <input
                        type="number"
                        step="any"
                        value={values[feature] ?? ''}
                        onChange={(e) => setValues({ ...values, [feature]: e.target.value })}
                        placeholder={`e.g. ${PRESETS[0].values[feature] ?? '0.0'}`}
                        required
                      />
                    </label>
                  ))}
                </div>

                {error && <div className="inline-error" style={{ marginTop: '12px' }}>{error}</div>}

                <div className="prediction-form-footer">
                  <span>
                    <Icon name="shield" size={14} />
                    Validates measurements, applies exact pipeline transforms, and infers class via {model.model_name}.
                  </span>
                  <LoadingButton
                    type="submit"
                    loading={busy}
                    loadingLabel="Inferring with ML Model…"
                    icon="droplet"
                  >
                    Analyze Water Quality
                  </LoadingButton>
                </div>
              </form>
            </>
          )}

          {/* Source 2: CSV File Upload */}
          {inputSource === 'csv' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="drop-zone" onClick={() => document.getElementById('csv-predict-file-input').click()}>
                <input
                  id="csv-predict-file-input"
                  type="file"
                  accept=".csv"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <div className="upload-circle">
                  <Icon name="upload" size={20} />
                </div>
                <strong>
                  <span>Click to upload CSV</span> or drag & drop
                </strong>
                <small>Requires columns: {required.join(', ')}</small>
              </div>

              {activeDataset && (
                <div className="batch-dataset">
                  <div>
                    <strong>Active Dataset: {activeDataset.file_name}</strong>
                    <div style={{ fontSize: '9px', color: '#7a8c91' }}>
                      {activeDataset.row_count} rows · {activeDataset.is_demo ? 'Demo sample' : 'User uploaded'}
                    </div>
                  </div>
                  <LoadingButton
                    variant="secondary"
                    loading={fileBusy}
                    loadingLabel="Predicting…"
                    onClick={predictCurrentDataset}
                    disabled={!isModelReady}
                  >
                    Predict Dataset
                  </LoadingButton>
                </div>
              )}

              {fileError && <div className="inline-error">{fileError}</div>}
            </div>
          )}

          {/* Source 3: Live IoT Sensors */}
          {inputSource === 'iot' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ fontSize: '11px', color: '#274b55' }}>Connected ESP32 & IoT Devices</strong>
                  <div style={{ fontSize: '9px', color: '#889da2' }}>
                    Select a sensor hardware node registered in PostgreSQL.
                  </div>
                </div>
                {selectedDevice && (
                  <span className="ws-live-pill">
                    <span className="ws-live-dot" />
                    Live Stream Connected
                  </span>
                )}
              </div>

              {iotDevices.length === 0 ? (
                <div className="compact-empty">
                  <Icon name="radio" size={24} />
                  <span>No IoT devices currently registered in PostgreSQL. Register a device in IoT Monitoring.</span>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {iotDevices.map((d) => (
                    <button
                      key={d.device_id}
                      type="button"
                      className={`preset-btn ${selectedDevice === d.device_id ? 'active' : ''}`}
                      onClick={() => setSelectedDevice(d.device_id)}
                      style={{
                        padding: '8px 12px',
                        background: selectedDevice === d.device_id ? '#e4f7f2' : '#f8faf9',
                        borderColor: selectedDevice === d.device_id ? '#0d9488' : '#dbe5e4',
                      }}
                    >
                      <b>{d.device_name}</b> <small>({d.device_id})</small>
                    </button>
                  ))}
                </div>
              )}

              {latestIotReading ? (
                <div style={{ background: '#f5f9f8', border: '1px solid #dce8e6', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#567279' }}>
                      Timestamp: {new Date(latestIotReading.timestamp).toLocaleTimeString()}
                    </span>
                    <Tag tone={latestIotReading.source === 'simulation' ? 'warning' : 'success'}>
                      {latestIotReading.source === 'simulation' ? 'SIMULATED DATA' : 'PHYSICAL SENSOR (ESP32)'}
                    </Tag>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', fontSize: '9.5px', marginBottom: '12px' }}>
                    {Object.entries(latestIotReading.features || {}).map(([k, v]) => (
                      <div key={k} style={{ background: '#fff', padding: '4px 6px', borderRadius: '4px', border: '1px solid #eef2f1' }}>
                        <span style={{ color: '#7a8e94' }}>{k}: </span>
                        <strong>{typeof v === 'number' ? v.toFixed(2) : v} {units[k] || ''}</strong>
                      </div>
                    ))}
                  </div>
                  <Button variant="primary" icon="arrow" onClick={applyIotReading}>
                    Populate Readings into Analyzer
                  </Button>
                </div>
              ) : (
                selectedDevice && (
                  <div className="compact-empty">
                    <span>No telemetry received yet for {selectedDevice}.</span>
                  </div>
                )
              )}
            </div>
          )}

          {/* Model Unavailable Notice */}
          {!isModelReady && inputSource === 'manual' && (
            loadingModel ? (
              <div style={{ padding: '36px 16px', textAlign: 'center', color: '#688289' }}>
                <div style={{ marginBottom: '8px' }}>
                  <Icon name="activity" size={24} />
                </div>
                <strong>Connecting to AquaGuard ML Model…</strong>
                <p style={{ fontSize: '11px', marginTop: '4px' }}>Loading classifier parameters from backend.</p>
              </div>
            ) : (
              <div className="artifact-missing">
                <div className="artifact-illustration">
                  <Icon name="cpu" size={28} />
                </div>
                <h4>Production Model Needed</h4>
                <p>
                  AquaGuard does not guess or fabricate predictions. Install your validated ML artifact and matching preprocessing pipeline at <code>backend/models/production_model.pkl</code>.
                </p>
                <Button
                  variant="secondary"
                  icon="flask"
                  onClick={() => (window.location.hash = '#laboratory')}
                >
                  Go to ML Laboratory
                </Button>
              </div>
            )
          )}
        </Card>

        {/* Inference Architecture Pipeline Card */}
        <div className="page-stack compact-stack">
          <Card className="prediction-info-card">
            <div className="eyebrow">INFERENCE PIPELINE</div>
            <h3>Validated ML Architecture</h3>
            <div className="pipeline-vertical">
              <div className="pipeline-node">
                <span>01</span>
                <div>
                  <b>Validated Input Readings</b>
                  <small>Exact 9 chemical & physical features with units</small>
                </div>
              </div>
              <i />
              <div className="pipeline-node">
                <span>02</span>
                <div>
                  <b>Compatible Preprocessing</b>
                  <small>Coercion and fitted training imputation</small>
                </div>
              </div>
              <i />
              <div className="pipeline-node">
                <span>03</span>
                <div>
                  <b>Trained Classifier Inference</b>
                  <small>{model?.model_name || 'Production Model'}</small>
                </div>
              </div>
              <i />
              <div className="pipeline-node pipeline-node-end">
                <span><Icon name="check" size={15} /></span>
                <div>
                  <b>Explainable AI with SHAP</b>
                  <small>Local feature attribution & PostgreSQL persistence</small>
                </div>
              </div>
            </div>
          </Card>

          {/* Batch Prediction Result Summary */}
          {fileResult && (
            <Card>
              <div className="card-head">
                <div>
                  <div className="eyebrow">BATCH PREDICTION RESULTS</div>
                  <h3>{fileResult.row_count} Processed Rows</h3>
                  <p>{fileResult.file_name} · {fileResult.model_name}</p>
                </div>
                <Button
                  variant="secondary"
                  icon="download"
                  onClick={() => downloadCsv('aquaguard_batch_predictions.csv', fileResult.results)}
                >
                  Export CSV
                </Button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                <div style={{ background: '#e8f7f2', padding: '10px', borderRadius: '8px', border: '1px solid #78dfca' }}>
                  <div style={{ fontSize: '9px', color: '#0d7a6e', fontWeight: '700' }}>POTABLE PREDICTIONS</div>
                  <strong style={{ fontSize: '20px', color: '#0a4b3f' }}>{fileResult.potable_count} rows</strong>
                </div>
                <div style={{ background: '#fff0ef', padding: '10px', borderRadius: '8px', border: '1px solid #f8a69e' }}>
                  <div style={{ fontSize: '9px', color: '#b91c1c', fontWeight: '700' }}>NON-POTABLE PREDICTIONS</div>
                  <strong style={{ fontSize: '20px', color: '#7f1d1d' }}>{fileResult.non_potable_count} rows</strong>
                </div>
              </div>

              <div className="table-scroll" style={{ maxHeight: '250px' }}>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Predicted Label</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fileResult.results.slice(0, 50).map((r) => (
                      <tr key={r.row}>
                        <td>{r.row}</td>
                        <td>
                          <Tag tone={r.predicted_label === 'POTABLE' ? 'success' : 'danger'}>
                            {r.predicted_label}
                          </Tag>
                        </td>
                        <td>{r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Single Prediction Result Display */}
      {result && (
        <Card className={`prediction-result-banner ${isPotable ? 'banner-potable' : 'banner-nonpotable'}`}>
          <div className="card-head">
            <div>
              <div className="eyebrow" style={{ color: isPotable ? '#0d7a6e' : '#b91c1c' }}>
                AUTHENTIC ML INFERENCE RESULT · {result.model_name}
              </div>
              <div className={`prediction-badge ${isPotable ? 'badge-potable' : 'badge-nonpotable'}`}>
                <Icon name={isPotable ? 'check' : 'close'} size={18} />
                <span>{result.status_text || (isPotable ? 'POTABLE — predicted potable by the ML model' : 'NON-POTABLE — predicted non-potable by the ML model')}</span>
              </div>
              <p style={{ color: isPotable ? '#164e43' : '#7f1d1d' }}>
                Inferred on {new Date(result.timestamp).toLocaleString()} using {result.model_name} ({result.model_status}).
              </p>
            </div>
            <Tag tone={result.persisted ? 'success' : 'neutral'}>
              {result.persisted ? 'SAVED TO POSTGRESQL' : 'UNSAVED (DB OFFLINE)'}
            </Tag>
          </div>

          {/* Probabilities / Confidence */}
          {hasProbabilities && (
            <div className="prob-bars-grid">
              <div className="prob-card">
                <div className="prob-header">
                  <span>POTABILITY PROBABILITY</span>
                  <strong style={{ color: '#0d9488' }}>
                    {((result.class_probabilities['POTABLE'] || 0) * 100).toFixed(1)}%
                  </strong>
                </div>
                <div className="prob-track">
                  <div
                    className="prob-fill-potable"
                    style={{ width: `${(result.class_probabilities['POTABLE'] || 0) * 100}%` }}
                  />
                </div>
              </div>
              <div className="prob-card">
                <div className="prob-header">
                  <span>NON-POTABLE PROBABILITY</span>
                  <strong style={{ color: '#ef4444' }}>
                    {((result.class_probabilities['NON-POTABLE'] || 0) * 100).toFixed(1)}%
                  </strong>
                </div>
                <div className="prob-track">
                  <div
                    className="prob-fill-nonpotable"
                    style={{ width: `${(result.class_probabilities['NON-POTABLE'] || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Input Readings Summary Grid */}
          <div style={{ marginTop: '14px', marginBottom: '14px' }}>
            <div style={{ fontSize: '9.5px', fontWeight: '700', marginBottom: '8px', color: isPotable ? '#0d7a6e' : '#991b1b' }}>
              INPUT MEASUREMENTS EVALUATED:
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
              {(result.input_readings || []).map((inp) => (
                <div
                  key={inp.feature}
                  style={{
                    background: 'rgba(255, 255, 255, 0.7)',
                    padding: '8px 10px',
                    borderRadius: '7px',
                    border: '1px solid rgba(0, 0, 0, 0.05)',
                  }}
                >
                  <div style={{ fontSize: '8.5px', color: '#688289' }}>{inp.feature}</div>
                  <strong style={{ fontSize: '12px', color: '#1a3c46' }}>
                    {inp.value} <small style={{ fontWeight: 'normal', color: '#8aa0a6' }}>{inp.unit}</small>
                  </strong>
                </div>
              ))}
            </div>
          </div>

          {/* Explainable AI (SHAP) Section */}
          {result.shap_explanation && (
            <div className="shap-card">
              <div className="card-head">
                <div>
                  <div className="eyebrow">EXPLAINABLE AI · SHAP ATTRIBUTION</div>
                  <h3>Why the Model Predicted {result.predicted_label}</h3>
                  <p>Shapley Additive Explanations calculate each feature's marginal contribution to the prediction.</p>
                </div>
                <div className="shap-nav">
                  <button
                    type="button"
                    className={`shap-tab ${shapTab === 'local' ? 'active' : ''}`}
                    onClick={() => setShapTab('local')}
                  >
                    Feature Contributions
                  </button>
                  <button
                    type="button"
                    className={`shap-tab ${shapTab === 'waterfall' ? 'active' : ''}`}
                    onClick={() => setShapTab('waterfall')}
                  >
                    Waterfall Step Plot
                  </button>
                  <button
                    type="button"
                    className={`shap-tab ${shapTab === 'global' ? 'active' : ''}`}
                    onClick={() => setShapTab('global')}
                  >
                    Global Model Importance
                  </button>
                </div>
              </div>

              {/* Narrative Summary */}
              {result.shap_explanation.summary_text && (
                <div className="shap-narrative">
                  <strong>Key Influencing Factors: </strong>
                  {result.shap_explanation.summary_text}
                </div>
              )}

              {/* Optional Generative AI Expert Advisory (Gemini / OpenAI) */}
              {result.shap_explanation.ai_expert_advisory && (
                <div className="shap-ai-advisory" style={{
                  background: 'linear-gradient(135deg, rgba(13, 148, 136, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%)',
                  border: '1px solid rgba(13, 148, 136, 0.28)',
                  borderRadius: '10px',
                  padding: '12px 14px',
                  marginBottom: '14px',
                  display: 'flex',
                  gap: '12px',
                  alignItems: 'flex-start'
                }}>
                  <span style={{
                    background: '#0d9488',
                    color: '#ffffff',
                    padding: '3px 8px',
                    borderRadius: '5px',
                    fontSize: '10px',
                    fontWeight: '700',
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                    whiteSpace: 'nowrap',
                    marginTop: '2px'
                  }}>
                    {result.shap_explanation.ai_expert_advisory.provider} AI Advisory
                  </span>
                  <span style={{ fontSize: '12.5px', color: '#134e4a', lineHeight: 1.55 }}>
                    {result.shap_explanation.ai_expert_advisory.advisory}
                  </span>
                </div>
              )}

              {/* View 1: Local SHAP Horizontal Bars */}
              {shapTab === 'local' && (
                <ShapContributionBars contributions={result.shap_explanation.contributions || []} />
              )}

              {/* View 2: Waterfall Step Analysis */}
              {shapTab === 'waterfall' && (
                <ShapWaterfallChart waterfall={result.shap_explanation.waterfall || []} />
              )}

              {/* View 3: Global Feature Importance */}
              {shapTab === 'global' && (
                <div>
                  {loadingGlobalShap ? (
                    <div className="compact-empty">Loading global SHAP baseline...</div>
                  ) : globalShap?.items ? (
                    <ShapGlobalSummary items={globalShap.items} />
                  ) : (
                    <div className="compact-empty">Global SHAP summary not available.</div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Safety Notice */}
          <div className="result-caveat" style={{ marginTop: '16px' }}>
            <Icon name="shield" size={15} />
            <span>
              <strong>Drinking Water Safety Notice: </strong>
              {result.safety_notice || 'A model prediction is not a certified drinking-water safety assessment. Laboratory testing and applicable drinking-water standards are necessary to establish safety.'}
            </span>
          </div>
        </Card>
      )}
    </div>
  );
}
