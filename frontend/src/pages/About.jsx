import React from 'react';
import Icon from '../components/Icons';
import { Button, Card, Notice, Tag } from '../components/UI';

const concepts=[
  {icon:'settings',title:'Preprocessing',purpose:'Read-only data quality checks and the user’s exact Python pipeline on a separate immutable copy.',where:'ML Laboratory → Preprocessing'},
  {icon:'chart',title:'Regression',purpose:'Linear, multiple linear, polynomial and locally weighted models for continuous numerical targets.',where:'ML Laboratory → Regression'},
  {icon:'tree',title:'Classification',purpose:'Decision Tree/CART, gain-ratio C4.5-style tree, Naive Bayes, KNN, Bagging, AdaBoost and Random Forest.',where:'ML Laboratory → Classification'},
  {icon:'activity',title:'K-Fold validation',purpose:'Reliability evaluation with per-fold scores, mean and standard deviation—not a predictor.',where:'Regression and Classification results'},
  {icon:'cpu',title:'PCA',purpose:'Standardized dimensionality reduction, explained variance, loadings and visualization.',where:'ML Laboratory → PCA'},
  {icon:'layers',title:'Clustering',purpose:'K-Means and agglomerative groups compared with internal validation metrics.',where:'ML Laboratory → Clustering'},
  {icon:'radio',title:'IoT',purpose:'PostgreSQL device registry, sensor history, device-wise analysis and clearly labeled simulation.',where:'IoT Monitoring'},
  {icon:'shield',title:'Production prediction',purpose:'One explicitly selected, validated artifact with its original preprocessing and feature contract.',where:'Demo Prediction'},
];

export default function About({health}) {
  return <div className="page-stack">
    <section className="about-hero"><div className="about-hero-mark"><Icon name="droplet" size={32}/></div><div className="eyebrow">AQUAGUARD PLATFORM</div><h2>One platform.<br/><em>Different jobs for different models.</em></h2><p>AquaGuard connects water-quality datasets, a real model deployment path, an ML teaching laboratory, and IoT history—without collapsing every algorithm into one prediction pipeline.</p><div className="about-hero-tags"><Tag tone="success">React · Vite</Tag><Tag tone="blue">FastAPI · scikit-learn</Tag><Tag tone="neutral">PostgreSQL</Tag></div></section>
    {health?.production_model?.available ? (
      <Notice tone="info" title="Installed production model and preprocessing pipeline">
        The production model (<strong>{health.production_model.model_name}</strong>) is active with its connected preprocessing pipeline. Evaluation experiments and telemetry run against real data without fabricated metrics.
      </Notice>
    ) : (
      <Notice tone="warning" title="No production model installed">
        Install a validated model artifact and feature configuration in <code>backend/models/</code> to enable production predictions. The bundled CSV is synthetic demonstration data only.
      </Notice>
    )}
    <Card className="architecture-card"><div className="card-head"><div><div className="eyebrow">TASK SEPARATION</div><h3>Each concept has a proper purpose</h3><p>Training experiments are separate from production inference.</p></div><Button variant="secondary" onClick={()=>window.open('/api/docs','_blank','noopener,noreferrer')}>Open API docs <Icon name="arrow" size={14}/></Button></div><div className="concept-grid">{concepts.map((item)=><div className="concept-card" key={item.title}><span className="concept-icon"><Icon name={item.icon} size={18}/></span><div><h4>{item.title}</h4><p>{item.purpose}</p><small>{item.where}</small></div></div>)}</div><div className="purpose-band"><span><b>Classification</b> → discrete labels</span><span><b>Regression</b> → continuous values</span><span><b>PCA</b> → reduce / visualize</span><span><b>Clustering</b> → discover groups</span><span><b>K-Fold</b> → evaluate reliability</span><span><b>IoT</b> → collect readings</span></div></Card>
    <div className="setup-grid">
      <Card>
        <div className="card-head">
          <div>
            <div className="eyebrow">MODEL INSTALLATION</div>
            <h3>Bring your validated model</h3>
            <p>Supported local artifact layout.</p>
          </div>
        </div>
        <pre className="setup-code">{`backend/models/
├── production_model.pkl
├── scaler.pkl                 # if separate
├── preprocessing.pkl          # if separate
├── feature_config.json        # exact ordered features & target labels
└── aquaguard_pipeline.pkl     # optional integrated pipeline`}</pre>
        <div className="setup-note">
          Models are stored on the backend filesystem—not in PostgreSQL. Run <code>python -m ml.train</code> to train a genuine classifier directly from a labelled CSV.
        </div>
      </Card>

      <Card>
        <div className="card-head">
          <div>
            <div className="eyebrow">SERVICE CONNECTIONS</div>
            <h3>Deployment readiness</h3>
            <p>Health checks report only verified states.</p>
          </div>
        </div>
        <div className="setup-status-list">
          <div>
            <span>FastAPI Backend</span>
            <Tag tone={health?.status === 'ok' ? 'success' : 'warning'} dot>
              {health?.status === 'ok' ? 'Connected' : 'Unavailable'}
            </Tag>
          </div>
          <div>
            <span>PostgreSQL Database</span>
            <Tag tone={health?.database?.available ? 'success' : 'warning'} dot>
              {health?.database?.available ? 'Connected' : 'Not configured'}
            </Tag>
          </div>
          <div>
            <span>Production ML Model</span>
            <Tag tone={health?.production_model?.available ? 'success' : 'neutral'} dot>
              {health?.production_model?.available ? health.production_model.model_name : 'Not installed'}
            </Tag>
          </div>
        </div>
        <div className="setup-note">
          Use <code>.env</code> for local settings and <code>http://localhost:8000/docs</code> for interactive Swagger documentation.
        </div>
      </Card>
    </div>

    {/* Full API Connection & ESP32 Integration Guide */}
    <Card style={{ marginTop: '16px' }}>
      <div className="card-head">
        <div>
          <div className="eyebrow">API CONNECTION & INTEGRATION GUIDE</div>
          <h3>How to Connect Devices & External Applications to the API</h3>
          <p>Everything you need to send water readings, retrieve predictions, and stream live IoT data.</p>
        </div>
        <Button variant="secondary" icon="arrow" onClick={() => window.open('/docs', '_blank')}>
          Interactive Swagger Docs (/docs)
        </Button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px', marginTop: '10px' }}>
        <div style={{ background: '#f8faf9', border: '1px solid #dce8e6', borderRadius: '9px', padding: '14px' }}>
          <strong style={{ fontSize: '11px', color: '#13545b', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Icon name="radio" size={14} /> 1. Where do I get the API URL?
          </strong>
          <p style={{ fontSize: '9.5px', color: '#5b747b', lineHeight: '1.6', marginTop: '6px' }}>
            • <b>Local development:</b> Your backend runs at <code>http://localhost:8000</code> or <code>http://127.0.0.1:8000</code>. On the same Wi-Fi network, replace localhost with your PC’s local IP (e.g., <code>http://192.168.1.100:8000</code>) so your ESP32 hardware can reach it.<br />
            • <b>Production deployment:</b> Deploy the FastAPI backend on Render, Railway, AWS, or your own VPS to get a public HTTPS URL (e.g. <code>https://your-aquaguard-api.onrender.com</code>).
          </p>
        </div>

        <div style={{ background: '#f8faf9', border: '1px solid #dce8e6', borderRadius: '9px', padding: '14px' }}>
          <strong style={{ fontSize: '11px', color: '#13545b', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Icon name="droplet" size={14} /> 2. Core API Endpoints
          </strong>
          <p style={{ fontSize: '9.5px', color: '#5b747b', lineHeight: '1.6', marginTop: '6px' }}>
            • <code>POST /api/predict</code> — Single reading inference + SHAP explanation<br />
            • <code>POST /api/predict-file</code> — Batch CSV file inference with row-by-row labels<br />
            • <code>POST /api/devices/{'{id}'}/readings</code> — Ingest live IoT telemetry (ESP32)<br />
            • <code>GET /api/prediction/shap-global</code> — Global feature importance ranking<br />
            • <code>WS /api/ws/iot</code> — Live real-time WebSocket telemetry stream
          </p>
        </div>

        <div style={{ background: '#f8faf9', border: '1px solid #dce8e6', borderRadius: '9px', padding: '14px' }}>
          <strong style={{ fontSize: '11px', color: '#13545b', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Icon name="cpu" size={14} /> 3. Do SHAP & AI Need API Keys?
          </strong>
          <p style={{ fontSize: '9.5px', color: '#5b747b', lineHeight: '1.6', marginTop: '6px' }}>
            • <b>SHAP & ML Inference:</b> Run <b>100% locally and offline</b> on your computer using <code>scikit-learn</code> and <code>shap</code>. No paid external API key is needed.<br />
            • <b>Optional Generative AI Advisory:</b> If you want natural-language biochemical advice, add <code>GEMINI_API_KEY</code> or <code>OPENAI_API_KEY</code> to your project's <code>.env</code> file.
          </p>
        </div>

        <div style={{ background: '#f8faf9', border: '1px solid #dce8e6', borderRadius: '9px', padding: '14px' }}>
          <strong style={{ fontSize: '11px', color: '#13545b', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Icon name="shield" size={14} /> 4. Where to Set Keys in .env
          </strong>
          <p style={{ fontSize: '9.5px', color: '#5b747b', lineHeight: '1.6', marginTop: '6px' }}>
            • Open <code>.env</code> in the project root directory.<br />
            • Add <code>GEMINI_API_KEY="AIzaSy..."</code> from Google AI Studio or <code>OPENAI_API_KEY="sk-..."</code>.<br />
            • For secure ESP32 sensor ingress, optionally set <code>AQUAGUARD_API_KEY="your-secret"</code> and send the <code>X-API-Key</code> header.
          </p>
        </div>
      </div>

      <div style={{ marginTop: '16px' }}>
        <div style={{ fontSize: '10.5px', fontWeight: '700', color: '#20434c', marginBottom: '8px' }}>
          Quick Connection Code Examples:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '12px' }}>
          <div>
            <div style={{ fontSize: '9px', fontWeight: '700', color: '#456a73', marginBottom: '4px' }}>
              Python (requests):
            </div>
            <pre style={{ background: '#0e2936', color: '#b2e2d8', padding: '10px', borderRadius: '7px', fontSize: '8.5px', overflow: 'auto', maxHeight: '160px' }}>
{`import requests

url = "http://localhost:8000/api/predict"
payload = {
    "source": "manual",
    "features": {
        "ph": 7.2, "Hardness": 180.0, "Solids": 14000.0,
        "Chloramines": 6.5, "Sulfate": 280.0, "Conductivity": 410.0,
        "Organic_carbon": 10.2, "Trihalomethanes": 55.0, "Turbidity": 2.8
    }
}
res = requests.post(url, json=payload).json()
print("Prediction:", res["predicted_label"])
print("SHAP Factors:", res["shap_explanation"]["summary_text"])`}
            </pre>
          </div>

          <div>
            <div style={{ fontSize: '9px', fontWeight: '700', color: '#456a73', marginBottom: '4px' }}>
              cURL (Terminal):
            </div>
            <pre style={{ background: '#0e2936', color: '#b2e2d8', padding: '10px', borderRadius: '7px', fontSize: '8.5px', overflow: 'auto', maxHeight: '160px' }}>
{`curl -X POST http://localhost:8000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "source": "manual",
    "features": {
      "ph": 7.2, "Hardness": 180, "Solids": 14000,
      "Chloramines": 6.5, "Sulfate": 280, "Conductivity": 410,
      "Organic_carbon": 10.2, "Trihalomethanes": 55, "Turbidity": 2.8
    }
  }'`}
            </pre>
          </div>
        </div>
      </div>
    </Card>

    <div className="about-bottom">
      <span>Demo data is synthetic; simulation is not telemetry.</span>
      <span>Model evaluation values are calculated from actual selected data.</span>
      <span>Original uploads are never overwritten.</span>
    </div>
  </div>;
}
