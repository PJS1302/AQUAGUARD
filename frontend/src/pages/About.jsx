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
    <Notice tone="warning" title="The user's original ML work was not supplied in this workspace">No trained artifact, scaler, preprocessing pipeline, feature configuration, training dataset, or evaluation report was provided. AquaGuard therefore does not claim a production model or report fabricated metrics. The bundled CSV is synthetic demonstration data only.</Notice>
    <Card className="architecture-card"><div className="card-head"><div><div className="eyebrow">TASK SEPARATION</div><h3>Each concept has a proper purpose</h3><p>Training experiments are separate from production inference.</p></div><Button variant="secondary" onClick={()=>window.open('/api/docs','_blank','noopener,noreferrer')}>Open API docs <Icon name="arrow" size={14}/></Button></div><div className="concept-grid">{concepts.map((item)=><div className="concept-card" key={item.title}><span className="concept-icon"><Icon name={item.icon} size={18}/></span><div><h4>{item.title}</h4><p>{item.purpose}</p><small>{item.where}</small></div></div>)}</div><div className="purpose-band"><span><b>Classification</b> → discrete labels</span><span><b>Regression</b> → continuous values</span><span><b>PCA</b> → reduce / visualize</span><span><b>Clustering</b> → discover groups</span><span><b>K-Fold</b> → evaluate reliability</span><span><b>IoT</b> → collect readings</span></div></Card>
    <div className="setup-grid"><Card><div className="card-head"><div><div className="eyebrow">MODEL INSTALLATION</div><h3>Bring your validated model</h3><p>Supported local artifact layout.</p></div></div><pre className="setup-code">{`backend/models/
├── production_model.pkl
├── scaler.pkl                 # if separate
├── preprocessing.pkl          # if separate
├── feature_config.json        # exact ordered features
└── aquaguard_pipeline.pkl     # optional integrated pipeline`}</pre><div className="setup-note">Models are stored on the backend filesystem—not in PostgreSQL. Do not commit private or sensitive artifacts.</div></Card><Card><div className="card-head"><div><div className="eyebrow">SERVICE CONNECTIONS</div><h3>Deployment readiness</h3><p>Health checks report only verified states.</p></div></div><div className="setup-status-list"><div><span>FastAPI</span><Tag tone={health?.status==='ok'?'success':'warning'} dot>{health?.status==='ok'?'Connected':'Unavailable'}</Tag></div><div><span>PostgreSQL</span><Tag tone={health?.database?.available?'success':'warning'} dot>{health?.database?.available?'Connected':'Not configured'}</Tag></div><div><span>Production model</span><Tag tone={health?.production_model?.available?'success':'neutral'} dot>{health?.production_model?.available?health.production_model.model_name:'Not installed'}</Tag></div></div><div className="setup-note">Use <code>.env.example</code> for local environment settings and the README for installation details.</div></Card></div>
    <div className="about-bottom"><span>Demo data is synthetic; simulation is not telemetry.</span><span>Model evaluation values are calculated from actual selected data.</span><span>Original uploads are never overwritten.</span></div>
  </div>;
}
