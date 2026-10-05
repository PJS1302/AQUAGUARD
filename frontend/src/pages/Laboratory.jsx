import React, { useEffect, useMemo, useState } from 'react';
import Icon from '../components/Icons';
import { ActualPredictedChart, ConfusionMatrix, Dendrogram, ScatterChart, TinyBars, VarianceBars } from '../components/Charts';
import { Button, Card, DataTable, EmptyState, FeaturePicker, LoadingButton, MetricCard, Notice, NumberField, SectionHeading, SelectField, TabBar, Tag } from '../components/UI';
import { compareClassifiers, promoteModel, runClassifier, runClustering, runPca, runRegression } from '../services/mlService';
import { downloadProcessedFile, getPreprocessingStatus, preprocessDataset } from '../services/datasetService';
import { downloadCsv } from '../utils/export';

const TABS = [
  {id:'preprocessing',label:'Preprocessing',icon:'settings'},
  {id:'regression',label:'Regression',icon:'chart'},
  {id:'classification',label:'Classification',icon:'tree'},
  {id:'pca',label:'PCA',icon:'cpu'},
  {id:'clustering',label:'Clustering',icon:'layers'},
  {id:'comparison',label:'Model comparison',icon:'activity'},
];
const CLASSIFIERS = [
  {value:'decision-tree',label:'Decision Tree / CART'}, {value:'c45',label:'C4.5 · gain ratio'},
  {value:'naive-bayes',label:'Gaussian Naive Bayes'}, {value:'knn',label:'K-Nearest Neighbors'},
  {value:'bagging',label:'Bagging'}, {value:'adaboost',label:'AdaBoost'}, {value:'random-forest',label:'Random Forest'},
];
const REGRESSORS = [
  {value:'linear',label:'Linear Regression · one feature'}, {value:'multiple',label:'Multiple Linear Regression'},
  {value:'polynomial',label:'Polynomial Regression'}, {value:'lwlr',label:'Locally Weighted Linear Regression'},
];
const CLUSTERERS = [{value:'compare',label:'Compare both'}, {value:'kmeans',label:'K-Means'}, {value:'hierarchical',label:'Hierarchical / Agglomerative'}];
const defaultTarget = (columns, numeric, prefer) => columns.includes(prefer) ? prefer : (numeric.at(-1) || columns.at(-1) || '');

function MetricValue({ label, value, percent = false }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return <MetricCard label={label} value="—" detail="Not returned" icon="chart" tone="blue"/>;
  const display = percent ? `${(Number(value)*100).toFixed(1)}%` : Number(value).toFixed(4);
  return <MetricCard label={label} value={display} detail="Calculated from this run" icon="chart" tone="blue"/>;
}

function CrossValidation({ data }) {
  if (!data) return null;
  return <Card className="cv-card"><div className="card-head"><div><div className="eyebrow">VALIDATION</div><h3>{data.folds ? `${data.folds}-Fold Cross Validation` : 'K-Fold Cross Validation'}</h3><p>{data.metric ? `Fold scores · ${data.metric}` : 'Actual fold evaluation'}</p></div><Tag tone={data.mean_score == null?'neutral':'success'}>{data.mean_score == null?'Unavailable':`Mean ${Number(data.mean_score).toFixed(4)}`}</Tag></div>{data.fold_scores?.length ? <div className="fold-score-grid">{data.fold_scores.map((score,i)=><div className="fold-score" key={i}><span>Fold {i+1}</span><strong>{Number(score).toFixed(4)}</strong></div>)}</div> : <div className="inline-message">{data.message || 'Fold scores were not returned for this run.'}</div>}{data.std_dev != null && <div className="cv-summary"><span>Standard deviation</span><strong>{Number(data.std_dev).toFixed(4)}</strong></div>}<div className="cv-explainer">K-Fold is an evaluation technique; it is not a separate prediction model.</div></Card>;
}

function ClassificationResult({ result, onExport }) {
  const m = result.metrics || {};
  return <div className="page-stack compact-stack"><div className="results-title-row"><div><div className="eyebrow">EVALUATION RESULT</div><h3>{result.display_name || result.algorithm}</h3><p>{result.complete_rows} complete processed samples · {result.preprocessing?.rows_removed_by_pipeline??0} rows removed by the user's pipeline</p></div><Button variant="secondary" icon="download" onClick={onExport}>Export prediction results</Button></div><div className="metrics-grid result-metrics"><MetricValue label="Accuracy" value={m.accuracy} percent/><MetricValue label="Precision · weighted" value={m.precision} percent/><MetricValue label="Recall · weighted" value={m.recall} percent/><MetricValue label="F1 · weighted" value={m.f1} percent/></div><div className="analysis-grid"><Card><div className="card-head"><div><div className="eyebrow">ERROR ANALYSIS</div><h3>Confusion matrix</h3><p>Rows are actual labels; columns are model predictions.</p></div></div><ConfusionMatrix data={m.confusion_matrix}/></Card><Card><div className="card-head"><div><div className="eyebrow">PREDICTION CHECK</div><h3>Holdout samples</h3><p>Actual and predicted labels from the test split.</p></div></div><DataTable columns={[{key:'index',label:'#'},{key:'actual',label:'Actual'},{key:'predicted',label:'Predicted'},{key:'match',label:'Result',render:(v,row)=>v?<Tag tone="success">Match</Tag>:<Tag tone="warning">Different</Tag>}]} rows={(result.actual||[]).slice(0,12).map((actual,i)=>({index:i+1,actual,predicted:result.predicted?.[i],match:String(actual)===String(result.predicted?.[i])}))}/></Card></div>{result.feature_importance?.length>0 && <Card><div className="card-head"><div><div className="eyebrow">MODEL INTERPRETATION</div><h3>Random Forest feature importance</h3><p>Impurity-based importance from the refit model; not causal effect.</p></div></div><TinyBars items={result.feature_importance}/></Card>}{result.tree && <Card><div className="card-head"><div><div className="eyebrow">{result.algorithm==='c45'?'C4.5 TREE STRUCTURE':'CART TREE STRUCTURE'}</div><h3>{result.algorithm==='c45'?'Gain-ratio split summary':'CART tree summary'}</h3><p>{result.method_note||result.tree_note}</p></div></div><pre className="tree-output">{typeof result.tree==='string'?result.tree:JSON.stringify(result.tree,null,2)}</pre></Card>}<CrossValidation data={result.cross_validation}/></div>;
}

function ComparisonResults({ comparison, onPromote, promoting, isDemo }) {
  if (!comparison) return null;
  const rows = (comparison.results || []).map((item)=>({...item,...(item.metrics||{}),cross_mean:item.cross_validation?.mean_score}));
  return <div className="page-stack compact-stack"><Notice tone="info" title="Compare actual experiments—not a hardcoded leaderboard">All displayed values were calculated from the selected dataset using the same stratified holdout setup. Review the split and class balance before choosing a production candidate.</Notice>{comparison.errors?.length>0 && <Notice tone="warning" title="Some algorithms could not run">{comparison.errors.map((item)=>`${item.algorithm}: ${item.message}`).join(' · ')}</Notice>}<Card className="comparison-card"><div className="card-head"><div><div className="eyebrow">CLASSIFIER BENCHMARK</div><h3>Model comparison</h3><p>Weighted precision, recall and F1; cross-validation is shown separately.</p></div><div className="comparison-head-actions"><Tag tone="success">{comparison.results?.length || 0} completed</Tag><Button variant="secondary" icon="download" onClick={()=>downloadCsv('aquaguard_model_comparison.csv',(comparison.results||[]).map((item)=>({model:item.display_name,accuracy:item.metrics?.accuracy,precision:item.metrics?.precision,recall:item.metrics?.recall,f1:item.metrics?.f1,cv_mean:item.cross_validation?.mean_score,cv_std:item.cross_validation?.std_dev}))) }>Export comparison</Button></div></div><DataTable columns={[{key:'display_name',label:'Model'},{key:'accuracy',label:'Accuracy',render:(v)=>v==null?'—':`${(Number(v)*100).toFixed(1)}%`},{key:'precision',label:'Precision',render:(v)=>v==null?'—':`${(Number(v)*100).toFixed(1)}%`},{key:'recall',label:'Recall',render:(v)=>v==null?'—':`${(Number(v)*100).toFixed(1)}%`},{key:'f1',label:'F1',render:(v)=>v==null?'—':`${(Number(v)*100).toFixed(1)}%`},{key:'cross_mean',label:'K-Fold mean',render:(v)=>v==null?'—':Number(v).toFixed(4)},{key:'action',label:'Action',render:(_,row)=><button className="table-action" disabled={isDemo||!row.production_eligible||promoting===row.algorithm||!row.experiment_id} onClick={()=>onPromote(row)} title={isDemo?'Demo data cannot be promoted':row.production_ineligible_reason||'Designate this evaluated model'}>{promoting===row.algorithm?'Saving…':'Set production'}</button>}]} rows={rows}/>{isDemo && <div className="demo-promotion-block"><Icon name="shield" size={16}/><span>Production promotion is disabled for the synthetic demo dataset. Upload a validated dataset to evaluate and designate a candidate.</span></div>}{!isDemo&&rows.some((row)=>!row.production_eligible)&&<div className="demo-promotion-block"><Icon name="shield" size={16}/><span>{rows.find((row)=>!row.production_eligible)?.production_ineligible_reason||'Promotion is unavailable until the exact preprocessing pipeline is compatible with raw inference.'}</span></div>}</Card></div>;
}

export default function Laboratory({ activeDataset, onLoadDemo, loadingDemo, initialTab = 'preprocessing', onModelPromoted, onNavigate }) {
  const [tab,setTab] = useState(initialTab);
  const [regTarget,setRegTarget] = useState('');
  const [classTarget,setClassTarget] = useState('');
  const [regFeatures,setRegFeatures] = useState([]);
  const [classFeatures,setClassFeatures] = useState([]);
  const [analysisFeatures,setAnalysisFeatures] = useState([]);
  const [regAlgorithm,setRegAlgorithm] = useState('linear');
  const [classAlgorithm,setClassAlgorithm] = useState('random-forest');
  const [clusterAlgorithm,setClusterAlgorithm] = useState('compare');
  const [degree,setDegree] = useState(2);
  const [bandwidth,setBandwidth] = useState(0.8);
  const [queryPoint,setQueryPoint] = useState({});
  const [folds,setFolds] = useState(5);
  const [knn,setKnn] = useState(5);
  const [estimators,setEstimators] = useState(100);
  const [learningRate,setLearningRate] = useState(0.8);
  const [clusters,setClusters] = useState(3);
  const [components,setComponents] = useState(3);
  const [pipelineStatus,setPipelineStatus] = useState(null);
  const [pipelineLoading,setPipelineLoading] = useState(false);
  const [result,setResult] = useState(null);
  const [comparison,setComparison] = useState(null);
  const [prepResult,setPrepResult] = useState(null);
  const [busy,setBusy] = useState(false);
  const [promoting,setPromoting] = useState('');
  const [error,setError] = useState('');
  const [message,setMessage] = useState('');

  useEffect(()=>setTab(initialTab),[initialTab]);
  useEffect(()=>{
    if (!activeDataset) return;
    const numeric=activeDataset.numeric_columns||[];
    const columns=activeDataset.columns||[];
    const reg=defaultTarget(columns,numeric,columns.includes('Hardness')?'Hardness':numeric.at(-1));
    const cls=defaultTarget(columns,numeric,'Potability');
    const analytical=numeric;
    setRegTarget(reg); setClassTarget(cls);
    setRegAlgorithm('linear'); setClassAlgorithm('random-forest'); setClusterAlgorithm('compare');
    setDegree(2); setBandwidth(0.8); setFolds(5); setKnn(5); setEstimators(100); setLearningRate(0.8); setClusters(3); setComponents(Math.max(1,Math.min(3,analytical.length||1)));
    setPipelineStatus(null);
    setPipelineLoading(true);
    getPreprocessingStatus().then(setPipelineStatus).catch((err)=>setPipelineStatus({available:false,reason:err.message||'Preprocessing status could not be loaded.'})).finally(()=>setPipelineLoading(false));
    setRegFeatures(numeric.filter((x)=>x!==reg).slice(0,1));
    setClassFeatures(columns.filter((x)=>x!==cls && (numeric.includes(x)|| (activeDataset.categorical_columns||[]).includes(x))).slice(0,8));
    setAnalysisFeatures(analytical.slice(0,Math.min(8,analytical.length)));
    const firstRow=activeDataset.preview?.[0]||{};
    setQueryPoint(Object.fromEntries(numeric.filter((name)=>name!==reg).map((name)=>[name,firstRow[name]??''])));
    setResult(null); setComparison(null); setPrepResult(null); setError(''); setMessage('');
  },[activeDataset?.id]);

  const columns=prepResult?.processed?.columns||activeDataset?.columns||[];
  const numeric=prepResult?.processed?.numeric_columns||activeDataset?.numeric_columns||[];
  const preprocessingReady=Boolean(pipelineStatus?.available&&prepResult?.processed&&prepResult.report?.artifact_sha256===pipelineStatus.artifact_sha256);
  const classFeatureOptions=columns.filter((x)=>x!==classTarget);
  const regFeatureOptions=numeric.filter((x)=>x!==regTarget);
  const analysisFeatureOptions=numeric;
  const targetOptions=columns.map((name)=>({value:name,label:`${name}${numeric.includes(name)?' · numeric':''}`}));

  function resetMessages(){setError('');setMessage('');setResult(null);setComparison(null);}
  function baseClassBody(){return {dataset_id:activeDataset.id,preprocessing_sha256:prepResult?.report?.artifact_sha256,target:classTarget,features:classFeatures,folds,k:knn,n_estimators:estimators,learning_rate:learningRate};}
  async function executeRegression(){
    resetMessages();setBusy(true);
    try { const query=regAlgorithm==='lwlr'&&regFeatures.every((name)=>queryPoint[name]!==''&&queryPoint[name]!==undefined&&Number.isFinite(Number(queryPoint[name])))?Object.fromEntries(regFeatures.map((name)=>[name,Number(queryPoint[name])])):undefined; const output=await runRegression(regAlgorithm,{dataset_id:activeDataset.id,preprocessing_sha256:prepResult?.report?.artifact_sha256,target:regTarget,features:regFeatures,degree,bandwidth,folds,query});setResult(output);setMessage(`${output.algorithm} completed using ${output.complete_rows} complete rows.`); }
    catch(e){setError(e.message||'Regression could not be completed.');}
    finally{setBusy(false);}
  }
  async function executeClassifier(){
    resetMessages();setBusy(true);
    try { const output=await runClassifier(classAlgorithm,baseClassBody());setResult(output);setMessage(`${output.display_name} completed. Review holdout and K-Fold metrics.`); }
    catch(e){setError(e.message||'Classification could not be completed.');}
    finally{setBusy(false);}
  }
  async function executeComparison(){
    resetMessages();setBusy(true);
    try {const output=await compareClassifiers(baseClassBody());setComparison(output);setMessage('Model comparison calculated from the selected dataset. No algorithm was automatically promoted.');}
    catch(e){setError(e.message||'Model comparison could not be completed.');}
    finally{setBusy(false);}
  }
  async function executePca(){
    resetMessages();setBusy(true);
    try {const output=await runPca({dataset_id:activeDataset.id,preprocessing_sha256:prepResult?.report?.artifact_sha256,features:analysisFeatures,n_components:components});setResult(output);setMessage(`PCA completed with ${output.complete_rows} complete rows.`);}
    catch(e){setError(e.message||'PCA could not be completed.');}
    finally{setBusy(false);}
  }
  async function executeClustering(){
    resetMessages();setBusy(true);
    try {const endpoint=clusterAlgorithm==='compare'?'compare':clusterAlgorithm;const output=await runClustering(endpoint,{dataset_id:activeDataset.id,preprocessing_sha256:prepResult?.report?.artifact_sha256,features:analysisFeatures,n_clusters:clusters});setResult(output);setMessage('Clustering labels and internal metrics were calculated from the selected data.');}
    catch(e){setError(e.message||'Clustering could not be completed.');}
    finally{setBusy(false);}
  }
  async function refreshPipelineStatus(){
    setPipelineLoading(true);
    try {setPipelineStatus(await getPreprocessingStatus());}
    catch(e){setPipelineStatus({available:false,reason:e.message||'Preprocessing status could not be loaded.'});}
    finally{setPipelineLoading(false);}
  }
  async function executePreprocess(){
    resetMessages();setPrepResult(null);setBusy(true);
    try {
      const output=await preprocessDataset(activeDataset.id);
      setPrepResult(output);
      const processedColumns=output.processed.columns||[];
      const processedNumeric=output.processed.numeric_columns||[];
      const reg=defaultTarget(processedColumns,processedNumeric,processedColumns.includes('Hardness')?'Hardness':processedNumeric.at(-1));
      const cls=defaultTarget(processedColumns,processedNumeric,'Potability');
      setRegTarget(reg);setClassTarget(cls);
      setRegFeatures(processedNumeric.filter((x)=>x!==reg).slice(0,1));
      setClassFeatures(processedColumns.filter((x)=>x!==cls).slice(0,8));
      setAnalysisFeatures(processedNumeric.slice(0,Math.min(8,processedNumeric.length)));
      setComponents(Math.max(1,Math.min(3,processedNumeric.length||1)));
      const firstRow=output.processed.preview?.[0]||{};
      setQueryPoint(Object.fromEntries(processedNumeric.filter((name)=>name!==reg).map((name)=>[name,firstRow[name]??''])));
      setMessage('The configured user pipeline created a separate processing copy. The uploaded original was not modified.');
    }
    catch(e){setError(e.message||'The user preprocessing pipeline could not create a processing copy.');}
    finally{setBusy(false);}
  }
  async function handlePromote(row){
    if (activeDataset?.is_demo||!row.production_eligible) return;
    const confirmed=window.confirm(`Set ${row.display_name} as AquaGuard's production classification model? Review its holdout and K-Fold metrics first. This changes the active model selection; previous artifact files are preserved.`);
    if(!confirmed)return;
    setPromoting(row.algorithm);setError('');setMessage('');
    try {const result=await promoteModel(row.experiment_id,row.algorithm);setMessage(`${result.model_name} is now selected as the production model. ${result.warning||''}`);onModelPromoted?.();}
    catch(e){setError(e.message||'Model promotion failed.');}
    finally{setPromoting('');}
  }
  function setTabAndClear(value){setTab(value);resetMessages();}
  function updateRegTarget(value){setRegTarget(value);setRegFeatures((prev)=>prev.filter((x)=>x!==value));}
  function updateClassTarget(value){setClassTarget(value);setClassFeatures((prev)=>prev.filter((x)=>x!==value));}
  function changeRegAlgorithm(value){setRegAlgorithm(value);setRegFeatures((prev)=>{const next=prev.filter((x)=>x!==regTarget);if(value==='linear')return next.slice(0,1);if(value==='multiple')return next.length>=2?next:regFeatureOptions.filter((x)=>!next.includes(x)).slice(0,2-next.length).concat(next);return next.length?next:regFeatureOptions.slice(0,2);});}

  if(!activeDataset){
    return <div className="page-stack"><TabBar items={TABS} active={tab} onChange={setTabAndClear}/><Card className="dataset-needed-card"><EmptyState icon="flask" title="Choose a dataset to begin the lab" action={<div className="empty-actions-row"><Button icon="database" onClick={onLoadDemo} disabled={loadingDemo}>{loadingDemo?'Loading sample…':'Load demo sample'}</Button><Button variant="secondary" icon="upload" onClick={()=>onNavigate('datasets')}>Upload a CSV</Button></div>}>Every experiment runs against a compatible uploaded or demo dataset. The bundled sample is explicitly synthetic and is never treated as live water data.</EmptyState></Card></div>;
  }

  return <div className="page-stack">
    <TabBar items={TABS} active={tab} onChange={setTabAndClear}/>
    <div className="active-dataset-strip"><div className="dataset-strip-icon"><Icon name="database" size={16}/></div><div><strong>{activeDataset.file_name}</strong><span>{activeDataset.row_count?.toLocaleString()} rows · {activeDataset.column_count} columns</span></div>{activeDataset.is_demo?<Tag tone="warning">DEMO · SAMPLE DATA</Tag>:<Tag tone="success">UPLOADED DATA</Tag>}<button className="strip-action" onClick={()=>onNavigate('datasets')}>Change dataset <Icon name="arrow" size={13}/></button></div>
    {activeDataset.is_demo && <Notice tone="warning" title="Synthetic sample data">Analyses are real computations over this demo CSV, but the sample's labels are not validated observations. Do not use its results for safety decisions or production promotion.</Notice>}
    {(!(Number(activeDataset.row_count)>0)||!(Number(activeDataset.column_count)>0))&&<Notice tone="error" title="This dataset is reporting zero rows or columns">The bundled demo should contain 73 rows and 10 columns. Reload it with Load demo dataset or re-upload a non-empty CSV; this count problem is separate from the missing preprocessing pipeline.</Notice>}
    {!pipelineLoading&&!pipelineStatus?.available&&<Notice tone="warning" title="ML experiments are paused until the exact preprocessing pipeline is installed">AquaGuard will not substitute its own missing-value handling, imputation, scaling, encoding, or transformations. Uploads can still be inspected; install the user's trusted pipeline to create a processing copy and run experiments.</Notice>}
    {pipelineStatus?.available&&!prepResult&&<Notice tone="info" title="Create the user-processed copy before running ML">Use the Preprocessing tab to apply the configured Python pipeline. Feature and target choices will then come from that processed copy.</Notice>}
    {pipelineStatus?.available&&prepResult&&prepResult.report?.artifact_sha256!==pipelineStatus.artifact_sha256&&<Notice tone="warning" title="Preprocessing pipeline changed">Create a fresh processing copy before running experiments. AquaGuard will not evaluate data with a pipeline different from the one used to build the visible copy.</Notice>}
    {error && <Notice tone="error" title="Analysis could not be completed">{error}</Notice>}
    {message && <Notice tone="success" title="Analysis status">{message}</Notice>}

    {tab==='preprocessing' && <div className="page-stack compact-stack">
      <div className="preprocess-pipeline-strip"><div className="pipeline-step-active"><span>01</span><b>Original dataset</b><small>Preserved unchanged</small></div><Icon name="arrow" size={14}/><div><span>02</span><b>User's Python pipeline</b><small>No app-invented steps</small></div><Icon name="arrow" size={14}/><div><span>03</span><b>Processing copy</b><small>Only the actual pipeline output</small></div></div>
      <div className="analysis-grid">
        <Card><div className="card-head"><div><div className="eyebrow">SOURCE DATA · READ ONLY</div><h3>Original dataset inspection</h3><p>These statistics describe the preserved upload; inspection does not alter it.</p></div><Tag tone="success">ORIGINAL · READ ONLY</Tag></div><div className="quality-overview"><div><strong>{activeDataset.row_count?.toLocaleString()??'—'}</strong><span>rows</span></div><div><strong>{activeDataset.column_count??'—'}</strong><span>columns</span></div><div><strong>{activeDataset.missing_total??'—'}</strong><span>missing cells</span></div><div><strong>{activeDataset.data_quality?.duplicate_rows??'—'}</strong><span>duplicate rows</span></div></div><div className="inline-message"><Icon name="shield" size={15}/>The raw CSV remains unchanged. Every ML run reads it again and preprocesses a separate copy.</div></Card>
        <Card className="preprocessing-options-card"><div className="card-head"><div><div className="eyebrow">USER PREPROCESSING PIPELINE</div><h3>Use the exact Python pipeline</h3><p>AquaGuard does not provide cleaning, imputation, deletion, scaling, encoding, or other default transforms.</p></div><div className="preprocess-icon"><Icon name="settings" size={18}/></div></div>
          {pipelineLoading?<Notice tone="info" title="Checking pipeline status">Reading the locally installed pipeline configuration.</Notice>:pipelineStatus?.available?<Notice tone="success" title={`Configured: ${pipelineStatus.pipeline_name}`}>{pipelineStatus.steps?.length?pipelineStatus.steps.join(' → '):'The artifact did not declare a displayable step list.'} · Artifact: {pipelineStatus.artifact_file}</Notice>:<Notice tone="warning" title="Exact preprocessing pipeline not installed">{pipelineStatus?.reason||'AquaGuard will not guess or substitute preprocessing steps.'} Install the trusted artifact at backend/models/user_preprocessing_pipeline.joblib. It must implement preprocess_frame(frame) and return a DataFrame with original row indexes preserved.</Notice>}
          <div className="pipeline-note"><Icon name="shield" size={14}/><span>No imputation or row deletion is added by AquaGuard. Any such behavior must come from the installed user's pipeline.</span></div>
          <div className="controls-footer"><Button variant="ghost" icon="refresh" disabled={pipelineLoading} onClick={refreshPipelineStatus}>Refresh pipeline status</Button><LoadingButton loading={busy} loadingLabel="Running user's pipeline…" disabled={!pipelineStatus?.available} icon="flask" onClick={executePreprocess}>Create processing copy</LoadingButton></div>
        </Card>
      </div>
      {prepResult && <Card><div className="card-head"><div><div className="eyebrow">PROCESSED DATASET</div><h3>Separate output from {prepResult.report.pipeline_name}</h3><p>The original file remains untouched. No additional AquaGuard preprocessing was applied.</p></div><Button variant="secondary" icon="download" onClick={()=>downloadProcessedFile(prepResult.process_id)}>Download processing copy</Button></div><div className="quality-overview"><div><strong>{prepResult.report.original_rows}</strong><span>original rows</span></div><div><strong>{prepResult.report.processed_rows}</strong><span>processed rows</span></div><div><strong>{prepResult.report.rows_removed_by_pipeline}</strong><span>rows removed by pipeline</span></div><div><strong>{prepResult.report.complete_processed_rows}</strong><span>complete processed rows</span></div></div><div className="inline-message"><Icon name="check" size={15}/>{prepResult.report.note}</div><DataTable columns={(prepResult.processed.columns||[]).slice(0,10).map((name)=>({key:name,label:name}))} rows={prepResult.processed.preview||[]}/></Card>}
    </div>}

    {tab==='regression' && <div className="page-stack compact-stack"><Notice tone="info" title="Regression is for continuous numerical targets">A binary or categorical label such as Potability belongs in classification. AquaGuard runs only after your configured preprocessing; any incomplete selected row blocks the experiment instead of being dropped or imputed.</Notice><Card className="analysis-controls"><div className="analysis-controls-head"><div><div className="eyebrow">REGRESSION LAB</div><h3>Configure a continuous-target experiment</h3><p>Choose the target and input features for one compatible run.</p></div><Tag tone="neutral">K-FOLD = EVALUATION</Tag></div><div className="control-grid"><SelectField label="Continuous target" value={regTarget} onChange={updateRegTarget} options={numeric.map((x)=>({value:x,label:x}))} hint="Requires at least 3 distinct numerical values."/><SelectField label="Regression algorithm" value={regAlgorithm} onChange={changeRegAlgorithm} options={REGRESSORS}/>{regAlgorithm==='polynomial'&&<NumberField label="Polynomial degree" value={degree} min={2} max={5} onChange={setDegree} hint="Degree 2–5; higher degree can overfit."/>}{regAlgorithm==='lwlr'&&<NumberField label="LWLR bandwidth" value={bandwidth} min={0.05} max={20} step={0.05} onChange={setBandwidth} hint="Gaussian-kernel bandwidth."/>}<NumberField label="K-Fold folds" value={folds} min={2} max={10} onChange={setFolds} hint="Evaluation technique, not a model."/></div><FeaturePicker label="Input features" options={regFeatureOptions} value={regFeatures} onChange={setRegFeatures}/>{regAlgorithm==='lwlr'&&<div className="lwlr-query"><div><b>Optional query point</b><small>Pre-filled from the demo preview when available. Edit each input value; if left incomplete, the first holdout row is used as the query.</small></div><div className="query-fields">{regFeatures.map((name)=><label className="field" key={name}><span className="field-label">{name}</span><input type="number" step="any" value={queryPoint[name]??''} onChange={(event)=>setQueryPoint({...queryPoint,[name]:event.target.value})}/></label>)}</div></div>}<div className="controls-footer"><span>Selected {regFeatures.length} feature{regFeatures.length===1?'':'s'} · ≥12 complete processed rows required</span><LoadingButton loading={busy} loadingLabel="Running experiment…" disabled={!preprocessingReady} icon="play" onClick={executeRegression}>Run regression</LoadingButton></div></Card>{result?.metrics && result.algorithm?.includes('Regression') && <div className="page-stack compact-stack"><div className="results-title-row"><div><div className="eyebrow">REGRESSION RESULTS</div><h3>{result.algorithm}</h3><p>{result.complete_rows} complete processed rows · {result.preprocessing?.rows_removed_by_pipeline??0} rows removed by the user's pipeline</p></div><Button variant="secondary" icon="download" onClick={()=>downloadCsv('aquaguard_regression_results.csv',(result.actual||[]).map((actual,i)=>({row:i+1,actual,predicted:result.predicted?.[i],mae:result.metrics.mae,mse:result.metrics.mse,rmse:result.metrics.rmse,r2:result.metrics.r2})))}>Export results</Button></div><div className="metrics-grid result-metrics"><MetricValue label="MAE" value={result.metrics.mae}/><MetricValue label="MSE" value={result.metrics.mse}/><MetricValue label="RMSE" value={result.metrics.rmse}/><MetricValue label="R²" value={result.metrics.r2}/></div>{result.query_point && <Notice tone="info" title={`LWLR query point · bandwidth ${result.bandwidth}`}>{JSON.stringify(result.query_point)} → prediction {Number(result.query_prediction).toFixed(4)}. A query prediction is not a safety classification.</Notice>}<div className="analysis-grid"><Card><div className="card-head"><div><div className="eyebrow">FIT QUALITY</div><h3>Actual vs predicted</h3><p>Holdout values from the deterministic train/test split.</p></div></div><ActualPredictedChart actual={result.actual} predicted={result.predicted}/></Card><Card><div className="card-head"><div><div className="eyebrow">MODEL COEFFICIENTS</div><h3>Fitted equation</h3><p>{result.degree?`Polynomial degree ${result.degree}`:'Linear model coefficients'}</p></div></div><div className="coefficient-list">{Object.entries(result.coefficients||{}).map(([name,value])=><div key={name}><span>{name}</span><strong>{Number(value).toFixed(5)}</strong></div>)}<div className="coefficient-intercept"><span>Intercept</span><strong>{Number(result.intercept).toFixed(5)}</strong></div></div></Card></div><DataTable columns={[{key:'row',label:'Row'},{key:'actual',label:'Actual'},{key:'predicted',label:'Predicted'}]} rows={(result.actual||[]).slice(0,10).map((actual,i)=>({row:i+1,actual,predicted:result.predicted?.[i]}))}/><CrossValidation data={result.cross_validation}/></div>}</div>}

    {tab==='classification' && <div className="page-stack compact-stack"><Notice tone="info" title="Classification predicts discrete labels">Select a target with categories such as Safe/Unsafe or the labels in your dataset. C4.5 uses entropy and gain ratio; Decision Tree / CART uses scikit-learn's CART implementation.</Notice><Card className="analysis-controls"><div className="analysis-controls-head"><div><div className="eyebrow">CLASSIFICATION LAB</div><h3>Configure a classifier</h3><p>Metrics are computed on a stratified holdout; at least 12 complete processed rows are required. K-Fold scores are reported separately.</p></div><Tag tone="neutral">CLASS LABELS</Tag></div><div className="control-grid"><SelectField label="Target label" value={classTarget} onChange={updateClassTarget} options={targetOptions}/><SelectField label="Classification algorithm" value={classAlgorithm} onChange={setClassAlgorithm} options={CLASSIFIERS}/><NumberField label="K for KNN" value={knn} min={1} max={200} onChange={setKnn} hint="Used by KNN only."/><NumberField label="Estimators" value={estimators} min={1} max={1000} onChange={setEstimators} hint="Used by ensemble methods."/><NumberField label="K-Fold folds" value={folds} min={2} max={10} onChange={setFolds}/><NumberField label="AdaBoost learning rate" value={learningRate} min={0.01} max={10} step={0.05} onChange={setLearningRate}/></div><FeaturePicker label="Input features" options={classFeatureOptions} value={classFeatures} onChange={setClassFeatures}/>{classAlgorithm==='c45'&&<div className="inline-message"><Icon name="info" size={15}/>This C4.5-style implementation uses numeric thresholds and entropy-based information gain ratio; it is not renamed CART.</div>}<div className="controls-footer"><span>Metrics are actual calculations on this dataset.</span><LoadingButton loading={busy} loadingLabel="Training & evaluating…" disabled={!preprocessingReady} icon="play" onClick={executeClassifier}>Run selected classifier</LoadingButton></div></Card>{result?.metrics && result.metrics.accuracy!==undefined && <ClassificationResult result={result} onExport={()=>downloadCsv('aquaguard_classification_predictions.csv',(result.actual||[]).map((actual,i)=>({row:i+1,actual,predicted:result.predicted?.[i],mae:result.metrics.mae,mse:result.metrics.mse,rmse:result.metrics.rmse,r2:result.metrics.r2})))}/>}</div>}

    {tab==='comparison' && <div className="page-stack compact-stack"><Notice tone="info" title="No automatic production selection">Review actual holdout and K-Fold results, then explicitly designate an eligible model. Metrics are not hardcoded. Demo sample experiments are never production-eligible.</Notice><Card className="analysis-controls"><div className="analysis-controls-head"><div><div className="eyebrow">MODEL COMPARISON</div><h3>Benchmark all classifiers</h3><p>All seven classification methods use the same user-processed target, feature set and evaluation strategy; at least 12 complete processed rows are required.</p></div><Tag tone="neutral">7 METHODS</Tag></div><div className="control-grid"><SelectField label="Target label" value={classTarget} onChange={updateClassTarget} options={targetOptions}/><NumberField label="K-Fold folds" value={folds} min={2} max={10} onChange={setFolds}/><NumberField label="K for KNN" value={knn} min={1} max={200} onChange={setKnn}/><NumberField label="Ensemble estimators" value={estimators} min={1} max={1000} onChange={setEstimators}/></div><FeaturePicker label="Shared input features" options={classFeatureOptions} value={classFeatures} onChange={setClassFeatures}/><div className="controls-footer"><span>Decision Tree / CART · C4.5 · Naive Bayes · KNN · Bagging · AdaBoost · Random Forest</span><LoadingButton loading={busy} loadingLabel="Evaluating models…" disabled={!preprocessingReady} icon="activity" onClick={executeComparison}>Compare classifiers</LoadingButton></div></Card><ComparisonResults comparison={comparison} onPromote={handlePromote} promoting={promoting} isDemo={Boolean(activeDataset.is_demo)}/></div>}

    {tab==='pca' && <div className="page-stack compact-stack"><Notice tone="info" title="PCA is exploratory—not your final predictor">PCA is an explicit exploratory analysis over the user-preprocessed numerical values. AquaGuard does not add scaling; PCA is not inserted into production unless your trained model was built with PCA.</Notice><Card className="analysis-controls"><div className="analysis-controls-head"><div><div className="eyebrow">PRINCIPAL COMPONENT ANALYSIS</div><h3>Select numeric inputs</h3><p>Incomplete selected inputs must already be handled by the user's pipeline; AquaGuard will not delete or impute rows.</p></div><Tag tone="neutral">UNSUPERVISED TRANSFORM</Tag></div><div className="control-grid"><NumberField label="Principal components" value={components} min={1} max={Math.max(1,Math.min(numeric.length,activeDataset.row_count||100))} onChange={setComponents} hint={`Maximum for this dataset: ${Math.min(numeric.length,activeDataset.row_count||100)}.`}/></div><FeaturePicker options={analysisFeatureOptions} value={analysisFeatures} onChange={setAnalysisFeatures}/><div className="controls-footer"><span>{analysisFeatures.length} numerical features selected</span><LoadingButton loading={busy} loadingLabel="Calculating components…" disabled={!preprocessingReady} icon="play" onClick={executePca}>Run PCA</LoadingButton></div></Card>{result?.explained_variance_ratio && <div className="page-stack compact-stack"><div className="results-title-row"><div><div className="eyebrow">PCA RESULTS</div><h3>{result.original_feature_count} original features → {result.component_count} components</h3><p>{result.complete_rows} complete processed samples · {result.preprocessing?.rows_removed_by_pipeline??0} rows removed by the user's pipeline</p></div><Button variant="secondary" icon="download" onClick={()=>downloadCsv('aquaguard_pca_scores.csv',result.scores||[])}>Export PCA scores</Button></div><div className="analysis-grid"><Card><div className="card-head"><div><div className="eyebrow">VARIANCE</div><h3>Explained variance</h3><p>Individual and cumulative contribution.</p></div></div><VarianceBars values={result.explained_variance_ratio} cumulative={result.cumulative_explained_variance}/></Card><Card><div className="card-head"><div><div className="eyebrow">COMPONENT SPACE</div><h3>PC1 vs PC2</h3><p>Projection of the selected samples.</p></div></div><ScatterChart points={(result.scores||[]).map((p)=>({x:p.pc1,y:p.pc2}))} colorByCluster={false} xLabel="PC1" yLabel="PC2"/></Card></div><Card><div className="card-head"><div><div className="eyebrow">LOADINGS</div><h3>Component contributions</h3><p>Signed feature weights in each principal component.</p></div></div><DataTable columns={[{key:'feature',label:'Feature'},...(result.explained_variance_ratio||[]).map((_,i)=>({key:`pc${i+1}`,label:`PC${i+1}`,render:(_,row)=>Number(row.components?.[i]).toFixed(4)}))]} rows={result.loadings}/></Card><Notice tone="info" title="Production boundary">{result.note}</Notice></div>}</div>}

    {tab==='clustering' && <div className="page-stack compact-stack"><Notice tone="info" title="Clusters are exploratory groups">Unsupervised cluster IDs do not mean Safe or Unsafe. Compare algorithms with the same selected numerical features and cluster count.</Notice><Card className="analysis-controls"><div className="analysis-controls-head"><div><div className="eyebrow">CLUSTERING LAB</div><h3>Explore sample groups</h3><p>K-Means and Ward agglomerative clustering use the user's processed numeric inputs as-is; AquaGuard adds no scaling or row deletion.</p></div><Tag tone="neutral">UNSUPERVISED</Tag></div><div className="control-grid"><SelectField label="Clustering method" value={clusterAlgorithm} onChange={setClusterAlgorithm} options={CLUSTERERS}/><NumberField label="Number of clusters" value={clusters} min={2} max={20} onChange={setClusters} hint="Must be less than the complete sample count."/></div><FeaturePicker options={analysisFeatureOptions} value={analysisFeatures} onChange={setAnalysisFeatures}/><div className="controls-footer"><span>Internal metrics: silhouette, Davies–Bouldin, Calinski–Harabasz</span><LoadingButton loading={busy} loadingLabel="Finding groups…" disabled={!preprocessingReady} icon="play" onClick={executeClustering}>Run clustering</LoadingButton></div></Card>{result?.automatic_comment && <Notice tone="success" title="Metric-based comparison">{result.automatic_comment}</Notice>}{(result?.points || result?.kmeans) && <ClusteringResult result={result} onExport={()=>{const rows=result.points?(result.points||[]).map((point)=>({sample:point.index+1,cluster:point.cluster,x:point.x,y:point.y})):Array.from({length:Math.min(result.kmeans?.labels?.length||0,result.hierarchical?.labels?.length||0)},(_,index)=>({sample:index+1,kmeans_cluster:result.kmeans.labels[index],hierarchical_cluster:result.hierarchical.labels[index]}));downloadCsv('aquaguard_cluster_assignments.csv',rows);}}/>}</div>}
  </div>;
}

function ClusteringResult({result,onExport}) {
  const compare=result.kmeans&&result.hierarchical;
  const variants=compare?[result.kmeans,result.hierarchical]:[result];
  return <div className="page-stack compact-stack"><div className="results-title-row"><div><div className="eyebrow">CLUSTERING RESULTS</div><h3>{compare?'K-Means vs Hierarchical':result.algorithm}</h3><p>{result.complete_rows||result.kmeans?.complete_rows} complete samples · {result.cluster_count} clusters requested</p></div><Button variant="secondary" icon="download" onClick={onExport}>Export cluster assignments</Button></div><div className="metrics-grid result-metrics">{variants.flatMap((item)=>[{label:`${item.algorithm} · Silhouette`,value:item.metrics?.silhouette},{label:`${item.algorithm} · Davies–Bouldin`,value:item.metrics?.davies_bouldin},{label:`${item.algorithm} · Calinski–Harabasz`,value:item.metrics?.calinski_harabasz}]).map((metric,index)=><MetricValue key={index} label={metric.label} value={metric.value}/>)}</div>{variants.map((item,index)=><div className="analysis-grid" key={item.algorithm}><Card><div className="card-head"><div><div className="eyebrow">SAMPLE GROUPS</div><h3>{item.algorithm} · sizes</h3><p>Cluster membership counts.</p></div></div><div className="cluster-size-list">{Object.entries(item.cluster_sizes||{}).map(([cluster,size])=><div key={cluster}><span className={`cluster-dot cluster-${Number(cluster)%6}`}/><span>Cluster {cluster}</span><b>{size.toLocaleString()} samples</b></div>)}</div>{item.cluster_centers && <div className="cluster-center-note"><Icon name="info" size={14}/>Centers use the values returned by the user's pipeline; AquaGuard applied no scaler.</div>}</Card><Card><div className="card-head"><div><div className="eyebrow">FEATURE SPACE</div><h3>Cluster visualization</h3><p>Coordinates show the first two user-processed features; AquaGuard added no scaling.</p></div></div><ScatterChart points={item.points} xLabel={item.features?.[0]||'Feature 1'} yLabel={item.features?.[1]||'Feature 2'}/></Card></div>)}{result.dendrogram && <Card><div className="card-head"><div><div className="eyebrow">HIERARCHICAL TREE</div><h3>Dendrogram</h3><p>Ward linkage tree; for large datasets a bounded prefix sample is plotted.</p></div></div><Dendrogram data={result.dendrogram}/></Card>}{compare&&result.hierarchical?.dendrogram&&<Card><div className="card-head"><div><div className="eyebrow">HIERARCHICAL TREE</div><h3>Ward dendrogram</h3><p>Truncated leaf display is calculated from the current dataset.</p></div></div><Dendrogram data={result.hierarchical.dendrogram}/></Card>}<Notice tone="info" title="Interpretation guardrail">{result.interpretation_note||'Cluster membership is an exploratory pattern, not a water safety determination.'}</Notice></div>;
}
