import React, { useEffect, useState } from 'react';
import Icon from '../components/Icons';
import { Button, Card, DataTable, EmptyState, Notice, Tag } from '../components/UI';
import { getPredictionHistory } from '../services/datasetService';
import { downloadCsv } from '../utils/export';

export default function History({ refreshToken }) {
  const [items,setItems]=useState([]);
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  async function load(){setBusy(true);setError('');try{const result=await getPredictionHistory(200);setItems(result.items||[]);}catch(e){setError(e.message||'Prediction history could not be loaded.');}finally{setBusy(false);}}
  useEffect(()=>{load();},[refreshToken]);
  return <div className="page-stack">{error&&<Notice tone="warning" title="Prediction history unavailable">{error} Prediction history requires PostgreSQL; no records are kept in a silent in-memory substitute.</Notice>}<Card><div className="card-head"><div><div className="eyebrow">POSTGRESQL AUDIT TRAIL</div><h3>Prediction records</h3><p>Manual, dataset and IoT prediction results saved by AquaGuard.</p></div><div className="history-actions"><Tag tone={error?'warning':'neutral'}>{error?'PostgreSQL unavailable':`${items.length} records loaded`}</Tag><Button variant="ghost" icon="refresh" onClick={load} disabled={busy}>Refresh</Button>{items.length>0&&<Button variant="secondary" icon="download" onClick={()=>downloadCsv('aquaguard_prediction_history.csv',items)}>Export CSV</Button>}</div></div>{items.length?<DataTable columns={[{key:'timestamp',label:'Timestamp',render:(v)=>new Date(v).toLocaleString()},{key:'source',label:'Source',render:(v)=><Tag tone={v==='iot'?'success':v==='dataset'?'blue':'neutral'}>{v}</Tag>},{key:'model_name',label:'Model'},{key:'prediction',label:'Output'},{key:'confidence',label:'Model-reported probability',render:(v)=>v==null?'Not exposed':`${(Number(v)*100).toFixed(1)}%`},{key:'dataset_id',label:'Dataset',render:(v)=>v?String(v).slice(0,8):'—'}]} rows={items}/>:<EmptyState icon="history" title={busy?'Loading prediction history…':'No saved predictions'}>Prediction records appear here after a model artifact is installed, PostgreSQL is configured, and predictions are run.</EmptyState>}</Card><Card className="history-policy"><div className="policy-icon"><Icon name="shield" size={17}/></div><div><h3>Traceable model outputs</h3><p>Stored records include source, model name, timestamp, original input features and model-reported probability when exposed. A missing probability remains blank; confidence is never fabricated.</p></div></Card></div>;
}
