import React, { useEffect, useMemo, useState } from 'react';
import Icon from '../components/Icons';
import { SparkLine } from '../components/Charts';
import { Button, Card, DataTable, LoadingButton, Notice, NumberField, Tag } from '../components/UI';
import { createDevice, getDevices, getReadings, sendReading } from '../services/iotService';

function normalizeFeatureRow(row, columns) {
  const blocked = new Set(['potability','safe','unsafe','target','label','class','quality']);
  const values={};
  for(const name of columns){if(blocked.has(String(name).toLowerCase()))continue;const value=row?.[name];if(value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value)))values[name]=Number(value);}
  return values;
}

export default function IoTMonitoring({ activeDataset, health, onRefreshDashboard }) {
  const [devices,setDevices]=useState([]);
  const [selected,setSelected]=useState('');
  const [readings,setReadings]=useState([]);
  const [deviceId,setDeviceId]=useState('AquaSensor-01');
  const [deviceName,setDeviceName]=useState('North intake sensor');
  const [location,setLocation]=useState('');
  const [busy,setBusy]=useState(false);
  const [simBusy,setSimBusy]=useState(false);
  const [error,setError]=useState('');
  const [message,setMessage]=useState('');
  const [loadError,setLoadError]=useState('');

  async function refreshDevices(){
    setLoadError('');
    try{const response=await getDevices();setDevices(response.items||[]);if(!selected&&response.items?.length)setSelected(response.items[0].device_id);}
    catch(e){setLoadError(e.message||'Could not load devices.');}
  }
  useEffect(()=>{refreshDevices();},[]);
  useEffect(()=>{if(!selected){setReadings([]);return;}getReadings(selected).then((r)=>setReadings(r.items||[])).catch((e)=>setLoadError(e.message));},[selected]);

  async function register(event){
    event.preventDefault();setError('');setMessage('');setBusy(true);
    try{const item=await createDevice({device_id:deviceId,device_name:deviceName,location:location||null});setMessage(`${item.device_name} registered in PostgreSQL.`);setSelected(item.device_id);await refreshDevices();onRefreshDashboard?.();}
    catch(e){setError(e.message||'Device registration failed.');}
    finally{setBusy(false);}
  }
  async function runSimulation(){
    if(!activeDataset||!devices.length)return;
    setError('');setMessage('');setSimBusy(true);
    try{
      const rows=(activeDataset.preview||[]).slice(0,3);
      const device=devices.find((d)=>d.device_id===selected)||devices[0];
      if(!device)throw new Error('Register a device before sending simulated readings.');
      let sent=0;
      for(const row of rows){
        const features=normalizeFeatureRow(row,activeDataset.numeric_columns||Object.keys(row));
        if(!Object.keys(features).length)continue;
        await sendReading(device.device_id,{timestamp:new Date(Date.now()+sent).toISOString(),source:'simulation',features});
        sent++;
      }
      setMessage(`${sent} deterministic sample row${sent===1?'':'s'} sent as Simulated IoT Data. No random sensor values were generated.`);
      const response=await getReadings(device.device_id);setReadings(response.items||[]);await refreshDevices();onRefreshDashboard?.();
    }catch(e){setError(e.message||'Simulation request failed.');}
    finally{setSimBusy(false);}
  }

  const registryReady=Boolean(health?.database?.available)&&!loadError;
  const online=devices.filter((d)=>d.status==='online').length;
  const selectedDevice=devices.find((d)=>d.device_id===selected);
  const recent=[...readings].reverse();
  const phValues=recent.map((r)=>r.features?.ph??r.features?.pH).filter((v)=>Number.isFinite(Number(v))).map(Number);
  return <div className="page-stack">
    {!health?.database?.available && <Notice tone="warning" title="PostgreSQL is required for persistent IoT records">Configure <code>DATABASE_URL</code> and start PostgreSQL before registering devices or saving sensor readings. Telemetry is not silently stored in a substitute database.</Notice>}
    {error&&<Notice tone="error" title="IoT action failed">{error}</Notice>}{message&&<Notice tone="success" title="IoT status">{message}</Notice>}{loadError&&<Notice tone="warning" title="Could not load device data">{loadError}</Notice>}
    <div className="iot-overview-grid"><Card className="iot-stat-card"><div className="iot-stat-icon"><Icon name="radio" size={18}/></div><div><span>Registered devices</span><strong>{registryReady?devices.length:'—'}</strong><small>PostgreSQL registry</small></div></Card><Card className="iot-stat-card"><div className="iot-stat-icon online"><Icon name="activity" size={18}/></div><div><span>Recently active</span><strong>{registryReady?online:'—'}</strong><small>Seen in last 10 minutes</small></div></Card><Card className="iot-stat-card"><div className="iot-stat-icon purple"><Icon name="database" size={18}/></div><div><span>Recent readings</span><strong>{registryReady&&selectedDevice?readings.length:'—'}</strong><small>{selectedDevice?.device_name||'Select a device'}</small></div></Card><Card className="iot-stat-card"><div className="iot-stat-icon amber"><Icon name="shield" size={18}/></div><div><span>Prediction pipeline</span><strong>{health?.production_model?.available?'Ready':'Unavailable'}</strong><small>{health?.production_model?.available?health.production_model.model_name:'Artifact not installed'}</small></div></Card></div>
    <div className="iot-main-grid"><Card><div className="card-head"><div><div className="eyebrow">DEVICE REGISTRY</div><h3>Register a water sensor</h3><p>Each device code must be unique. IoT history is stored in PostgreSQL.</p></div><div className="iot-form-icon"><Icon name="radio" size={18}/></div></div><form className="device-form" onSubmit={register}><label className="field"><span className="field-label">Device ID / code</span><input required maxLength={128} value={deviceId} onChange={(e)=>setDeviceId(e.target.value)} placeholder="AquaSensor-01"/></label><label className="field"><span className="field-label">Display name</span><input required maxLength={200} value={deviceName} onChange={(e)=>setDeviceName(e.target.value)} placeholder="North intake sensor"/></label><label className="field"><span className="field-label">Location <span className="muted">(optional)</span></span><input maxLength={300} value={location} onChange={(e)=>setLocation(e.target.value)} placeholder="Site / coordinates"/></label><LoadingButton loading={busy} loadingLabel="Registering…" icon="plus" disabled={!health?.database?.available}>Register device</LoadingButton></form><div className="api-example"><div className="api-example-head"><span>REAL SENSOR API · JSON</span><Tag tone="neutral">POST /devices/{'{device_id}'}/readings</Tag></div><pre>{`{
  "device_id": "AquaSensor-01",
  "timestamp": "2026-10-05T15:30:00+05:30",
  "ph": 7.2,
  "turbidity": 3.8,
  "conductivity": 420
}`}</pre><small>Flat numeric fields or a <code>features</code> object are accepted. Production inference requires a matching installed model pipeline.</small></div></Card>
      <Card className="simulation-card"><div className="card-head"><div><div className="eyebrow">DEMO MODE</div><h3>IoT simulation</h3><p>Submit existing sample rows to exercise the IoT endpoint.</p></div><span className="simulated-stamp"><span/> SIMULATED</span></div><div className="simulation-illustration"><div className="sim-ring"><Icon name="radio" size={28}/></div><div className="sim-pulse pulse-one"/><div className="sim-pulse pulse-two"/><span className="sim-node node-a"/><span className="sim-node node-b"/></div><div className="simulation-source">{activeDataset ? <><Icon name="database" size={14}/><span>Source: {activeDataset.file_name}</span></> : <><Icon name="info" size={14}/><span>Load a dataset to send deterministic sample rows</span></>}</div><LoadingButton variant="secondary" loading={simBusy} loadingLabel="Sending simulated readings…" icon="play" disabled={!activeDataset||!devices.length||!health?.database?.available} onClick={runSimulation}>Send sample reading(s)</LoadingButton><div className="simulation-disclaimer">These are existing demo CSV rows—not live measurements and not randomly generated sensor readings. Every record is labeled <b>simulation</b>.</div></Card></div>
    <div className="iot-bottom-grid"><Card><div className="card-head"><div><div className="eyebrow">REGISTERED SENSORS</div><h3>Device status</h3><p>Online is determined by a last-seen timestamp within 10 minutes.</p></div><Button variant="ghost" icon="refresh" onClick={refreshDevices}>Refresh</Button></div>{devices.length? <div className="device-list">{devices.map((device)=><button key={device.id} className={`device-row ${selected===device.device_id?'device-row-active':''}`} onClick={()=>setSelected(device.device_id)}><span className={`device-row-icon ${device.status==='online'?'device-online':''}`}><Icon name="radio" size={17}/></span><span className="device-row-info"><strong>{device.device_name}</strong><small>{device.location||device.device_id}</small></span><Tag tone={device.status==='online'?'success':'neutral'} dot>{device.status}</Tag><Icon name="chevron" size={15}/></button>)}</div>:<div className="compact-empty"><Icon name="radio" size={20}/><span>{health?.database?.available?'No devices registered yet.':'Connect PostgreSQL to load the device registry.'}</span></div>}</Card><Card><div className="card-head"><div><div className="eyebrow">DEVICE TELEMETRY</div><h3>{selectedDevice?.device_name||'Readings'}</h3><p>{selectedDevice?.location||selectedDevice?.device_id||'Choose a registered device to view its history.'}</p></div>{selectedDevice&&<Tag tone="neutral">{readings.length} records</Tag>}</div>{readings.length>0 ? <><div className="iot-trend-summary"><strong>{phValues.length?phValues.at(-1).toFixed(2):'—'}</strong><span>latest pH</span><SparkLine values={phValues} height={70}/></div><DataTable columns={[{key:'timestamp',label:'Timestamp',render:(v)=>new Date(v).toLocaleString()},{key:'source',label:'Source',render:(v)=>v==='simulation'?<Tag tone="warning">Simulated</Tag>:<Tag tone="success">Sensor</Tag>},{key:'features',label:'Reading'}]} rows={readings.slice(0,8)}/></>:<div className="compact-empty"><Icon name="activity" size={20}/><span>No readings stored for this device yet.</span></div>}</Card></div>
  </div>;
}
