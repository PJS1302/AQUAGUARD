import React from 'react';

const PALETTE = ['#0e9384', '#3264d4', '#ed8b37', '#8754c6', '#d45462', '#58a76e', '#4ca5cb', '#c7a235'];

export function SparkLine({ values = [], color = '#0e9384', height = 70, fill = true }) {
  if (!values.length) return <div className="chart-empty">No readings recorded yet.</div>;
  const clean = values.map((v) => Number(v)).filter(Number.isFinite);
  if (!clean.length) return <div className="chart-empty">No numeric readings available.</div>;
  const min = Math.min(...clean), max = Math.max(...clean), range = max - min || 1;
  const points = clean.map((value, index) => `${(index / Math.max(1, clean.length - 1)) * 400},${height - 8 - ((value - min) / range) * (height - 20)}`).join(' ');
  return <svg className="spark-svg" viewBox={`0 0 400 ${height}`} preserveAspectRatio="none" role="img" aria-label="Reading trend">
    {[0.25, 0.5, 0.75].map((step) => <line key={step} x1="0" x2="400" y1={height * step} y2={height * step} stroke="#edf1f2" strokeWidth="1"/>)}
    {fill && <polygon points={`0,${height} ${points} 400,${height}`} fill={`${color}12`}/>}
    <polyline points={points} fill="none" stroke={color} strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round"/>
    {clean.map((v, i) => { const x=(i / Math.max(1, clean.length - 1))*400; const y=height-8-((v-min)/range)*(height-20); return <circle key={i} cx={x} cy={y} r="3.2" fill="#fff" stroke={color} strokeWidth="2"/>; })}
  </svg>;
}

export function ScatterChart({ points = [], height = 250, colorByCluster = true, xLabel = 'Component 1', yLabel = 'Component 2' }) {
  if (!points.length) return <div className="chart-empty">Run an analysis to visualize actual sample points.</div>;
  const xs = points.map((p) => Number(p.x)).filter(Number.isFinite), ys = points.map((p) => Number(p.y)).filter(Number.isFinite);
  if (!xs.length || !ys.length) return <div className="chart-empty">No numeric coordinates returned.</div>;
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = { l: 40, r: 16, t: 16, b: 32 }, width = 620, plotW = width-pad.l-pad.r, plotH = height-pad.t-pad.b;
  const xx = (x) => pad.l + ((Number(x)-xMin)/(xMax-xMin || 1))*plotW;
  const yy = (y) => pad.t + plotH - ((Number(y)-yMin)/(yMax-yMin || 1))*plotH;
  const shown = points.slice(0, 1000);
  return <div className="chart-frame"><svg viewBox={`0 0 ${width} ${height}`} className="scatter-svg" role="img" aria-label={`${xLabel} versus ${yLabel}`}>
    {[0,1,2,3].map((i)=><line key={i} x1={pad.l} x2={width-pad.r} y1={pad.t+i*plotH/3} y2={pad.t+i*plotH/3} stroke="#edf1f2"/>)}
    {[0,1,2,3].map((i)=><line key={`v${i}`} x1={pad.l+i*plotW/3} x2={pad.l+i*plotW/3} y1={pad.t} y2={pad.t+plotH} stroke="#f1f4f5"/>)}
    <line x1={pad.l} x2={pad.l} y1={pad.t} y2={pad.t+plotH} stroke="#cdd8da"/><line x1={pad.l} x2={width-pad.r} y1={pad.t+plotH} y2={pad.t+plotH} stroke="#cdd8da"/>
    {shown.map((p,i)=><circle key={i} cx={xx(p.x)} cy={yy(p.y)} r="4" fill={colorByCluster ? PALETTE[Math.abs(Number(p.cluster)||0)%PALETTE.length] : '#0e9384'} fillOpacity=".78" stroke="#fff" strokeWidth=".8"><title>{`${xLabel}: ${Number(p.x).toFixed(3)} · ${yLabel}: ${Number(p.y).toFixed(3)}${p.cluster != null ? ` · cluster ${p.cluster}` : ''}`}</title></circle>)}
    <text x={width/2} y={height-5} textAnchor="middle" className="svg-label">{xLabel}</text><text transform={`translate(12 ${height/2}) rotate(-90)`} textAnchor="middle" className="svg-label">{yLabel}</text>
  </svg></div>;
}

export function ActualPredictedChart({ actual = [], predicted = [], height = 250 }) {
  if (!actual.length || !predicted.length) return <div className="chart-empty">Actual-versus-predicted values will appear after a regression run.</div>;
  const pairs = actual.map((a,i)=>({x:Number(a), y:Number(predicted[i])})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
  if (!pairs.length) return <div className="chart-empty">No comparable predictions returned.</div>;
  const min=Math.min(...pairs.flatMap(p=>[p.x,p.y])), max=Math.max(...pairs.flatMap(p=>[p.x,p.y])), range=max-min||1;
  const pad={l:44,r:18,t:18,b:36}, width=620, plotW=width-pad.l-pad.r, plotH=height-pad.t-pad.b;
  const pos=(v)=>((v-min)/range);
  return <div className="chart-frame"><svg viewBox={`0 0 ${width} ${height}`} className="scatter-svg" role="img" aria-label="Actual versus predicted chart">
    {[0,1,2,3].map(i=><line key={i} x1={pad.l} x2={width-pad.r} y1={pad.t+i*plotH/3} y2={pad.t+i*plotH/3} stroke="#edf1f2"/>)}
    <line x1={pad.l} y1={pad.t+plotH} x2={width-pad.r} y2={pad.t} stroke="#bdc9cb" strokeDasharray="5 5"/>
    {pairs.slice(0,300).map((p,i)=><circle key={i} cx={pad.l+pos(p.x)*plotW} cy={pad.t+plotH-pos(p.y)*plotH} r="4" fill="#0e9384" fillOpacity=".72"/>)}
    <text x={width/2} y={height-5} textAnchor="middle" className="svg-label">Actual</text><text transform={`translate(12 ${height/2}) rotate(-90)`} textAnchor="middle" className="svg-label">Predicted</text>
  </svg></div>;
}

export function VarianceBars({ values = [], cumulative = [] }) {
  if (!values.length) return <div className="chart-empty">Run PCA to calculate explained variance.</div>;
  return <div className="variance-chart">{values.map((value,index)=><div className="variance-row" key={index}><div className="variance-label">PC{index+1}</div><div className="variance-track"><div className="variance-fill" style={{width:`${Math.max(0,Math.min(100,value*100))}%`}}/></div><div className="variance-number">{(value*100).toFixed(1)}%</div><div className="variance-cumulative">{((cumulative[index]||0)*100).toFixed(1)}% cumulative</div></div>)}</div>;
}

export function Dendrogram({ data }) {
  if (!data?.icoord?.length) return <div className="chart-empty">Dendrogram is not available for this run.</div>;
  const width=620, height=210, maxX=Math.max(...data.icoord.flat()), maxY=Math.max(...data.dcoord.flat())||1;
  const lines=data.icoord.map((x,index)=>({x, y:data.dcoord[index]}));
  return <div className="chart-frame"><svg viewBox={`0 0 ${width} ${height}`} className="dendrogram-svg" role="img" aria-label="Hierarchical clustering dendrogram">
    {[0,1,2,3].map(i=><line key={i} x1="28" x2={width-10} y1={12+i*45} y2={12+i*45} stroke="#eff2f3"/>)}
    {lines.map((line,index)=>{const points=line.x.map((x,i)=>`${28+(x/maxX)*(width-40)},${height-14-(line.y[i]/maxY)*(height-35)}`).join(' ');return <polyline key={index} points={points} fill="none" stroke={PALETTE[index%PALETTE.length]} strokeWidth="1.7"/>;})}
    <text x="34" y="15" className="svg-label">merge distance</text><text x={width/2} y={height-1} textAnchor="middle" className="svg-label">sample groups · {data.sample_rows} plotted rows</text>
  </svg></div>;
}

export function ConfusionMatrix({ data }) {
  if (!data?.values?.length) return null;
  const n=data.labels.length;
  return <div className="confusion-wrap"><div className="confusion-axis">Predicted →</div><table className="confusion-table"><thead><tr><th>Actual ↓</th>{data.labels.map((label)=><th key={String(label)}>{String(label)}</th>)}</tr></thead><tbody>{data.values.map((row,i)=><tr key={i}><th>{String(data.labels[i])}</th>{row.map((value,j)=><td key={j} className={i===j?'confusion-hit':''}>{value}</td>)}</tr>)}</tbody></table></div>;
}

export function TinyBars({ items, valueKey = 'importance', labelKey = 'feature' }) {
  if (!items?.length) return <div className="chart-empty">No feature importance values were returned.</div>;
  const max=Math.max(...items.map(x=>Number(x[valueKey])||0))||1;
  return <div className="tiny-bars">{items.slice(0,8).map((item,index)=><div className="tiny-bar-row" key={index}><span>{item[labelKey]}</span><div className="tiny-bar-track"><i style={{width:`${(Number(item[valueKey])/max)*100}%`}}/></div><b>{Number(item[valueKey]).toFixed(3)}</b></div>)}</div>;
}
