import React from 'react';
import Icon from './Icons';

export function Button({ children, onClick, variant = 'primary', icon, disabled = false, size = '', type = 'button', className = '' }) {
  return <button type={type} className={`button button-${variant} ${size ? `button-${size}` : ''} ${className}`} onClick={onClick} disabled={disabled}>{icon && <Icon name={icon} size={16}/>}<span>{children}</span></button>;
}

export function Tag({ children, tone = 'neutral', dot = false, className = '' }) {
  return <span className={`tag tag-${tone} ${className}`}>{dot && <i className="tag-dot"/>}{children}</span>;
}

export function Card({ children, className = '', flush = false }) {
  return <section className={`card ${flush ? 'card-flush' : ''} ${className}`}>{children}</section>;
}

export function SectionHeading({ eyebrow, title, description, action, className = '' }) {
  return <div className={`section-heading ${className}`}><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h2>{title}</h2>{description && <p>{description}</p>}</div>{action && <div className="section-action">{action}</div>}</div>;
}

export function MetricCard({ label, value, detail, icon, tone = 'mint', footnote }) {
  return <Card className="metric-card"><div className={`metric-icon metric-icon-${tone}`}><Icon name={icon} size={18}/></div><div className="metric-body"><div className="metric-label">{label}</div><div className="metric-value">{value ?? '—'}</div>{detail && <div className="metric-detail">{detail}</div>}{footnote && <div className="metric-footnote">{footnote}</div>}</div></Card>;
}

export function Notice({ tone = 'info', title, children, icon }) {
  const iconName = icon || (tone === 'success' ? 'check' : tone === 'warning' ? 'info' : 'info');
  return <div className={`notice notice-${tone}`}><div className="notice-icon"><Icon name={iconName} size={17}/></div><div><strong>{title}</strong>{children && <div className="notice-copy">{children}</div>}</div></div>;
}

export function EmptyState({ icon = 'database', title, children, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon name={icon} size={22}/></div><h3>{title}</h3>{children && <p>{children}</p>}{action && <div className="empty-action">{action}</div>}</div>;
}

export function SelectField({ label, value, onChange, options, disabled = false, hint, className = '' }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span><span className="select-wrap"><select value={value ?? ''} disabled={disabled} onChange={(event) => onChange(event.target.value)}>{options.map((option) => typeof option === 'string' ? <option key={option} value={option}>{option}</option> : <option key={option.value} value={option.value} disabled={option.disabled}>{option.label}</option>)}</select><Icon name="down" size={14}/></span>{hint && <small>{hint}</small>}</label>;
}

export function NumberField({ label, value, onChange, min, max, step = 1, hint, suffix }) {
  return <label className="field"><span className="field-label">{label}</span><span className="number-wrap"><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))}/>{suffix && <span className="number-suffix">{suffix}</span>}</span>{hint && <small>{hint}</small>}</label>;
}

export function FeaturePicker({ options, value = [], onChange, label = 'Input features', disabled = false }) {
  const toggle = (name) => onChange(value.includes(name) ? value.filter((item) => item !== name) : [...value, name]);
  return <div className="field feature-field"><span className="field-label">{label} <span className="muted">({value.length} selected)</span></span>{options.length === 0 ? <div className="feature-empty">Upload a dataset to choose features.</div> : <div className="feature-chips">{options.map((name) => <button key={name} type="button" disabled={disabled} onClick={() => toggle(name)} className={`feature-chip ${value.includes(name) ? 'selected' : ''}`} aria-pressed={value.includes(name)}><span className="feature-check">{value.includes(name) ? <Icon name="check" size={12}/> : '+'}</span>{name}</button>)}</div>}</div>;
}

export function DataTable({ columns, rows, empty = 'No rows to display.' }) {
  if (!rows?.length) return <div className="table-empty">{empty}</div>;
  return <div className="table-scroll"><table><thead><tr>{columns.map((col) => <th key={col.key}>{col.label}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={row.id || row.row || index}>{columns.map((col) => <td key={col.key}>{col.render ? col.render(row[col.key], row) : formatCell(row[col.key])}</td>)}</tr>)}</tbody></table></div>;
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') return <span className="muted">—</span>;
  if (typeof value === 'object') return <code className="inline-code">{JSON.stringify(value)}</code>;
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(3);
  return String(value);
}

export function PageHeader({ breadcrumb, title, description, actions }) {
  return <div className="page-header"><div className="breadcrumbs">{breadcrumb?.map((item, i) => <React.Fragment key={item}>{i > 0 && <span className="crumb-separator">/</span>}<span className={i === breadcrumb.length - 1 ? 'crumb-active' : ''}>{item}</span></React.Fragment>)}</div><div className="page-heading-row"><div><h1>{title}</h1>{description && <p>{description}</p>}</div>{actions && <div className="page-actions">{actions}</div>}</div></div>;
}

export function TabBar({ items, active, onChange }) {
  return <div className="tab-bar" role="tablist">{items.map((item) => <button key={item.id} role="tab" aria-selected={active === item.id} className={`tab-button ${active === item.id ? 'active' : ''}`} onClick={() => onChange(item.id)}>{item.icon && <Icon name={item.icon} size={15}/>}<span>{item.label}</span>{item.count != null && <span className="tab-count">{item.count}</span>}</button>)}</div>;
}

export function LoadingButton({ loading, children, loadingLabel = 'Working…', ...props }) {
  return <Button {...props} disabled={props.disabled || loading}>{loading && <span className="spinner"/>}{loading ? loadingLabel : children}</Button>;
}
