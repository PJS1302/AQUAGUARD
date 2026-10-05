import React, { useState } from 'react';
import Icon from './Icons';
import { Tag } from './UI';

const NAV_GROUPS = [
  { label: 'WORKSPACE', items: [
    { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
    { id: 'prediction', label: 'Demo Prediction', icon: 'droplet' },
    { id: 'datasets', label: 'Datasets & Upload', icon: 'database' },
    { id: 'laboratory', label: 'ML Laboratory', icon: 'flask' },
  ] },
  { label: 'ANALYSIS LABS', items: [
    { id: 'regression', label: 'Regression', icon: 'chart' },
    { id: 'classification', label: 'Classification', icon: 'tree' },
    { id: 'clustering', label: 'Clustering', icon: 'layers' },
    { id: 'pca', label: 'PCA Analysis', icon: 'cpu' },
  ] },
  { label: 'OPERATIONS', items: [
    { id: 'iot', label: 'IoT Monitoring', icon: 'radio' },
    { id: 'history', label: 'Prediction History', icon: 'history' },
    { id: 'about', label: 'About AquaGuard', icon: 'book' },
  ] },
];

export const PAGE_META = {
  dashboard: { title: 'Water intelligence, at a glance', eyebrow: 'OVERVIEW', description: 'A live view of your monitoring and analytics workspace.' },
  prediction: { title: 'Production prediction', eyebrow: 'MODEL INFERENCE', description: 'Run the validated model artifact with its original feature pipeline.' },
  datasets: { title: 'Datasets & uploads', eyebrow: 'DATA WORKSPACE', description: 'Inspect source files, keep originals intact, and create separate analysis copies.' },
  laboratory: { title: 'ML Laboratory', eyebrow: 'EXPERIMENT WORKSPACE', description: 'Run each learning method for the task it was designed to solve.' },
  regression: { title: 'Regression analysis', eyebrow: 'CONTINUOUS TARGETS', description: 'Model a numerical water parameter with interpretable regression experiments.' },
  classification: { title: 'Classification analysis', eyebrow: 'DISCRETE LABELS', description: 'Evaluate classification algorithms against your selected target labels.' },
  clustering: { title: 'Clustering analysis', eyebrow: 'UNSUPERVISED PATTERNS', description: 'Explore natural sample groups without treating clusters as safety labels.' },
  pca: { title: 'PCA analysis', eyebrow: 'DIMENSIONALITY REDUCTION', description: 'Reduce numerical features for exploration and visualization.' },
  iot: { title: 'IoT monitoring', eyebrow: 'DEVICE OPERATIONS', description: 'Register sensors, persist readings, and keep simulated data clearly separated.' },
  history: { title: 'Prediction history', eyebrow: 'AUDIT TRAIL', description: 'Review predictions persisted to PostgreSQL.' },
  about: { title: 'AquaGuard platform', eyebrow: 'ABOUT', description: 'An ML lab and water-quality analytics system—not just a prediction page.' },
};

export default function Layout({ page, onNavigate, children, health, activeDataset, onLoadDemo, loadingDemo }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const meta = PAGE_META[page] || PAGE_META.dashboard;
  const databaseAvailable = health?.database?.available;
  const apiAvailable = Boolean(health?.status === 'ok');
  const selected = activeDataset?.file_name;
  return <div className="app-shell">
    {mobileOpen && <button className="mobile-scrim" onClick={() => setMobileOpen(false)} aria-label="Close menu"/>}
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <div className="brand-row"><div className="brand-symbol"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M20 4C15.5 11 9 17.7 9 24.2a11 11 0 0 0 22 0C31 17.7 24.5 11 20 4Z" fill="none" stroke="currentColor" strokeWidth="2.2"/><path d="M14.8 25.2a5.5 5.5 0 0 0 5.2 5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/><circle cx="29" cy="10" r="2" fill="currentColor"/></svg></div><div><div className="brand-name">Aqua<span>Guard</span></div><div className="brand-subtitle">WATER INTELLIGENCE</div></div><button className="mobile-close" onClick={() => setMobileOpen(false)}><Icon name="close"/></button></div>
      <div className="workspace-pill"><div className="workspace-avatar">AG</div><div className="workspace-meta"><strong>Research workspace</strong><span>Local deployment</span></div><Icon name="down" size={14}/></div>
      <nav className="sidebar-nav" aria-label="Primary navigation">{NAV_GROUPS.map((group)=><div className="nav-group" key={group.label}><div className="nav-label">{group.label}</div>{group.items.map((item)=><button key={item.id} className={`nav-item ${page===item.id?'nav-item-active':''}`} onClick={()=>{onNavigate(item.id);setMobileOpen(false)}}><Icon name={item.icon} size={17}/><span>{item.label}</span>{item.id==='laboratory' && <span className="nav-badge">6</span>}</button>)}</div>)}</nav>
      <div className="sidebar-spacer"/>
      <div className="sidebar-status"><div className="sidebar-status-title">SYSTEM STATUS</div><div className="status-line"><span className={`status-light ${apiAvailable?'green':'gray'}`}/><span>API</span><b>{apiAvailable?'Connected':'Checking'}</b></div><div className="status-line"><span className={`status-light ${databaseAvailable?'green':'amber'}`}/><span>PostgreSQL</span><b>{databaseAvailable?'Ready':'Setup needed'}</b></div></div>
      <div className="sidebar-footer"><div className="user-avatar">AQ</div><div className="user-copy"><strong>AquaGuard</strong><span>Analytics console</span></div><button className="quiet-icon" title="About AquaGuard" onClick={()=>onNavigate('about')}><Icon name="settings" size={17}/></button></div>
    </aside>
    <main className="main-shell">
      <header className="topbar"><button className="mobile-menu" onClick={()=>setMobileOpen(true)} aria-label="Open menu"><Icon name="menu" size={20}/></button><div className="topbar-context"><span className="topbar-label">AQUAGUARD PLATFORM</span><span className="topbar-divider">/</span><span className="topbar-current">{meta.eyebrow}</span></div><div className="topbar-actions"><div className={`api-pill ${apiAvailable?'api-on':'api-off'}`}><span className="status-light"/>{apiAvailable?'API online':'API offline'}</div>{selected && <div className="active-dataset-pill"><Icon name="database" size={14}/><span>{selected}</span>{activeDataset?.is_demo && <span className="demo-dot">DEMO</span>}</div>}<button className="topbar-icon" aria-label="Notifications" title="No unread alerts"><Icon name="bell" size={18}/></button><div className="topbar-avatar">AG</div></div></header>
      <div className="page-content"><div className="page-heading-row global-page-heading"><div><div className="breadcrumbs"><span>Workspace</span><span className="crumb-separator">/</span><span className="crumb-active">{meta.eyebrow}</span></div><h1>{meta.title}</h1><p>{meta.description}</p></div><div className="page-actions"><button className="button button-secondary button-sm" onClick={onLoadDemo} disabled={loadingDemo}>{loadingDemo?<span className="spinner"/>:<Icon name="database" size={15}/>}<span>{loadingDemo?'Loading sample…':'Load demo dataset'}</span></button></div></div>{children}</div>
    </main>
  </div>;
}
