import React, { useCallback, useState } from 'react';
import Layout from './components/Layout';
import { Notice } from './components/UI';
import Dashboard from './pages/Dashboard';
import DatasetWorkspace from './pages/DatasetWorkspace';
import Laboratory from './pages/Laboratory';
import PredictionPage from './pages/PredictionPage';
import IoTMonitoring from './pages/IoTMonitoring';
import History from './pages/History';
import About from './pages/About';
import { usePlatformData } from './hooks/usePlatformData';
import { loadDemoDataset, uploadDataset } from './services/datasetService';

const TAB_FOR_PAGE = { regression:'regression', classification:'classification', clustering:'clustering', pca:'pca' };

export default function App() {
  const [page,setPage]=useState('dashboard');
  const [activeDataset,setActiveDataset]=useState(null);
  const [refreshToken,setRefreshToken]=useState(0);
  const {health,dashboard}=usePlatformData(refreshToken);
  const [loadingDemo,setLoadingDemo]=useState(false);
  const [globalError,setGlobalError]=useState('');
  const [toast,setToast]=useState('');

  const refresh=useCallback(()=>setRefreshToken((value)=>value+1),[]);
  const handleUpload=useCallback(async(file)=>{
    setGlobalError('');
    try{
      const result=await uploadDataset(file);
      setActiveDataset(result);
      setToast(`${result.file_name} uploaded; original file preserved.`);
      window.setTimeout(()=>setToast(''),5000);
      refresh();
      return result;
    }catch(error){setGlobalError(error.message||'Dataset upload failed.');throw error;}
  },[refresh]);

  const handleDemoLoad=useCallback(async()=>{
    setGlobalError('');setLoadingDemo(true);
    try{
      const result=await loadDemoDataset();
      setActiveDataset(result);
      setPage('laboratory');
      setToast('Demo Dataset — Sample Data loaded. It is synthetic and not live water data.');
      window.setTimeout(()=>setToast(''),6500);
      refresh();
      return result;
    }catch(error){setGlobalError(error.message||'Demo dataset could not be loaded. Is the API online?');return null;}
    finally{setLoadingDemo(false);}
  },[refresh]);

  const afterPromote=()=>{refresh();setToast('Production model selection refreshed.');window.setTimeout(()=>setToast(''),5000);};
  let content;
  if(page==='dashboard') content=<Dashboard dashboard={dashboard} health={health} activeDataset={activeDataset} refreshing={false} onRefresh={refresh} onNavigate={setPage} onLoadDemo={handleDemoLoad}/>;
  else if(page==='datasets') content=<DatasetWorkspace activeDataset={activeDataset} onUploaded={handleUpload} onLoadDemo={handleDemoLoad} loadingDemo={loadingDemo} onNavigate={setPage}/>;
  else if(page==='prediction') content=<PredictionPage health={health} activeDataset={activeDataset} onRefresh={refresh}/>;
  else if(page==='iot') content=<IoTMonitoring activeDataset={activeDataset} health={health} onRefreshDashboard={refresh}/>;
  else if(page==='history') content=<History refreshToken={refreshToken}/>;
  else if(page==='about') content=<About health={health}/>;
  else content=<Laboratory activeDataset={activeDataset} onLoadDemo={handleDemoLoad} loadingDemo={loadingDemo} initialTab={TAB_FOR_PAGE[page]||'preprocessing'} onModelPromoted={afterPromote} onNavigate={setPage}/>;

  return <Layout page={page} onNavigate={(next)=>{setPage(next);setGlobalError('');}} health={health} activeDataset={activeDataset} onLoadDemo={handleDemoLoad} loadingDemo={loadingDemo}>
    {globalError&&<div className="global-alert"><Notice tone="error" title="Could not complete the action">{globalError}</Notice><button onClick={()=>setGlobalError('')} aria-label="Dismiss">×</button></div>}
    {toast&&<div className="toast-message"><span className="toast-check">✓</span>{toast}<button onClick={()=>setToast('')} aria-label="Dismiss">×</button></div>}
    {content}
  </Layout>;
}
