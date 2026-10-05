import { useEffect, useState } from 'react';
import { getDashboard, getHealth } from '../services/dashboardService';

export function usePlatformData(refreshToken) {
  const [health,setHealth]=useState(null);
  const [dashboard,setDashboard]=useState(null);
  useEffect(()=>{
    let alive=true;
    const load=async()=>{
      const [healthResult,dashboardResult]=await Promise.allSettled([getHealth(),getDashboard()]);
      if(!alive)return;
      setHealth(healthResult.status==='fulfilled'?healthResult.value:null);
      setDashboard(dashboardResult.status==='fulfilled'?dashboardResult.value:null);
    };
    load();
    const timer=setInterval(load,30000);
    return()=>{alive=false;clearInterval(timer);};
  },[refreshToken]);
  return {health,dashboard};
}
