import { api } from './api';

export const getHealth = () => api.get('/health');
export const getDashboard = () => api.get('/dashboard');
