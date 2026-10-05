import { api } from './api';

export const getDevices = () => api.get('/devices');
export const createDevice = (body) => api.post('/devices', body);
export const getDevice = (id) => api.get(`/devices/${encodeURIComponent(id)}`);
export const getReadings = (id, limit = 100) => api.get(`/devices/${encodeURIComponent(id)}/readings?limit=${limit}`);
export const sendReading = (id, body) => api.post(`/devices/${encodeURIComponent(id)}/readings`, body);
