import { api } from './api';

export async function uploadDataset(file, { isDemo = false } = {}) {
  const data = new FormData();
  data.append('file', file);
  data.append('is_demo', String(isDemo));
  const uploaded = await api.form('/datasets/upload', data);
  // Re-read the saved original so the workspace uses the backend's authoritative
  // row/column counts rather than a stale or partial upload response.
  let inspected;
  try {
    inspected = await api.get(`/datasets/${encodeURIComponent(uploaded.id)}`);
  } catch (error) {
    if (Number(uploaded.row_count) > 0 && Number(uploaded.column_count) > 0) return uploaded;
    throw error;
  }
  const dataset = { ...uploaded, ...inspected };
  if (!(Number(dataset.row_count) > 0) || !(Number(dataset.column_count) > 0)) {
    throw new Error('The backend saved this CSV but reports zero data rows or columns. Check the CSV contents and confirm the frontend is connected to the correct AquaGuard API.');
  }
  return dataset;
}

export async function loadDemoDataset() {
  const response = await fetch('/demo/demo_water_quality.csv');
  if (!response.ok) throw new Error('The bundled demo CSV could not be loaded.');
  const blob = await response.blob();
  const file = new File([blob], 'demo_water_quality.csv', { type: 'text/csv' });
  return uploadDataset(file, { isDemo: true });
}

export const getDatasets = () => api.get('/datasets');
export const inspectDataset = (id) => api.get(`/datasets/${encodeURIComponent(id)}`);
export const getPreprocessingStatus = () => api.get('/preprocessing/status');
export const preprocessDataset = (id) => api.post(`/datasets/${encodeURIComponent(id)}/preprocess`, {});
export const predictFile = (id) => {
  const data = new FormData();
  data.append('dataset_id', id);
  return api.form('/predict-file', data);
};
export const getPredictionHistory = (limit = 100) => api.get(`/predictions?limit=${limit}`);

export function downloadProcessedFile(processId) {
  window.location.href = `/api/datasets/processed/${encodeURIComponent(processId)}`;
}

export function downloadOriginalFile(datasetId) {
  window.location.href = `/api/datasets/${encodeURIComponent(datasetId)}/original`;
}
