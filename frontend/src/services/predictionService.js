import { api } from './api';

export const predictOne = (features, source = 'manual') => api.post('/predict', { features, source });
export const predictDataset = (datasetId) => {
  const data = new FormData();
  data.append('dataset_id', datasetId);
  return api.form('/predict-file', data);
};
export const predictUploadFile = (file) => {
  const data = new FormData();
  data.append('file', file);
  return api.form('/predict-file', data);
};
export const getGlobalShap = () => api.get('/prediction/shap-global');
