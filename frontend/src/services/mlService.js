import { api } from './api';

const regressionPath = {
  linear: '/regression/linear',
  multiple: '/regression/multiple',
  polynomial: '/regression/polynomial',
  lwlr: '/regression/lwlr',
};
export const runRegression = (algorithm, body) => api.post(regressionPath[algorithm], body);

const classificationPath = {
  'decision-tree': '/classification/decision-tree',
  c45: '/classification/c45',
  'naive-bayes': '/classification/naive-bayes',
  knn: '/classification/knn',
  bagging: '/classification/bagging',
  adaboost: '/classification/adaboost',
  'random-forest': '/classification/random-forest',
};
export const runClassifier = (algorithm, body) => api.post(classificationPath[algorithm], body);
export const compareClassifiers = (body) => api.post('/classification/compare', body);
export const promoteModel = (experimentId, modelName) => api.post('/classification/production', { experiment_id: experimentId, model_name: modelName });
export const runPca = (body) => api.post('/pca/analyze', body);
export const runClustering = (algorithm, body) => api.post(`/clustering/${algorithm}`, body);
