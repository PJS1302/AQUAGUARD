import React, { useEffect, useState } from 'react';
import Icon from '../components/Icons';
import { Button, Card, DataTable, EmptyState, Notice, Tag } from '../components/UI';
import { getPredictionHistory } from '../services/datasetService';
import { downloadCsv } from '../utils/export';

export default function History({ refreshToken }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    setError('');
    try {
      const result = await getPredictionHistory(200);
      setItems(result.items || []);
    } catch (e) {
      setError(e.message || 'Prediction history could not be loaded.');
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
  }, [refreshToken]);

  return (
    <div className="page-stack">
      {error && (
        <Notice tone="warning" title="Prediction history unavailable">
          {error} Prediction history requires PostgreSQL; no records are kept in a silent in-memory substitute.
        </Notice>
      )}

      <Card>
        <div className="card-head">
          <div>
            <div className="eyebrow">POSTGRESQL AUDIT TRAIL</div>
            <h3>Prediction History & Audit Records</h3>
            <p>Manual, batch dataset, and ESP32 IoT predictions stored in PostgreSQL.</p>
          </div>
          <div className="history-actions">
            <Tag tone={error ? 'warning' : 'neutral'}>
              {error ? 'PostgreSQL unavailable' : `${items.length} records loaded`}
            </Tag>
            <Button variant="ghost" icon="refresh" onClick={load} disabled={busy}>
              Refresh
            </Button>
            {items.length > 0 && (
              <Button
                variant="secondary"
                icon="download"
                onClick={() => downloadCsv('aquaguard_prediction_history.csv', items)}
              >
                Export CSV
              </Button>
            )}
          </div>
        </div>

        {items.length ? (
          <DataTable
            columns={[
              {
                key: 'timestamp',
                label: 'Timestamp',
                render: (v) => new Date(v).toLocaleString(),
              },
              {
                key: 'source',
                label: 'Input Source',
                render: (v) => (
                  <Tag tone={v === 'iot' ? 'success' : v === 'dataset' ? 'blue' : 'neutral'}>
                    {v === 'iot' ? 'IoT SENSOR' : v === 'dataset' ? 'CSV BATCH' : 'MANUAL'}
                  </Tag>
                ),
              },
              { key: 'model_name', label: 'Model' },
              {
                key: 'prediction',
                label: 'AI Potability Output',
                render: (v, row) => {
                  const label =
                    row.predicted_label || (String(v) === '1' ? 'POTABLE' : 'NON-POTABLE');
                  return (
                    <Tag tone={label === 'POTABLE' ? 'success' : 'danger'}>
                      {label}
                    </Tag>
                  );
                },
              },
              {
                key: 'confidence',
                label: 'Model Confidence',
                render: (v) => (v == null ? 'Not exposed' : `${(Number(v) * 100).toFixed(1)}%`),
              },
              {
                key: 'features',
                label: 'Evaluated Readings',
                render: (f) =>
                  f
                    ? Object.entries(f)
                        .slice(0, 3)
                        .map(([k, val]) => `${k}: ${typeof val === 'number' ? val.toFixed(1) : val}`)
                        .join(', ')
                    : '—',
              },
            ]}
            rows={items}
          />
        ) : (
          <EmptyState
            icon="history"
            title={busy ? 'Loading prediction history…' : 'No saved predictions'}
          >
            Prediction records appear here after an AI prediction is run and persisted to PostgreSQL.
          </EmptyState>
        )}
      </Card>

      <Card className="history-policy">
        <div className="policy-icon">
          <Icon name="shield" size={17} />
        </div>
        <div>
          <h3>Traceable Model Outputs</h3>
          <p>
            Stored records include input source, model name, timestamp, original input features, and model-reported probabilities. Drinking water decisions require certified physical laboratory analysis.
          </p>
        </div>
      </Card>
    </div>
  );
}
