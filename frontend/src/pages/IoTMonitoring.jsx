import React, { useEffect, useMemo, useState } from 'react';
import Icon from '../components/Icons';
import { SparkLine } from '../components/Charts';
import { Button, Card, DataTable, LoadingButton, Notice, Tag } from '../components/UI';
import { createDevice, getDevices, getReadings, sendReading } from '../services/iotService';

function normalizeFeatureRow(row, columns) {
  const blocked = new Set(['potability', 'safe', 'unsafe', 'target', 'label', 'class', 'quality']);
  const values = {};
  for (const name of columns) {
    if (blocked.has(String(name).toLowerCase())) continue;
    const value = row?.[name];
    if (value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))) {
      values[name] = Number(value);
    }
  }
  return values;
}

export default function IoTMonitoring({ activeDataset, health, onRefreshDashboard, onLoadDemo }) {
  const [devices, setDevices] = useState([]);
  const [selected, setSelected] = useState('');
  const [readings, setReadings] = useState([]);
  const [deviceId, setDeviceId] = useState('ESP32-WaterGuard-01');
  const [deviceName, setDeviceName] = useState('Main Water Reservoir Sensor');
  const [location, setLocation] = useState('Station A - Intake Point');
  const [busy, setBusy] = useState(false);
  const [simBusy, setSimBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [loadError, setLoadError] = useState('');
  const [wsConnected, setWsConnected] = useState(false);
  const [showHardwareGuide, setShowHardwareGuide] = useState(false);

  // WebSocket Live Stream Connection
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    function connectWs() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/api/ws/iot`;

      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setWsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'new_reading') {
              // Prepend to current readings if device matches or no filter
              setReadings((prev) => [data, ...prev]);
              onRefreshDashboard?.();
            }
          } catch (e) {}
        };

        ws.onclose = () => {
          setWsConnected(false);
          reconnectTimeout = setTimeout(connectWs, 3000);
        };

        ws.onerror = () => {
          setWsConnected(false);
        };
      } catch (e) {
        setWsConnected(false);
      }
    }

    connectWs();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, [onRefreshDashboard]);

  async function refreshDevices() {
    setLoadError('');
    try {
      const response = await getDevices();
      setDevices(response.items || []);
      if (!selected && response.items?.length) setSelected(response.items[0].device_id);
    } catch (e) {
      setLoadError(e.message || 'Could not load devices.');
    }
  }

  useEffect(() => {
    refreshDevices();
  }, []);

  useEffect(() => {
    if (!selected) {
      setReadings([]);
      return;
    }
    getReadings(selected)
      .then((r) => setReadings(r.items || []))
      .catch((e) => setLoadError(e.message));
  }, [selected]);

  async function register(event) {
    event.preventDefault();
    setError('');
    setMessage('');
    setBusy(true);
    try {
      const item = await createDevice({
        device_id: deviceId,
        device_name: deviceName,
        location: location || null,
      });
      setMessage(`${item.device_name} registered in PostgreSQL.`);
      setSelected(item.device_id);
      await refreshDevices();
      onRefreshDashboard?.();
    } catch (e) {
      setError(e.message || 'Device registration failed.');
    } finally {
      setBusy(false);
    }
  }

  async function runSimulation() {
    if (!activeDataset || !devices.length) return;
    setError('');
    setMessage('');
    setSimBusy(true);
    try {
      const rows = (activeDataset.preview || []).slice(0, 3);
      const device = devices.find((d) => d.device_id === selected) || devices[0];
      if (!device) throw new Error('Register a device before sending simulated readings.');

      let sent = 0;
      for (const row of rows) {
        const features = normalizeFeatureRow(row, activeDataset.numeric_columns || Object.keys(row));
        if (!Object.keys(features).length) continue;
        await sendReading(device.device_id, {
          timestamp: new Date(Date.now() + sent).toISOString(),
          source: 'simulation',
          features,
        });
        sent++;
      }
      setMessage(`${sent} sample row(s) sent as Simulated IoT Data. Labeled source: 'simulation'.`);
      const response = await getReadings(device.device_id);
      setReadings(response.items || []);
      await refreshDevices();
      onRefreshDashboard?.();
    } catch (e) {
      setError(e.message || 'Simulation request failed.');
    } finally {
      setSimBusy(false);
    }
  }

  const registryReady = Boolean(health?.database?.available) && !loadError;
  const online = devices.filter((d) => d.status === 'online').length;
  const selectedDevice = devices.find((d) => d.device_id === selected);
  const recent = [...readings].reverse();
  const phValues = recent
    .map((r) => r.features?.ph ?? r.features?.pH)
    .filter((v) => Number.isFinite(Number(v)))
    .map(Number);

  return (
    <div className="page-stack">
      {!health?.database?.available && (
        <Notice
          tone="warning"
          title="PostgreSQL is required for persistent IoT records"
        >
          Telemetry history and device registrations are persisted in PostgreSQL. Make sure PostgreSQL is online.
        </Notice>
      )}

      {error && <Notice tone="error" title="IoT action failed">{error}</Notice>}
      {message && <Notice tone="success" title="IoT status">{message}</Notice>}
      {loadError && <Notice tone="warning" title="Could not load device data">{loadError}</Notice>}

      {/* Top Stat Cards Grid */}
      <div className="iot-overview-grid">
        <Card className="iot-stat-card">
          <div className="iot-stat-icon">
            <Icon name="radio" size={18} />
          </div>
          <div>
            <span>Registered devices</span>
            <strong>{registryReady ? devices.length : '—'}</strong>
            <small>PostgreSQL registry</small>
          </div>
        </Card>

        <Card className="iot-stat-card">
          <div className="iot-stat-icon online">
            <Icon name="activity" size={18} />
          </div>
          <div>
            <span>Recently active</span>
            <strong>{registryReady ? online : '—'}</strong>
            <small>Seen in last 10 minutes</small>
          </div>
        </Card>

        <Card className="iot-stat-card">
          <div className="iot-stat-icon purple">
            <Icon name="database" size={18} />
          </div>
          <div>
            <span>Live WebSocket</span>
            <strong style={{ fontSize: '13px', color: wsConnected ? '#0d9488' : '#eab308' }}>
              {wsConnected ? 'Connected (Live)' : 'Polling Fallback'}
            </strong>
            <small>{wsConnected ? 'Real-time broadcast' : 'Reconnecting…'}</small>
          </div>
        </Card>

        <Card className="iot-stat-card">
          <div className="iot-stat-icon amber">
            <Icon name="shield" size={18} />
          </div>
          <div>
            <span>AI Model Pipeline</span>
            <strong>{health?.production_model?.available ? 'Active' : 'Unavailable'}</strong>
            <small>{health?.production_model?.available ? health.production_model.model_name : 'No artifact'}</small>
          </div>
        </Card>
      </div>

      {/* Main Grid: Device Registration & Simulation */}
      <div className="iot-main-grid">
        <Card>
          <div className="card-head">
            <div>
              <div className="eyebrow">DEVICE REGISTRY</div>
              <h3>Register Physical ESP32 / Sensor</h3>
              <p>Each device code must be unique. Telemetry is saved to PostgreSQL.</p>
            </div>
            <div className="iot-form-icon">
              <Icon name="radio" size={18} />
            </div>
          </div>

          <form className="device-form" onSubmit={register}>
            <label className="field">
              <span className="field-label">Device ID / Hardware Code</span>
              <input
                required
                maxLength={128}
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                placeholder="ESP32-WaterGuard-01"
              />
            </label>
            <label className="field">
              <span className="field-label">Display Name</span>
              <input
                required
                maxLength={200}
                value={deviceName}
                onChange={(e) => setDeviceName(e.target.value)}
                placeholder="Main Water Reservoir Sensor"
              />
            </label>
            <label className="field">
              <span className="field-label">Location <span className="muted">(optional)</span></span>
              <input
                maxLength={300}
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="Station A - Intake Point"
              />
            </label>
            <LoadingButton type="submit" loading={busy} loadingLabel="Registering…" icon="plus">
              Register Device
            </LoadingButton>
          </form>

          {/* Quick Hardware Guide Toggle */}
          <div style={{ marginTop: '14px', borderTop: '1px solid #edf2f1', paddingTop: '12px' }}>
            <Button
              variant="secondary"
              icon="book"
              onClick={() => setShowHardwareGuide(!showHardwareGuide)}
            >
              {showHardwareGuide ? 'Hide ESP32 Connection Guide' : 'View ESP32 Hardware & API Guide'}
            </Button>
          </div>

          {showHardwareGuide && (
            <div style={{ marginTop: '12px', background: '#f5f9f9', border: '1px solid #dce8e7', borderRadius: '8px', padding: '12px' }}>
              <div style={{ fontSize: '10px', fontWeight: '700', color: '#1a434c', marginBottom: '6px' }}>
                ESP32 C++ HTTP POST Example:
              </div>
              <pre style={{ background: '#0e2b38', color: '#bce4dc', padding: '10px', borderRadius: '6px', fontSize: '9px', overflow: 'auto', maxHeight: '160px' }}>
{`#include <WiFi.h>
#include <HTTPClient.h>

const char* serverUrl = "http://YOUR_SERVER_IP:8000/api/devices/ESP32-WaterGuard-01/readings";

void sendTelemetry() {
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\\"device_id\\":\\"ESP32-WaterGuard-01\\","
                   "\\"source\\":\\"sensor\\","
                   "\\"features\\":{"
                   "\\"ph\\":7.25,"
                   "\\"Hardness\\":180.0,"
                   "\\"Turbidity\\":2.8"
                   "}}";

  int httpCode = http.POST(payload);
  http.end();
}`}
              </pre>
            </div>
          )}
        </Card>

        {/* Simulated Telemetry Card */}
        <Card className="simulation-card">
          <div className="card-head">
            <div>
              <div className="eyebrow">DEMO MODE · ZERO INVENTED NUMBERS</div>
              <h3>IoT Simulation</h3>
              <p>Submit deterministic rows from the demo CSV for end-to-end testing.</p>
            </div>
            <span className="simulated-stamp">
              <span /> SIMULATED
            </span>
          </div>

          <div className="simulation-illustration">
            <div className="sim-ring">
              <Icon name="radio" size={28} />
            </div>
            <div className="sim-pulse pulse-one" />
            <div className="sim-pulse pulse-two" />
            <span className="sim-node node-a" />
            <span className="sim-node node-b" />
          </div>

          <div className="simulation-source">
            {activeDataset ? (
              <>
                <Icon name="database" size={14} />
                <span>Source: {activeDataset.file_name}</span>
              </>
            ) : (
              <>
                <Icon name="info" size={14} />
                <span>Load demo dataset to send verified sample rows</span>
              </>
            )}
          </div>

          <LoadingButton
            variant="secondary"
            loading={simBusy}
            loadingLabel="Submitting Simulated Data…"
            icon="play"
            disabled={!activeDataset || !devices.length}
            onClick={runSimulation}
          >
            Send Simulated Sample Readings
          </LoadingButton>

          <div className="simulation-disclaimer">
            These readings are existing demo CSV rows—never fabricated or claimed to be live sensor telemetry. Every stored record is strictly labeled <b>simulation</b>.
          </div>
        </Card>
      </div>

      {/* Bottom Grid: Devices List & Live Telemetry History */}
      <div className="iot-bottom-grid">
        <Card>
          <div className="card-head">
            <div>
              <div className="eyebrow">REGISTERED SENSORS</div>
              <h3>Device Status</h3>
              <p>Live devices seen within the last 10 minutes.</p>
            </div>
            <Button variant="ghost" icon="refresh" onClick={refreshDevices}>Refresh</Button>
          </div>

          {devices.length ? (
            <div className="device-list">
              {devices.map((device) => (
                <button
                  key={device.id}
                  className={`device-row ${selected === device.device_id ? 'device-row-active' : ''}`}
                  onClick={() => setSelected(device.device_id)}
                >
                  <span className={`device-row-icon ${device.status === 'online' ? 'device-online' : ''}`}>
                    <Icon name="radio" size={17} />
                  </span>
                  <span className="device-row-info">
                    <strong>{device.device_name}</strong>
                    <small>{device.location || device.device_id}</small>
                  </span>
                  <Tag tone={device.status === 'online' ? 'success' : 'neutral'} dot>
                    {device.status}
                  </Tag>
                  <Icon name="chevron" size={15} />
                </button>
              ))}
            </div>
          ) : (
            <div className="compact-empty">
              <Icon name="radio" size={20} />
              <span>No devices registered yet.</span>
            </div>
          )}
        </Card>

        <Card>
          <div className="card-head">
            <div>
              <div className="eyebrow">TELEMETRY & AI PREDICTIONS</div>
              <h3>{selectedDevice?.device_name || 'Telemetry Stream'}</h3>
              <p>{selectedDevice?.location || selectedDevice?.device_id || 'Select a registered sensor.'}</p>
            </div>
            {wsConnected && (
              <span className="ws-live-pill">
                <span className="ws-live-dot" />
                Live Stream
              </span>
            )}
          </div>

          {readings.length > 0 ? (
            <>
              <div className="iot-trend-summary">
                <strong>{phValues.length ? phValues.at(-1).toFixed(2) : '—'}</strong>
                <span>Latest pH Measurement</span>
                <SparkLine values={phValues} height={70} />
              </div>
              <DataTable
                columns={[
                  {
                    key: 'timestamp',
                    label: 'Timestamp',
                    render: (v) => new Date(v).toLocaleTimeString(),
                  },
                  {
                    key: 'source',
                    label: 'Data Source',
                    render: (v) =>
                      v === 'simulation' ? (
                        <Tag tone="warning">SIMULATED</Tag>
                      ) : (
                        <Tag tone="success">ESP32 SENSOR</Tag>
                      ),
                  },
                  {
                    key: 'prediction',
                    label: 'ML AI Prediction',
                    render: (v, row) => {
                      const p = v?.predicted_label || (row.prediction?.prediction != null ? (String(row.prediction.prediction) === '1' ? 'POTABLE' : 'NON-POTABLE') : null);
                      if (!p) return <span className="muted">—</span>;
                      return <Tag tone={p === 'POTABLE' ? 'success' : 'danger'}>{p}</Tag>;
                    },
                  },
                  {
                    key: 'features',
                    label: 'Readings Sample',
                    render: (f) =>
                      f
                        ? Object.entries(f)
                            .slice(0, 3)
                            .map(([k, val]) => `${k}: ${typeof val === 'number' ? val.toFixed(1) : val}`)
                            .join(', ')
                        : '—',
                  },
                ]}
                rows={readings.slice(0, 10)}
              />
            </>
          ) : (
            <div className="compact-empty">
              <Icon name="activity" size={20} />
              <span>No readings received yet for this sensor.</span>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
