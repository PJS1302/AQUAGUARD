from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_backend_exposes_api_routes_under_api_prefix():
    health = client.get('/api/health')
    assert health.status_code == 200
    payload = health.json()
    assert payload.get('status') in {'ok', 'healthy'}

    datasets = client.get('/api/datasets')
    assert datasets.status_code == 200
