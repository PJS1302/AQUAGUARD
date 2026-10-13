from __future__ import annotations

import asyncio
import json
import unittest

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from backend.routes.readings import post_reading
from backend.schemas.device import DeviceCreate


def _request_for(device_id: str, body: dict) -> Request:
    raw = json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    path = f"/devices/{device_id}/readings"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    return Request(scope, receive)


class IoTContractTests(unittest.TestCase):
    def test_device_registration_trims_values_and_rejects_blank_ids_or_names(self):
        device = DeviceCreate(device_id=" sensor-1 ", device_name=" Intake ", location="   ")
        self.assertEqual(device.device_id, "sensor-1")
        self.assertEqual(device.device_name, "Intake")
        self.assertIsNone(device.location)
        with self.assertRaises(ValidationError):
            DeviceCreate(device_id="   ", device_name="Intake")
        with self.assertRaises(ValidationError):
            DeviceCreate(device_id="sensor-1", device_name="  ")

    def test_reading_endpoint_rejects_invalid_measurements_before_database_access(self):
        invalid_bodies = [
            {"source": "sensor", "features": {"ph": "invalid"}},
            {"source": "sensor", "features": {"ph": float("nan")}},
            {"source": "sensor", "features": {"ph": 10**400}},
            {"source": "sensor", "features": {"ph": True}},
            {"source": "sensor", "features": {" ": 7.0}},
            {"source": "sensor", "features": {" ph ": 7.0, "ph": 7.1}},
            {"source": "unknown", "features": {"ph": 7.0}},
            {"device_id": "another-device", "features": {"ph": 7.0}},
        ]
        for body in invalid_bodies:
            with self.subTest(body=body):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(post_reading("sensor-1", _request_for("sensor-1", body)))
                self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
