"""WebSocket endpoint for real-time IoT monitoring and dashboard streaming."""
from __future__ import annotations

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/iot")
async def websocket_iot_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Keep connection open and handle incoming client pings or heartbeats
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.info("WebSocket connection closed: %s", exc)
        ws_manager.disconnect(websocket)
