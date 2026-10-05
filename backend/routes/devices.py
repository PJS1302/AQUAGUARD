from __future__ import annotations
from fastapi import APIRouter
from backend.schemas.device import DeviceCreate
from backend.services.device_service import get_device, list_devices, register_device

router = APIRouter(prefix="/devices")

@router.get("")
def devices():
    return {"items": list_devices()}

@router.post("")
def create_device(request: DeviceCreate):
    return register_device(request.device_id, request.device_name, request.location)

@router.get("/{device_id}")
def device(device_id: str):
    return get_device(device_id)

