from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.connection import init_db
from backend.routes import health, prediction, dataset, regression, classification, clustering, pca, devices, readings, dashboard, preprocessing, iot_ws

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="AquaGuard API",
    description="Water quality monitoring, ML experiments, production prediction, and IoT analytics.",
    version="1.0.0",
    lifespan=lifespan,
)
raw_origins = os.getenv("AQUAGUARD_CORS_ORIGINS", "*").strip()
origins = [item.strip() for item in raw_origins.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" in origins else origins + ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for route_module in (health, prediction, dataset, regression, classification, clustering, pca, devices, readings, dashboard, preprocessing, iot_ws):
    app.include_router(route_module.router, prefix="/api")

# Serve React frontend static files when built
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")
else:
    @app.get("/")
    def root():
        return {"application": "AquaGuard", "docs": "/docs", "health": "/api/health"}
