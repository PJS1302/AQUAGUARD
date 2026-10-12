from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.connection import init_db
from backend.routes import health, prediction, dataset, regression, classification, clustering, pca, devices, readings, dashboard, preprocessing

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
if "*" in origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

for route_module in (health, prediction, dataset, regression, classification, clustering, pca, devices, readings, dashboard, preprocessing):
    app.include_router(route_module.router, prefix="/api")

@app.get("/")
def root():
    return {"application": "AquaGuard", "docs": "/docs", "health": "/health"}
