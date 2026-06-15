"""FastAPI application for Stellar Predictor web interface."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.metadata import version as _pkg_version
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from stellar_predictor.web.routes import systems, predictions, visualizations
from stellar_predictor.web.tasks import TaskManager
from stellar_predictor.web.websocket import router as ws_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.task_manager = TaskManager()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stellar Predictor",
        description="Predict unknown celestial bodies through gravitational perturbation analysis",
        version=_pkg_version("stellar-predictor"),
        lifespan=lifespan,
    )

    app.include_router(systems.router, prefix="/api")
    app.include_router(predictions.router, prefix="/api")
    app.include_router(visualizations.router, prefix="/api")
    app.include_router(ws_router)

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
