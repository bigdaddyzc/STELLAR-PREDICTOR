"""Analysis task endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from stellar_predictor.data.known_systems import system_keys
from stellar_predictor.web.schemas import AnalysisRequest, TaskStatusResponse

router = APIRouter()


@router.post("/analyze")
async def start_analysis(request: Request, body: AnalysisRequest):
    # Reject unknown systems up front instead of creating a task that fails.
    if body.system not in system_keys():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown system: {body.system}. Supported: {', '.join(system_keys())}",
        )
    task_manager = request.app.state.task_manager
    task_id = task_manager.create_task(body)
    loop = asyncio.get_event_loop()
    task_manager.submit(task_id, body, loop)
    return {"task_id": task_id, "status": "submitted"}


@router.get("/analyze/{task_id}")
async def get_analysis_status(request: Request, task_id: str):
    task_manager = request.app.state.task_manager
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        result=task.result,
        error=task.error,
    )
