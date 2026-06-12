"""Analysis task endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from stellar_predictor.web.schemas import AnalysisRequest, TaskStatusResponse

router = APIRouter()


@router.post("/analyze")
async def start_analysis(request: Request, body: AnalysisRequest):
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
        return {"error": "Task not found"}
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        result=task.result,
        error=task.error,
    )
