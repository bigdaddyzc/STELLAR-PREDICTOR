"""Visualization data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/viz/distribution/{task_id}")
async def get_distribution_plot(request: Request, task_id: str):
    task = request.app.state.task_manager.get_task(task_id)
    if not task:
        return {"error": "Task not found"}
    if task.status != "complete":
        return {"error": "Task not complete", "status": task.status}
    if not task.distribution_plot_data:
        return {"error": "No distribution plot data"}
    return task.distribution_plot_data


@router.get("/viz/tb-fit/{task_id}")
async def get_tb_visualization(request: Request, task_id: str):
    task = request.app.state.task_manager.get_task(task_id)
    if not task:
        return {"error": "Task not found"}
    if task.status != "complete":
        return {"error": "Task not complete", "status": task.status}
    if not task.tb_plot_data:
        return {"error": "No TB fit data available"}
    return task.tb_plot_data


@router.get("/viz/spacing/{task_id}")
async def get_spacing_visualization(request: Request, task_id: str):
    task = request.app.state.task_manager.get_task(task_id)
    if not task:
        return {"error": "Task not found"}
    if task.status != "complete":
        return {"error": "Task not complete", "status": task.status}
    if not task.spacing_plot_data:
        return {"error": "No spacing data available"}
    return task.spacing_plot_data
