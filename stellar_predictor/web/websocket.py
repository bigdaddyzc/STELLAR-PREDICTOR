"""WebSocket endpoint for real-time prediction progress."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/analyze/{task_id}")
async def analysis_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()

    task_manager = websocket.app.state.task_manager
    task = task_manager.get_task(task_id)

    if not task:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    last_progress = -1.0
    try:
        while task.status not in ("complete", "failed"):
            if task.progress != last_progress:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.status,
                    "stage": task.stage,
                    "progress": task.progress,
                })
                last_progress = task.progress
            await asyncio.sleep(0.5)

        await websocket.send_json({
            "task_id": task_id,
            "status": task.status,
            "stage": task.stage,
            "progress": task.progress,
            "error": task.error,
        })
    except WebSocketDisconnect:
        pass
