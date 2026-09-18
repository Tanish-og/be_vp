import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.routes import build_live_slots

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await websocket.send_json({
                "success": True,
                "slots": build_live_slots(),
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return