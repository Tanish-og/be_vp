import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.routes import build_live_slots

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            try:
                slots = build_live_slots()
            except Exception as e:
                print(f"[WebSocket] Error building slots: {e}")
                slots = []

            await websocket.send_json({
                "success": True,
                "slots": slots,
                "detections": 0,
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"[WebSocket] Connection error: {e}")
        return