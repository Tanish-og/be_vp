from fastapi import FastAPI
from api.routes import router
from fastapi.middleware.cors import CORSMiddleware
from websocket import router as websocket_router


app = FastAPI(
    title="VisionPark Backend",
    description="AI Smart Parking System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

app.include_router(websocket_router)