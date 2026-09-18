import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "").strip()

client = None


def get_events_collection():
    global client
    if client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI is not configured")
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return client["parking_db"]["parking_events"]


def is_mongo_connected():
    try:
        get_events_collection().database.client.admin.command("ping")
        return True
    except (PyMongoError, RuntimeError):
        return False


def save_event(slot, status, detections):
    get_events_collection().insert_one({
        "slot": slot,
        "status": status,
        "detections": detections,
        "timestamp": datetime.utcnow()
    })


def get_all_events():
    return list(
        get_events_collection().find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1)
    )