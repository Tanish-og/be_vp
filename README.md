# VisionPark AI — Intelligent Parking Management

![Overview page](./aotc-visionparkai/parking-dashboard/public/visionpark_overview.png)

VisionPark AI is a real-time, AI-powered smart parking management system that leverages YOLOv8 for vehicle detection and computer vision for dynamic slot occupancy monitoring. It comes with a modern React/Next.js dashboard and a FastAPI WebSocket backend.

**Deployed Dashboard:** https://visionpark.vercel.app (Note: Due to hardware restrictions, the YOLO inference backend must run locally for full functionality. The deployed dashboard provides a preview of the frontend UI).

## 🚀 Features

*   **Real-time Vehicle Detection:** Uses YOLOv8 to identify vehicles in video streams or live camera feeds at ~30 FPS.
*   **Dynamic Polygon Mapping:** A built-in admin tool lets you draw and save custom polygon geometries for each parking slot.
*   **Geometry-based Occupancy:** Analyzes bounding box intersection with defined polygons (3-of-5 points rule) to determine if a slot is OCCUPIED or FREE.
*   **Live WebSocket Feed:** Streams detection data and slot statuses directly to the frontend dashboard in real-time.
*   **Camera Integration:** A dedicated `/camera` page accesses the device camera via `getUserMedia`, pushing frames to the backend and overlaying bounding boxes + polygons live on the video feed.
*   **Temporal Debouncing:** Implements a 3-second smoothing window to prevent flicker from transient detections.

## 🛠️ Architecture

### Backend (`/Backend`)
*   **FastAPI:** High-performance async Python web framework.
*   **YOLOv8 (Ultralytics):** State-of-the-art object detection model.
*   **OpenCV:** Image processing and decoding.
*   **WebSockets:** For bi-directional, real-time communication of JSON payloads and binary image frames.
*   **MongoDB:** For persistent storage of parking events and analytics (optional, gracefully degraded if offline).

### Frontend (`/aotc-visionparkai/parking-dashboard`)
*   **Next.js 16 (App Router):** React framework for the modern web.
*   **Tailwind CSS:** Utility-first styling with custom glassmorphism components.
*   **Lucide React:** Beautiful, consistent icon set.

## ⚙️ Getting Started

### Prerequisites
*   Python 3.9+
*   Node.js 18+ (Node 20 recommended)
*   (Optional) MongoDB instance running on default port 27017

### 1. Start the Backend
```bash
cd Backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```
The FastAPI backend will start at `http://127.0.0.1:8002`.

### 2. Start the Frontend
```bash
cd aotc-visionparkai/parking-dashboard
npm install
npm run dev
```
The Next.js dashboard will be available at `http://localhost:3000`.

## 📷 Usage Guide

1.  **Admin Setup:** Navigate to `http://localhost:3000/admin`. Upload a reference image of your parking lot and draw polygons for each slot. Save the configuration.
2.  **Live Map:** Go to `http://localhost:3000/live` to see a real-time visualization of slot statuses based on the backend processing video files or camera feeds.
3.  **Camera Feed:** Navigate to `http://localhost:3000/camera` to activate your device's webcam. The system will encode frames, stream them to the backend, run YOLO inference, and project the results back onto the live video view.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
