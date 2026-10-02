"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  Camera,
  CameraOff,
  Radio,
  Car,
  CheckCircle2,
  XCircle,
  Zap,
  AlertCircle,
  RefreshCw,
  Maximize2,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Types                                                                */
/* ------------------------------------------------------------------ */
type Box = {
  bbox: [number, number, number, number];
  confidence: number;
};

type Slot = {
  id: string;
  status: string;
  points: number[][];
};

type ServerMsg = {
  success: boolean;
  detections: number;
  slots: Slot[];
  boxes: Box[];
  message?: string;
};

/* ------------------------------------------------------------------ */
/*  Constants                                                            */
/* ------------------------------------------------------------------ */
const WS_CAMERA_URL =
  process.env.NEXT_PUBLIC_WS_URL?.replace("/ws", "/ws/camera") ||
  "ws://127.0.0.1:8002/ws/camera";

// How many ms to wait between frames sent to the backend
const FRAME_INTERVAL_MS = 200; // ~5 fps inference cadence

/* ------------------------------------------------------------------ */
/*  Helper: capture a JPEG blob from a <video> element                  */
/* ------------------------------------------------------------------ */
function captureJpeg(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  quality = 0.75
): Promise<Blob | null> {
  return new Promise((resolve) => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return resolve(null);
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(resolve, "image/jpeg", quality);
  });
}

/* ------------------------------------------------------------------ */
/*  StatCard                                                             */
/* ------------------------------------------------------------------ */
function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div
      className="glass rounded-xl p-4 flex items-center gap-3"
      style={{ flex: "1 1 0" }}
    >
      <div
        className="rounded-xl flex items-center justify-center shrink-0"
        style={{
          width: 40,
          height: 40,
          background: `${color}18`,
          border: `1px solid ${color}30`,
        }}
      >
        <Icon size={18} style={{ color }} />
      </div>
      <div>
        <div
          style={{
            fontSize: 10,
            color: "var(--text-muted)",
            fontWeight: 700,
            letterSpacing: "0.1em",
            marginBottom: 2,
          }}
        >
          {label}
        </div>
        <div style={{ fontSize: 22, fontWeight: 900, color: "var(--text-primary)", lineHeight: 1 }}>
          {value}
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  Main Page                                                            */
/* ================================================================== */
export default function CameraPage() {
  /* --- Refs --- */
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null); // off-screen capture canvas
  const overlayRef = useRef<HTMLCanvasElement>(null); // on-screen annotation canvas
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  /* --- State --- */
  const [cameraActive, setCameraActive] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [detections, setDetections] = useState(0);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);

  const fpsCounterRef = useRef(0);
  const lastFpsRef = useRef(Date.now());

  /* ---------------------------------------------------------------- */
  /*  Draw overlay boxes + polygon slots onto the overlay canvas        */
  /* ---------------------------------------------------------------- */
  const drawOverlay = useCallback(
    (currentBoxes: Box[], currentSlots: Slot[]) => {
      const overlay = overlayRef.current;
      const video = videoRef.current;
      if (!overlay || !video) return;

      const vw = video.videoWidth || overlay.width;
      const vh = video.videoHeight || overlay.height;
      overlay.width = overlay.clientWidth;
      overlay.height = overlay.clientHeight;

      const scaleX = overlay.width / vw;
      const scaleY = overlay.height / vh;

      const ctx = overlay.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      /* --- draw slot polygons --- */
      for (const slot of currentSlots) {
        if (!slot.points || slot.points.length < 3) continue;
        const occupied = slot.status === "Occupied";
        ctx.beginPath();
        ctx.moveTo(slot.points[0][0] * scaleX, slot.points[0][1] * scaleY);
        for (let i = 1; i < slot.points.length; i++) {
          ctx.lineTo(slot.points[i][0] * scaleX, slot.points[i][1] * scaleY);
        }
        ctx.closePath();
        ctx.fillStyle = occupied ? "rgba(244,63,94,0.22)" : "rgba(16,185,129,0.18)";
        ctx.fill();
        ctx.strokeStyle = occupied ? "#f43f5e" : "#10b981";
        ctx.lineWidth = 2;
        ctx.stroke();

        /* label */
        const cx =
          (slot.points.reduce((s, p) => s + p[0], 0) / slot.points.length) * scaleX;
        const cy =
          (slot.points.reduce((s, p) => s + p[1], 0) / slot.points.length) * scaleY;
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "white";
        ctx.shadowColor = "rgba(0,0,0,0.7)";
        ctx.shadowBlur = 4;
        ctx.fillText(slot.id, cx, cy);
        ctx.shadowBlur = 0;
      }

      /* --- draw bounding boxes --- */
      for (const box of currentBoxes) {
        const [x1, y1, x2, y2] = box.bbox;
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

        // confidence label
        const labelText = `${(box.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 10px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "rgba(56,189,248,0.85)";
        ctx.fillRect(x1 * scaleX - 1, y1 * scaleY - 17, ctx.measureText(labelText).width + 8, 16);
        ctx.fillStyle = "white";
        ctx.fillText(labelText, x1 * scaleX + 3, y1 * scaleY - 15);
      }
    },
    []
  );

  /* ---------------------------------------------------------------- */
  /*  Connect WebSocket                                                  */
  /* ---------------------------------------------------------------- */
  const connectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const ws = new WebSocket(WS_CAMERA_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (event) => {
      try {
        const msg: ServerMsg = JSON.parse(event.data);
        if (msg.success) {
          setSlots(msg.slots ?? []);
          setBoxes(msg.boxes ?? []);
          setDetections(msg.detections ?? 0);
          drawOverlay(msg.boxes ?? [], msg.slots ?? []);

          // FPS counter
          fpsCounterRef.current += 1;
          const now = Date.now();
          if (now - lastFpsRef.current >= 1000) {
            setFps(fpsCounterRef.current);
            fpsCounterRef.current = 0;
            lastFpsRef.current = now;
          }
        }
      } catch {
        /* ignore parse errors */
      }
    };

    wsRef.current = ws;
  }, [drawOverlay]);

  /* ---------------------------------------------------------------- */
  /*  Start camera                                                       */
  /* ---------------------------------------------------------------- */
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      connectWs();
      setCameraActive(true);

      // Start sending frames
      intervalRef.current = setInterval(async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ws = wsRef.current;
        if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

        const t0 = performance.now();
        const blob = await captureJpeg(video, canvas);
        if (!blob) return;

        const buf = await blob.arrayBuffer();
        ws.send(buf);
        setInferenceMs(Math.round(performance.now() - t0));
      }, FRAME_INTERVAL_MS);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Camera access failed: ${msg}`);
    }
  }, [connectWs]);

  /* ---------------------------------------------------------------- */
  /*  Stop camera                                                        */
  /* ---------------------------------------------------------------- */
  const stopCamera = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    streamRef.current = null;
    setCameraActive(false);
    setWsConnected(false);
    setSlots([]);
    setBoxes([]);
    setDetections(0);
    setFps(0);
    // Clear overlay
    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext("2d");
      ctx?.clearRect(0, 0, overlay.width, overlay.height);
    }
  }, []);

  /* Cleanup on unmount */
  useEffect(() => () => stopCamera(), [stopCamera]);

  /* ---------------------------------------------------------------- */
  /*  Derived values                                                     */
  /* ---------------------------------------------------------------- */
  const totalSlots = slots.length;
  const occupied = slots.filter((s) => s.status === "Occupied").length;
  const available = totalSlots - occupied;

  /* ================================================================ */
  /*  Render                                                             */
  /* ================================================================ */
  return (
    <div className="flex flex-col gap-6 p-6 lg:p-8 h-full">
      {/* ── Header ── */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-black gradient-text">Camera Feed</h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            Live camera input with real-time YOLO detection &amp; slot occupancy
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* WS status badge */}
          <div
            className="flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold"
            style={{
              background: wsConnected ? "rgba(16,185,129,0.1)" : "rgba(100,116,139,0.1)",
              border: `1px solid ${wsConnected ? "rgba(16,185,129,0.3)" : "rgba(100,116,139,0.2)"}`,
              color: wsConnected ? "var(--accent-green)" : "var(--text-muted)",
            }}
          >
            <span
              className="rounded-full"
              style={{
                width: 7,
                height: 7,
                background: wsConnected ? "var(--accent-green)" : "var(--text-muted)",
                display: "inline-block",
                animation: wsConnected ? "pulse-dot 2s infinite" : "none",
              }}
            />
            {wsConnected ? "AI Connected" : "AI Offline"}
          </div>

          {/* Toggle button */}
          <button
            id="camera-toggle-btn"
            onClick={cameraActive ? stopCamera : startCamera}
            className={cameraActive ? "btn-danger" : "btn-primary"}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            {cameraActive ? (
              <>
                <CameraOff size={15} />
                Stop Camera
              </>
            ) : (
              <>
                <Camera size={15} />
                Start Camera
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div
          className="glass rounded-xl p-4 flex items-start gap-3"
          style={{ borderColor: "rgba(244,63,94,0.3)", background: "rgba(244,63,94,0.06)" }}
        >
          <AlertCircle size={18} style={{ color: "var(--accent-red)", marginTop: 1, flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, color: "var(--accent-red)" }}>
              Camera Error
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{error}</div>
          </div>
          <button
            className="btn-ghost ml-auto"
            style={{ padding: "4px 12px", fontSize: 12 }}
            onClick={() => {
              setError(null);
              startCamera();
            }}
          >
            <RefreshCw size={13} style={{ display: "inline", marginRight: 4 }} />
            Retry
          </button>
        </div>
      )}

      {/* ── Stat cards ── */}
      <div className="flex gap-3">
        <StatCard label="TOTAL SLOTS" value={totalSlots} icon={Car} color="#38bdf8" />
        <StatCard label="OCCUPIED" value={occupied} icon={XCircle} color="#f43f5e" />
        <StatCard label="AVAILABLE" value={available} icon={CheckCircle2} color="#10b981" />
        <StatCard label="DETECTIONS" value={detections} icon={Zap} color="#8b5cf6" />
        <StatCard label="INF/s" value={fps} icon={Radio} color="#f59e0b" />
      </div>

      {/* ── Main video + overlay area ── */}
      <div className="flex gap-5" style={{ flex: 1, minHeight: 0 }}>
        {/* Video feed */}
        <div
          className="glass flex-1 overflow-hidden rounded-2xl"
          style={{ position: "relative", minHeight: 400 }}
        >
          {/* Header bar */}
          <div
            className="flex items-center justify-between px-5 py-3"
            style={{ borderBottom: "1px solid rgba(30,41,59,0.9)" }}
          >
            <div className="flex items-center gap-2">
              <Radio size={15} style={{ color: "var(--accent-blue)" }} />
              <span style={{ fontWeight: 700, fontSize: 14 }}>Live Camera Feed</span>
            </div>
            <div className="flex items-center gap-3">
              {cameraActive && (
                <div
                  className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold"
                  style={{ background: "rgba(244,63,94,0.12)", border: "1px solid rgba(244,63,94,0.25)", color: "#f43f5e" }}
                >
                  <span
                    className="rounded-full"
                    style={{ width: 6, height: 6, background: "#f43f5e", display: "inline-block", animation: "pulse-dot 1s infinite" }}
                  />
                  LIVE
                </div>
              )}
              {inferenceMs != null && cameraActive && (
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {inferenceMs}ms capture
                </span>
              )}
            </div>
          </div>

          {/* Camera / placeholder */}
          <div
            style={{
              position: "relative",
              background: "rgba(5,13,26,0.95)",
              height: "calc(100% - 50px)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {/* Video element */}
            <video
              ref={videoRef}
              id="camera-video"
              playsInline
              muted
              autoPlay
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                display: cameraActive ? "block" : "none",
              }}
            />

            {/* SVG annotation overlay */}
            <canvas
              ref={overlayRef}
              id="camera-overlay"
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                display: cameraActive ? "block" : "none",
              }}
            />

            {/* Placeholder (camera not active) */}
            {!cameraActive && (
              <div className="flex flex-col items-center gap-4 text-center p-8" style={{ maxWidth: 340 }}>
                <div
                  className="rounded-full flex items-center justify-center"
                  style={{
                    width: 80,
                    height: 80,
                    background: "rgba(56,189,248,0.08)",
                    border: "1px solid rgba(56,189,248,0.2)",
                  }}
                >
                  <Camera size={36} style={{ color: "var(--accent-blue)", opacity: 0.6 }} />
                </div>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>
                    Camera Feed Inactive
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.7 }}>
                    Click <strong style={{ color: "var(--accent-blue)" }}>Start Camera</strong> to
                    activate your device camera. The AI engine will detect vehicles and overlay
                    slot occupancy in real-time.
                  </div>
                </div>
                <button
                  id="camera-start-placeholder-btn"
                  className="btn-primary"
                  onClick={startCamera}
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <Camera size={15} />
                  Start Camera
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Side panel — slot list */}
        <div
          className="glass rounded-2xl flex flex-col overflow-hidden"
          style={{ width: 220, minHeight: 400 }}
        >
          <div
            className="flex items-center gap-2 px-4 py-3"
            style={{ borderBottom: "1px solid rgba(30,41,59,0.9)" }}
          >
            <Maximize2 size={14} style={{ color: "var(--accent-blue)" }} />
            <span style={{ fontWeight: 700, fontSize: 13 }}>Slot Status</span>
          </div>

          <div className="flex-1 overflow-y-auto p-3" style={{ gap: 6, display: "flex", flexDirection: "column" }}>
            {slots.length === 0 ? (
              <div
                className="flex flex-col items-center justify-center gap-3 h-full text-center"
                style={{ color: "var(--text-muted)", fontSize: 12, padding: "32px 16px" }}
              >
                <AlertCircle size={24} style={{ opacity: 0.4 }} />
                <span>Start camera to see real-time slot occupancy</span>
              </div>
            ) : (
              slots.map((slot) => (
                <div
                  key={slot.id}
                  className="rounded-xl flex items-center justify-between px-3 py-2"
                  style={{
                    background:
                      slot.status === "Occupied"
                        ? "rgba(244,63,94,0.08)"
                        : "rgba(16,185,129,0.08)",
                    border: `1px solid ${slot.status === "Occupied" ? "rgba(244,63,94,0.2)" : "rgba(16,185,129,0.2)"}`,
                    transition: "all 0.3s ease",
                  }}
                >
                  <span style={{ fontWeight: 700, fontSize: 13 }}>{slot.id}</span>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: slot.status === "Occupied" ? "var(--accent-red)" : "var(--accent-green)",
                      letterSpacing: "0.05em",
                    }}
                  >
                    {slot.status === "Occupied" ? "OCCUPIED" : "FREE"}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Off-screen canvas for JPEG encoding */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
