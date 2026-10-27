import { useCallback, useEffect, useRef, useState } from "react";

interface DetectionBox {
  label: string;
  score: number;
  bbox_xyxy: number[];
}

interface StreamResult {
  reading: {
    person_count: number;
    small_luggage: number;
    large_luggage: number;
    confidence: number;
    detection_count: number;
  };
  detections: DetectionBox[];
  state_emitted: boolean;
  reason_codes: string[];
  model_version: string;
  state?: { mode: string; proposed_mode: string | null };
}

const SEND_INTERVAL_MS = 250;

export function CameraPanel({
  vehicleId,
  onResult,
}: {
  vehicleId: string;
  onResult: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const boxesRef = useRef<DetectionBox[]>([]);
  const latestRef = useRef<StreamResult | null>(null);
  const sendingRef = useRef(false);
  const timerRef = useRef<number | undefined>(undefined);

  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<StreamResult | null>(null);

  const stop = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = undefined;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setActive(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  const sendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || sendingRef.current) return;
    sendingRef.current = true;
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",", 2)[1];
      const response = await fetch(`/api/vision/stream-frame/${vehicleId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame_jpeg_base64: base64 }),
      });
      if (!response.ok) {
        setError(`frame rejected: ${response.status}`);
        return;
      }
      const result = (await response.json()) as StreamResult;
      latestRef.current = result;
      boxesRef.current = result.detections;
      setLastResult(result);
      setError(null);
      if (result.state_emitted) onResult();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      sendingRef.current = false;
    }
  }, [vehicleId, onResult]);

  const drawLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const context = canvas.getContext("2d");
    if (!context || video.readyState < 2) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    for (const box of boxesRef.current) {
      const [x1, y1, x2, y2] = box.bbox_xyxy;
      context.strokeStyle = box.label === "person" ? "#7ee2a8" : "#f4d58a";
      context.lineWidth = 3;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      context.fillStyle = context.strokeStyle;
      context.font = "16px sans-serif";
      context.fillText(`${box.label} ${Math.round(box.score * 100)}%`, x1 + 4, y1 + 20);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    const animation = window.setInterval(drawLoop, 66);
    const sender = window.setInterval(() => void sendFrame(), SEND_INTERVAL_MS);
    return () => {
      window.clearInterval(animation);
      window.clearInterval(sender);
    };
  }, [active, drawLoop, sendFrame]);

  const start = useCallback(async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = media;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = media;
      await video.play();
      setError(null);
      setActive(true);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? `camera unavailable: ${caught.message}`
          : "camera unavailable",
      );
    }
  }, []);

  return (
    <div className="camera-panel">
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={canvasRef} className="camera-canvas" data-testid="camera-canvas" />
      {!active && (
        <button disabled={false} onClick={() => void start()}>
          Start live camera
        </button>
      )}
      {active && (
        <button className="danger" onClick={stop}>
          Stop camera
        </button>
      )}
      {error && <p className="badge offline">{error}</p>}
      {lastResult && (
        <div className="row">
          <span>
            {lastResult.reading.person_count} persons · {lastResult.reading.small_luggage} small /{" "}
            {lastResult.reading.large_luggage} large bags · conf{" "}
            {(lastResult.reading.confidence * 100).toFixed(0)}%
            {lastResult.state ? ` · cabin ${lastResult.state.mode}` : ""}
          </span>
          <span className="muted">
            {lastResult.model_version} · {lastResult.reason_codes.join(", ") || "tracking"}
          </span>
        </div>
      )}
    </div>
  );
}
