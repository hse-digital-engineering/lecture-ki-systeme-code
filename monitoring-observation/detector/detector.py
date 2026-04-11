"""
Litter detection service.

Captures frames from a webcam (CAMERA_MODE=webcam) or generates synthetic
frames (CAMERA_MODE=synthetic), runs YOLO v8 detection, and publishes results
to a Zenoh message bus.

All spans, metrics and logs are exported via OpenTelemetry.
"""
import json
import logging
import os
import random
import sys
import time
from collections import deque

import cv2
import numpy as np
import zenoh
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


# ── Inference setup ──────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n")

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("detector")
#logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose output; change to INFO for less noise

# ── OpenTelemetry setup ──────────────────────────────────────────────────────
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "yolo-detector")


resource = Resource.create({ResourceAttributes.SERVICE_NAME: SERVICE_NAME})

# Traces
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True))
)

trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(SERVICE_NAME)

# Metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True), export_interval_millis=5000
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(SERVICE_NAME)

# Logs
log_provider = LoggerProvider(resource=resource)
log_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTEL_ENDPOINT, insecure=True))
)
set_logger_provider(log_provider)
logging.getLogger().addHandler(LoggingHandler(logger_provider=log_provider))

# ── Metrics instruments ───────────────────────────────────────────────────────
inference_latency = meter.create_histogram(
    "inference_duration_seconds", description="YOLO inference latency", unit="s"
)

detection_latency = meter.create_histogram(
    "detection_duration_seconds", description="Total detection latency (preprocessing + inference)", unit="s"
)

detection_counter = meter.create_counter(
    "detections_total", description="Total detections per class"
)
confidence_hist = meter.create_histogram(
    "detection_confidence", description="YOLO confidence scores", unit="1"
)
corrupt_frames = meter.create_counter(
    "corrupt_frames_total", description="Frames rejected by preprocessing"
)
frame_brightness = meter.create_histogram(
    "frame_brightness", description="Mean pixel brightness", unit="1"
)
frames_processed = meter.create_counter(
    "frames_processed_total", description="Total frames processed"
)

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    from ultralytics import YOLO
    logger.info("Loading YOLO v8n model…")
    MODEL = YOLO(MODEL_NAME + ".pt")
    logger.info("Model loaded.")

# ── Frame sources ─────────────────────────────────────────────────────────────
def webcam_frames():
    """Yield JPEG-encoded frames from /dev/video0."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam — falling back to synthetic mode")
        yield from synthetic_frames()
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed, retrying…")
                time.sleep(0.1)
                continue
            logger.debug("Captured frame from webcam (size=%dx%d)", frame.shape[1], frame.shape[0])
            
            #resize for faster processing
            frame = cv2.resize(frame, (640, 480))

            logger.debug("Resized frame from webcam (size=%dx%d)", frame.shape[1], frame.shape[0])

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield buf.tobytes()

            
            time.sleep(1 / 10)  # ~10 fps
    finally:
        cap.release()


LITTER_CLASSES = ["bottle", "can", "paper", "plastic bag", "cigarette", "cup"]

def synthetic_frames():
    """Yield synthetic frame payloads (noise images) for demo without webcam."""
    logger.info("Running in synthetic camera mode")
    frame_id = 0
    while True:
        # Generate a plausible-looking noisy BGR frame
        h, w = 480, 640
        img = np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield buf.tobytes()
        frame_id += 1
        time.sleep(0.1)  # 10 fps


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_frame(image_bytes: bytes) -> np.ndarray | None:
    with tracer.start_as_current_span("preprocess-frame") as span:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                corrupt_frames.add(1, {"reason": "decode_error"})
                span.set_status(trace.StatusCode.ERROR, "Failed to decode frame")
                return None
            brightness = float(img.mean()) / 255.0
            frame_brightness.record(brightness)
            span.set_attribute("frame.size_bytes", len(image_bytes))
            span.set_attribute("frame.brightness", round(brightness, 3))

            #── SLOW_MODE: uncomment to simulate intermittent slow preprocessing ──
            if random.random() < 0.20:
                time.sleep(0.5)
                logger.warning("Simulated slow preprocessing for this frame")
                span.set_attribute("slow_preprocessing", True)
                span.add_event("Simulated slow preprocessing", {"duration_ms": 500})

            return img
        except Exception as exc:
            corrupt_frames.add(1, {"reason": "exception"})
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            raise


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(img: np.ndarray, frame_bytes: bytes) -> dict:
    with tracer.start_as_current_span("yolo-inference") as span:
        span.set_attribute("frame.size_bytes", len(frame_bytes))
        span.set_attribute("model.name", "MODEL_NAME")

        t0 = time.perf_counter()
        results = MODEL(img, verbose=False)
        duration = time.perf_counter() - t0

        boxes = results[0].boxes
        conf_scores = boxes.conf.tolist() if len(boxes) > 0 else []
        class_ids = boxes.cls.tolist() if len(boxes) > 0 else []

        span.set_attribute("detections.count", len(conf_scores))
        span.set_attribute("inference.duration_ms", round(duration * 1000, 1))

        inference_latency.record(duration, {"model": "MODEL_NAME"})
        frames_processed.add(1)

        detections = []
        for conf, cls_id in zip(conf_scores, class_ids):
            cls_name = MODEL.names[int(cls_id)]
            detection_counter.add(1, {"class": cls_name})
            confidence_hist.record(conf, {"model": "MODEL_NAME"})
            detections.append({"class": cls_name, "confidence": round(conf, 3)})

        return {
            "detections": detections,
            "latency_ms": round(duration * 1000, 1),
            "model": "MODEL_NAME",
        }


# ── Synthetic inference (no YOLO) ─────────────────────────────────────────────
def run_synthetic_inference() -> dict:
    """Produce plausible detection results without a real model."""
    with tracer.start_as_current_span("yolo-inference") as span:
        t0 = time.perf_counter()
        time.sleep(random.gauss(0.04, 0.005))  # ~40ms ± 5ms
        duration = time.perf_counter() - t0

        n = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        detections = []
        for _ in range(n):
            cls = random.choice(LITTER_CLASSES)
            conf = max(0.25, min(0.99, random.gauss(0.72, 0.12)))
            detection_counter.add(1, {"class": cls})
            confidence_hist.record(conf, {"model": "synthetic"})
            detections.append({"class": cls, "confidence": round(conf, 3)})

        span.set_attribute("detections.count", n)
        span.set_attribute("inference.duration_ms", round(duration * 1000, 1))
        span.set_attribute("model.name", "synthetic")
        inference_latency.record(duration, {"model": "synthetic"})
        frames_processed.add(1)
        return {"detections": detections, "latency_ms": round(duration * 1000, 1), "model": "synthetic"}


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    camera_mode = os.getenv("CAMERA_MODE", "webcam").lower()
    use_real_model = camera_mode == "webcam"

    logger.info(f"Starting detector with CAMERA_MODE={camera_mode}")

    if use_real_model:
        load_model()

    zenoh_router = os.getenv("ZENOH_ROUTER", "localhost:7447")
    logger.info(f"Connecting to Zenoh router at {zenoh_router}…")

    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([f"tcp/{zenoh_router}"]))
    session = zenoh.open(conf)
    logger.info("Zenoh session open. Starting detection loop.")

    frame_source = webcam_frames() if use_real_model else synthetic_frames()

    for frame_bytes in frame_source:
        with tracer.start_as_current_span("process-frame") as root_span:
            try:

                overall_t0 = time.perf_counter()


                if use_real_model:
                    img = preprocess_frame(frame_bytes)
                    if img is None:
                        continue
                    result = run_inference(img, frame_bytes)
                else:
                    result = run_synthetic_inference()

                n = len(result["detections"])
                logger.info(
                    f"Processed frame: {n} detection(s) in {result['latency_ms']:.1f} ms"
                )
                root_span.set_attribute("detections.count", n)

                session.put(
                    "litter/detections",
                    json.dumps(result).encode(),
                    encoding=zenoh.Encoding.APPLICATION_JSON,
                )
                # Publish the raw JPEG frame so the dashboard can display it
                session.put(
                    "litter/frame",
                    frame_bytes,
                    encoding=zenoh.Encoding.IMAGE_JPEG
                )

                overal_latency = time.perf_counter() - overall_t0
                detection_latency.record(overal_latency)
                root_span.set_attribute("detection.duration_ms", round(overal_latency * 1000, 1))
                logger.info(f"Total detection latency (preprocessing + inference): {overal_latency:.3f} s")

            except Exception as exc:
                logger.exception("Frame processing failed")
                root_span.record_exception(exc)
                root_span.set_status(trace.StatusCode.ERROR, str(exc))

    session.close()


if __name__ == "__main__":
    main()
