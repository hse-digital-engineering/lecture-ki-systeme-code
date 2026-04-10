"""
FastAPI service for the litter detection system.

- Subscribes to Zenoh detections in a background thread
- Persists detections to PostgreSQL
- Exposes REST endpoints for the Streamlit dashboard
- Instrumented with OpenTelemetry (auto-instrumentation via opentelemetry-instrument wrapper)
"""
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import zenoh
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from opentelemetry import trace
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("app")
tracer = trace.get_tracer("fastapi-service")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://admin:secret@localhost:5432/litter_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    class_name = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=True)
    model = Column(String(64), nullable=True)


def init_db(retries: int = 10, delay: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(engine)
            logger.info("Database schema ready.")
            return
        except Exception as exc:
            logger.warning(f"DB not ready yet (attempt {attempt}/{retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to database after repeated attempts.")


# ── Zenoh subscriber ──────────────────────────────────────────────────────────
_latest: dict = {"detections": [], "latency_ms": 0.0, "model": "unknown", "ts": None}
_latest_frame: bytes | None = None
_history: list[dict] = []
_history_lock = threading.Lock()

ZENOH_ROUTER = os.getenv("ZENOH_ROUTER", "localhost:7447")


def _start_zenoh_subscriber():
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", json.dumps([f"tcp/{ZENOH_ROUTER}"]))
    session = zenoh.open(conf)
    logger.info(f"Zenoh subscriber connected to {ZENOH_ROUTER}")

    def on_detection(sample):
        global _latest
        with tracer.start_as_current_span("zenoh-receive") as span:
            try:
                payload = json.loads(bytes(sample.payload))
                payload["ts"] = datetime.now(timezone.utc).isoformat()
                _latest = payload

                n = len(payload.get("detections", []))
                span.set_attribute("detections.count", n)
                span.set_attribute("message.key", str(sample.key_expr))

                db: Session = SessionLocal()
                try:
                    for det in payload.get("detections", []):
                        row = Detection(
                            class_name=det["class"],
                            confidence=det["confidence"],
                            latency_ms=payload.get("latency_ms"),
                            model=payload.get("model"),
                        )
                        db.add(row)
                    db.commit()
                finally:
                    db.close()

                with _history_lock:
                    _history.append(payload)
                    if len(_history) > 500:
                        _history.pop(0)

                logger.info(
                    f"Received {n} detection(s) from Zenoh, latency={payload.get('latency_ms')}ms"
                )
            except Exception as exc:
                logger.exception("Failed to handle Zenoh message")
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))

    def on_frame(sample):
        global _latest_frame
        _latest_frame = bytes(sample.payload)

    session.declare_subscriber("litter/detections", on_detection)
    session.declare_subscriber("litter/frame", on_frame)
    logger.info("Zenoh subscribers declared on 'litter/detections' and 'litter/frame'")
    # Keep the subscriber thread alive
    while True:
        time.sleep(1)


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    t = threading.Thread(target=_start_zenoh_subscriber, daemon=True)
    t.start()
    logger.info("FastAPI service started.")
    yield
    logger.info("FastAPI service shutting down.")


app = FastAPI(title="Litter Detection API", lifespan=lifespan)


class HealthResponse(BaseModel):
    status: str
    db: str
    zenoh: str


class DetectionItem(BaseModel):
    class_name: str
    confidence: float
    timestamp: Optional[str] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
def health():
    db_ok = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = "error"

    return {"status": "ok", "db": db_ok, "zenoh": "connected"}


@app.get("/api/frame")
def get_frame():
    """Return the latest JPEG frame from the detector."""
    if _latest_frame is None:
        raise HTTPException(status_code=404, detail="No frame received yet")
    return Response(content=_latest_frame, media_type="image/jpeg")


@app.get("/api/latest")
def get_latest():
    """Return the most recent detection result."""
    return _latest


@app.get("/api/detections", response_model=list[DetectionItem])
def get_detections(limit: int = 50):
    """Return recent detections from the database."""
    with tracer.start_as_current_span("db-query-detections") as span:
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(Detection)
                .order_by(Detection.timestamp.desc())
                .limit(limit)
                .all()
            )
            span.set_attribute("db.result.count", len(rows))
            return [
                DetectionItem(
                    class_name=r.class_name,
                    confidence=r.confidence,
                    timestamp=r.timestamp.isoformat() if r.timestamp else None,
                    latency_ms=r.latency_ms,
                    model=r.model,
                )
                for r in rows
            ]
        finally:
            db.close()


@app.get("/api/stats")
def get_stats():
    """Return aggregate detection statistics."""
    with tracer.start_as_current_span("db-query-stats") as span:
        db: Session = SessionLocal()
        try:
            from sqlalchemy import func

            total = db.query(func.count(Detection.id)).scalar() or 0
            avg_conf = db.query(func.avg(Detection.confidence)).scalar()
            avg_lat = db.query(func.avg(Detection.latency_ms)).scalar()
            by_class = (
                db.query(Detection.class_name, func.count(Detection.id))
                .group_by(Detection.class_name)
                .all()
            )
            span.set_attribute("db.total_detections", total)
            return {
                "total_detections": total,
                "avg_confidence": round(float(avg_conf), 3) if avg_conf else 0.0,
                "avg_latency_ms": round(float(avg_lat), 1) if avg_lat else 0.0,
                "by_class": {cls: cnt for cls, cnt in by_class},
            }
        finally:
            db.close()


@app.delete("/api/detections")
def clear_detections():
    """Clear all detection records (useful for demo resets)."""
    db: Session = SessionLocal()
    try:
        deleted = db.query(Detection).delete()
        db.commit()
        with _history_lock:
            _history.clear()
        return {"deleted": deleted}
    finally:
        db.close()
