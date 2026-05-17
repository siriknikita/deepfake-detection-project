"""FastAPI app: model registry + image / video detection.

Endpoints:
- ``GET  /api/health``    — liveness probe.
- ``GET  /api/models``    — the selectable model registry.
- ``POST /api/detect``    — multipart upload (``file`` + repeated
  ``model_ids``). Images are scored synchronously and the result is
  returned inline. Videos are slow (a physics solve per sampled frame),
  so a background job is created and ``{"job_id": ...}`` is returned.
- ``GET  /api/jobs/{id}`` — poll a video job's status / result.

Local/LAN demo posture: permissive CORS, no auth. The Vite dev server
proxies ``/api`` here so the browser talks same-origin in dev.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forge_detect.serving_infer import (
    InferenceError,
    infer_image,
    infer_video,
    list_models,
)

from .jobs import JOBS

app = FastAPI(title="Hyperplane-Forge Detector", version="0.1.0")

# Local demo: allow any localhost origin so `vite dev` (5173) and any
# LAN device can reach the API without per-origin config.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\]|.*\.local)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _is_video(file: UploadFile) -> bool:
    if (file.content_type or "").startswith("video/"):
        return True
    name = (file.filename or "").lower()
    return name.endswith(_VIDEO_EXTS)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> list[dict[str, object]]:
    return list_models()


@app.post("/api/detect")
async def detect(
    file: Annotated[UploadFile, File(description="Image or video to analyse")],
    model_ids: Annotated[list[str], Form(description="Selected model ids")],
) -> JSONResponse:
    """Score an upload through every selected model.

    Image → inline ``{filename, results}``. Video → ``{job_id, kind}``;
    poll ``GET /api/jobs/{job_id}``. Client-fixable problems (unknown
    model, undecodable file, no ffmpeg) return 400; server faults 500.
    """
    payload = await file.read()
    name = file.filename

    if _is_video(file):
        job = JOBS.submit(
            lambda progress: infer_video(payload, model_ids, progress),
        )
        return JSONResponse(content={"job_id": job.id, "kind": "video", "filename": name})

    try:
        result = infer_image(payload, model_ids)
    except InferenceError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 — surface any fault as JSON, not HTML
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"filename": name, "results": result})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """Poll a video job. 404 if unknown; otherwise its status view."""
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "unknown job id"})
    view: dict[str, Any] = job.view()
    return JSONResponse(content=view)
