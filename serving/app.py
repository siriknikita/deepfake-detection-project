"""FastAPI app: model registry + single-image / video detection.

Endpoints:
- ``GET  /api/health``  — liveness probe.
- ``GET  /api/models``  — the selectable model registry.
- ``POST /api/detect``  — multipart upload (``file`` + repeated ``model_ids``)
  run through every selected model; returns a per-model comparison.

Local/LAN demo posture: permissive CORS, no auth. The Vite dev server
also proxies ``/api`` here so the browser talks same-origin in dev.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forge_detect.serving_infer import InferenceError, infer_image, list_models

app = FastAPI(title="Hyperplane-Forge Detector", version="0.1.0")

# Local demo: allow any localhost origin so `vite dev` (5173) and any
# LAN device can reach the API without per-origin config.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\]|.*\.local)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Run the uploaded media through every selected model.

    ``model_ids`` is a repeated form field (one entry per selected model).
    Errors the client can fix (unknown model, no face, bad file) come back
    as 400; genuine server faults as 500. Phase 0 returns 501 from the
    not-yet-implemented adapter — the wiring is exercised end-to-end.
    """
    payload = await file.read()
    try:
        result = infer_image(payload, model_ids)
    except InferenceError as exc:
        message = str(exc)
        status = 501 if "not implemented" in message else 400
        return JSONResponse(status_code=status, content={"error": message})
    except Exception as exc:  # noqa: BLE001 — surface any fault as JSON, not HTML
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"filename": file.filename, "results": result})
