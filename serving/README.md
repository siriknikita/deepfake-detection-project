# Web app — backend (`serving/`) + frontend (`web/`)

A local/LAN demo UI for comparing the project's deepfake detectors on a
single image or video. The backend is a thin FastAPI layer over
[`forge_detect.serving_infer`](../python/forge_detect/serving_infer.py)
(the model registry + inference adapter); the frontend is a Vite + React
SPA under [`web/`](../web).

```
web/  ──HTTP /api──▶  serving/app.py  ──in-process──▶  forge_detect.serving_infer
                                                       └─ model registry (4 models)
                                                       └─ infer_image / infer_video
```

## One-time install

```bash
make serve-install     # uv sync --extra serve  (FastAPI, uvicorn)
make web-install       # npm install in web/
```

The CNN models also need the Rust extension and the local checkpoints
(`runs_mac/...`); the heuristic model needs only `make dev-install`.

## Run (two terminals)

```bash
make serve             # backend  → http://127.0.0.1:8000
make web               # frontend → http://localhost:5173
```

The Vite dev server proxies `/api` to the backend, so the browser stays
same-origin. Open the frontend URL and the model list loads from
`GET /api/models`.

## API

| Method | Path          | Purpose                                            |
|--------|---------------|----------------------------------------------------|
| GET    | `/api/health` | Liveness probe.                                    |
| GET    | `/api/models` | Selectable model registry + per-model availability.|
| POST   | `/api/detect` | Multipart `file` + repeated `model_ids`. Image → inline `{filename, results}`. Video → `{job_id, kind}`. |
| GET    | `/api/jobs/{id}` | Poll a video job: `status`, `done`/`total`, `result`/`error`. |

## Status

- **Phase 0 (done):** scaffold, registry, `/api/models`, frontend boots.
- **Phase 1 (done):** heuristic single-image path — `detect()` + the
  6-panel manifold figure.
- **Phase 2 (done):** CNN single-image inference adapter — MTCNN
  face-crop, one shared physics solve, frequency maps, channel assembly
  through the public `ChannelSource` contract, all 3 checkpoints scored.
- **Phase 3 (done):** video input — ffmpeg samples `_VIDEO_FRAMES`
  evenly-spaced frames, each scored, mean-pooled to a video verdict;
  visuals rendered for the most-suspicious frame; per-frame P(fake)
  returned for a timeline sparkline. Runs as a background job
  (`serving/jobs.py`) with progress polling.
- **Phase 4:** UX polish + error handling.

`POST /api/detect` (image) returns: `input` (face crop), `face_detected`,
`peak` (highest score + which model), and `models[]` — each with
`score`/`verdict`, the heuristic `panel` figure, and `channels` previews
of what the model sees. The heuristic row is diagnostic-only (no
calibrated score without the Phase-1 GBC pickle, which is not on this
checkout). Client errors (bad image, unknown/unavailable model) → `400`.
