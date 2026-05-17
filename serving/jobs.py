"""Minimal in-process job store for slow (video) inference.

Image detection is fast and stays synchronous. Video decodes + scores
many frames (a physics PDE solve per frame), which would time out an
HTTP request, so it runs on a daemon thread and the client polls
``GET /api/jobs/{id}``. Single-process, single-worker uvicorn for the
local/LAN demo — a plain dict + lock is sufficient; no broker needed.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ProgressFn = Callable[[int, int], None]
JobFn = Callable[[ProgressFn], dict[str, Any]]


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, fn: JobFn) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job

        def worker() -> None:
            job.status = "running"

            def progress(done: int, total: int) -> None:
                job.done, job.total = done, total

            try:
                job.result = fn(progress)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — surfaced to the client
                job.error = str(exc)
                job.status = "error"

        threading.Thread(target=worker, daemon=True).start()
        return job


JOBS = JobStore()
