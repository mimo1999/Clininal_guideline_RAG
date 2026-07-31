"""Lightweight progress tracking for the pipeline's compute-heavy loops
(ingestion, chunking, index building, evaluation generation+judging). Each
tracker writes a small JSON file per task_id under data_corpus/progress/ --
that file is the entire interface. Polling it (see progress_server.py) never
touches the running job: it's a separate process reading a file the job
happens to be writing, so it cannot slow down or otherwise disrupt the
compute it's reporting on.

Writes are atomic (write-to-temp + os.replace, which is atomic on both POSIX
and Windows) so a poll never reads a half-written file, and throttled to a
handful per second so frequent advance() calls (e.g. once per embedding
batch) don't turn into their own I/O bottleneck.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock

PROGRESS_DIR = Path(__file__).parent.parent / "data_corpus" / "progress"
_MIN_WRITE_INTERVAL_S = 0.25


class ProgressTracker:
    def __init__(self, task_id: str, total: int = 0, stage: str = "starting"):
        self.task_id = task_id
        self.total = total
        self.current = 0
        self.stage = stage
        self.done = False
        self.error: str | None = None
        self._lock = Lock()
        self._last_write = 0.0
        self._write(force=True)

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = total
        self._write(force=True)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
        self._write(force=True)

    def advance(self, n: int = 1) -> None:
        with self._lock:
            self.current += n
        self._write()

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self.done = True
            self.error = error
            if error is None and self.total:
                self.current = self.total
        self._write(force=True)

    def _write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_write) < _MIN_WRITE_INTERVAL_S:
            return
        self._last_write = now

        with self._lock:
            payload = {
                "task_id": self.task_id,
                "stage": self.stage,
                "current": self.current,
                "total": self.total,
                "pct": round(100 * self.current / self.total, 1) if self.total else None,
                "done": self.done,
                "error": self.error,
                "updated_at": time.time(),
            }

        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = PROGRESS_DIR / f".{self.task_id}.tmp"
        final_path = PROGRESS_DIR / f"{self.task_id}.json"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, final_path)


def read_progress(task_id: str) -> dict | None:
    path = PROGRESS_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Tiny race window if read lands mid os.replace on some platforms --
        # caller just polls again; there's no partial-write state to recover.
        return None


def list_progress() -> dict[str, dict]:
    if not PROGRESS_DIR.exists():
        return {}
    result = {}
    for path in PROGRESS_DIR.glob("*.json"):
        try:
            result[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return result
