"""Async HTTP polling API for compute-task progress (asyncio, stdlib only --
no new dependency, consistent with the project's fully-local/offline
constraint; binds localhost only).

This is a SEPARATE process from any ingestion/chunking/index-build/eval job.
It only reads the small JSON files those jobs write via common/progress.py --
it never touches GPU/CPU compute, so polling it cannot slow down or disrupt
whatever it's reporting on.

Usage:
    python -m common.progress_server                    # serves on :8765
    curl http://127.0.0.1:8765/progress                  # every tracked task
    curl http://127.0.0.1:8765/progress/<task_id>        # one task
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

from .progress import list_progress, read_progress

DEFAULT_PORT = 8765


def _response(status: int, body: dict | list) -> bytes:
    payload = json.dumps(body).encode("utf-8")
    reason = "OK" if status == 200 else "Not Found" if status == 404 else "Internal Server Error"
    headers = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8")
    return headers + payload


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line = await reader.readline()
        # Drain and discard the rest of the request headers -- GET-only API,
        # no request body is ever expected.
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break

        parts = request_line.decode("utf-8", errors="ignore").split()
        path = urlparse(parts[1]).path if len(parts) >= 2 else "/"

        if path == "/progress":
            writer.write(_response(200, list_progress()))
        elif path.startswith("/progress/"):
            task_id = path[len("/progress/"):]
            result = read_progress(task_id)
            if result is None:
                writer.write(_response(404, {"error": f"no progress for task_id={task_id!r}"}))
            else:
                writer.write(_response(200, result))
        else:
            writer.write(_response(404, {"error": "unknown path", "try": ["/progress", "/progress/<task_id>"]}))
    except Exception as e:
        writer.write(_response(500, {"error": str(e)}))
    finally:
        await writer.drain()
        writer.close()


async def serve(port: int = DEFAULT_PORT) -> None:
    server = await asyncio.start_server(_handle, host="127.0.0.1", port=port)
    print(f"Progress polling API on http://127.0.0.1:{port}/progress")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    import sys

    chosen_port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    asyncio.run(serve(chosen_port))
