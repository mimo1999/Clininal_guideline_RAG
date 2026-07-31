"""Entrypoint: `python -m webapp.run_webapp`.

Runs uvicorn programmatically (not via `uvicorn webapp.main:app` on the CLI)
specifically so the app can stash a reference to the running Server object
on app.state -- that's what lets the /api/shutdown route trigger a real
graceful shutdown (server.should_exit = True) from inside a request handler,
the portable way to do this from within uvicorn's own event loop rather than
reaching for a raw os.kill signal.
"""

from __future__ import annotations

import uvicorn

from webapp.main import app


def main() -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="info")
    server = uvicorn.Server(config)
    app.state.server = server
    server.run()


if __name__ == "__main__":
    main()
