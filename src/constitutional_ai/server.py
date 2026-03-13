"""Local HTTP server that provides both the GUI and JSON API endpoints."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from constitutional_ai.config import AppConfig, DEFAULT_CONFIG_PATH, load_config, merge_config, save_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.utils import normalize_chat_history


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    """Write a JSON payload to the active HTTP response."""
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppState:
    """Thread-safe mutable state used by the HTTP handler."""

    def __init__(self, config_path: Path) -> None:
        """Initialize with a config file path and load current state."""
        self.config_path = config_path
        self._lock = threading.Lock()
        self.config = load_config(config_path)

    def get_config(self) -> AppConfig:
        """Return a deep-copied config snapshot."""
        with self._lock:
            return AppConfig.from_mapping(self.config.to_dict())

    def set_config(self, payload: dict[str, Any]) -> AppConfig:
        """Merge payload into config, persist it, and return updated config."""
        with self._lock:
            self.config = merge_config(self.config, payload)
            save_config(self.config, self.config_path)
            return AppConfig.from_mapping(self.config.to_dict())


class ConstitutionalHandler(BaseHTTPRequestHandler):
    """HTTP request handler for static GUI files and API endpoints."""

    server: "ConstitutionalHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        """Serve static assets and the config endpoint."""
        if self.path in {"/", "/index.html"}:
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if self.path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
            return
        if self.path == "/api/config":
            config = self.server.state.get_config().to_dict()
            _json_response(self, HTTPStatus.OK, {"ok": True, "config": config})
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        """Handle API mutation routes."""
        payload = self._read_json_body()
        if payload is None:
            return

        if self.path == "/api/turn":
            self._handle_turn(payload)
            return

        if self.path == "/api/config":
            updated = self.server.state.set_config(payload).to_dict()
            _json_response(self, HTTPStatus.OK, {"ok": True, "config": updated})
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        """Write concise request logs to stdout."""
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _read_json_body(self) -> dict[str, Any] | None:
        """Decode and validate a JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid Content-Length."})
            return None

        body = self.rfile.read(content_length)
        try:
            parsed = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Body must be valid JSON."})
            return None

        if not isinstance(parsed, dict):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "JSON body must be an object."})
            return None
        return parsed

    def _serve_static(self, name: str, content_type: str) -> None:
        """Serve a static file packaged under constitutional_ai/static."""
        resource = files("constitutional_ai.static").joinpath(name)
        content = resource.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_turn(self, payload: dict[str, Any]) -> None:
        """Execute a constitutional turn using current config plus request overrides."""
        user_text = payload.get("user_text")
        if not isinstance(user_text, str) or not user_text.strip():
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "user_text is required."})
            return

        thread_messages = normalize_chat_history(payload.get("thread_messages", []))
        config = self.server.state.get_config()

        # Allow per-request overrides without mutating persisted server config.
        override_payload = {
            "settings": payload.get("settings"),
            "rules": payload.get("rules"),
            "rules_text": payload.get("rules_text"),
            "prompts": payload.get("prompts"),
        }
        runtime_config = merge_config(config, override_payload)

        try:
            turn = run_constitutional_turn(
                user_text=user_text.strip(),
                thread_messages=thread_messages,
                config=runtime_config,
            )
        except Exception as exc:  # noqa: BLE001
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        _json_response(self, HTTPStatus.OK, {"ok": True, "turn": turn.to_dict()})


class ConstitutionalHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying application state for request handlers."""

    def __init__(self, server_address: tuple[str, int], state: AppState) -> None:
        """Initialize the threaded server with shared mutable state."""
        super().__init__(server_address, ConstitutionalHandler)
        self.state = state


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the GUI server entrypoint."""
    parser = argparse.ArgumentParser(description="Run the constitutional-ai local GUI server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--config", default=None, help="Path to shared config JSON")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the local server and block until interrupted."""
    args = build_parser().parse_args(argv)
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    state = AppState(config_path)

    server = ConstitutionalHTTPServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving constitutional-ai GUI at {url}")
    print(f"Using config file: {config_path}")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
