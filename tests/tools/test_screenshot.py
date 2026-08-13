"""Tests for the visual proof pipeline (tools/screenshot.py, 2026-08-13)."""

from __future__ import annotations

import http.server
import socket
import threading

from openjarvis.tools.screenshot import (
    capture_screenshot,
    find_dev_server_url,
    ram_available_mb,
)


def test_ram_available_mb_returns_positive_number():
    """Sanity check against the real /proc/meminfo on this machine."""
    mb = ram_available_mb()
    assert mb > 0
    assert mb < 10_000_000  # sane upper bound, not an overflow/parse bug


def test_find_dev_server_url_none_when_nothing_listening():
    # Port 47_291 is not a real dev-server default and almost certainly
    # free in any test environment.
    assert find_dev_server_url("no server mentioned here") is None


def test_find_dev_server_url_parses_explicit_url_from_text():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        url = find_dev_server_url(f"Server started at http://localhost:{port}/")
        assert url == f"http://127.0.0.1:{port}"


def test_find_dev_server_url_probes_candidate_ports_as_fallback():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 3000))
        srv.listen(1)
        # No explicit URL in the text -- must fall back to port probing.
        url = find_dev_server_url("Implémentation terminée, tests passés.")
        assert url == "http://127.0.0.1:3000"


def test_find_dev_server_url_excludes_jarvis_own_port():
    # Port 8000 is JARVIS's own -- must never be mistaken for a mission's
    # dev server, whether or not anything is actually listening there.
    assert find_dev_server_url("http://localhost:8000/health") is None


def test_capture_screenshot_skips_when_ram_too_tight():
    path, reason = capture_screenshot(
        "http://127.0.0.1:1", "/tmp/should_not_be_created.png",
        min_ram_mb=10**9,  # impossibly high -- always fails the guard
    )
    assert path is None
    assert "RAM" in reason


def test_capture_screenshot_reports_failure_cleanly_never_raises():
    # Nothing listens on this port -- must fail gracefully, not raise.
    path, reason = capture_screenshot(
        "http://127.0.0.1:1", "/tmp/should_not_be_created.png",
        min_ram_mb=1.0,  # trivially satisfied so the RAM guard isn't what fails this
        nav_timeout_ms=2000,
    )
    assert path is None
    assert reason  # some human-readable explanation, not empty


def test_capture_screenshot_real_end_to_end(tmp_path):
    """Real integration test: a tiny local HTTP server + real headless
    Chromium (installed 2026-08-13) -> a real PNG file on disk."""
    handler = http.server.SimpleHTTPRequestHandler

    class _Handler(handler):
        def do_GET(self):  # noqa: N802
            body = b"<html><body><h1>JARVIS test page</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence test output
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out_path = str(tmp_path / "shot.png")
        path, reason = capture_screenshot(
            f"http://127.0.0.1:{port}/", out_path, min_ram_mb=1.0
        )
        assert path == out_path, reason
        assert "ok" in reason
        data = open(out_path, "rb").read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG file signature
        assert len(data) > 100  # not an empty/corrupt file
    finally:
        server.shutdown()
        thread.join(timeout=5)
