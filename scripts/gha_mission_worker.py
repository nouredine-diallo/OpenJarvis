#!/usr/bin/env python3
"""JARVIS Phase A, cloud fallback worker (runs inside GitHub Actions).

Deliberately minimal: a single Groq call answering/researching the
mission's goal, reported back to the control plane. This is NOT the full
Mission Engine's multi-phase coding_pr pipeline (Setup/Implement/Test/
Review/Ship) -- that stays a PC-only capability for now, since it needs
real shell execution and the Claude/Gemini *subscription* CLIs, which
authenticate via a browser OAuth session tied to the PC and have no
credentials available in a fresh, ephemeral GitHub Actions runner.

What this proves and provides: JARVIS keeps answering research/reasoning
questions even when the PC is completely off, using only stdlib (no
dependency install step, so the job starts fast) and the same free Groq
model tier the PC uses by default.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

GROQ_MODEL = "qwen/qwen3.6-27b"  # matches config.toml's default_model


def ask_groq(api_key: str, goal: str) -> str:
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        method="POST",
        data=json.dumps(
            {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": goal}],
                "max_tokens": 2048,
            }
        ).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        payload = json.loads(resp.read())
    raw = payload["choices"][0]["message"]["content"]
    import re

    return re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()


def report_completion(control_plane_url: str, shared_secret: str, mission_id: str, status: str, report: str) -> None:
    url = control_plane_url.rstrip("/") + f"/missions/{mission_id}/complete"
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps({"status": status, "report": report}).encode(),
        headers={
            "content-type": "application/json",
            "x-control-plane-secret": shared_secret,
            "user-agent": "JARVIS-GHA-Worker/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as resp:
        resp.read()


def main() -> int:
    mission_id = os.environ["MISSION_ID"]
    goal = os.environ["MISSION_GOAL"]
    groq_key = os.environ["GROQ_API_KEY"]
    control_plane_url = os.environ["CONTROL_PLANE_URL"]
    shared_secret = os.environ["CONTROL_PLANE_SHARED_SECRET"]

    try:
        answer = ask_groq(groq_key, goal)
        status = "SUCCEEDED"
        report = answer
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as exc:
        status = "FAILED"
        report = f"Cloud worker (GitHub Actions) error: {exc}"

    print(f"[gha-worker] mission={mission_id} status={status}")
    print(report)

    try:
        report_completion(control_plane_url, shared_secret, mission_id, status, report)
    except Exception as exc:  # noqa: BLE001
        print(f"[gha-worker] failed to report completion: {exc}", file=sys.stderr)
        return 1

    return 0 if status == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
