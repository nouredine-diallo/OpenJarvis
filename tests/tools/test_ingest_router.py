"""Tests for the universal ingestion router (Brique 1,
docs/SPEC_BRIQUE1_INGESTION.md).

Follows this codebase's existing convention (see test_screenshot.py's
test_capture_screenshot_real_end_to_end) of exercising real pipelines
directly rather than mocking everything away -- PDF/image/text/local-repo
fixtures are fast and dependency-free once installed, so they run for
real. Network-dependent pipelines (web fetch, GitHub clone, YouTube) get
one real, lightweight case each for the same reason live coverage caught
real bugs during implementation (a tuple-vs-str mismatch in read_document,
a missing local-directory case in the router) that a fully-mocked suite
would have missed entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.tools.ingest_router import (
    CONTENT_TYPES,
    IngestResult,
    detect_content_type,
    ingest_content,
)


class TestDetectContentType:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("https://github.com/nouredine-diallo/OpenJarvis", "repo"),
            ("https://github.com/nouredine-diallo/OpenJarvis/tree/main", "repo"),
            ("https://github.com/nouredine-diallo/OpenJarvis.git", "repo"),
            ("https://youtube.com/watch?v=abc123", "video"),
            ("https://youtu.be/abc123", "video"),
            ("/tmp/report.pdf", "pdf"),
            ("/tmp/photo.PNG", "image"),
            ("/tmp/photo.jpg", "image"),
            ("/tmp/audio.mp3", "audio"),
            ("/tmp/audio.webm", "audio"),
            ("/tmp/notes.md", "text"),
            ("https://example.com/page", "web"),
            ("", "unknown"),
            ("   ", "unknown"),
            ("random gibberish with no markers", "unknown"),
        ],
    )
    def test_classification(self, source, expected):
        assert detect_content_type(source) == expected

    def test_local_directory_is_repo(self, tmp_path: Path):
        """Spec §3.1 covers 'repo GitHub OU dossier de code' -- a local
        folder must classify as repo too, not just GitHub URLs. Found
        missing live while testing against a real local repo fixture."""
        (tmp_path / "src").mkdir()
        assert detect_content_type(str(tmp_path)) == "repo"

    def test_nonexistent_local_path_is_unknown(self):
        assert detect_content_type("/does/not/exist/at/all") == "unknown"

    def test_all_content_types_are_reachable(self):
        """CONTENT_TYPES is the public contract -- every entry except
        'unknown' should be producible by some real input."""
        assert set(CONTENT_TYPES) == {
            "repo", "pdf", "video", "image", "audio", "web", "text", "unknown",
        }


class TestIngestContentDispatch:
    def test_unknown_type_fails_cleanly(self):
        result = ingest_content("gibberish, not a real source")
        assert result.success is False
        assert result.content_type == "unknown"
        assert result.error

    def test_pipeline_exception_never_propagates(self, monkeypatch):
        """ingest_content must never raise -- a broken pipeline becomes a
        failed IngestResult, not an exception into the caller (the agent's
        tool-calling loop, or a mission step)."""
        import openjarvis.tools.ingest_router as router

        def _boom(source, *, workspace=""):
            raise RuntimeError("pipeline exploded")

        monkeypatch.setitem(
            router.ingest_content.__globals__, "_ingest_text", _boom
        )
        # Force dispatch through the text pipeline via a .md path.
        result = ingest_content("/tmp/whatever.md")
        assert result.success is False
        assert "pipeline exploded" in result.error or "RuntimeError" in result.error

    def test_successful_result_is_credential_stripped(self, tmp_path: Path):
        secret_file = tmp_path / "notes.md"
        secret_file.write_text("token: ghp_" + "a" * 36)
        result = ingest_content(str(secret_file))
        assert result.success is True
        assert "ghp_" not in result.content
        assert "[REDACTED:github_token]" in result.content


class TestPdfPipeline:
    def _make_pdf(self, tmp_path: Path, text: str) -> Path:
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        out = tmp_path / "test.pdf"
        doc.save(str(out))
        doc.close()
        return out

    def test_real_pdf_text_extraction(self, tmp_path: Path):
        pdf_path = self._make_pdf(tmp_path, "Never deploy on a Friday")
        result = ingest_content(str(pdf_path))
        assert result.success is True
        assert result.mode == "extrait"
        assert "deploy" in result.content.lower()
        assert result.metadata["tool"] == "pymupdf"

    def test_missing_pdf_fails_cleanly(self):
        result = ingest_content("/no/such/file.pdf")
        assert result.success is False
        assert "introuvable" in result.error.lower()

    def test_ram_guard_blocks_scanned_pdf_ocr(self, tmp_path: Path, monkeypatch):
        import openjarvis.tools.ingest_router as router

        monkeypatch.setattr(router, "_ram_guard", lambda min_mb, name: f"RAM insuffisante pour {name}")
        # A PDF with zero extractable text forces the scanned/OCR branch.
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()  # blank page, no text layer
        pdf_path = tmp_path / "scanned.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = ingest_content(str(pdf_path))
        assert result.success is False
        assert "RAM" in result.error


class TestImagePipeline:
    def _make_image(self, tmp_path: Path, text: str) -> Path:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (400, 100), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 40), text, fill="black")
        out = tmp_path / "test.png"
        img.save(out)
        return out

    def test_real_image_ocr(self, tmp_path: Path):
        img_path = self._make_image(tmp_path, "Never deploy on a Friday")
        result = ingest_content(str(img_path))
        assert result.success is True
        assert result.mode == "ocr"
        assert len(result.content) > 0
        assert result.metadata["needs_vision_escalation"] is False

    def test_missing_image_fails_cleanly(self):
        result = ingest_content("/no/such/file.png")
        assert result.success is False

    def test_low_ocr_yield_signals_vision_escalation(self, tmp_path: Path):
        """Decision D1: OCR first, vision escalation only when OCR yields
        little -- verified via the actual threshold logic, not mocked."""
        # A blank image: OCR should find ~0 characters.
        from PIL import Image

        blank = tmp_path / "blank.png"
        Image.new("RGB", (200, 100), color="white").save(blank)
        result = ingest_content(str(blank))
        assert result.success is True
        assert result.metadata["needs_vision_escalation"] is True
        assert result.mode == "ocr_insuffisant"


class TestRepoPipeline:
    def test_local_directory_via_repomix(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')\n")
        result = ingest_content(str(tmp_path))
        assert result.success is True
        assert result.content_type == "repo"
        assert result.metadata["tool"] == "repomix"
        assert "main.py" in result.content

    def test_real_small_public_github_repo(self):
        """One real network case, deliberately a tiny well-known repo to
        keep this fast -- same rationale as the module docstring."""
        result = ingest_content("https://github.com/octocat/Hello-World")
        assert result.success is True
        assert result.content_type == "repo"
        assert len(result.content) > 0

    def test_ram_guard_blocks_repomix(self, monkeypatch, tmp_path: Path):
        import openjarvis.tools.ingest_router as router

        monkeypatch.setattr(router, "_ram_guard", lambda min_mb, name: "RAM insuffisante")
        result = ingest_content(str(tmp_path))
        assert result.success is False
        assert "RAM" in result.error


class TestWebPipeline:
    def test_real_fetch(self):
        result = ingest_content("https://example.com")
        assert result.success is True
        assert result.content_type == "web"
        assert len(result.content) > 0

    def test_ssrf_blocked_url_fails_cleanly(self):
        result = ingest_content("http://169.254.169.254/latest/meta-data/")
        assert result.success is False
        assert "SSRF" in result.error


class TestTextPipeline:
    def test_real_markdown_file(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("# Title\n\nSome content.")
        result = ingest_content(str(f))
        assert result.success is True
        assert "Some content" in result.content
        assert result.metadata["file_type"] == "markdown"

    def test_missing_file_fails_cleanly(self):
        result = ingest_content("/no/such/file.md")
        assert result.success is False


class TestAudioPipeline:
    def _make_wav(self, tmp_path: Path, seconds: float = 1.0) -> Path:
        import struct
        import wave

        out = tmp_path / "test.wav"
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            n_frames = int(16000 * seconds)
            wf.writeframes(struct.pack("<%dh" % n_frames, *([0] * n_frames)))
        return out

    def test_real_local_audio_transcription(self, tmp_path: Path):
        """Silence transcribes to empty/near-empty text, but must not
        error -- this proves the faster-whisper wiring (decision: wire
        the local provider, previously 'not yet implemented') actually
        works end-to-end, not just that the file is readable."""
        wav_path = self._make_wav(tmp_path)
        result = ingest_content(str(wav_path))
        assert result.success is True
        assert result.mode == "transcription"
        assert result.metadata["language"] is not None

    def test_missing_audio_fails_cleanly(self):
        result = ingest_content("/no/such/file.mp3")
        assert result.success is False

    def test_ram_guard_blocks_audio_transcription(self, tmp_path: Path, monkeypatch):
        import openjarvis.tools.ingest_router as router

        monkeypatch.setattr(router, "_ram_guard", lambda min_mb, name: "RAM insuffisante")
        wav_path = self._make_wav(tmp_path)
        result = ingest_content(str(wav_path))
        assert result.success is False
        assert "RAM" in result.error


class TestVideoPipeline:
    def test_real_video_with_subtitles(self):
        """'Me at the zoo' -- the first YouTube video, 19s, has auto
        captions. Chosen deliberately for speed and stability as a test
        fixture (unlikely to ever be removed).

        Skips (doesn't fail) on YouTube's own bot/rate-limit blocking
        rather than asserting success -- found live: repeated real
        requests during development triggered "HTTP Error 429" / "Sign in
        to confirm you're not a bot", which is YouTube's behavior, not a
        bug in this pipeline (which correctly returned a graceful
        success=False with a clear error in that case)."""
        result = ingest_content("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        if not result.success and (
            "429" in result.error or "not a bot" in result.error.lower()
        ):
            pytest.skip(f"YouTube blocked the request (rate limit/bot check): {result.error[:150]}")
        assert result.success is True
        assert result.mode == "sous-titres"
        assert result.metadata["title"]
        assert len(result.content) > 20

    def test_metadata_only_never_claims_content_was_analyzed(self, monkeypatch):
        """Anti-hallucination (PROJET_JARVIS.md §3, spec §3.4): if neither
        subtitles nor transcription succeed, the result must say so
        explicitly, not silently look like analyzed content."""
        import openjarvis.tools.ingest_router as router

        def _fake_extract_info(url, *, download=False):
            return {
                "title": "A video with no captions",
                "duration": 120,
                "uploader": "someone",
                "upload_date": "20260101",
                "subtitles": {},
                "automatic_captions": {},
            }

        class _FakeYdl:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                return _fake_extract_info(url, download=download)

        monkeypatch.setattr(router, "_ram_guard", lambda min_mb, name: "skip transcription in this test")

        import yt_dlp

        monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda opts=None: _FakeYdl())

        result = router._ingest_video("https://youtube.com/watch?v=fake")
        assert result.success is True
        assert result.mode == "métadonnées"
        assert "NON analysé" in result.content
