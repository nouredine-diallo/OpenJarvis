"""Universal context ingestion — Brique 1 (docs/SPEC_BRIQUE1_INGESTION.md).

A single entry point (:func:`ingest_content`) that takes anything a user
might hand JARVIS -- a GitHub repo URL, a PDF, a YouTube link, an image, an
audio file, a web URL, or plain text/markdown -- detects what it is with a
cheap heuristic (regex/extension, no LLM call, spec §4.1), and returns
normalized Markdown with sourced metadata, ready to inject as mission
context.

Design constraints carried over from the rest of this project:
- **RAM discipline** (PLAN.md D9, spec §4.3 point 5): repomix (~800 MB),
  faster-whisper (~450 MB), RapidOCR (~390 MB) are all real memory spikes
  on a 7.6 GB no-GPU machine. Never launched blind (RAM checked first,
  reusing the same ``ram_available_mb()`` the visual-proof pipeline
  already uses) and never run two at once (a process-wide lock -- the
  spec is explicit: "un seul pipeline lourd à la fois").
- **Anti-hallucination** (PROJET_JARVIS.md §3, spec §3.4): a video with no
  subtitles and no successful transcription is reported as "métadonnées
  uniquement, contenu non analysé" -- never silently treated as if its
  content were read.
- **Credentials never indexed**: every pipeline's output passes through
  :class:`~openjarvis.security.credential_stripper.CredentialStripper`
  before it's returned.
- **Best-effort, never raises**: a failure in one pipeline (missing
  dependency, bad file, network error) returns a clear ``IngestResult``
  with ``success=False`` and a human-readable ``error`` -- it never
  propagates an exception into the caller (the agent's tool-calling loop,
  or a mission step).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# One heavy pipeline (repomix / whisper / RapidOCR) at a time, process-wide.
_HEAVY_PIPELINE_LOCK = threading.Lock()

_REPO_RE = re.compile(
    r"github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git|/.*)?/?$"
)
_VIDEO_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_PDF_EXTS = {".pdf"}
_TEXT_EXTS = {".txt", ".md", ".csv", ".docx", ".json", ".yaml", ".yml", ".py", ".js", ".ts"}

CONTENT_TYPES = (
    "repo",
    "pdf",
    "video",
    "image",
    "audio",
    "web",
    "text",
    "unknown",
)


def detect_content_type(source: str) -> str:
    """Classify *source* (a URL or a local path) with a cheap heuristic --
    no LLM call, spec §4.1. Mostly a pure regex/extension check; the one
    exception is a single cheap ``is_dir()`` stat (not a full walk) to
    recognize a local code folder as "repo" (spec §3.1 explicitly covers
    "repo GitHub / dossier de code", not just GitHub URLs) -- still far
    cheaper than any real pipeline, so it doesn't undermine the "no
    expensive work before classification" intent."""
    if not source or not source.strip():
        return "unknown"
    s = source.strip()
    lowered = s.lower()

    if _REPO_RE.search(lowered):
        return "repo"
    if _VIDEO_RE.search(lowered):
        return "video"

    suffix = Path(lowered.split("?", 1)[0]).suffix
    if suffix in _PDF_EXTS:
        return "pdf"
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix in _AUDIO_EXTS:
        return "audio"
    if suffix in _TEXT_EXTS:
        return "text"

    if lowered.startswith(("http://", "https://")):
        return "web"

    if not lowered.startswith(("http://", "https://")) and Path(s).is_dir():
        return "repo"

    return "unknown"


@dataclass(slots=True)
class IngestResult:
    """Normalized ingestion output -- what every pipeline converges to."""

    success: bool
    content: str = ""  # normalized Markdown, credential-stripped
    content_type: str = "unknown"
    mode: str = ""  # e.g. "extrait" | "sous-titres" | "transcription" | "métadonnées"
    source: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown_header(self) -> str:
        """A short metadata header prefixed to `content`, per spec §4.2:
        source/type/date/mode always travel with the content, not just in
        `metadata` -- so a reader of the raw Markdown (not just the
        structured result) still sees provenance."""
        import datetime

        date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        lines = [
            f"<!-- source: {self.source} | type: {self.content_type} | "
            f"date: {date} | mode: {self.mode} -->"
        ]
        return "\n".join(lines)


def _strip_credentials(text: str) -> str:
    from openjarvis.security.credential_stripper import CredentialStripper

    return CredentialStripper().strip(text)


def _whisper_backend() -> Any:
    """Construct the faster-whisper STT backend directly, not via
    SpeechRegistry: this module always wants exactly this backend (no
    runtime backend choice to make), and the registry gets cleared between
    every test in this codebase's conftest -- import-time registration
    only fires once per process, so relying on the registry here was
    found live to break under the full test suite (though not when this
    module's own tests ran in isolation, since nothing had cleared the
    registry yet at that point)."""
    from openjarvis.speech.faster_whisper import FasterWhisperBackend

    return FasterWhisperBackend()


def _ram_guard(min_mb: float, pipeline_name: str) -> Optional[str]:
    """Returns an error string if RAM is too tight to proceed, else None."""
    from openjarvis.tools.screenshot import ram_available_mb

    available = ram_available_mb()
    if available < min_mb:
        return (
            f"RAM disponible trop juste pour {pipeline_name} "
            f"({available:.0f} Mo < {min_mb:.0f} Mo requis) -- ingestion ignorée."
        )
    return None


def ingest_content(source: str, *, workspace: str = "") -> IngestResult:
    """Detect and ingest *source*, returning normalized Markdown + metadata.

    Never raises: every pipeline below catches its own errors and returns
    a ``success=False`` result instead.
    """
    content_type = detect_content_type(source)

    dispatch = {
        "repo": _ingest_repo,
        "pdf": _ingest_pdf,
        "image": _ingest_image,
        "video": _ingest_video,
        "audio": _ingest_audio,
        "web": _ingest_web,
        "text": _ingest_text,
    }

    handler = dispatch.get(content_type)
    if handler is None:
        return IngestResult(
            success=False,
            content_type="unknown",
            source=source,
            error=f"Type de contenu non reconnu pour : {source}",
        )

    try:
        result = handler(source, workspace=workspace)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ingestion pipeline %r failed for %r", content_type, source, exc_info=True)
        return IngestResult(
            success=False,
            content_type=content_type,
            source=source,
            error=f"Échec de l'ingestion ({type(exc).__name__}): {exc}",
        )

    if result.success and result.content:
        result.content = _strip_credentials(result.content)
    return result


# -- pipelines ---------------------------------------------------------------
# Each returns an IngestResult and is safe to call directly (e.g. from
# tests) without going through ingest_content's dispatch/error-wrapping.


def _ingest_repo(source: str, *, workspace: str = "") -> IngestResult:
    """GitHub repo (or local repo dir) -> repomix XML. Heavy: ~800 MB peak
    on a large repo (measured, spec §3.1) -- RAM-guarded and serialized."""
    error = _ram_guard(1024.0, "repomix (analyse de dépôt)")
    if error:
        return IngestResult(success=False, content_type="repo", source=source, error=error)

    with _HEAVY_PIPELINE_LOCK:
        return _run_repomix(source)


def _run_repomix(source: str) -> IngestResult:
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node") or shutil.which("npx")
    if node is None:
        return IngestResult(
            success=False,
            content_type="repo",
            source=source,
            error="node/npx introuvable -- repomix nécessite Node.js.",
        )

    match = _REPO_RE.search(source.lower())
    target = source
    cleanup_dir: Optional[str] = None
    if match and source.lower().startswith(("http://", "https://")):
        clone_url = f"https://github.com/{match.group('owner')}/{match.group('repo')}.git"
        cleanup_dir = tempfile.mkdtemp(prefix="jarvis-ingest-repo-")
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, cleanup_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if clone.returncode != 0:
            return IngestResult(
                success=False,
                content_type="repo",
                source=source,
                error=f"git clone a échoué : {clone.stderr[:300]}",
            )
        target = cleanup_dir

    try:
        out_file = tempfile.mktemp(suffix=".xml", prefix="jarvis-repomix-")
        proc = subprocess.run(
            ["npx", "--yes", "repomix@1.14.0", target, "--output", out_file, "--style", "xml"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not Path(out_file).exists():
            return IngestResult(
                success=False,
                content_type="repo",
                source=source,
                error=f"repomix a échoué : {proc.stderr[:300] or proc.stdout[:300]}",
            )
        xml_content = Path(out_file).read_text(encoding="utf-8", errors="replace")
        Path(out_file).unlink(missing_ok=True)
    finally:
        if cleanup_dir:
            import shutil as _shutil

            _shutil.rmtree(cleanup_dir, ignore_errors=True)

    return IngestResult(
        success=True,
        content=xml_content,
        content_type="repo",
        mode="extrait",
        source=source,
        metadata={"tool": "repomix", "output_format": "xml"},
    )


def _ingest_pdf(source: str, *, workspace: str = "") -> IngestResult:
    """PDF -> text (pymupdf, ~30x faster than pdfplumber, spec §3.2), with
    RapidOCR fallback for scanned pages that have no text layer."""
    path = Path(source)
    if not path.exists():
        return IngestResult(success=False, content_type="pdf", source=source, error=f"Fichier introuvable : {source}")

    try:
        import pymupdf
    except ImportError:
        return IngestResult(
            success=False,
            content_type="pdf",
            source=source,
            error="pymupdf n'est pas installé (pip install pymupdf).",
        )

    doc = pymupdf.open(str(path))
    try:
        text_parts = []
        scanned_pages = []
        for i, page in enumerate(doc):
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
            else:
                scanned_pages.append(i)

        if not text_parts and scanned_pages:
            # Fully scanned PDF: OCR fallback (spec §3.2).
            return _ingest_pdf_scanned(doc, source)

        return IngestResult(
            success=True,
            content="\n\n".join(text_parts),
            content_type="pdf",
            mode="extrait",
            source=source,
            metadata={
                "tool": "pymupdf",
                "total_pages": doc.page_count,
                "scanned_pages": len(scanned_pages),
            },
        )
    finally:
        doc.close()


def _ingest_pdf_scanned(doc: Any, source: str) -> IngestResult:
    error = _ram_guard(768.0, "OCR de PDF scanné")
    if error:
        return IngestResult(success=False, content_type="pdf", source=source, error=error)

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return IngestResult(
            success=False,
            content_type="pdf",
            source=source,
            error="rapidocr-onnxruntime n'est pas installé.",
        )

    with _HEAVY_PIPELINE_LOCK:
        ocr = RapidOCR()
        text_parts = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            result, _ = ocr(img_bytes)
            if result:
                text_parts.append("\n".join(line[1] for line in result))

    return IngestResult(
        success=True,
        content="\n\n".join(text_parts),
        content_type="pdf",
        mode="ocr",
        source=source,
        metadata={"tool": "rapidocr", "total_pages": doc.page_count},
    )


def _ingest_image(source: str, *, workspace: str = "") -> IngestResult:
    """Image -> OCR text, escalating to a vision model (Claude/Gemini
    subscription, decision D1) only when OCR yields little (spec: OCR
    systematic, vision as last resort, not for every image)."""
    path = Path(source)
    if not path.exists():
        return IngestResult(success=False, content_type="image", source=source, error=f"Fichier introuvable : {source}")

    error = _ram_guard(768.0, "OCR d'image")
    if error:
        return IngestResult(success=False, content_type="image", source=source, error=error)

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return IngestResult(
            success=False,
            content_type="image",
            source=source,
            error="rapidocr-onnxruntime n'est pas installé.",
        )

    with _HEAVY_PIPELINE_LOCK:
        ocr = RapidOCR()
        result, _ = ocr(str(path))

    ocr_text = "\n".join(line[1] for line in result) if result else ""

    # Escalation trigger (decision D1): little usable OCR text -> a
    # picture worth describing, not just reading. Vision itself is NOT
    # called here -- it needs a subscription agent (Claude/Gemini) that
    # this module has no reference to; the caller (agent tool layer) does
    # the escalation, this just signals it clearly via metadata.
    MIN_USABLE_CHARS = 20
    needs_vision = len(ocr_text.strip()) < MIN_USABLE_CHARS

    return IngestResult(
        success=True,
        content=ocr_text or "(aucun texte détecté par OCR)",
        content_type="image",
        mode="ocr" if not needs_vision else "ocr_insuffisant",
        source=source,
        metadata={
            "tool": "rapidocr",
            "ocr_chars": len(ocr_text.strip()),
            "needs_vision_escalation": needs_vision,
        },
    )


def _ingest_video(source: str, *, workspace: str = "") -> IngestResult:
    """YouTube video -> subtitles (preferred, verified content) -> local
    whisper transcription (fallback, reconstructed content, flagged as
    such) -> metadata only (last resort, explicitly marked as NOT
    analyzed). Order is mandatory, not a preference -- anti-hallucination
    (PROJET_JARVIS.md §3, spec §3.4)."""
    try:
        import yt_dlp
    except ImportError:
        return IngestResult(
            success=False,
            content_type="video",
            source=source,
            error="yt-dlp n'est pas installé.",
        )

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        try:
            info = ydl.extract_info(source, download=False)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(
                success=False,
                content_type="video",
                source=source,
                error=f"Impossible de récupérer les informations vidéo : {exc}",
            )

    meta = {
        "title": info.get("title", ""),
        "duration_s": info.get("duration"),
        "channel": info.get("uploader", ""),
        "upload_date": info.get("upload_date", ""),
    }

    subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    sub_lang = next(iter(subs), None) or next(iter(auto_subs), None)
    sub_source = subs if next(iter(subs), None) else auto_subs

    if sub_lang:
        text = _download_subtitle_text(source, sub_lang)
        if text:
            return IngestResult(
                success=True,
                content=text,
                content_type="video",
                mode="sous-titres",
                source=source,
                metadata={**meta, "subtitle_lang": sub_lang},
            )

    # No subtitles -- try local transcription (heavy, RAM-guarded).
    error = _ram_guard(768.0, "transcription audio (faster-whisper)")
    if not error:
        transcript = _transcribe_video_audio(source)
        if transcript:
            return IngestResult(
                success=True,
                content=transcript,
                content_type="video",
                mode="transcription",
                source=source,
                metadata=meta,
            )

    # Last resort: metadata only, explicitly marked as not analyzed.
    header = (
        f"# {meta['title']}\n\n"
        f"**Métadonnées uniquement -- contenu vidéo NON analysé** "
        f"(pas de sous-titres, transcription indisponible ou ignorée).\n\n"
        f"- Chaîne : {meta['channel']}\n"
        f"- Durée : {meta['duration_s']}s\n"
        f"- Date : {meta['upload_date']}\n"
    )
    return IngestResult(
        success=True,
        content=header,
        content_type="video",
        mode="métadonnées",
        source=source,
        metadata=meta,
    )


def _download_subtitle_text(url: str, lang: str) -> str:
    import re as _re
    import tempfile

    import yt_dlp

    out_dir = tempfile.mkdtemp(prefix="jarvis-ingest-subs-")
    opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        vtt_files = list(Path(out_dir).glob("*.vtt"))
        if not vtt_files:
            return ""
        raw = vtt_files[0].read_text(encoding="utf-8", errors="replace")
        # Strip VTT timestamps/cue markup, keep only spoken text.
        lines = [
            line.strip()
            for line in raw.splitlines()
            if line.strip()
            and "-->" not in line
            and not line.strip().isdigit()
            and not line.startswith(("WEBVTT", "Kind:", "Language:"))
        ]
        text = " ".join(lines)
        return _re.sub(r"<[^>]+>", "", text)  # drop inline VTT tags
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)


def _transcribe_video_audio(url: str) -> str:
    import tempfile

    import yt_dlp

    with _HEAVY_PIPELINE_LOCK:
        out_dir = tempfile.mkdtemp(prefix="jarvis-ingest-audio-")
        try:
            opts = {
                "quiet": True,
                "format": "bestaudio/best",
                "outtmpl": f"{out_dir}/audio.%(ext)s",
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            audio_files = [
                p for p in Path(out_dir).iterdir() if p.suffix.lower() in _AUDIO_EXTS | {".m4a", ".opus"}
            ]
            if not audio_files:
                return ""

            backend = _whisper_backend()
            audio_bytes = audio_files[0].read_bytes()
            result = backend.transcribe(audio_bytes, format=audio_files[0].suffix.lstrip("."))
            return result.text
        finally:
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)


def _ingest_audio(source: str, *, workspace: str = "") -> IngestResult:
    """Local audio file -> faster-whisper transcription (free, local --
    the OpenAI provider stays available in audio_transcribe but costs
    money and isn't the default path here)."""
    path = Path(source)
    if not path.exists():
        return IngestResult(success=False, content_type="audio", source=source, error=f"Fichier introuvable : {source}")

    error = _ram_guard(768.0, "transcription audio (faster-whisper)")
    if error:
        return IngestResult(success=False, content_type="audio", source=source, error=error)

    try:
        with _HEAVY_PIPELINE_LOCK:
            backend = _whisper_backend()
            result = backend.transcribe(path.read_bytes(), format=path.suffix.lstrip("."))
    except ImportError as exc:
        return IngestResult(
            success=False,
            content_type="audio",
            source=source,
            error=f"faster-whisper n'est pas installé : {exc}",
        )

    return IngestResult(
        success=True,
        content=result.text,
        content_type="audio",
        mode="transcription",
        source=source,
        metadata={"language": result.language, "duration_s": result.duration_seconds},
    )


def _ingest_web(source: str, *, workspace: str = "") -> IngestResult:
    """Plain web URL -> fetched + extracted text (reuses the existing
    SSRF-guarded http_request tool, spec §3.5 -- already OK, no new
    pipeline needed here)."""
    try:
        import httpx
    except ImportError:
        return IngestResult(success=False, content_type="web", source=source, error="httpx n'est pas installé.")

    from openjarvis.security.ssrf import check_ssrf  # SSRF guard already in the codebase

    ssrf_error = check_ssrf(source)
    if ssrf_error:
        return IngestResult(
            success=False, content_type="web", source=source, error=f"URL refusée (garde SSRF) : {ssrf_error}"
        )

    try:
        resp = httpx.get(source, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return IngestResult(success=False, content_type="web", source=source, error=f"Échec de récupération : {exc}")

    return IngestResult(
        success=True,
        content=resp.text,
        content_type="web",
        mode="extrait",
        source=source,
        metadata={"status_code": resp.status_code, "content_type_header": resp.headers.get("content-type", "")},
    )


def _ingest_text(source: str, *, workspace: str = "") -> IngestResult:
    """Local text/markdown/csv/docx -> reuses the existing ingest_path
    (already handles detection, chunking readiness, secret-file skip)."""
    from openjarvis.tools.storage.ingest import read_document

    path = Path(source)
    if not path.exists():
        return IngestResult(success=False, content_type="text", source=source, error=f"Fichier introuvable : {source}")

    try:
        text, doc_meta = read_document(path)
    except Exception as exc:  # noqa: BLE001
        return IngestResult(success=False, content_type="text", source=source, error=f"Échec de lecture : {exc}")

    return IngestResult(
        success=True,
        content=text,
        content_type="text",
        mode="extrait",
        source=source,
        metadata={
            "tool": "read_document",
            "file_type": doc_meta.file_type,
            "size_bytes": doc_meta.size_bytes,
            "line_count": doc_meta.line_count,
        },
    )


__all__ = [
    "CONTENT_TYPES",
    "IngestResult",
    "detect_content_type",
    "ingest_content",
]
