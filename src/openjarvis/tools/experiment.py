"""Benchmark two or more implementations and pick a winner on evidence.

Brique 4 (docs/SPEC_BRIQUE4_BENCHMARK.md). The engine could already
*measure* LLM calls (``bench/``, ``evals/``), but nothing could compare
candidate implementations of a task -- so "solution A is faster than B"
was only ever the model's opinion. This runs both and reports numbers.

Design decisions that matter:

* **Sandboxed, always.** Variant code is untrusted by construction, so
  each run happens in a throwaway container with the network cut, a
  memory cap, a CPU cap and a timeout -- the same posture as
  ``code_interpreter_docker``. Never on the host.
* **Sequential, never parallel.** Running variants concurrently on an
  8-core laptop makes them fight for CPU and corrupts the very numbers
  the tool exists to produce (spec §4.3 point 5).
* **Warmup then N runs, reported as p50/p95.** A single run on a noisy
  machine (this one swaps) says nothing; the spread is part of the
  answer.
* **A failing variant is an error, never a winner** -- it is excluded
  from ranking rather than scoring "infinitely fast" by crashing early.
* **Time and peak RAM only.** Per-process CPU% is too noisy to decide
  between variants (decision D2), so it is deliberately not collected.

Measurement runs inside the container using only the standard library
(``time.perf_counter`` + ``resource.getrusage``), which keeps the image
plain ``python:3.12-slim`` with nothing to install at run time.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_RUNS = 5
DEFAULT_WARMUP = 1
MAX_VARIANTS = 4
MAX_RUNS = 50
CONTAINER_TIMEOUT_S = 120
MEMORY_LIMIT = "512m"
CPU_LIMIT = "1"

#: Harness executed inside the container. Times ``run()`` over N
#: iterations after a warmup and reports peak RSS, using only stdlib so
#: the image needs no pip install.
_HARNESS = '''
import json, resource, sys, time

_ns = {}
try:
    exec(VARIANT_CODE, _ns)
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)

fn = _ns.get("run")
if not callable(fn):
    print(json.dumps({"error": "variant code must define a callable run()"}))
    sys.exit(0)

try:
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)

# ru_maxrss is KiB on Linux.
peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"times_s": times, "peak_rss_mb": peak_kib / 1024.0}))
'''


@dataclass(slots=True)
class VariantResult:
    name: str
    times_s: List[float] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    error: str = ""
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.times_s)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_variant(
    name: str,
    code: str,
    *,
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
    image: str = DEFAULT_IMAGE,
    timeout_s: int = CONTAINER_TIMEOUT_S,
) -> VariantResult:
    """Execute one variant in a throwaway sandboxed container."""
    result = VariantResult(name=name)

    program = (
        f"VARIANT_CODE = {code!r}\nRUNS = {int(runs)}\nWARMUP = {int(warmup)}\n" + _HARNESS
    )
    args = [
        "docker", "run", "--rm", "-i",
        "--name", f"jarvis-experiment-{uuid.uuid4().hex[:10]}",
        "--label", "openjarvis-sandbox=true",
        "--network", "none",          # untrusted code never reaches the network
        "--memory", MEMORY_LIMIT,
        # --memory alone is NOT a real cap: without an equal
        # --memory-swap, Docker lets the container spill into swap
        # instead of OOM-killing it. Verified live on this host -- a
        # 900 MB allocation under "--memory 512m" succeeded and
        # touched every page. On a 7.6 GB laptop that already hit OOM
        # twice (PLAN.md D9), a runaway variant swapping the machine
        # to a crawl is exactly what this sandbox exists to prevent.
        "--memory-swap", MEMORY_LIMIT,
        "--cpus", CPU_LIMIT,
        "--pids-limit", "256",
        image, "python", "-c", program,
    ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result.error = f"timeout après {timeout_s}s"
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"exécution impossible : {exc}"
        return result

    if proc.returncode != 0:
        result.error = (proc.stderr or proc.stdout or "échec du conteneur").strip()[:300]
        return result

    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        result.error = f"sortie illisible : {(proc.stdout or '')[:200]}"
        return result

    if payload.get("error"):
        result.error = str(payload["error"])[:300]
        return result

    result.times_s = [float(t) for t in payload.get("times_s", [])]
    result.peak_rss_mb = float(payload.get("peak_rss_mb", 0.0))
    if result.times_s:
        from openjarvis.bench._stats import compute_stats

        result.stats = compute_stats("time_s", result.times_s)
    return result


def _format_table(results: List[VariantResult]) -> str:
    rows = [
        "| Variante | Statut | Temps moyen | p50 | p95 | RAM peak | Runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r.ok:
            rows.append(f"| {r.name} | ❌ erreur | — | — | — | — | 0 |")
            continue
        rows.append(
            f"| {r.name} | ✅ | {r.stats.get('mean_time_s', 0)*1000:.2f} ms "
            f"| {r.stats.get('p50_time_s', 0)*1000:.2f} ms "
            f"| {r.stats.get('p95_time_s', 0)*1000:.2f} ms "
            f"| {r.peak_rss_mb:.1f} Mo | {len(r.times_s)} |"
        )
    return "\n".join(rows)


def _write_raw_log(task: str, results: List[VariantResult]) -> str:
    """Persist raw per-run measurements as evidence (Quality Gate, B5)."""
    out_dir = get_config_dir() / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"experiment_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "task": task,
                        "variant": r.name,
                        "times_s": r.times_s,
                        "peak_rss_mb": r.peak_rss_mb,
                        "error": r.error,
                        "stats": r.stats,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return str(path)


@ToolRegistry.register("experiment")
class ExperimentTool(BaseTool):
    """Compare implementations by actually running them, in a sandbox."""

    tool_id = "experiment"
    #: Injected post-build (cli/serve.py) so a benchmark conclusion can be
    #: written to long-term memory as a sourced decision (Brique 2), rather
    #: than being measured once and forgotten.
    _memory_backend: Optional[Any] = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="experiment",
            description=(
                "Benchmark 2-4 candidate implementations of the same task by "
                "really running them in a sandbox, and return a comparison "
                "table (mean/p50/p95 time, peak RAM, errors). Use this when a "
                "choice between concrete implementations actually matters and "
                "you would otherwise be guessing -- not for every small task, and "
                "not for questions that cannot be settled by a micro-benchmark "
                "(e.g. 'React vs Vue' for a whole project). Each variant must "
                "be self-contained Python defining a callable run()."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What is being compared, and the success criterion.",
                    },
                    "variants": {
                        "type": "array",
                        "description": (
                            "2-4 variants: [{'name': 'regex', 'code': 'def run(): ...'}]. "
                            "Each code block must define run() and be self-contained."
                        ),
                        "items": {"type": "object"},
                    },
                    "runs": {"type": "integer", "description": "Measured runs per variant (default 5)."},
                    "warmup": {"type": "integer", "description": "Warmup runs (default 1)."},
                },
                "required": ["task", "variants"],
            },
            category="analysis",
        )

    def execute(self, **params: Any) -> ToolResult:
        task = str(params.get("task", "") or "").strip()
        variants = params.get("variants") or []
        if not task:
            return ToolResult(tool_name=self.tool_id, content="No task provided.", success=False)
        if not isinstance(variants, list) or len(variants) < 2:
            return ToolResult(
                tool_name=self.tool_id,
                content="Il faut au moins 2 variantes à comparer.",
                success=False,
            )
        if len(variants) > MAX_VARIANTS:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Trop de variantes ({len(variants)} > {MAX_VARIANTS}).",
                success=False,
            )
        if not _docker_available():
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    "Docker n'est pas disponible — le benchmark exige un bac à sable "
                    "(du code non vérifié ne doit jamais tourner sur la machine hôte)."
                ),
                success=False,
            )

        runs = max(1, min(int(params.get("runs", DEFAULT_RUNS)), MAX_RUNS))
        warmup = max(0, int(params.get("warmup", DEFAULT_WARMUP)))

        results: List[VariantResult] = []
        for v in variants:  # sequential on purpose -- see module docstring
            name = str(v.get("name") or f"variante{len(results)+1}")
            code = str(v.get("code") or "")
            if not code.strip():
                results.append(VariantResult(name=name, error="code vide"))
                continue
            results.append(run_variant(name, code, runs=runs, warmup=warmup))

        ok = [r for r in results if r.ok]
        log_path = _write_raw_log(task, results)

        lines = [f"# Benchmark — {task}", "", _format_table(results), ""]
        if ok:
            winner = min(ok, key=lambda r: r.stats.get("mean_time_s", float("inf")))
            lines.append(f"**Gagnant : {winner.name}**")
            if len(ok) > 1:
                others = [r for r in ok if r is not winner]
                slowest = max(others, key=lambda r: r.stats.get("mean_time_s", 0))
                w = winner.stats.get("mean_time_s", 0) or 1e-12
                lines.append(
                    f"({slowest.name} est {slowest.stats.get('mean_time_s', 0)/w:.2f}× plus lent)"
                )
            self._remember(task, winner.name, ok)
        else:
            lines.append("**Aucune variante n'a abouti** — aucune ne peut être recommandée.")

        failed = [r for r in results if not r.ok]
        if failed:
            lines.append("")
            lines.append("## Échecs")
            for r in failed:
                lines.append(f"- **{r.name}** : {r.error}")

        lines += ["", f"Mesures brutes : `{log_path}`"]

        return ToolResult(
            tool_name=self.tool_id,
            content="\n".join(lines),
            success=True,
            metadata={
                "log_path": log_path,
                "variants": len(results),
                "succeeded": len(ok),
                "runs": runs,
            },
        )

    def _remember(self, task: str, winner: str, ok: List[VariantResult]) -> None:
        """Record the conclusion as a sourced decision in long-term memory.

        Decision D3: a benchmark whose result is never remembered is dead
        data -- the next time the same choice comes up, JARVIS would guess
        again. Best-effort: never let a memory failure break the benchmark.
        """
        if self._memory_backend is None:
            return
        try:
            detail = ", ".join(
                f"{r.name} {r.stats.get('mean_time_s', 0)*1000:.2f}ms" for r in ok
            )
            self._memory_backend.store(
                f"Benchmark « {task} » : {winner} retenu ({detail}).",
                source="experiment",
                metadata={"kind": "decision"},
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to record benchmark decision in memory", exc_info=True)


__all__ = ["ExperimentTool", "VariantResult", "run_variant"]
