"""Tests for the benchmark/experiment tool (Brique 4).

The sandbox tests run real containers on purpose: the whole value of this
tool rests on its isolation actually holding, and that is precisely the
kind of claim that turns out to be false when only asserted. One such
failure was found this way -- ``--memory`` alone did not cap anything
because Docker silently allowed swap.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from openjarvis.tools.experiment import (
    ExperimentTool,
    VariantResult,
    _format_table,
    _write_raw_log,
    run_variant,
)

FAST = "def run():\n    return sum(range(1000))"
SLOW = "def run():\n    return sum(range(200000))"
BROKEN = "def run():\n    raise ValueError('boom')"
NO_RUN = "x = 1"


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


requires_docker = pytest.mark.skipif(not _docker_ok(), reason="Docker unavailable")


class TestFormatting:
    def test_table_marks_failed_variant_without_numbers(self):
        table = _format_table([VariantResult(name="cassée", error="boom")])
        assert "❌" in table
        assert "cassée" in table

    def test_table_renders_stats_for_successful_variant(self):
        r = VariantResult(
            name="ok",
            times_s=[0.001, 0.002],
            peak_rss_mb=12.5,
            stats={"mean_time_s": 0.0015, "p50_time_s": 0.0015, "p95_time_s": 0.002},
        )
        table = _format_table([r])
        assert "1.50 ms" in table
        assert "12.5 Mo" in table

    def test_raw_log_is_written_as_jsonl_evidence(self, tmp_path, monkeypatch):
        import openjarvis.tools.experiment as exp

        monkeypatch.setattr(exp, "get_config_dir", lambda: tmp_path)
        path = _write_raw_log("ma tâche", [VariantResult(name="a", times_s=[0.1])])
        lines = Path(path).read_text().strip().splitlines()
        assert json.loads(lines[0])["variant"] == "a"
        assert json.loads(lines[0])["task"] == "ma tâche"


class TestToolValidation:
    def test_requires_a_task(self):
        assert ExperimentTool().execute(variants=[{"name": "a", "code": FAST}]).success is False

    def test_requires_at_least_two_variants(self):
        result = ExperimentTool().execute(task="t", variants=[{"name": "a", "code": FAST}])
        assert result.success is False
        assert "2 variantes" in result.content

    def test_rejects_too_many_variants(self):
        variants = [{"name": str(i), "code": FAST} for i in range(6)]
        result = ExperimentTool().execute(task="t", variants=variants)
        assert result.success is False
        assert "Trop de variantes" in result.content

    def test_refuses_to_run_without_docker(self, monkeypatch):
        """Untrusted variant code must never fall back to running on the
        host when the sandbox is unavailable."""
        import openjarvis.tools.experiment as exp

        monkeypatch.setattr(exp, "_docker_available", lambda: False)
        result = ExperimentTool().execute(
            task="t", variants=[{"name": "a", "code": FAST}, {"name": "b", "code": SLOW}]
        )
        assert result.success is False
        assert "bac à sable" in result.content


@requires_docker
class TestSandboxIsolation:
    def test_network_is_unreachable(self):
        code = (
            "import socket\n"
            "def run():\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)"
        )
        r = run_variant("net", code, runs=1, warmup=0)
        assert r.ok is False
        assert "Network is unreachable" in r.error or "unreachable" in r.error.lower()

    def test_memory_cap_is_actually_enforced(self):
        """Regression guard for a real finding: with ``--memory`` but no
        ``--memory-swap``, Docker let a container allocate and touch 900 MB
        under a 512 MB "limit" by spilling into swap. On a 7.6 GB machine
        that already OOM'd twice, that made the cap decorative."""
        code = (
            "def run():\n"
            "    b = bytearray(900 * 1024 * 1024)\n"
            "    for i in range(0, len(b), 4096):\n"
            "        b[i] = 1"
        )
        r = run_variant("hog", code, runs=1, warmup=0)
        assert r.ok is False

    def test_timeout_is_enforced(self):
        code = "import time\ndef run():\n    time.sleep(60)"
        r = run_variant("slow", code, runs=1, warmup=0, timeout_s=5)
        assert r.ok is False
        assert "timeout" in r.error.lower()


@requires_docker
class TestRealBenchmark:
    def test_measures_and_ranks_variants(self):
        tool = ExperimentTool()
        result = tool.execute(
            task="somme sur une plage",
            runs=3,
            variants=[{"name": "petit", "code": FAST}, {"name": "grand", "code": SLOW}],
        )
        assert result.success is True
        assert result.metadata["succeeded"] == 2
        # The genuinely cheaper variant must win on measurement.
        assert "**Gagnant : petit**" in result.content
        assert Path(result.metadata["log_path"]).exists()

    def test_failing_variant_is_never_the_winner(self):
        tool = ExperimentTool()
        result = tool.execute(
            task="t",
            runs=2,
            variants=[{"name": "cassée", "code": BROKEN}, {"name": "ok", "code": FAST}],
        )
        assert result.success is True
        assert "**Gagnant : ok**" in result.content
        assert "## Échecs" in result.content
        assert result.metadata["succeeded"] == 1

    def test_all_variants_failing_recommends_nothing(self):
        tool = ExperimentTool()
        result = tool.execute(
            task="t",
            runs=1,
            variants=[{"name": "a", "code": BROKEN}, {"name": "b", "code": NO_RUN}],
        )
        assert "Aucune variante n'a abouti" in result.content
        assert "Gagnant" not in result.content

    def test_variant_without_run_function_is_an_error(self):
        r = run_variant("norun", NO_RUN, runs=1, warmup=0)
        assert r.ok is False
        assert "run()" in r.error

    def test_empty_code_is_rejected_before_running(self):
        tool = ExperimentTool()
        result = tool.execute(
            task="t", runs=1, variants=[{"name": "vide", "code": ""}, {"name": "ok", "code": FAST}]
        )
        assert "vide" in result.content


@requires_docker
class TestMemoryIntegration:
    def test_conclusion_is_recorded_as_a_sourced_decision(self):
        """Decision D3: a benchmark nobody remembers is dead data."""

        class FakeMemory:
            def __init__(self):
                self.stored = []

            def store(self, content, *, source="", metadata=None):
                self.stored.append((content, source, metadata or {}))
                return "id"

        tool = ExperimentTool()
        memory = FakeMemory()
        tool._memory_backend = memory
        tool.execute(
            task="comparer deux sommes",
            runs=2,
            variants=[{"name": "petit", "code": FAST}, {"name": "grand", "code": SLOW}],
        )

        assert len(memory.stored) == 1
        content, source, metadata = memory.stored[0]
        assert "petit" in content
        assert source == "experiment"
        assert metadata["kind"] == "decision"

    def test_memory_failure_never_breaks_the_benchmark(self):
        class BrokenMemory:
            def store(self, *a, **k):
                raise RuntimeError("memory down")

        tool = ExperimentTool()
        tool._memory_backend = BrokenMemory()
        result = tool.execute(
            task="t", runs=2, variants=[{"name": "a", "code": FAST}, {"name": "b", "code": SLOW}]
        )
        assert result.success is True
