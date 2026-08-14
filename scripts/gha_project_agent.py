#!/usr/bin/env python3
"""Work on a real GitHub project from GitHub Actions -- no PC required.

This is the piece that makes "PC éteint" mean *working* rather than
*chatting*. It runs on a GitHub-hosted runner (4 CPU / 16 GB, free and
unlimited on public repos -- more machine than the laptop it replaces),
so it keeps working when the user is away and their PC is off.

Two modes, deliberately separated by risk:

``analyze``
    Read-only. Clone the repo, compress it into LLM-readable context
    (repomix), optionally read an objective file, and report where the
    project stands versus that objective. Nothing is written, so this is
    safe to run on anything.

``implement``
    Clone, understand, produce a patch, apply it, run the tests, and open
    a pull request. Never pushes to the default branch: the output is
    always a PR the user reviews. With Groq doing the coding (the free,
    zero-cost path), the code genuinely needs that review -- which is
    exactly why a PR, not a direct push, is the only acceptable output.

Honest constraint recorded here so nobody rediscovers it the hard way:
the Claude/Gemini *subscriptions* cannot authenticate inside a runner
(browser OAuth tied to the user's machine), so cloud-side coding quality
is Groq quality, not Claude quality.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# NOT qwen3.6 here, despite it being the engine default: it reasons
# inline in <think> and, under the free tier's tight per-minute token
# budget, spends the entire output allowance thinking and returns an
# empty answer (observed live -- a run "succeeded" with a blank
# report). llama-3.3-70b answers directly, so every token of a scarce
# budget goes to the answer instead of to discarded reasoning.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq returns HTTP 413 when a SINGLE request exceeds the free tier's
# 8k tokens-per-minute budget -- not just when the body is physically
# large. Found live: 60k chars failed, then 18k+tree still failed,
# because input AND requested output count against the same minute.
# ~8k chars of context (~2k tokens) + a 1.5k-token answer leaves
# headroom. This is a real ceiling of the zero-cost path: big repos
# are read partially, and the prompt says so out loud rather than
# letting the model answer as if it had seen everything.
MAX_CONTEXT_CHARS = 8_000
MAX_TREE_CHARS = 2_500
# input + requested output must both fit inside ONE minute's 8k budget.
ANSWER_TOKENS = 1_536
# Writing whole files needs more room than a short analysis answer.
IMPLEMENT_TOKENS = 3_072
UA = "JARVIS-GHA-ProjectAgent/1.0"


def log(msg: str) -> None:
    print(f"[jarvis] {msg}", flush=True)


def run(args: List[str], *, cwd: Optional[str] = None, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {proc.stderr[:400]}")
    return proc


def strip_reasoning(text: str) -> str:
    """Remove qwen3.6's inline chain of thought, closed or truncated.

    The unclosed case matters: when the model spends its whole budget
    reasoning it never emits </think>, and a closing-tag-only regex leaves
    the entire block in place -- which is how raw reasoning once got
    presented as a finished answer elsewhere in this project.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def ask_groq(api_key: str, system: str, user: str, *, max_tokens: int = ANSWER_TOKENS) -> str:
    body = json.dumps(
        {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "user-agent": UA,  # Groq 403s the default urllib UA
        },
    )
    log(f"groq request: ~{len(system) + len(user)} chars in, {max_tokens} tokens out")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return strip_reasoning(payload["choices"][0]["message"]["content"])


def build_context(repo_dir: str) -> str:
    """Compress the repo into LLM-readable context via repomix.

    Falls back to a plain file listing if repomix is unavailable -- a
    degraded context beats refusing to work.
    """
    out = Path(repo_dir) / ".jarvis-context.xml"
    proc = run(
        ["npx", "--yes", "repomix@1.14.0", ".", "--output", str(out), "--style", "xml"],
        cwd=repo_dir,
    )
    if proc.returncode == 0 and out.exists():
        text = out.read_text(encoding="utf-8", errors="replace")
        out.unlink(missing_ok=True)
        if len(text) > MAX_CONTEXT_CHARS:
            tree = run(["git", "ls-files"], cwd=repo_dir).stdout[:MAX_TREE_CHARS]
            text = (
                text[:MAX_CONTEXT_CHARS]
                + "\n\n[CONTEXTE TRONQUÉ — dépôt trop gros pour être lu en entier. "
                "Arborescence complète ci-dessous ; dis-le si ta réponse est limitée "
                "par ce que tu n'as pas pu lire.]\n\nARBORESCENCE COMPLÈTE :\n" + tree
            )
        return text

    log("repomix unavailable, falling back to a file listing")
    listing = run(["git", "ls-files"], cwd=repo_dir).stdout
    return f"Fichiers du dépôt :\n{listing[:MAX_CONTEXT_CHARS]}"


def read_objective(repo_dir: str, objective_file: str) -> str:
    if not objective_file:
        return ""
    path = Path(repo_dir) / objective_file
    if not path.exists():
        return f"(fichier d'objectif introuvable : {objective_file})"
    return path.read_text(encoding="utf-8", errors="replace")[:4_000]


ANALYZE_SYSTEM = (
    "Tu es JARVIS, l'assistant technique de Nourredine. On te donne le code "
    "d'un de ses projets et, éventuellement, son objectif final. Tu dois dire "
    "OÙ EN EST le projet par rapport à cet objectif.\n"
    "RÈGLES :\n"
    "- Fonde-toi UNIQUEMENT sur le code fourni. N'invente aucun fichier ni "
    "fonctionnalité que tu ne vois pas.\n"
    "- Structure : 1) ce qui est fait, 2) ce qui manque, 3) prochaines étapes "
    "concrètes par ordre de priorité, avec une estimation d'effort.\n"
    "- Sois concret et cite les fichiers réels.\n"
    "- Réponds en français, de façon dense et sans remplissage."
)

# Whole files, not a diff. Asking a small free model for a byte-perfect
# unified diff does not work in practice -- the first real run died on
# "corrupt patch at line 34", because @@ hunk line counts have to be
# exactly right and models get them wrong. Full file contents need no
# line arithmetic, so they either parse or they don't.
IMPLEMENT_SYSTEM = (
    "Tu es JARVIS, développeur sur le projet de Nourredine. On te donne le "
    "code du projet et une tâche à implémenter.\n"
    "RÈGLES STRICTES :\n"
    "- Réponds UNIQUEMENT avec un objet JSON, sans texte autour, de la forme :\n"
    '  {"files": [{"path": "chemin/relatif.ts", "content": "contenu COMPLET"}], '
    '"summary": "ce que tu as fait"}\n'
    "- Donne le contenu ENTIER de chaque fichier, pas un extrait ni un diff.\n"
    "- Privilégie la création de nouveaux fichiers plutôt que la réécriture "
    "de gros fichiers existants.\n"
    "- Respecte le style et les conventions visibles dans le code fourni.\n"
    "- N'invente pas d'API absente du code fourni.\n"
    "- Si la tâche est infaisable, renvoie : {\"impossible\": \"raison\"}"
)


def do_analyze(repo_dir: str, task: str, objective: str, api_key: str) -> str:
    context = build_context(repo_dir)
    prompt = f"PROJET :\n{context}\n\n"
    if objective:
        prompt += f"OBJECTIF FINAL (fourni par l'utilisateur) :\n{objective}\n\n"
    prompt += f"DEMANDE :\n{task or 'Où en est ce projet par rapport à son objectif ?'}"
    return ask_groq(api_key, ANALYZE_SYSTEM, prompt)


def do_implement(repo_dir: str, task: str, objective: str, api_key: str) -> Dict[str, Any]:
    context = build_context(repo_dir)
    prompt = f"PROJET :\n{context}\n\n"
    if objective:
        prompt += f"OBJECTIF FINAL :\n{objective}\n\n"
    prompt += f"TÂCHE À IMPLÉMENTER :\n{task}"

    raw = ask_groq(api_key, IMPLEMENT_SYSTEM, prompt, max_tokens=IMPLEMENT_TOKENS)

    # Models habitually wrap JSON in fences despite instructions.
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"applied": False, "reason": f"réponse illisible du modèle : {raw[:300]}"}
    try:
        # strict=False tolerates literal newlines/tabs inside JSON strings.
        # Models emit real line breaks in code payloads instead of \n escapes
        # (seen live: a perfectly good React component rejected purely on
        # encoding). The content is what matters here, not RFC-strict JSON.
        payload = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError as exc:
        return {"applied": False, "reason": f"JSON invalide ({exc}) : {raw[:300]}"}

    if payload.get("impossible"):
        return {"applied": False, "reason": f"jugé infaisable : {payload['impossible']}"}

    files = payload.get("files") or []
    if not files:
        return {"applied": False, "reason": "le modèle n'a produit aucun fichier"}

    written = []
    for f in files:
        rel = str(f.get("path") or "").lstrip("/")
        content = f.get("content")
        # Path traversal guard: a generated path must stay inside the repo.
        target = (Path(repo_dir) / rel).resolve()
        if not rel or content is None or not str(target).startswith(str(Path(repo_dir).resolve())):
            return {"applied": False, "reason": f"chemin de fichier refusé : {rel!r}"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    log(f"wrote {len(written)} file(s): {', '.join(written)}")

    tests = run(["python", "-m", "pytest", "-q", "--timeout=120"], cwd=repo_dir)
    return {
        "applied": True,
        "files": written,
        "summary": payload.get("summary", ""),
        "tests_passed": tests.returncode == 0,
        "tests_output": (tests.stdout or tests.stderr)[-2000:],
    }


def open_pull_request(repo_dir: str, repo: str, branch: str, title: str, body: str, token: str) -> str:
    run(["git", "config", "user.name", "JARVIS"], cwd=repo_dir)
    run(["git", "config", "user.email", "jarvis@users.noreply.github.com"], cwd=repo_dir)
    run(["git", "checkout", "-b", branch], cwd=repo_dir)
    run(["git", "add", "-A"], cwd=repo_dir)
    commit = run(["git", "commit", "-m", title], cwd=repo_dir)
    if commit.returncode != 0:
        return ""

    push = run(["git", "push", "origin", branch], cwd=repo_dir)
    if push.returncode != 0:
        log(f"push failed: {push.stderr[:300]}")
        return ""

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/pulls",
        data=json.dumps({"title": title, "body": body, "head": branch, "base": _default_branch(repo_dir)}).encode(),
        method="POST",
        headers={
            "accept": "application/vnd.github+json",
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "user-agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("html_url", "")
    except urllib.error.HTTPError as exc:
        log(f"PR creation failed: {exc.read()[:300]!r}")
        return ""


def _default_branch(repo_dir: str) -> str:
    proc = run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=repo_dir)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().rsplit("/", 1)[-1]
    return "main"


def report(control_plane_url: str, secret: str, mission_id: str, status: str, body: str) -> None:
    if not control_plane_url or not mission_id:
        return
    req = urllib.request.Request(
        control_plane_url.rstrip("/") + f"/missions/{mission_id}/complete",
        data=json.dumps({"status": status, "report": body}).encode(),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-control-plane-secret": secret,
            "user-agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        log(f"could not report completion: {exc}")


def main() -> int:
    mode = os.environ.get("MODE", "analyze").strip()
    repo = os.environ.get("TARGET_REPO", "").strip()
    task = os.environ.get("TASK", "").strip()
    objective_file = os.environ.get("OBJECTIVE_FILE", "").strip()
    mission_id = os.environ.get("MISSION_ID", "").strip()
    api_key = os.environ["GROQ_API_KEY"]
    gh_token = os.environ.get("PROJECT_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
    cp_url = os.environ.get("CONTROL_PLANE_URL", "")
    cp_secret = os.environ.get("CONTROL_PLANE_SHARED_SECRET", "")

    if not repo:
        log("TARGET_REPO is required")
        return 1

    work = Path("/tmp/jarvis-target")
    clone_url = f"https://x-access-token:{gh_token}@github.com/{repo}.git"
    log(f"cloning {repo}")
    clone = run(["git", "clone", "--depth", "50", clone_url, str(work)])
    if clone.returncode != 0:
        msg = (
            f"Impossible de cloner {repo}. Le jeton GitHub ne couvre "
            f"probablement pas ce dépôt.\n{clone.stderr[:300]}"
        )
        log(msg)
        report(cp_url, cp_secret, mission_id, "FAILED", msg)
        return 1

    objective = read_objective(str(work), objective_file)

    try:
        if mode == "analyze":
            answer = do_analyze(str(work), task, objective, api_key)
            if not answer.strip():
                # An empty answer previously still reported SUCCEEDED, so a
                # blank report looked like a finished analysis. Absence of
                # an answer is a failure, and must read as one.
                msg = (
                    f"# {repo} — analyse non aboutie\n\n"
                    "Le modèle n'a produit aucune réponse exploitable "
                    "(budget de tokens épuisé). Réessaie dans une minute."
                )
                print(msg)
                report(cp_url, cp_secret, mission_id, "FAILED", msg)
                return 1
            body = f"# État du projet {repo}\n\n{answer}"
            print(body)
            report(cp_url, cp_secret, mission_id, "SUCCEEDED", body)
            return 0

        result = do_implement(str(work), task, objective, api_key)
        if not result["applied"]:
            body = f"# {repo} — implémentation non aboutie\n\n{result['reason']}"
            print(body)
            report(cp_url, cp_secret, mission_id, "FAILED", body)
            return 1

        branch = f"jarvis/{mission_id or 'task'}-{os.environ.get('GITHUB_RUN_ID', '0')}"
        tests_line = (
            "✅ tests verts" if result.get("tests_passed") else "❌ tests en échec (voir la PR)"
        )
        pr_body = (
            f"Tâche : {task}\n\n{tests_line}\n\n"
            f"```\n{result.get('tests_output', '')[:1500]}\n```\n\n"
            "_Généré par JARVIS depuis GitHub Actions (PC éteint). "
            "Code produit par un modèle gratuit : à relire avant fusion._"
        )
        pr_url = open_pull_request(str(work), repo, branch, f"JARVIS: {task[:60]}", pr_body, gh_token)

        body = (
            f"# {repo} — implémentation proposée\n\n"
            f"{tests_line}\n\n"
            + (f"Pull request : {pr_url}" if pr_url else "⚠️ PR non créée (jeton insuffisant ?)")
        )
        print(body)
        report(cp_url, cp_secret, mission_id, "SUCCEEDED" if pr_url else "FAILED", body)
        return 0 if pr_url else 1
    except Exception as exc:  # noqa: BLE001
        msg = f"Erreur pendant le travail sur {repo} : {type(exc).__name__}: {exc}"
        log(msg)
        report(cp_url, cp_secret, mission_id, "FAILED", msg)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
