# Overnight agent rules (shared by every Jarvis 2.0 subagent)

Read `claude.md` at the repo root first. Everything in it still applies, with the
git exceptions listed below. Then read the specific source and tests you'll touch.

## Machine constraints (measured 2026-11-17)

- CPU: AMD Athlon Gold 3150U, **2 cores**. RAM: **5.9 GB**. No CUDA GPU.
- Python **3.14.0** in `.venv` (always use `.\.venv\Scripts\python.exe`). Windows 11.
- PostgreSQL 18 runs as a Windows service. Ollama runs locally with `qwen2.5:1.5b`,
  `qwen2.5:0.5b`, `nomic-embed-text`, `moondream`, `granite3.2-vision`.
- Don't load two large models at once. Measure latency and memory **on this
  machine** before adopting any model. `keep_alive: "0"` likely exists to save
  RAM, so measure before changing it.

## Hard rules

1. **Never read, print, or copy `.env`.** The backend loads it itself; use that
   to run real integration checks. Put new settings in `.env.example` with safe
   defaults, and keep defaults in code so nothing breaks if `.env` lacks them.
2. Keep everything local-first: no paid or cloud AI APIs. Downloading open model
   weights is fine. Use **permissive licenses only** (Apache-2.0, MIT, or BSD).
   No AGPL or GPL models or libraries.
3. Don't store raw audio or camera images, whether on disk, in the DB, or in logs.
4. Person detection is presence-only. **No facial recognition**, ever.
5. Surface errors. No fake success-shaped fallbacks.
6. Stay inside your assigned scope. If you find a bug outside it, record it in
   your report instead of fixing it, unless it blocks your work.

## Git workflow for tonight (overrides claude.md's "don't commit")

- Work **directly on `main`**. Don't create branches.
- After **each completed step** (each numbered step in your agent definition),
  do the following:
  1. Run backend tests:
     `.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v`
  2. If you touched frontend files: `$env:OPENSSL_CONF = ''; npm run build; npm run lint`
  3. Only if everything passes: stage files **by explicit path** (never
     `git add -A` or `git add .`; never stage `claude.md`, `.env`, `.claude/settings.local.json`,
     model weights, or caches), then commit and push:
     `git commit -m "<short plain message>"` then `git push origin main`
- Commit messages should be short and plain, like the existing history (e.g.
  `add memory eval harness`).
- **NEVER add `Co-Authored-By`, "Generated with Claude", or any AI attribution
  line** to commits. The owner is the only author.
- Never force-push, never `reset --hard`, never rewrite history, never use `--no-verify`.
- If tests fail and you can't fix them, don't commit the broken step. Note it in
  your report and continue with whatever else is safe.
- If a push fails (e.g., network), keep the local commit and note it in the report.

## Testing standard

- Every behavior change needs unit tests (Python `unittest`; mock external services).
- Also run **real** checks where possible: start the backend with
  `.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000`
  as a background process, call endpoints with real Ollama and PostgreSQL, then stop it.
  Use a dedicated test session UUID and delete any test rows you create.
- Physical camera and microphone can't be tested tonight. Say so plainly, and add
  a manual check for the morning.
- Put large weights under `backend/models/` (git-ignored). Provide a download
  script that verifies a SHA-256 checksum. Run it so the weights exist on this machine.

## Report (required)

Write `docs/overnight/<your-agent-name>.md` and commit it with your final step. Use these sections:

1. **What I built**: features, files, and endpoints.
2. **Technologies**: what was added and why.
3. **Decisions and reasoning**: written in plain language, because the owner
   is learning the system.
4. **Tradeoffs and alternatives considered.**
5. **Measurements**: latency, memory, and accuracy numbers with how you measured them.
6. **Tests**: list them, and keep mocked unit tests separate from real-service checks.
7. **Known limitations and open issues.**
8. **Morning manual checklist**: what the owner should click or say to verify it.
9. **Handoff notes** for the next agent: APIs, payload shapes, gotchas.

Also update `README.md` (setup/run) and the status/checklists in `claude.md` on
disk if your work changes them (claude.md is untracked, so edit it but don't stage it).
