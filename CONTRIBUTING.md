# Contributing to watermarks-remover

Thanks for helping keep the skill accurate and the cleaners reliable. The
project is a small Python skill (`skills/remove-ai-marks/`) plus tests —
focused PRs land fastest.

## Who can do what

| Action | Who |
| --- | --- |
| Open issues | Anyone |
| Suggest a release | Anyone (use the **Release suggestion** issue template) |
| Open pull requests | Anyone (fork the repo) |
| Approve and merge pull requests | Maintainer only (`@guillaumemeyer`) |

`main` is protected. A change needs a pull request, a passing **CI** check
(`test`), and an approving review from the code owner before merge. Only the
maintainer can give that approval. Direct pushes to `main` are blocked for
non-admins.

To suggest a release without a code change: open a **Release suggestion**
issue.

## Prerequisites

- **Python 3.10+** (stdlib only for the skill scripts; optional rewrite backends
  use HTTP to local Ollama / OpenAI-compatible endpoints)
- From the repo root: `python3 -m pytest -q` should pass before you open a PR
- Optional for manual file checks: [`c2patool`](https://github.com/contentauth/c2pa-rs/tree/main/cli), [`exiftool`](https://exiftool.org/) (PDF)

## Layout

| Path | Role |
| --- | --- |
| `skills/remove-ai-marks/SKILL.md` | Agent skill entry (workflow, ethics) — remote client over HTTP |
| `.claude-plugin/` | Claude Code plugin manifest + single-plugin marketplace (`make plugin-validate`) |
| `install_skill.py` | Multi-host skill installer (Claude Code, Cowork bundle, Cursor) |
| `skills/remove-ai-marks/references/` | Vendors, mark classes, matrix, ethics |
| `service/scripts/` | Layer A/B hooks + image/container cleaners + `server.py` HTTP service |
| `service/Dockerfile*` | Container images (core + optional backends) |
| `compose.yaml` | Local full-stack bring-up |
| `tests/` | Pytest suite and fixtures |
| `.github/workflows/ci.yml` | CI job `test` |
| `.github/workflows/release-images.yml` | GHCR image publishing on `v*` tags |

## Layers (what to change where)

1. **Layer A (Unicode / format controls)** — deterministic scripts under
   `service/scripts/` (`text_unicode.py`, `clean_text.py`, `inspect_text.py`). Prefer
   tests with fixtures in `tests/fixtures/`.
2. **Layer B (statistical rewrite)** — guidance in `SKILL.md` plus optional
   `rewrite_text.py` (print-prompt default; ollama / openai-compatible). No
   bundled model. Keep ethics-aware.
3. **Files (C2PA / EXIF / XMP / props)** — `image_meta.py` (PNG/JPEG/AVIF/HEIC/...),
   `container_meta.py` (SVG/PDF/DOCX/ODT/HTML/MD), `av_meta.py` (MP4/MOV/WAV/MP3),
   unified `inspect_file.py` / `clean_file.py`. Preserve document body / pixels /
   waveform; strip provenance metadata only.

## Checklist for a change

- [ ] Behaviour matches `SKILL.md` / `references/removal-matrix.md` when relevant
- [ ] Unit tests updated or added under `tests/`
- [ ] `python3 -m pytest -q` passes
- [ ] Docs updated (README and/or skill references) if user-facing behaviour
      changes
- [ ] No drive-by refactors unrelated to the fix or feature

## PR expectations

- Stay focused and match existing style (stdlib-first scripts, clear CLI flags)
- Do not commit secrets, private user files, or large binary fixtures unless
  needed and redacted
- Respect `references/ethics.md`: this tool is for content the user owns

Questions? Open an issue describing the input type (text / image / document)
and which layer failed or is missing.

## Community

- [Code of Conduct](CODE_OF_CONDUCT.md) — expected behaviour in the project
- [Security policy](SECURITY.md) — how to report vulnerabilities privately
- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md) and
  [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates
