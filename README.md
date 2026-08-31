```
_ _ _ ____ ___ ____ ____ _  _ ____ ____ _  _ ____    ____ ____ _  _ ____ _  _ ____ ____
| | | |__|  |  |___ |__/ |\/| |__| |__/ |_/  [__  __ |__/ |___ |\/| |  | |  | |___ |__/
|_|_| |  |  |  |___ |  \ |  | |  | |  \ | \_ ___]    |  \ |___ |  | |__|  \/  |___ |  \
```

# watermarks-remover

<!-- logo: figlet -d .figlet -f cybermedium -w 120 "watermarks-remover" -->

[![CI](https://github.com/guillaumemeyer/watermarks-remover/actions/workflows/ci.yml/badge.svg)](https://github.com/guillaumemeyer/watermarks-remover/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/guillaumemeyer/watermarks-remover)](https://github.com/guillaumemeyer/watermarks-remover/releases)
[![Stars](https://img.shields.io/github/stars/guillaumemeyer/watermarks-remover)](https://github.com/guillaumemeyer/watermarks-remover/stargazers)
[![Forks](https://img.shields.io/github/forks/guillaumemeyer/watermarks-remover)](https://github.com/guillaumemeyer/watermarks-remover/forks)

Agent skill + stdlib Python service to strip **multi-vendor AI provenance marks** from text and files — for privacy and hygiene on content **you own**. The skill is a thin client: it drives the machinery over HTTP, so the agent host needs no Python.

| Layer | Target | How |
| --- | --- | --- |
| **A** | Invisible Unicode, exotic spaces, bidi, tag chars | Deterministic Python scripts |
| **B** | Statistical (token-sampling) text watermarks | Agent rewrite + optional `rewrite_text.py` hook |
| **Files** | C2PA / EXIF / XMP / doc props | PNG, JPEG, WebP, AVIF, HEIC, BMP, GIF, TIFF, SVG, PDF, DOCX, XLSX, PPTX, EPUB, ODT, HTML, Markdown, MP4/MOV/M4A/M4V, WAV, MP3, FLAC |

Vendors / ecosystems (class-level): **Claude**, **Gemini / SynthID-Text**, **OpenAI** provenance surfaces, **open-LLM** Kirchenbauer-style (green-list) and keyed-Gumbel / EXP (Aaronson) marks.

**Latest release:** [v0.6.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.6.0)

Skill path: [`skills/remove-ai-marks/`](skills/remove-ai-marks/)  
Service path: [`service/`](service/)  
(migration: formerly `remove-claude-marks`; slash alias `/remove-claude-marks` still documented)

## Install (agent skill)

The skill ships **no code** — it calls the service over HTTP. Install the skill (markdown only) and start the service, then set `WATERMARKS_SERVICE_URL` if it is not `http://127.0.0.1:8765`.

In Claude Code, the fastest route is the bundled
[plugin marketplace](#claude-code-plugin-marketplace) — no clone, and it updates
in place. Everywhere else, one installer covers every supported host
(Python 3.10+ stdlib, no dependencies):

```bash
python3 install_skill.py --skill remove-ai-marks --target claude-code
```

| Host | Target | Lands in |
| --- | --- | --- |
| Claude Code (personal) | `--target claude-code` | `~/.claude/skills/<skill>` (honors `CLAUDE_CONFIG_DIR`) |
| Claude Code (project) | `--target claude-project --project-dir PATH` | `PATH/.claude/skills/<skill>` |
| Cowork, claude.ai, cloud sessions, routines | `--target cowork` | `dist/<skill>.zip` to upload under **Customize → Skills** |
| Cursor | `--target cursor` (default) | `~/.cursor/skills/<skill>` |

Shipped skills: `remove-ai-marks` (full, service-backed) and
`clean-user-facing-text` (text only, self-contained). `--list` prints them.
Existing installations are preserved unless you pass `--force`; replacement is
staged first and the previous install is kept as a uniquely named backup.
`--link` symlinks this checkout instead of copying, so edits are picked up
live. On Windows, use `py install_skill.py ...`; the `install-skill.sh` wrapper
is provided for macOS/Linux shells.

Before writing anything, the installer validates the skill against the
[Agent Skills](https://agentskills.io) packaging rules that claude.ai uploads
and the Skills API enforce: spec-only frontmatter (`name`, `description`,
`license`, `compatibility`, `metadata`, `allowed-tools`), a lowercase hyphenated
`name` of at most 64 characters matching the directory, a non-empty
`description` of at most 1024 characters. The Cowork bundle additionally has
to fit the 30 MB upload limit, which the packager enforces.

### Automatic cleaning via hook (deterministic)

A skill is an instruction: the model decides whether to invoke it, and the
model is the thing producing the marks. A **hook** is executed by the harness
on every matching tool call, cooperation not required. That makes the hook the
deterministic half of this workflow.

The plugin registers a `PostToolUse` hook on `Write|Edit|MultiEdit|NotebookEdit`
that runs [`service/scripts/hook_written_file.py`](service/scripts/hook_written_file.py)
against the file the agent just wrote. Two modes, matching the pre-commit
convention of check-by-default:

| Mode | Behaviour |
| --- | --- |
| `check` (default) | Reports provenance marks, leaves the file alone. Findings go to the model (exit 2), so it can offer to clean them. |
| `clean` | Strips the marks in place, then tells the model the file on disk changed. |

Set the mode from the plugin's settings (**Hook mode** in `/plugin manage`,
read by the hook as `CLAUDE_PLUGIN_OPTION_HOOK_MODE`), or with
`WATERMARKS_HOOK_MODE=clean` in the environment. The hook command deliberately
does **not** interpolate `${user_config.hook_mode}`: Claude Code refuses to run
a hook that references an option the user has never opened `/plugin manage` to
set — a declared `default` does not satisfy it — so interpolating it would mean
the hook silently never runs on a fresh install. Detection reuses `audit_lib`'s
`scan_file` / `is_actionable`, so the hook, the pre-commit gate, and the CI
SARIF export agree on what counts as actionable; cleaning shells out to
`clean_file.py`, so no cleaning logic is duplicated. `clean` mode writes to a
sibling temp file and swaps only on a real difference, so files that were
already clean keep their mtime and don't retrigger file watchers.

Without the plugin, wire it in `~/.claude/settings.json` (or a project
`.claude/settings.json`) yourself:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["/path/to/watermarks-remover/service/scripts/hook_written_file.py",
                     "--mode", "check"],
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

On Windows, replace `python3` with `py`.

**What a hook cannot do.** No hook can rewrite the assistant's chat message
before you read it. Claude Code's `Stop` hook receives `last_assistant_message`
read-only, and there is no pre-send filter for final responses — the same limit
this project already documents for Cursor rules. So the deterministic guarantee
covers **files the agent writes**, plus the
[pre-commit gate](#pre-commit-hook) for anything on its way into git. Text that
only ever exists in the chat transcript still depends on the skill workflow,
which is model-instruction-based and therefore best-effort.

### Claude Code plugin (marketplace)

The repository is also a Claude Code **plugin** and a single-plugin
**marketplace** (`.claude-plugin/`), so both skills install and update in two
commands, no clone or script required:

```
/plugin marketplace add guillaumemeyer/watermarks-remover
/plugin install watermarks-remover@watermarks-remover
```

The skills then load namespaced: `/watermarks-remover:remove-ai-marks` and
`/watermarks-remover:clean-user-facing-text` (the bare `/remove-ai-marks` also
works when nothing else claims the name). `/plugin marketplace update
watermarks-remover` pulls later versions. The same works from the CLI with
`claude plugin marketplace add …` / `claude plugin install …`, and from a local
checkout by passing a path instead of `owner/repo`.

Maintainers: `make plugin-validate` runs `claude plugin validate . --strict`
against both manifests; `tests/test_plugin_manifest.py` covers the same files
without needing the CLI.

### Claude Code

```bash
# Personal — available in all your projects
python3 install_skill.py --skill remove-ai-marks --target claude-code
# or: make install-claude-code-skill

# Project — commit .claude/skills/ to share it with the repo
python3 install_skill.py --skill remove-ai-marks --target claude-project \
  --project-dir /path/to/project
# or: make install-claude-project-skill PROJECT=/path/to/project
```

Claude Code picks up personal and project skills without a restart; `/skills`
lists what it loaded. Invoke with `/remove-ai-marks` or ask to “strip AI
watermarks / C2PA / Claude marks / SynthID-class text.” A project install is
also what [cloud sessions](https://code.claude.com/docs/en/cloud-environments)
read, since they clone the repository and load its `.claude/skills/`.

### Cowork (and claude.ai, cloud sessions, routines)

Cowork sessions do **not** read `~/.claude/skills` on your machine — they load
the skills enabled for your claude.ai account, synced when the session starts.
So install there by uploading a bundle:

```bash
python3 install_skill.py --skill remove-ai-marks --target cowork
# writes dist/remove-ai-marks.zip   (make package-cowork-skill)
```

Then, in the Claude Desktop app, open **Customize → Skills → Add** and upload
the zip (the same skill settings on claude.ai work too). The bundle is
reproducible and contains a single top-level `remove-ai-marks/` directory with
`SKILL.md` at its root, which is the layout the upload expects.

Service reachability matters more here than in a local install: the skill is a
thin HTTP client, so the session must be able to reach `WATERMARKS_SERVICE_URL`.
Cowork sessions that run locally on your machine reach a local `make serve`;
cloud sessions and routines run remotely and need a service URL reachable from
there (and `WATERMARKS_SERVER_API_KEY` set on it). If you want a skill with no
service at all, upload `clean-user-facing-text` instead — it is text-only and
ships its own scripts:

```bash
python3 install_skill.py --skill clean-user-facing-text --target cowork
```

### Grok

```bash
# Grok Build / project-local
mkdir -p .grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" .grok/skills/remove-ai-marks

# User-global Grok
mkdir -p ~/.grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" ~/.grok/skills/remove-ai-marks
```

### Optional text-only skill

[`skills/clean-user-facing-text/`](skills/clean-user-facing-text/) is a
self-contained skill for authorized manuscripts, documentation, and web
copy. It excludes image, C2PA, service, and external-model tooling, and runs
its own vendored Layer A scripts instead of calling the service.

```bash
python3 install_skill.py --skill clean-user-facing-text --target claude-code
python3 install_skill.py --skill clean-user-facing-text --target cursor
```

Skill invocation is model-selected. Projects that explicitly adopt this
workflow in Cursor can also copy the optional rule:

```bash
mkdir -p /path/to/project/.cursor/rules
cp integrations/cursor/clean-user-facing-text.mdc \
  /path/to/project/.cursor/rules/clean-user-facing-text.mdc
```

For all projects, put the same instruction in Cursor **User Rules** instead.
Rules improve consistency but remain model instructions; Cursor does not expose
a deterministic pre-send filter for final chat responses.

### Start the service

The fastest path is a local HTTP server (Python 3.10+ stdlib only — no deps, no Docker):

```bash
make serve                 # http://127.0.0.1:8765
# or directly:
python3 service/scripts/server.py --host 127.0.0.1 --port 8765
```

### Windows (no Docker)

See [docs/windows-autostart.md](docs/windows-autostart.md) for auto-starting the service at Windows login without Docker.

For the whole infra (core + optional harness/heavy backends), see [Docker / compose](#docker--compose) below.

Optional system tools (auto-used when present — preinstalled in the core Docker image):

| Tool | Role |
| --- | --- |
| [`c2patool`](https://github.com/contentauth/c2pa-rs/tree/main/cli) | Inspect C2PA manifests |
| [`exiftool`](https://exiftool.org/) | Residual metadata strip (esp. **PDF**) |
| [`qpdf`](https://qpdf.sourceforge.io/) | Structural PDF rebuild — **required** for a real PDF strip (see below) |

Core scripts need **Python 3.10+** stdlib only. Layer B model calls are optional.

## Quick use (scripts)

```bash
SCRIPTS=service/scripts

# Unified inspect / clean
python3 "$SCRIPTS/inspect_file.py" draft.md
python3 "$SCRIPTS/clean_file.py" draft.md -o draft.cleaned.md
python3 "$SCRIPTS/clean_file.py" photo.png -o photo.cleaned.png
python3 "$SCRIPTS/clean_file.py" notes.docx -o notes.cleaned.docx

# Text Layer A
python3 "$SCRIPTS/inspect_text.py" draft.md
python3 "$SCRIPTS/clean_text.py" draft.md -o draft.cleaned.md --stats

# Layer B rewrite hook (default: print prompt only — no model required)
python3 "$SCRIPTS/rewrite_text.py" draft.md --backend print-prompt --strength paraphrase
# Optional local Ollama (loopback only by default — remote endpoints require
# WATERMARKS_REWRITE_ALLOW_REMOTE=1 or --allow-remote):
# WATERMARKS_REWRITE_BACKEND=ollama WATERMARKS_REWRITE_MODEL=llama3.2 \
#   python3 "$SCRIPTS/rewrite_text.py" draft.md -o draft.rewritten.md
# API keys are read from WATERMARKS_REWRITE_API_KEY only (never argv).

# Images
python3 "$SCRIPTS/inspect_image.py" shot.png
python3 "$SCRIPTS/clean_image.py" shot.png -o shot.cleaned.png
```

### Text tools refuse binary input

`inspect_text.py`, `clean_text.py` and `rewrite_text.py` operate on text. Pointed
at a `.docx`, `.pdf` or image they used to decode the compressed bytes and report
whatever codepoints fell out — noise that tracks the compression, not the
content — and `clean_text.py` then wrote those mangled bytes back, destroying the
file. They now refuse binary input and name the tool that handles it:

```bash
python3 "$SCRIPTS/inspect_text.py" report.docx
# refusing to treat report.docx as text: it looks like a ZIP container (DOCX, ODT, …).
# Use inspect_file.py / clean_file.py, which route by format,
# or pass --force-text to scan the raw bytes anyway.
```

Detection is by magic number plus a control-byte ratio, so text in encodings
other than UTF-8 keeps working. `--force-text` overrides it everywhere.

### Unrecognized formats are never auto-cleaned

`classify()` labels bytes that match no supported text, image or container
format as **`unknown`** — it no longer falls back to "text". In auto mode
`clean_file.py` refuses such files (exit 2, no output written) instead of
decoding them as UTF-8 and writing back mangled bytes; `--as text` or
`--force-text` are the explicit opt-ins. `inspect_file.py` reports the file
as `unknown` (exit 0), and the HTTP service answers `/inspect` with
`kind: "unknown"` but rejects `/clean` of unknown formats (400 — send a
filename with a known extension, e.g. `notes.txt`).

## HTTP service

The same machinery runs as a stdlib HTTP service (`service/scripts/server.py`) — the interface the skill uses and the way any web app can integrate without vendoring:

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"ok": true, "version": ...}` |
| GET | `/capabilities` | — | optional tools / backends usable (each tool is version-probed, not just found on `PATH`) |
| GET | `/openapi.json` | — | dynamically generated OpenAPI 3.0.3 spec |
| POST | `/inspect` | `{"file": "<base64>", "name": "notes.md"}` | `{"ok", "kind", "suspicious", "report"}` |
| POST | `/detect` | `{"file": "<base64>", "name": "notes.txt"}` | `{"ok", "kind", "detections": [...]}` |
| POST | `/clean` | `{"file": "<base64>", "name": "notes.md", "options": {...}}` | `{"ok", "kind", "cleaned": "<base64>", "report"}` |
| POST | `/inspect/batch` | `{"files": [{"file": "<base64>", "name": "notes.md"}, ...]}` | `{"ok", "results": [{"name", "ok", "kind", "suspicious", "report"}, ...]}` |
| POST | `/clean/batch` | `{"files": [{"file": "<base64>", "name": "notes.md", "options": {...}}, ...]}` | `{"ok", "results": [{"name", "ok", "kind", "cleaned": "<base64>", "report"}, ...]}` |

Batch endpoints loop the same per-file pipeline as `/inspect` and `/clean`, capped at `WATERMARKS_MAX_BATCH_FILES` files per request (default 50). A malformed entry (bad base64, unknown option, unrecognized format) surfaces as that entry's `"ok": false` with an `"error"` string — it never aborts the rest of the batch.

```bash
WM="http://127.0.0.1:8765"
curl -s "$WM/health"                       # {"ok": true, "version": "..."}
curl -s "$WM/openapi.json"                 # machine-readable OpenAPI 3.0.3 contract
curl -s -X POST "$WM/clean" -H 'Content-Type: application/json' \
  -d "{\"file\": \"$(base64 < notes.md | tr -d '\n')\", \"name\": \"notes.md\"}"
```

The service routes by filename extension then magic bytes, so text / image / container are auto-detected. Set `WATERMARKS_SERVER_API_KEY` to require `Authorization: Bearer <key>` on every request. Loopback-only bind by default (`--host` to override); intended for a trusted network.

### Watermark detection (`/detect` and `detect_before` / `detect_after`)

Detection is a separate step from cleaning — the service never calls vendor
APIs unless you ask it to:

- **`POST /detect`** runs the configured watermark detectors on a file.
  Text → vendor detectors + stylometry; image → SynthID pixel score.
- **`/inspect`** accepts an opt-in `"detect": true` flag that appends
  detector results to the text report (and can flip `suspicious`).
- **`/clean`** accepts `"detect_before"` / `"detect_after"` options to
  score the input and the cleaned output, so you can measure what a clean
  actually changed.

Text detectors (see `/capabilities` → `text_detectors`):

| Detector | Activated by | Notes |
| --- | --- | --- |
| `markllm` | `MARKLLM_DIR` (host checkout) | Research harness (KGW / SynthID schemes), same-config-only — not a vendor oracle. |
| `gumbel` | `WATERMARKS_GUMBEL_KEY` | Model-free same-key replay of the keyed-Gumbel (Aaronson EXP) scheme (see `detect_gumbel.py`), stdlib-only — self-hosted engines such as arbi-serve; same-key-only, not a vendor oracle. |
| `claude-text` | — (placeholder) | Anthropic has announced a watermark detection API; this seam activates when it ships. |

Image scoring: when `WATERMARKS_SYNTHID_SCORER_URL` is set, the service
scores images through the `wr-synthid-score` sidecar (heavy profile); with a
local `REVERSE_SYNTHID_DIR` it uses the checkout directly. Detection is
fail-soft: unconfigured, timed-out, or errored detectors report
`{"available": false, "error": ...}` and never block cleaning.

## Docker / compose

Published images (GHCR):

| Image tag | Contents | Published? |
| --- | --- | --- |
| `ghcr.io/guillaumemeyer/watermarks-remover:<tag>` / `:latest` | Core HTTP service + all cleaners + exiftool / qpdf / c2patool | Yes |
| `…:markllm-<tag>` / `:markllm-latest` | MarkLLM text-watermark harness (Apache-2.0 upstream) | Yes |
| `…:markdiffusion-<tag>` / `:markdiffusion-latest` | MarkDiffusion image harness (Apache-2.0 upstream) | Yes |
| `watermarks-remover-ctrlregen:local` | CtrlRegen pixel removal — **never published** (`noai-watermark` ships no LICENSE) | Local build only |
| `watermarks-remover-synthid-scorer:local` | reverse-SynthID scorer — **never published** (non-commercial Research License) | Local build only (CLI scorer + optional `wr-synthid-score` HTTP sidecar under the `heavy` profile) |

Build and run the core service:

```bash
make docker-core-build
docker run --rm -p 127.0.0.1:8765:8765 --read-only --tmpfs /tmp watermarks-remover
# any CLI stays runnable by overriding the command:
docker run --rm -v "$(pwd):/data" watermarks-remover \
  /app/scripts/clean_file.py /data/notes.md -o /data/notes.cleaned.md
```

Whole-infra bring-up:

```bash
docker compose up -d                         # core HTTP service only
docker compose --profile harness up -d       # + markllm / markdiffusion
docker compose --profile heavy up -d         # + ctrlregen / synthid (local builds)
docker compose --profile harness --profile heavy up -d   # all services
```

The compose stack maps the core service to `127.0.0.1:8765`. The harness/heavy services are one-shot CLIs — invoke with `docker compose run --rm <service> …` when you need verification or pixel work.

Validate the running stack (exit code only, no output on success):

```bash
make compose-check        # or: ./compose-check.sh
```

Checks `wr-core` via `GET /health` and runs each harness/heavy service with `--help`, requiring exit `0`.

### Configuration (env vars for docker compose)

**Nothing is required to clean arbitrary text** — the core service works out of the box:

```bash
echo "Hello\u200bWorld\u00ad!" > /tmp/sample.txt
curl -s -X POST http://127.0.0.1:8765/clean -H 'Content-Type: application/json' \
  -d "{\"file\": \"$(base64 < /tmp/sample.txt | tr -d '\n')\", \"name\": \"sample.txt\"}"
```

Languages whose typography relies on a non-breaking space (French `« … »`, the
space before `; : ! ?`) should pass `"options": {"normalize_spaces": false}`, the
HTTP equivalent of `clean_text.py --no-normalize-spaces`. Invisible carriers are
still removed; only the space rewrite is skipped.

Everything else is optional and lives in a `.env` file at the repo root. `docker compose` **auto-loads `.env`** and interpolates the `${VAR}` references in `compose.yaml` from it (shell exports win over `.env` if both are set).

```bash
cp .env.example .env       # then edit
docker compose up -d       # picks up .env automatically
```

`.env` is **gitignored** (deny-by-default) — never commit it. For host-side CLI runs (`rewrite_text.py`, the skill), export the same file into the environment:

```bash
set -a; . ./.env; set +a; python3 service/scripts/rewrite_text.py /tmp/x.txt -o /tmp/x.rewritten.txt
```

| Var | Reaches | Purpose |
| --- | --- | --- |
| `WATERMARKS_SERVER_API_KEY` | `wr-core` (via compose `environment`) | Require `Authorization: Bearer <key>` on the HTTP API |
| `WATERMARKS_GEMINI_*` | — | Removed Aug 2026: Google retired SynthID text watermarking on the API (see `vendor-notes.md`) |
| `WATERMARKS_SYNTHID_SCORER_URL` | `wr-core` | Point core at the `wr-synthid-score` sidecar for SynthID image scoring (e.g. `http://wr-synthid-score:8766` under the heavy profile) |
| `WATERMARKS_SYNTHID_SCORER_API_KEY` | `wr-core` + `wr-synthid-score` | Shared bearer key for the scorer sidecar (empty = no auth) |
| `WATERMARKS_MARKLLM_SCHEME` | `text_detectors.py` (host) | MarkLLM scheme for `/detect`: `kgw` (default) / `synthid` |
| `HF_TOKEN` | harness/heavy services | Hugging Face token for gated models |
| `WATERMARKS_SERVICE_URL` | client only (skill / curl) | Where to reach the service; default `http://127.0.0.1:8765` |
| `WATERMARKS_REWRITE_BACKEND` | `rewrite_text.py` hook | `print-prompt` (default) / `ollama` / `openai-compatible` |
| `WATERMARKS_REWRITE_MODEL` | `rewrite_text.py` hook | Model name (e.g. `deepseek-v4-flash`) |
| `WATERMARKS_REWRITE_BASE_URL` | `rewrite_text.py` hook | API base (e.g. `https://api.deepseek.com`) |
| `WATERMARKS_REWRITE_API_KEY` | `rewrite_text.py` hook | API key — env only, never on argv |
| `WATERMARKS_REWRITE_ALLOW_REMOTE` | `rewrite_text.py` hook | `1` to allow non-loopback endpoints |
| `WATERMARKS_REWRITE_REASONING_EFFORT` | `rewrite_text.py` hook | `none` (default) / `low` / `medium` / `high` / `off` |
| `WATERMARKS_GUMBEL_KEY` | `detect_gumbel.py` / `text_detectors.py` | Secret key for keyed-Gumbel (EXP) same-key replay (e.g. `0x…`); preferred over argv — never logged |

Layer B is agent-orchestrated in the skill (it rewrites with its own model), so the `WATERMARKS_REWRITE_*` vars are only needed when driving `rewrite_text.py` directly.

Images publish automatically on `v*` tags via [`.github/workflows/release-images.yml`](.github/workflows/release-images.yml).

## Optional SynthID pixel scoring

`inspect_image.py` and `clean_image.py` can report a pixel-domain SynthID
confidence score when an external checkout of
[`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID)
is available. The scorer is **not bundled**: it is loaded at runtime from your
checkout, and its code remains under the upstream project's non-commercial
Research License.

### Option 1: one-command bootstrap (no Docker)

```bash
SCRIPTS=service/scripts

# Clones upstream, creates a venv, and installs scorer-only dependencies.
"$SCRIPTS/setup_synthid.sh"

# Score an image (default checkout: ~/reverse-SynthID).
REVERSE_SYNTHID_DIR=~/reverse-SynthID \
~/reverse-SynthID/.venv/bin/python "$SCRIPTS/score_synthid.py" shot.png

# Or surface the score from inspect / clean (same venv Python).
REVERSE_SYNTHID_DIR=~/reverse-SynthID \
~/reverse-SynthID/.venv/bin/python "$SCRIPTS/inspect_image.py" shot.png
```

`setup_synthid.sh` accepts `--dir PATH`, `--ref REF`, and `--full` (install the
full upstream `requirements.txt`, which adds `torch`/`diffusers` for the
upstream VAE bypass this project does not use).

On Windows use `setup_synthid.ps1` (`-Dir`, `-Ref`, `-Full`), which creates the
venv at `.venv\Scripts\` — the layout `image_meta.py` already looks for on
`os.name == "nt"`.

### Option 2: local Docker build

```bash
make docker-synthid-build
# Run unprivileged and with a read-only rootfs; the scorer only needs to read
# /data and write to stdout/tmp.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --read-only --tmpfs /tmp \
  -v "$(pwd):/data" \
  watermarks-remover-synthid-scorer /data/shot.png
```

The image is built locally from the upstream source at build time. It is not
published, so it does not redistribute the upstream code.

### Option 3: HTTP scorer sidecar (docker compose)

Under the `heavy` profile the compose stack also runs the scorer as an HTTP
sidecar (`wr-synthid-score`) so the **published core service** can score
images before/after cleaning without bundling the non-commercial upstream
code. Point `wr-core` at it and share a bearer key (see `.env.example`):

```bash
# .env
WATERMARKS_SYNTHID_SCORER_URL=http://wr-synthid-score:8766
WATERMARKS_SYNTHID_SCORER_API_KEY=change-me

docker compose --profile heavy up -d
```

Then `POST /clean` with `{"options": {"detect_before": true,
"detect_after": true}}` returns `synthid_before` / `synthid_after` in the
report, and `POST /detect` on an image returns the SynthID score. Fail-soft:
if the sidecar is down or unconfigured, reports carry
`{"available": false, "error": ...}` and cleaning still succeeds.

V4 scoring uses `artifacts/spectral_codebook_v4.npz` from the upstream checkout
(`220 MB). This is **detection/scoring only** — it does not remove pixel
watermarks.

## Optional CtrlRegen pixel removal

For **pixel-domain** image watermarks (SynthID-class, StegaStamp, Tree-Ring,
StableSignature), an optional external backend runs the CtrlRegen pipeline
(ControlNet + DINOv2 IP-Adapter controllable regeneration). The backend is
[`mertizci/noai-watermark`](https://github.com/mertizci/noai-watermark), a
maintained reimplementation of the ICLR 2025
[CtrlRegen](https://arxiv.org/abs/2410.05470) method with automatic tiling.

The backend is **not bundled** and ships no LICENSE file, so it is treated as
all-rights-reserved: it is cloned at a pinned commit and loaded at runtime.
Its research-era dependency pins (`requirements-ctrlregen.txt` — e.g.
`transformers==4.37.2`, `diffusers==0.27.2`) carry published advisories and
are intentionally not current, so they are only ever installed inside the
dedicated venv this script creates and never into the main service image;
`setup_ctrlregen.sh` also re-verifies the pinned commit on existing
checkouts, not just fresh clones.

### Bootstrap

```bash
SCRIPTS=service/scripts

# Clones upstream (pinned commit), creates a venv, installs torch + deps.
"$SCRIPTS/setup_ctrlregen.sh"

# Standalone removal (default checkout: ~/noai-watermark).
NOAI_WATERMARK_DIR=~/noai-watermark \
~/noai-watermark/.venv/bin/python "$SCRIPTS/clean_ctrlregen.py" shot.png -o shot.ctrlregen.png
```

On Windows use `setup_ctrlregen.ps1` (same flags as `-Dir`, `-Ref`, `-Python`);
the venv lands in `.venv\Scripts\`, which `clean_image.py` already resolves.
It probes the published PyTorch wheel indices and picks the highest one at or
below the CUDA version `nvidia-smi` prints that actually exists — that number
is the maximum the *driver* supports, and drivers are backward compatible, so a
driver reporting 13.1 (no published `cu131`) installs `cu130`. Below compute
capability 7.5 it forces `cu126`, the last index whose wheels still carry
Maxwell/Pascal/Volta kernels. It installs `torch` **and** `torchvision`
together from that index so the dependency install cannot swap them for CPU
builds from PyPI, then verifies after install that `torch.cuda.is_available()`
is true — if a GPU was detected but torch ends up CPU-only, the script warns
loudly and exits non-zero instead of pretending the setup succeeded.

### From `clean_image.py`

```bash
NOAI_WATERMARK_DIR=~/noai-watermark \
~/noai-watermark/.venv/bin/python "$SCRIPTS/clean_image.py" shot.png \
  -o shot.cleaned.png --remove-pixel ctrlregen
```

Order of operations: metadata strip first, then CtrlRegen pixel removal, then
an optional reverse-SynthID before/after score (when `REVERSE_SYNTHID_DIR` is
also set).

**Strength is conservative by default** (`--ctrlregen-strength 0.25`), because
higher strength removes more watermark but regenerates more of the image.
Documented presets: `0.15` minimal / `0.25` default / `0.35` balanced /
`0.5` aggressive / `0.7` max (backend default is 0.5). `--ctrlregen-steps`
defaults to 50 (effective denoising steps ≈ steps × strength).

### Image size (512×512 native limit)

CtrlRegen is a 512×512 Stable Diffusion 1.5 ControlNet. The backend resolves
this for arbitrary inputs, so no extra tiling is exposed here:

- **≤512 px:** single pass — center-crop/resize to 512, regenerate, resize back.
- **>512 px:** automatic overlapping tiling (512 px tiles, 192 px overlap),
  width/height aligned to multiples of 8, then cosine-blended seams.
- **Either path:** output is resized to the original size and color-matched to
  the original image.

Very large images (e.g. 4K) produce many tiles, so runs scale with tile count
(slower and higher VRAM). Pre-downscale large inputs when practical; tile size
and overlap are hardcoded upstream and are not exposed as flags.

### Compute, gated models, and verification

Expect ~10 GB of model downloads; a GPU is strongly recommended and CPU runs
are slow. Some upstream models are gated, so export `HF_TOKEN` (env only —
never argv). `clean_ctrlregen.py` refuses to auto-install dependencies; run
`setup_ctrlregen.sh` first.

There is no local detector for StegaStamp/Tree-Ring/StableSignature, so the
only local signal is the reverse-SynthID score (a surrogate). When available,
`clean_image.py --remove-pixel ctrlregen` reports that score before/after; the
official Google SynthID check remains the final authority.

### Docker

```bash
make docker-ctrlregen-build
docker run --rm -e HF_TOKEN="$HF_TOKEN" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd):/data" \
  watermarks-remover-ctrlregen /data/shot.png -o /data/shot.ctrlregen.png
```

## Optional MarkLLM text-watermark verification

For **controlled experiments**, an optional external harness wraps
[`THU-BPM/MarkLLM`](https://github.com/THU-BPM/MarkLLM) (Apache-2.0) to
watermark test text and re-detect it after a Layer B rewrite — e.g. prove that
a KGW (Kirchenbauer, your "open-LLM" row) or SynthID-Text (Gemini row) mark
disappears under your rewrite. It is a **verification harness, not an oracle**:
MarkLLM detection is only valid against the *same* scheme config + keys used at
generation, and it cannot certify a vendor detector will fail.

The backend is **not bundled**. `setup_markllm.sh` clones upstream at a pinned
commit, creates a venv, and installs pinned deps (torch + transformers); the
scoring model (default `facebook/opt-1.3b`, Apache-2.0) downloads from Hugging
Face on first run.

```bash
SCRIPTS=service/scripts

# Bootstrap (clones upstream, creates ~/MarkLLM/.venv, installs deps).
"$SCRIPTS/setup_markllm.sh"

# Generate watermarked + unwatermarked sample text under the KGW scheme.
MARKLLM_DIR=~/MarkLLM \
  ~/MarkLLM/.venv/bin/python "$SCRIPTS/detect_text_watermark.py" watermark prompt.txt \
    --scheme kgw -o wm.txt -o2 plain.txt

# Detect the scheme mark in a text file.
MARKLLM_DIR=~/MarkLLM \
  ~/MarkLLM/.venv/bin/python "$SCRIPTS/detect_text_watermark.py" detect wm.txt --scheme kgw --json
```

**Verification around a Layer B rewrite:** pass `--markllm-scheme` to
`rewrite_text.py` (with `--markllm-dir`), and it records the MarkLLM detection
before/after plus a `cleared` flag:

```bash
export WATERMARKS_REWRITE_BACKEND=ollama WATERMARKS_REWRITE_MODEL=llama3.2
MARKLLM_DIR=~/MarkLLM \
  python3 "$SCRIPTS/rewrite_text.py" wm.txt -o wm.rewritten.txt \
    --markllm-scheme kgw --markllm-dir "$HOME/MarkLLM" --json-stats
```

**Detection-guided iterative rewriting:** Layer B now rewrites iteratively and
stops as soon as an attempt passes evaluation. Each evaluation round generates
`--candidates` variants (default **1**, `WATERMARKS_REWRITE_CANDIDATES`)
and `--max-loops` caps how many rounds run before the best-effort variant is
returned (default **1**, `WATERMARKS_REWRITE_LOOPS`). Each variant is one
rewrite call plus one evaluation, and a round exits early on the first attempt
the evaluator reports as not watermarked — so raising `--max-loops` retries
new variants until an evaluation passes (a typical clean rewrite costs one
attempt). The evaluator is chosen by priority:

1. **MarkLLM** — same-config research detection, when `--markllm-scheme` is
   passed (with `--markllm-dir`). A vendor-detector slot is reserved above
   MarkLLM for Google's SynthID-text detector, which Google retired on its API
   in Aug 2026 — a future vendor endpoint can plug in there.
2. **bigram-Jaccard lexical divergence** — when no detector is configured; no
   pass/fail verdict, so every attempt is generated and the most lexically
   diverged one is selected (the original behavior).

`--json-stats` reports the evaluator, attempts made, pass/fail, and per-attempt
records:

```json
{
  "evaluator": "markllm",
  "candidates": 1,
  "max_loops": 2,
  "attempts_made": 2,
  "passed": true,
  "candidate_scores": [
    {
      "lexical_divergence": 0.91,
      "selection_score": 0.91,
      "selected": false,
      "passed": false,
      "evaluation": {"detector": "markllm", "available": true, "scheme": "kgw",
                     "is_watermarked": true, "score": 4.3, "threshold": 3.0}
    },
    {
      "lexical_divergence": 0.84,
      "selection_score": 0.84,
      "selected": true,
      "passed": true,
      "evaluation": {"detector": "markllm", "available": true, "scheme": "kgw",
                     "is_watermarked": false, "score": 1.7, "threshold": 3.0}
    }
  ],
  "markllm": {"scheme": "kgw", "before": {"...": "..."}, "after": {"...": "..."},
              "cleared": true, "note": "same-config only"}
}
```

A detector that is unconfigured, times out, or errors yields an
`"available": false` entry with an `error` reason and never fails the
rewrite — that attempt simply cannot pass, and the loop falls back to
lexical-divergence selection. When the max is exhausted without a pass, the
least-watermarked (lowest score) attempt is returned as best-effort with a
note.

If the backend is unconfigured or its deps are missing, the rewrite proceeds
and the report notes verification was unavailable. A GPU is recommended; CPU
runs work but are slow, and the model download is a few GB.

Hardening knobs:

- `--offline` on the adapter (or any MarkLLM run) loads the scoring model from
  the Hugging Face cache only — zero network egress; fails fast if not cached.
  Custom remote code is never executed (transformers `trust_remote_code` is
  never enabled).
- `WATERMARKS_MARKLLM_RLIMIT_AS=<bytes>` (env, POSIX) applies an address-space
  limit to the MarkLLM detector subprocess. Off by default because torch/CUDA
  usually needs large address spaces.
- Config files are capped at 1 MiB; the upstream checkout and the base image
  are pinned by SHA/digest.

### Docker

```bash
make docker-markllm-build
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd):/data" \
  watermarks-remover-markllm detect /data/wm.txt --scheme kgw --json
```

### Keyed-Gumbel (Aaronson EXP) same-key verification

[ARBI's technical report](https://arbicity.com/news/ai-text-watermarking-for-self-hosted-ai/) describes the
keyed-Gumbel ("exponential") text watermark — now shipping in the open-source
arbi-serve engine (`ARBI_WATERMARK_KEY`) — where the sampler's noise is derived
from a keyed hash of the last 4-token context window. Detection is a
**model-free replay**: recompute `u = PRF(Hash(key, window), token)` from the
text alone and test the Gamma tail, so it needs no GPU, model, or logits.
This repo ships that detector as `detect_gumbel.py` (stdlib-only; the p-value
is the exact Poisson-sum identity for an integer Gamma shape):

```bash
# Text mode (deterministic word/run tokenizer) — quick checks and rewrite-loop
# evaluation; exact replay against a real engine needs its tokenizer:
python3 service/scripts/detect_gumbel.py draft.txt --key 0x... --json

# Exact replay: pass the engine's token ids (JSON array or one per line).
python3 service/scripts/detect_gumbel.py ids.json --tokens --key 0x... --json
```

Same honesty caveat as MarkLLM: this is a **same-key replay** — valid only
against the same key, tokenizer, and PRF layout used at generation, and a
negative result establishes nothing. The HMAC-SHA256 layout here is an
auditable instantiation, not bit-compatible with any specific engine kernel
(see the module docstring for what to adapt for exact replay).

**Detection-guided rewriting:** pass `--gumbel-key` to `rewrite_text.py`
(env: `WATERMARKS_GUMBEL_KEY`, preferred) and the iterative rewrite loop is
driven by the same-key Gumbel replay — evaluator priority becomes gumbel >
MarkLLM > lexical divergence — with a `gumbel.before/after/cleared` report:

```bash
export WATERMARKS_REWRITE_BACKEND=ollama WATERMARKS_REWRITE_MODEL=llama3.2
export WATERMARKS_GUMBEL_KEY=0x...
python3 "$SCRIPTS/rewrite_text.py" wm.txt -o wm.rewritten.txt --json-stats
```

The key never appears in stats or logs. Self-hosted operators who hold their
engine's key can verify a rewrite cleared a Gumbel mark; everyone else treats
Layer B as best-effort only.

## Optional SynthID-text removal benchmark

[`bench_synthid_text.py`](service/scripts/bench_synthid_text.py) measures how
effectively a Layer B rewrite clears SynthID-text-class watermarks and at
what cost. It generates watermarked + unwatermarked samples with the MarkLLM
SynthID scheme (same-config detection, sanity-gated), runs your rewrite
variants (strength × max rewrite attempts; the loop stops early on pass) plus
controls (no-removal, Layer-A-only, optional re-stamp check), and writes a
shareable `report.md` /
`results.json` / `results.csv`. Full guide:
[`docs/synthid-text-benchmark.md`](docs/synthid-text-benchmark.md).

Requires a MarkLLM checkout (`setup_markllm.sh` / `MARKLLM_DIR`) and a
rewrite backend. **The rewriting model is an LLM you configure** — the same
`rewrite_text.py` backend the skill uses. MarkLLM's default
`facebook/opt-1.3b` (`--markllm-model`) is only the watermark
generator/detector; it never rewrites. Configure the rewrite model via env
vars or benchmark flags (they mirror the
[config table](#configuration-env-vars-for-docker-compose) above):

| Env var | Benchmark flag | Default | Meaning |
| --- | --- | --- | --- |
| `WATERMARKS_REWRITE_BACKEND` | `--rewrite-backend` | `ollama` | `ollama` or `openai-compatible` |
| `WATERMARKS_REWRITE_MODEL` | `--rewrite-model` | *(required)* | The LLM that performs the rewrite (e.g. `llama3.2`, `deepseek-v4-flash`) |
| `WATERMARKS_REWRITE_BASE_URL` | `--rewrite-base-url` | `http://127.0.0.1:11434` | Endpoint; the Ollama default is loopback |
| `WATERMARKS_REWRITE_API_KEY` | `--rewrite-api-key` | — | API key (env-only in the child process, never argv) |
| `WATERMARKS_REWRITE_ALLOW_REMOTE=1` | `--rewrite-allow-remote` | off | Required to send content to non-loopback endpoints |

```bash
# Ollama (loopback):
python3 service/scripts/bench_synthid_text.py --markllm-dir ~/MarkLLM \
  --rewrite-backend ollama --rewrite-model llama3.2

# OpenAI-compatible API (remote):
WATERMARKS_REWRITE_API_KEY=... python3 service/scripts/bench_synthid_text.py \
  --markllm-dir ~/MarkLLM --rewrite-backend openai-compatible \
  --rewrite-model deepseek-v4-flash --rewrite-base-url https://api.deepseek.com \
  --rewrite-allow-remote
```

Use a **non-origin model** for rewriting (do not rewrite with the same
watermarked model that generated the text) or the rewrite can re-stamp the
output; `--restamp-control` measures this.

## Optional MarkDiffusion image-watermark harness

For **controlled experiments on images**, an optional external harness wraps
[`THU-BPM/MarkDiffusion`](https://github.com/THU-BPM/MarkDiffusion) (Apache-2.0),
a *generative watermarking* toolkit for latent diffusion models (it embeds marks
— it does not remove them). We use it for three things:

1. **Verification harness** (like MarkLLM, but for images): watermark a test
   image with a scheme, run removal, and re-detect with the *same* scheme config
   — e.g. prove a Tree-Ring-class mark clears under your pipeline. It is a
   **verification harness, not an oracle**: detection requires the generating
   model (and keys for key-based schemes), so it cannot certify a vendor
   detector will fail on an arbitrary image.
2. **Optional pixel-removal engine**: its `DiffusionPurification` regeneration
   attack is exposed as `clean_image.py --remove-pixel diffusion`, an
   alternative to CtrlRegen. It is **blind** regeneration (no ControlNet
   conditioning), so it drifts image content more than CtrlRegen — conservative
   strength default (`0.3`), treated as a fallback/comparison, never a
   guarantee.
3. **Local same-scheme detector** for Tree-Ring-class marks, partially filling
   the "no local detector for StegaStamp/Tree-Ring/StableSignature" gap (it
   covers Tree-Ring/Ring-ID/Gaussian-Shading etc., not StegaStamp /
   StableSignature / SynthID-media).

The backend is **not bundled**. `setup_markdiffusion.sh` creates a venv and
installs `markdiffusion==1.0.2` from PyPI (pinned), with torch installed from
the right platform index; `--checkout` installs an editable clone at a pinned
commit instead. The Stable Diffusion model (default
`huanzi05/stable-diffusion-2-1-base`) downloads from Hugging Face on first run.

```bash
SCRIPTS=service/scripts

# Bootstrap (PyPI pin default; creates ~/markdiffusion/.venv, installs deps).
"$SCRIPTS/setup_markdiffusion.sh"

# 1. Generate a Tree-Ring watermarked image (+ unwatermarked control).
echo "a red fox in snow" > /tmp/prompt.txt
MARKDIFFUSION_DIR=~/markdiffusion \
  ~/markdiffusion/.venv/bin/python "$SCRIPTS/markdiffusion_harness.py" watermark \
    /tmp/prompt.txt -o wm.png -o2 plain.png --scheme tr --json

# 2. Remove with the DiffusionPurification regeneration attack.
MARKDIFFUSION_DIR=~/markdiffusion \
  ~/markdiffusion/.venv/bin/python "$SCRIPTS/markdiffusion_harness.py" purify \
    wm.png -o wm.purified.png --purification-strength 0.3 --json

# 3. Re-detect with the SAME scheme config.
MARKDIFFUSION_DIR=~/markdiffusion \
  ~/markdiffusion/.venv/bin/python "$SCRIPTS/markdiffusion_harness.py" detect \
    wm.purified.png --scheme tr --detector-type l1_distance --json
```

Or run purification as part of the normal image pipeline:

```bash
MARKDIFFUSION_DIR=~/markdiffusion \
  ~/markdiffusion/.venv/bin/python "$SCRIPTS/clean_image.py" shot.png \
    -o shot.cleaned.png --remove-pixel diffusion
```

Hardening knobs mirror the MarkLLM harness: `--offline` loads the model from
the Hugging Face cache only (zero network egress, no remote code), `HF_TOKEN`
is env-only (never argv), algorithm configs are capped at 1 MiB, and the
subprocess gets the same higher resource caps as CtrlRegen.

### Docker

```bash
make docker-markdiffusion-build
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd):/data" \
  watermarks-remover-markdiffusion detect /data/wm.png --scheme tr --json
```

The image installs a CPU torch; CUDA users should run `setup_markdiffusion.sh`
on the host instead. Model downloads still hit the HF hub on first run.

## Coverage matrix

| Channel | Claude | Gemini/SynthID | OpenAI | Open-LLM |
| --- | --- | --- | --- | --- |
| Unicode / edit-based text | Layer A | Layer A | Layer A | Layer A |
| **Statistical sampling text** | Layer B best-effort (Claude seam when Anthropic's detection API ships) | Layer B best-effort (+ MarkLLM same-config harness; Google retired the vendor detector Aug 2026) | Layer B if present | Layer B best-effort + optional MarkLLM harness |
| C2PA / file metadata | Yes (listed formats) | Yes when present | Yes when present | Yes when present |
| Pixel image marks | Out of scope | Optional SynthID score + CtrlRegen removal (external); optional MarkDiffusion same-scheme detect + DiffusionPurification removal (external) | Out of scope | Optional CtrlRegen / MarkDiffusion removal (external) |
| Training backdoors | Out of scope | Out of scope | Out of scope | Out of scope |

Details: [`skills/remove-ai-marks/references/vendor-notes.md`](skills/remove-ai-marks/references/vendor-notes.md), [`mark-classes.md`](skills/remove-ai-marks/references/mark-classes.md).

---

## How text marking works (short)

Modern LLM watermarks often hide a signal in **which tokens are chosen** (generative / sampling bias), not only in invisible characters. Edit-based schemes inject Unicode or synonym rules. File schemes attach **C2PA** or generator metadata.

- **Layer A** removes edit-based Unicode carriers (testable).
- **Layer B** attacks sampling watermarks via heavy rewrite (best-effort; literature-standard attacks such as paraphrase / back-translation).
- **File cleaners** strip C2PA/XMP/props from supported containers.

Until vendors ship public detectors and keys, **no tool can honestly certify** “this fails the official check.” Reports must separate verifiable vs best-effort work.

Prefer a **non-origin** model for Layer B (do not rewrite Claude text with Claude if you are trying to avoid re-stamping).

---

## Disclaimer: what removing a text watermark costs

Text watermarks live in **the wording itself**: the signal is spread across token choices, so nearly every sentence carries a little of it. Two consequences follow, and they are why Layer B is honestly described as *best-effort* rather than a magic eraser.

1. **Removal means rewording, not restructuring.** Shuffling paragraphs, changing headings, or light touch-ups barely move the signal. Stripping a statistical mark requires rewriting a substantial fraction of the text — sentence by sentence, not section by section.

2. **Rewording degrades the copy.** Any rewrite replaces the original word choices with the rewriting model's, which flattens tone, voice, and precision. On production copy (SEO, marketing, client work) that degradation is real and often visible to the people who care most about the writing. It is like taking text from a top-tier model and asking a less capable model to rewrite it from scratch: the result cannot exceed the rewrite model's ceiling.

Which leads to the honest full-circle question:

> If the plan is to rewrite the text with a cheaper model anyway, why pay for a premium model in the first place? Generating directly with the cheaper model is simpler, cheaper, and produces the same — or better — end result.

Layer B makes sense when you specifically want the premium model's **thinking and drafting** and accept a rewrite pass to satisfy a hygiene or privacy requirement — not as a cheap route to mark-free text.

**When to skip Layer B:**

- **Quality matters more than hygiene:** use the lossless path — Layer A Unicode scrub plus the file metadata cleaners — and keep the original prose.
- **Rewriting anyway:** use a **non-origin** model (rewriting with the origin model can re-stamp the text), and remember residual risk remains — no tool can certify a vendor detector will fail.

---

## File formats

| Format | Inspect | Clean |
| --- | --- | --- |
| PNG / JPEG / WebP | C2PA chunks / APP11 / RIFF `C2PA`, AI XMP hints | Drop metadata segments |
| AVIF / HEIC | ISOBMFF `jumb` / XMP `uuid` boxes | Drop boxes |
| BMP | Trailing non-image bytes (no standardized channel) | Truncate trailing metadata, fix file-size field |
| GIF | Comment / XMP application extensions | Drop comment & XMP, keep `NETSCAPE2.0` loop |
| TIFF (classic + BigTIFF) | IFD tags: XMP, EXIF, GPS, IPTC, MakerNote | Drop tags, zero payloads, keep strips |
| SVG | `<metadata>`, XMP | Strip blocks |
| PDF | Byte/XMP + optional tools | **exiftool** then **qpdf**, then **ghostscript** for metadata inside embedded images; each missing tool degrades a different layer (document strip, structural rewrite, embedded images) |
| DOCX | docProps / customXml | Scrub props, drop customXml |
| EPUB | OPF metadata, XHTML meta/JSON-LD, embedded media | Scrub OPF, strip XHTML meta, clean media + Layer A (skips encrypted parts) |
| ODT | meta.xml | Drop generator / AI-ish meta |
| HTML | meta, JSON-LD, data-ai* | Strip tags/attrs |
| Markdown | YAML frontmatter AI keys | Drop keys + Layer A body |
| MP4 / MOV / M4A / M4V | ISOBMFF `jumb`/`uuid` boxes (same mechanism as AVIF/HEIC) + `moov/udta` generator tags | Drop boxes |
| WAV | RIFF `C2PA` / `LIST INFO` chunks, embedded `id3\x20` chunk | Drop chunks |
| MP3 | ID3v2 frames (v2.3/v2.4 per-frame; v2.2 whole-tag) | Drop matched frames or whole tag |
| FLAC | C2PA manifest in an ID3v2 `GEOB` frame | Drop the matched frame or whole ID3v2 tag |

FLAC support covers C2PA's standardized ID3v2 carrier. Native FLAC metadata blocks,
Vorbis Comments, and waveform-domain watermarks are left untouched.

#### Why PDF needs qpdf, not just exiftool

ExifTool writes PDFs **incrementally**. `exiftool -all=` appends a
`%BeginExifToolUpdate` block that frees the Info object and drops `/Info` from
the trailer — but the original metadata bytes stay in the file verbatim, and
exiftool itself can undo the edit with `-PDF-update:all=`. The command exits
`0`, viewers show no metadata, and the file gets *larger*, which is the tell.

For a provenance-stripping tool that is a silent leak, so `clean_pdf` follows
the exiftool pass with `qpdf --linearize`, which re-serializes the document
from its object graph and drops the now-unreferenced objects. Without `qpdf`
installed the clean still runs, but it says so:

```
warning: exiftool PDF edits are incremental — the original metadata bytes
remain recoverable; install qpdf for a structural rewrite
```

#### Why qpdf is not enough for images inside the PDF

Both passes above work on the document: the Info dictionary, the XMP packet,
the object graph. Neither descends into an image XObject, so a scan or a
Photoshop export — a page that *is* one big JPEG — keeps whatever the image
carries. On a real Photoshop-exported PDF that leaves 27 tags in place after a
"successful" clean, `IFD0:Software`, the capture timestamps and a preview
thumbnail among them; a C2PA manifest attached to the same image survives it
too.

So `clean_pdf` adds a third pass, `deep_images`, driven by Ghostscript's
`pdfwrite`. It runs in two rungs and stops as soon as the file is clean:

1. **Lossless.** `pdfwrite` with pass-through rebuilds the document from the
   object graph while copying the compressed image data byte-for-byte — verified
   by hashing the streams before and after. This clears everything the PDF
   wrapped around the image. Pass-through covers the codecs Ghostscript supports
   for it, JPEG (DCTDecode) and JPEG2000 (JPXDecode); Flate, CCITT and LZW
   images are decoded and re-encoded, which is lossless in practice for those
   codecs but not byte-identical. `never` is the option for a document whose
   streams must survive untouched.
2. **Re-encode, only on evidence.** Anything living in the JPEG's own APPn
   segments — EXIF in APP1, a C2PA manifest in APP11, Photoshop resources in
   APP13 — travels with the bytes it is attached to, so pass-through preserves
   it. Rung 2 runs the same pass with pass-through off, and only when rung 1
   demonstrably left something behind: an AI/C2PA marker in any mode, or, under
   `always`, any surviving APPn metadata. APP0 (JFIF) and APP2 (ICC) are left
   alone — the first is structural and the second decides how the colours are
   read. Pixels are spent on evidence, never on suspicion.

`deep_images` takes `auto` (default: rung 1 only when markers survived the
document strip, then rung 2 if they survive that), `always` (rung 1 for every
PDF, escalating to rung 2 for camera and editor EXIF too), `lossless` (rung 1
only — never recompress, and report whatever survives through the usual
`still_has_c2pa` / `post_findings` fields) and `never`. An unrecognised value is
rejected rather than quietly treated as `auto`. The report says which rungs ran
via `meta.deep_image_pass` and `meta.images_reencoded`, and when the pass is
skipped it names the option that would go further:

```text
deep image pass not needed for AI/C2PA markers; pass deep_images="always"
to also clear non-AI EXIF inside images
```

Without Ghostscript installed the clean still runs and says what it could not
reach:

```text
warning: metadata inside embedded images left in place; install ghostscript
for the deep image pass
```

Pixel-domain watermark **removal** is now available as an optional external CtrlRegen backend (see above); it is a regenerating remover, not a guarantee. **C2PA soft binding** (in-content watermark that can re-link a remote Content Credentials manifest after metadata is stripped) remains **out of scope**. Stripping hard-bound C2PA does **not** clear those channels.

### Residual risk after a clean

This tool reports **verifiable** removals (Unicode counts, metadata actions) and **best-effort** Layer B rewrites. It cannot certify that vendor detectors will fail.

To check residual signals yourself (optional, external):

| Channel | What we remove | What may remain | External check (examples) |
| --- | --- | --- | --- |
| Hard-bound C2PA / EXIF / XMP | Yes | Soft-bound / pixel marks | [c2patool](https://github.com/contentauth/c2pa-rs/tree/main/cli), [Content Credentials verify](https://contentcredentials.org/verify) |
| SynthID-class media | Optional pixel removal (external CtrlRegen); local score otherwise | Audio/video watermark; residual pixel watermark after removal | Provider tools (e.g. [Google SynthID](https://deepmind.google/science/synthid/) / Vertex detector where offered); optional local [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) scorer |
| Statistical text | Best-effort rewrite | Strong marks after light edit | No public universal detector; vendor tools when available |

Industry two-layer context (C2PA + imperceptible watermark): [Institute of AI PM guide](https://www.institutepm.com/knowledge-hub/ai-content-provenance-watermarking).

---

## Removal options (summary)

| Option | Removes | Notes |
| --- | --- | --- |
| Unicode scrub (Layer A) | ZWSP, bidi, tags, exotic spaces, … | Safe default for text |
| Rewrite (Layer B) | Statistical token marks (best-effort) | Always offered by skill; costs style — see [Disclaimer](#disclaimer-what-removing-a-text-watermark-costs) |
| Container/metadata strip | File provenance | See format table |
| CtrlRegen pixel removal (optional) | Pixel-domain image marks (SynthID-class, StegaStamp, Tree-Ring, StableSignature) | External backend; heavy compute; conservative strength default |
| DiffusionPurification pixel removal (optional) | Pixel-domain image marks (Tree-Ring-class) | MarkDiffusion backend; blind regeneration (more drift than CtrlRegen); conservative strength default |
| Open-weight local models | Avoid re-stamping with origin model | Operational alternative |

Matrix: [`skills/remove-ai-marks/references/removal-matrix.md`](skills/remove-ai-marks/references/removal-matrix.md).

## Ethics and disclaimer

See [`skills/remove-ai-marks/references/ethics.md`](skills/remove-ai-marks/references/ethics.md). For privacy and research on **your** content — not academic fraud or false “human-written” claims.

**Responsible use:** This project is for content you own or are authorized to process. Users must adhere to local regulations and use it responsibly. The developers disclaim any liability for potential misuse by users.

## Ecosystem

Third-party projects that wrap or complement this repository, listed for discoverability only. **They are not maintained, endorsed, or supported by this project.** This project does not review their code, vouch for their behavior or guarantees, or take responsibility for anything you install or run from this list. Each project is governed by its own license, maintainers, and documentation — read those before using it.

### MetaClean — desktop GUI

[MetaClean](https://github.com/Moresyl/metaclean) is an independent MIT-licensed Rust/Tauri desktop application (Windows, macOS, Linux) providing a packaged native GUI for drag-and-drop metadata cleaning, with a system tray and Explorer integration. It is a separate codebase: it does not call this repository's Python service, and its supported formats and cleaning guarantees differ from this project's. See its README for details.

### unmark-web — browser web UI

[unmark-web](https://github.com/ivanusto/unmark-web) is an independent, MIT-licensed static web client. It removes invisible Unicode marks from text and strips provenance metadata from images entirely in the browser, and can optionally call this repository's HTTP service for the formats it does not handle locally. It is a separate codebase and is not affiliated with this project; see its README for scope and limits.

### Adding a project

To register a project here, open a PR adding a short entry — project name, what it wraps or adds, and a link to its own repository. Keep entries brief and factual; do not claim compatibility with, or endorsement by, this project. A listed project should build on or integrate this repository — for example, by calling its service or reusing its detection engine — rather than merely address the same problem independently. Please avoid names that start with or closely resemble `watermarks-remover` — look-alike names make it hard to tell which project is which.

## Pre-commit hook

CI gating already exists (`audit_dir.py`'s SARIF export, see [Coverage matrix](#coverage-matrix) context) — the [pre-commit](https://pre-commit.com/) hooks below catch the same class of problem earlier, before a marked file is even committed. Both wrap the existing CLIs (`audit_dir.py` / `clean_file.py`) — no separate detection logic.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/guillaumemeyer/watermarks-remover
    rev: v0.5.0   # pin to a tag/commit
    hooks:
      - id: watermarks-remover-check   # fails the commit if marks are found
      # - id: watermarks-remover-clean # opt-in: cleans staged files in place instead
```

`watermarks-remover-check` fails the commit and lists findings; `watermarks-remover-clean` is opt-in and rewrites staged files in place (exits 1 so you review the diff and re-stage — the same convention as auto-fixing hooks like `ruff --fix`). When the cleaner cannot process a file at all — it crashed, was killed, or produced no report — `watermarks-remover-clean` names that file and exits 3 instead, so a cleaner that failed is never mistaken for an already-clean file. Run either by hand with `python3 service/scripts/check_staged.py <files...>` / `clean_staged.py <files...>`.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest          # or: make test
make smoke                          # quick CLI smoke on fixtures
```

## Changelog

### [v0.6.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.6.0) — wider format coverage, Layer A hardening, plugin & hook distribution, and detection-guided rewriting

**Format & container coverage**

- **AVIF / HEIC**: native stdlib metadata and C2PA stripping (#84, #85)
- **BMP / GIF / TIFF**: stdlib detection, inspection, and metadata cleaning — GIF comment/XMP extensions are dropped while `NETSCAPE2.0` looping and other animation chunks are preserved; TIFF IFD metadata (XMP/EXIF/GPS/IPTC/MakerNote) is dropped with payloads zeroed and strip offsets kept, for both classic and BigTIFF; BMP trailing metadata is truncated with the file-size field rewritten (#107)
- **EPUB**: stdlib container cleaning — OPF metadata and XHTML meta/JSON-LD scrubbed, embedded raster/SVG media stripped, Layer A applied to XHTML body text, marker-carrying metadata parts dropped, and OCF-encrypted parts passed through untouched (#107)
- **XLSX / PPTX / DOCX (OOXML)**: native stdlib container metadata, text, and embedded media scrubbing; always empty DOCX `docProps` provenance fields; prune dangling relationships after `customXml` removal; run Layer A over DOCX/ODT body text; decode XML entities before Layer A scrub (#91, #100, #76, #83, #73, #80, #74, #81, #142)
- **SGML/vector containers**: linear-time metadata stripping for SVG/ODT (GHSA-7vpp-96qp-j9wh) (#147); recursively inspect and clean embedded raster data URIs in SVGs, HTML, and Markdown (#87, #88)
- **Audio / video**: AI/C2PA metadata stripping for MP4/MOV, WAV, and MP3 (#139); WAV RIFF C2PA chunk detection and removal; FLAC C2PA metadata support; reject partial ID3v2 frame parsing (#232); preserve MP4 media offsets when stripping metadata (#183)
- **PDF**: reach metadata that lives inside embedded images and stop resizing the PDF to strip XMP; run the deep-image pass whether or not exiftool is installed; honour JPEG marker fill bytes and share one segment walker
- **PNG**: detect AI generator product names in PNG text metadata; detect AI markers in compressed PNG text (#127); keep the truncated tail instead of dropping it in png/isobmff strips (#182)

**Layer A (invisible Unicode) hardening**

- **Consolidated Layer A hardening** (#133): strip reserved `Default_Ignorable` code points without a legitimate interchange use (`U+2065`, `U+FFF0`–`U+FFF8`, `U+E0000`, `U+E0080`–`U+E00FF`, `U+E01F0`–`U+E0FFF` — reported as `reserved_ignorable`), the 66 noncharacters (`U+FDD0`–`U+FDEF` plus `U+FFFE`/`U+FFFF` per plane — reported as `noncharacter`), and three blank-rendering Default_Ignorable carriers the `Cf` catch-all never saw (`U+180F`, `U+3164`, `U+FFA0`). Each has the same in-context preservation as its already-covered siblings, so partial-syllable text is not corrupted, and each is applied to both the service engine and the vendored lightweight-skill copy
- **Stop stripping visible-layout format controls next to their own script**: Egyptian hieroglyph quadrat controls (`U+13430`–`U+1343F`), Duployan shorthand controls (`U+1BCA0`–`U+1BCA3`), and musical beam/tie/slur/phrase controls (`U+1D173`–`U+1D17A`) are now preserved when adjacent to their own script and still stripped (and flagged) when floating between unrelated text; `--strip-emoji-glue` paranoid mode still strips them everywhere
- **Emoji / script polish**: preserve VS16 after emoji singletons outside the block ranges; preserve script joiners, flag emoji, and Arabic Cf marks; preserve multilingual Unicode during text cleanup (#34)

**Layer B rewriting & watermark detection**

- **Iterative, detection-guided Layer B rewriting**: each round generates `--candidates` variants (default 1, `WATERMARKS_REWRITE_CANDIDATES`) and `--max-loops` (default 1, `WATERMARKS_REWRITE_LOOPS`) caps the evaluation rounds, stopping as soon as an attempt passes detection. Evaluator priority: MarkLLM (`--markllm-scheme`) > bigram-Jaccard lexical divergence (fallback). `rewrite_text.py --json-stats` now reports `evaluator` / `max_loops` / `attempts_made` / `passed` and per-attempt `candidate_scores` (#153)
- **Keyed-Gumbel (Aaronson EXP) same-key verification**: new stdlib-only `detect_gumbel.py` implements the model-free replay test (u = PRF(Hash(key, window), token); exact Gamma-tail p-value; repeated-window masking) with no GPU, model, or logits. `rewrite_text.py --gumbel-key` (env `WATERMARKS_GUMBEL_KEY`, preferred) makes it the iterative-loop evaluator (priority: gumbel > markllm > lexical divergence) and it is exposed as `gumbel` in `/capabilities` and `/detect`. Same-key-only — not a vendor oracle; the key is never logged (#190)
- **Benchmarks**: multi-scheme MarkLLM text benchmark and detection (#188) and a reproducible SynthID-text removal benchmark (#145); default variants `paraphrase:3`; report and CSV carry attempts per document (`mean_attempts` / `att`, `attempts` / `evaluator` / `passed` columns); `--rewrite-loops` mirrors `--max-loops`
- **Detection**: vendor text-watermark detection (Gemini SynthID, Claude seam, MarkLLM) plus a SynthID image scorer sidecar (#109); new zero-LLM statistical and stylometric AI text detector for CI and audits (#68, #69)

**Distribution: plugin, hooks, and skill installs**

- **The repository is now a Claude Code plugin and a single-plugin marketplace** (`.claude-plugin/plugin.json` + `marketplace.json`), so both skills install with `/plugin marketplace add guillaumemeyer/watermarks-remover` then `/plugin install watermarks-remover@watermarks-remover`, and update in place. `make plugin-validate` runs `claude plugin validate . --strict`; `tests/test_plugin_manifest.py` checks the manifests without the CLI
- **`install_skill.py` grew a `--target`** (`claude-code`, `claude-project`, `cowork`, `cursor`) and a `--skill` selector covering both shipped skills, plus `--list`, `--link`, and `CLAUDE_CONFIG_DIR`. The `cowork` target builds a reproducible upload bundle (`dist/<skill>.zip`, single top-level skill directory); every target validates against the Agent Skills packaging rules and the 30 MB upload limit. New `make` targets: `install-claude-code-skill`, `install-claude-code-text-skill`, `install-claude-project-skill`, `package-cowork-skill`, `package-cowork-text-skill`
- **Deterministic auto-cleaning via a `PostToolUse` hook** (`hooks/hooks.json` + `service/scripts/hook_written_file.py`): after the agent writes a file the harness runs the hook whether or not the model cooperates. `check` (default) reports marks to the model; `clean` strips them in place and tells the model the file moved, swapping only on a real difference so clean files keep their mtime. Mode comes from the plugin's `hook_mode` setting or `WATERMARKS_HOOK_MODE`; detection reuses `audit_lib.scan_file` / `is_actionable`, so the hook, the pre-commit gate, and the CI SARIF export agree. A hook still cannot rewrite the assistant's chat message — no such hook point exists — so that path stays best-effort
- **Pre-commit hook integration** for staged-file checking/cleaning (#138); lightweight Cursor text skill (#35); `clean-user-facing-text`'s description no longer names Cursor as the only host

**HTTP service**

- Batch endpoints: `POST /clean/batch`, `/inspect/batch` (#137) and `POST /detect/batch` (#151)
- Preserve image format extensions in `/clean` and use safe writes in `av_meta` (#150); use portable base64 in the `/detect` curl example (and fix the macOS `realpath` portability in the bootstraps, #185)

**Audit / inspection & security**

- `audit_dir.py` gained multi-worker concurrency and SARIF 2.1.0 export (#101, #102)
- Route website binary formats to their real scanners (#177); refuse DTD/entity bombs in the sitemap parser (GHSA-pjg6-92pm-mmcf) (#146); a crashed cleaner blocks the commit instead of reading as clean (#179); an unreadable text file is a failed scan, not a clean one (#169)

**Reliability & correctness fixes**

- A second `--in-place` run preserves the original `.bak`; keep collected evidence when a later zip member fails to read (#175); truncated ISOBMFF containers still run the C2PA byte-scan fallback (#176); distinguish a failed cleaner from an already-clean file (#159, #161); treat a failed c2patool run as inconclusive rather than "no C2PA" (#156); validate clean option types (#111); never auto-select MPS device for text watermark detection (#99); macOS portability — pure `--json` stdout for the SynthID scorer and BSD `realpath` probe (#70); fix a Windows `subprocess_creationflags` path in `_ghostscript_usable` and stop child processes from opening a console window on Windows
- Behavioural hardening: preserve benign JPEG comments keep-mode; fix the `bench-synthid-text` swallowed flag; simplify flag passthrough for the Ghostscript probe and clean_text unneeded noqa (lint)

**CI / tooling / docs**

- Ruff linting and formatting with CI enforcement (#103); add macOS to the test matrix (#152); add a CodeRabbit config for automated PR reviews (#222); CODEOWNERS for CODE_OF_CONDUCT/LICENSE and main-review owners; attribute copyright to Guillaume Meyer and contributors (#228)
- Docs: voice-preserving rewrite guidance and protecting voice/accessibility choices; Ecosystem additions (ClaudeWatermarks, unmark-web) and a note discouraging look-alike names; arXiv 2402.14904 reference; Windows auto-start guide via Task Scheduler; portable base64 in curl examples; pin the vendored Cursor-skill text engine to the service copy (#96)

### Unreleased

- Pre-commit clean hook (`watermarks-remover-clean` / `clean_staged.py`): use content digests (`SHA-256`) and active action detection so clean files on disk are recognized without demanding infinite re-staging (#173)
- **OOXML container preservation**: keep `<AppVersion>` intact in `docProps/app.xml` during DOCX, XLSX, and PPTX metadata cleaning to satisfy ECMA-376 schema constraints and avoid Microsoft Word/Office "unreadable content" errors (#283)

### [v0.5.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.5.0) — service & Docker distribution, HTTP API, and verification harnesses

**Service / Docker distribution**

- **Skill/service split**: the skill (`skills/remove-ai-marks/`) is now a code-free remote client over HTTP; all implementation moved to `service/scripts/` and runs behind `server.py`, a stdlib HTTP entrypoint (`/health`, `/inspect`, `/clean`, `/capabilities`)
- **HTTP service**: `service/scripts/server.py` exposes the cleaning pipeline over JSON/base64; hardening mirrors the CLIs (size caps, binary guard, atomic writes, loopback default, optional `WATERMARKS_SERVER_API_KEY` bearer auth)
- **OpenAPI**: `GET /openapi.json` serves a dynamically generated OpenAPI 3.0.3 spec (built from the route table + live config, so it never drifts from the real endpoints); CI validates it with `openapi-spec-validator`
- **Core Docker image** (`service/Dockerfile`): full cleaning service with exiftool / qpdf / c2patool preinstalled; any CLI stays runnable by overriding the command
- **Docker / compose**: `compose.yaml` brings up the whole infra (`core` always; `markllm` / `markdiffusion` behind `profile: harness`; `ctrlregen` / `synthid` behind `profile: heavy` as local-only builds); services are prefixed `wr-`; harness/heavy services default to `command: ["--help"]` so `docker compose up --profile harness --profile heavy` exits cleanly (one-shot CLIs are run with `docker compose run`); new `make compose-check` / `compose-check.sh` validates the running stack (exit code only)
- **GHCR publishing**: `.github/workflows/release-images.yml` publishes `core`, `markllm`, `markdiffusion` images on `v*` tags; `ctrlregen` / `synthid` are never published (upstream licensing)
- **Env configuration**: `.env.example` + service configuration guide; `docker compose` auto-loads `.env`; `.env` is gitignored (deny-by-default)
- **Repo hygiene**: `.gitignore` and `service/.dockerignore` are now deny-by-default — only explicitly allowed paths can be committed or sent in a build context (image contexts only ship `service/scripts/`, which is all the Dockerfiles COPY)
- Tests: `tests/test_http_server.py` (13 cases) for the HTTP service; all suites re-pointed at `service/scripts/`

**MarkDiffusion image-watermark harness (optional)**

- New optional harness (external `THU-BPM/MarkDiffusion`, Apache-2.0): `markdiffusion_harness.py` with `watermark` / `detect` / `purify` subcommands for nine image schemes (Tree-Ring, Ring-ID, ROBIN, WIND, SFW, Gaussian-Shading, GaussMarker, PRC, SEAL)
- `clean_image.py --remove-pixel diffusion` runs the MarkDiffusion `DiffusionPurification` regeneration attack as an alternative pixel-removal engine (conservative strength 0.3 default)
- `setup_markdiffusion.sh` bootstrap (PyPI pin `1.0.2`; `--checkout` editable clone at pinned commit) + `requirements-markdiffusion.txt` + `Dockerfile.markdiffusion` and Makefile `bootstrap-markdiffusion` / `smoke-markdiffusion` / `docker-markdiffusion-build` / `docker-markdiffusion-help`
- Mock-based tests (`tests/test_markdiffusion_harness.py`) — no torch in CI; `references/markdiffusion.md` reference doc
- Docs: same-scheme-only verification caveat (not a vendor-detector oracle) and blind-regeneration drift caveat in README, SKILL.md, `removal-matrix.md`, `markdiffusion.md`

**MarkLLM text-watermark harness (optional)**

- New optional harness (external `THU-BPM/MarkLLM` checkout, Apache-2.0): `detect_text_watermark.py` with `detect` / `watermark` subcommands for KGW and SynthID schemes
- `rewrite_text.py --markllm-scheme` runs before/after detection around a Layer B rewrite and per-candidate detection when `--candidates N>1` (env-gated; reports `cleared`)
- `setup_markllm.sh` bootstrap + `requirements-markllm.txt` (pinned deps) + `Dockerfile.markllm` and Makefile `bootstrap-markllm` / `smoke-markllm` / `docker-markllm-build` / `docker-markllm-help`
- Hardening: `--offline` cache-only model loading (no HF egress, no remote code), 1 MiB config cap, optional `WATERMARKS_MARKLLM_RLIMIT_AS` on the rewrite subprocess, pinned torch in the Dockerfile, and clone-SHA verification in `Dockerfile.markllm`
- Mock-based tests (`tests/test_markllm_detect.py`, 21 cases) — no torch in CI; verification-harness caveat (same-config-only, not a vendor-detector oracle) documented in README, SKILL.md, `removal-matrix.md`, `vendor-notes.md`

**Fixes and polish**

- **Layer B**: `rewrite_text.py` now sends `reasoning_effort: "none"` by default for `openai-compatible` backends (`--reasoning-effort` / `WATERMARKS_REWRITE_REASONING_EFFORT`; `off` omits it). Reasoning models like `deepseek-v4-flash` otherwise burn ~100s of chain-of-thought on a one-line rewrite (9,894 vs 12 completion tokens)
- **Fix markllm image build**: `requirements-markllm.txt` pinned `tokenizers==0.23.1`, which conflicts with `transformers==5.15.0` (caps `tokenizers<=0.23.0`; no 0.23.0 release exists) — now pinned `tokenizers==0.22.2`; torch moved to the CPU wheel index (`torch==2.13.0.*`) so the image is CPU-only like `Dockerfile.markdiffusion`
- **Fix ctrlregen image build**: the 2023-era research pins (`safetensors==0.4.3`, `transformers==4.37.2` → `tokenizers<0.19`) ship no Python 3.14 wheels, so the base image is now `python:3.11-slim` (digest-pinned, multi-arch)
- **Fix harness images at runtime**: `Dockerfile.markllm` and `Dockerfile.markdiffusion` never copied `common.py` into `/app` (pre-existing bug) — added
- **WebP**: stdlib-only inspection and metadata cleaning for RIFF `C2PA`, XMP, EXIF, and ICC profile chunks (#37)
- **BMP / GIF / TIFF**: stdlib-only detection, inspection, and metadata cleaning — GIF comment/XMP extensions are dropped while `NETSCAPE2.0` looping is preserved; TIFF IFD metadata (XMP/EXIF/GPS/IPTC/MakerNote) is dropped with payloads zeroed and strip offsets kept, for both classic and BigTIFF; BMP trailing metadata is truncated with the file-size field rewritten
- **EPUB**: stdlib-only container cleaning — OPF metadata and XHTML meta/JSON-LD scrubbed, embedded raster/SVG media stripped, Layer A applied to XHTML body text, marker-carrying metadata parts dropped, and OCF-encrypted parts passed through untouched
- **Filename sanitization**: HTTP service refuses unsafe client-supplied output names
- **Fix markdown frontmatter cleaner** crashing on and leaking nested AI keys (#25)
- **Text tools refuse binary input**; `--force-text` overrides (#24)
- **`--json` no longer suppresses the residual-signal exit code** (#30)
- **`inspect_file` prints the filename** in its output (#50)
- **Preserve mixed-case CMS generator meta tags** (#42)
- **Preserve load-bearing script invisibles, strip PUA** in Layer A (#38, #52)
- **Preserve script joiners, flag emoji, and Arabic Cf marks** in Layer A (#28)
- **Harden website audit against SSRF and gzip bombs** (#49)
- **SECURITY.md** only references the private advisories channel (#51)
- **Windows**: PowerShell ports of the setup bootstraps (#40)
- **Docs**: add stars/forks shields and drop star-history chart; add MarkLLM to README references; pull request template; plan for Docker CLI + API deployment

### [v0.4.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.4.0) — pixel removal, finding confidence, Windows & false-positive fixes

**Optional CtrlRegen pixel removal (external backend)**

- Optional pixel-domain watermark removal via an external `mertizci/noai-watermark` checkout: `clean_ctrlregen.py` adapter + `setup_ctrlregen.sh` bootstrap (pinned commit, sparse checkout, venv, SHA verification), plus `Dockerfile.ctrlregen` and `make bootstrap-ctrlregen` / `docker-ctrlregen-build` / `smoke-ctrlregen`
- `clean_image.py --remove-pixel ctrlregen` runs metadata strip → CtrlRegen removal → optional reverse-SynthID before/after score; `inspect_image.py` hints at the flag on a high SynthID score
- Conservative default strength `0.25` (presets 0.15/0.25/0.35/0.5/0.7); the 512×512-native pipeline is auto-tiled by the backend for larger images; the torch subprocess gets higher env-overridable resource caps
- Backend is never bundled: `noai-watermark` ships no LICENSE file (treated as all-rights-reserved), and its auto-install/restart code paths are bypassed by using `CtrlRegenEngine` directly

**Finding confidence and aggregate audits**

- Findings are now classified `confirmed` / `probable` / `informational` / `likely_false_positive`, exposed in text/image/container JSON and human reports
- New `audit_dir.py` (recursive tree) and `audit_website.py` (sitemap discovery + crawl) aggregate reports; documented in SKILL.md

**False-positive fixes**

- DOCX: scan only `docProps`/`customXml`, not the visible body (#14)
- Text Layer A: preserve emoji `VS16`/`ZWJ` after an emoji base; new `--strip-emoji-glue` paranoid flag (#22)
- HTML: treat CMS generator tags as informational, not AI metadata (#13)
- PDF: exclude stream payloads from the AI-marker byte scan (#13)
- Inspect reports note unsupported/best-effort paths

**Windows support**

- Gate POSIX-only `preexec_fn` and `os.fchmod` so writes and optional tools run on Windows (#15, #23)
- Reconfigure stdio to UTF-8 so redirected Windows streams no longer raise on invisible Unicode; Windows CI leg + CLI smoke run (#23)

**Docs and supply chain**

- README CtrlRegen section + research references (CtrlRegen, UnMarker, forensic-stealth caveat), responsible-use disclaimer; SKILL/matrix/vendor-notes/ethics updates
- Dependabot config + security-path CODEOWNERS; bump scipy/numpy/opencv-python/scikit-learn/pywavelets and the base image to Python 3.14-slim
- Mock-based CtrlRegen tests (no torch in CI)

### [v0.3.2](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.3.2) — security hardening (safe writes, HTTP client, CI supply chain)

- **Safe, atomic output writes**: every cleaner now writes via temp-file + atomic rename (`safe_write_bytes` / `safe_write_text`), refuses symlinked destinations, and creates `.bak` backups through the same safe path — pre-placed symlinks (e.g. in `/tmp` or download dirs) can no longer redirect a clean write onto an arbitrary file
- **`rewrite_text.py` HTTP client hardening**: redirects are refused outright, so an API key in the `Authorization` header can never be re-sent to an unvalidated host; non-loopback endpoints are **denied by default** (opt in with `--allow-remote` or `WATERMARKS_REWRITE_ALLOW_REMOTE=1`); only http(s) schemes are accepted; `--api-key` was removed — keys are env-only via `WATERMARKS_REWRITE_API_KEY`
- **Resource caps**: default max input 1 GiB → 256 MiB, new 64 MiB stdin cap, DOCX/ODT zip budget 512 MiB → 128 MiB, and `RLIMIT_AS`/`RLIMIT_FSIZE` applied to exiftool/c2patool/SynthID subprocesses (all caps env-overridable)
- **Supply chain**: CI actions SHA-pinned with `permissions: contents: read`, pinned dev deps (`requirements-dev.txt`), a `pip-audit` step, and a new CodeQL workflow; the Docker image now runs as an unprivileged user with pip pinned
- **Scorer deps**: Pillow bumped 10.4.0 → 12.3.0 (24 known CVEs); API usage verified against the pinned upstream commit
- Tests: 18 new security regression tests (60 total, all passing)

### [v0.3.1](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.3.1) — stronger Layer B statistical-watermark rewrite

- `rewrite_text.py` default paraphrase now performs an explicit **word-choice + syntax** attack (clause order, connectors, transition words, sentence boundaries, function words) rather than a generic rewrite
- New `--strength humanize`: zero-shot "write like a human" pass targeting formulaic AI-style phrasing
- New `--strength code`: rewrites comments, docstrings, and string literals, and renames local identifiers while preserving behavior and public API names
- Structural pass now emits "natural, varied human prose" instead of AI-typical "clear professional style"
- New `--temperature` (default `0.9`) for both Ollama and OpenAI-compatible backends
- New `--candidates N`: generates N rewrites and selects the most lexically diverged (bigram Jaccard distance) with a length-drift guard
- Stronger model hygiene: prefer local open-weight models and avoid any known-watermarked vendor, not just the suspected origin
- Residual-risk reporting now distinguishes short/highly predictable text (lower risk) from long, high-entropy prose (higher risk)
- Docs updated in `SKILL.md`, `removal-matrix.md`, and `vendor-notes.md`; tests cover new prompts, divergence scoring, and candidate selection

### [v0.3.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.3.0) — optional SynthID pixel scoring

- Optional pixel-domain SynthID scorer via an external [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID) checkout (`score_synthid.py`); surfaced in `inspect_image.py` / `clean_image.py` with `REVERSE_SYNTHID_DIR` or `--synthid-dir`
- `setup_synthid.sh` bootstrap (scorer-only dependencies; `--full` installs upstream requirements); `Dockerfile.synthid` plus `make docker-synthid-build` / `docker-synthid-help`
- Makefile `smoke-synthid` and `bootstrap-synthid` targets
- Tests for the scorer adapter, CLI unavailable path, JSON parsing, and runtime errors
- Docs: detection/scoring only (no pixel removal); upstream code is not bundled and remains under its non-commercial Research License

### [v0.2.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.2.0) — c2patool false-positive fix

- `image_meta.py`: `has_manifest` no longer flags `Error: No claim found` / `No JUMBF data found` as a manifest (operator-precedence bug: the negative markers now veto every positive branch)
- New `tests/test_c2patool_report.py` (4 cases: no claim, no JUMBF, genuine manifest, tool absent)
- Docs: fixed `c2patool` links (repo moved to `contentauth/c2pa-rs`); added a disclaimer on the quality cost of text-watermark removal

### [v0.1.0](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.1.0) — packaging polish + provenance honesty

- `Makefile` (`test` / `smoke` / `install-skill`) and `pytest.ini`
- Fixture samples for Markdown, HTML, SVG; PDF degraded-clean test
- Docs: industry **two-layer** model (hard-bound C2PA vs soft binding / SynthID-media)
- README residual-risk table + links to external verify tools
- Reference: Institute of AI PM C2PA/SynthID guide
- Soft-binding and pixel/audio/video watermarks explicitly out of scope in skill/matrix/ethics

### [v0.0.1](https://github.com/guillaumemeyer/watermarks-remover/releases/tag/v0.0.1) — initial multi-vendor release

- Agent skill `remove-ai-marks` (replaces Claude-only `remove-claude-marks`)
- **Layer A:** invisible Unicode / bidi / tag chars / space homoglyphs (`inspect_text` / `clean_text`)
- **Layer B:** rewrite guidance + optional `rewrite_text.py` (print-prompt, Ollama, OpenAI-compatible)
- **Files:** C2PA/AI metadata strip for PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, Markdown
- Unified `inspect_file.py` / `clean_file.py`
- Multi-vendor docs (Claude, Gemini/SynthID-class, OpenAI, open-LLM)
- Stdlib-first scripts; optional `c2patool` / `exiftool`

## License

MIT — see [LICENSE](LICENSE).

## Bibliography

- [How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content) (Anthropic)
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4) (SynthID-Text, Nature 2024)
- Google AI for Developers, [*SynthID safeguards*](https://ai.google.dev/responsible/docs/safeguards/synthid) (Gemini API docs)
- [C2PA](https://c2pa.org/) / [c2patool](https://github.com/contentauth/c2pa-rs/tree/main/cli)
- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226)
- Evseev, D. (Arbitration City), [*Accurate, Costless, and Invisible AI Text Watermarking for Self-Hosted AI Inference*](https://arbicity.com/news/ai-text-watermarking-for-self-hosted-ai/) (technical report, August 2026) — keyed-Gumbel watermarking shipped in the open-source arbi-serve engine, with exact-test detection and speculative-decoding support — [PDF](https://arbicity.com/news/ai-text-watermarking-for-self-hosted-ai/ARBI-Watermark-Technical-Paper.pdf)
- [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) (unified toolkit for evaluating LLM watermarking algorithms)
- Pan et al., [*MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models*](https://arxiv.org/abs/2509.10569) (JMLR) — the embedding toolkit this repo's optional image-watermark harness wraps — [code](https://github.com/THU-BPM/MarkDiffusion), [docs](https://markdiffusion.readthedocs.io)
- Zhang et al., [*Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models*](https://arxiv.org/abs/2311.04378v5) (ICML 2024)
- Sander et al., [*Watermarking Makes Language Models Radioactive*](https://arxiv.org/abs/2402.14904) — watermarks survive fine-tuning and mark downstream models trained on watermarked data
- Pan et al., [*Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?*](https://arxiv.org/abs/2502.11598) — watermark-based provenance and protection against knowledge distillation
- [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text) (research reference; not used for detection here)
- [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) (research reference)
- ETH Zurich SRI, [*Probing SynthID*](https://www.sri.inf.ethz.ch/blog/probingsynthid) (research blog on the detectability of SynthID watermarks)
- Liu et al., [*Image Watermarks are Removable Using Controllable Regeneration from Clean Noise*](https://arxiv.org/abs/2410.05470) (ICLR 2025) — the pixel-regeneration method the optional CtrlRegen backend implements — [code](https://github.com/yepengliu/CtrlRegen)
- Kassis & Hengartner, [*UnMarker: A Universal Attack on Defensive Image Watermarking*](https://arxiv.org/abs/2405.08363) (arXiv:2405.08363; IEEE S&P 2025) — a universal watermark attack compared on a different metric than CtrlRegen
- Goonatilake & Ateniese, [*Removing the Watermark Is Not Enough: Forensic Stealth in Generative-AI Watermark Removal*](https://arxiv.org/abs/2605.09203) (arXiv:2605.09203) — motivates the conservative-strength default: removal can still leave forensic traces
- [mertizci/noai-watermark](https://github.com/mertizci/noai-watermark) (CLI/Python toolkit for SynthID/StableSignature/TreeRing removal and AI metadata stripping)
- [0xROOTPLS/DeSynth](https://github.com/0xROOTPLS/DeSynth) (SynthID removal for OpenAI/Google images)
- Institute of AI PM, [*AI Content Provenance and Watermarking: The PM's Guide to C2PA and SynthID*](https://www.institutepm.com/knowledge-hub/ai-content-provenance-watermarking) (two-layer industry model: C2PA + imperceptible watermark / soft binding; SB 942 / EU AI Act Art. 50 context)
