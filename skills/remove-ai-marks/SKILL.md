---
name: remove-ai-marks
description: >
  Remove multi-vendor AI provenance marks: invisible Unicode (Layer A), statistical
  text watermarks via rewrite (Layer B, always offer), and C2PA/EXIF/XMP/container
  metadata on images (PNG/JPEG/WebP/AVIF/HEIC/GIF/TIFF), documents
  (SVG/PDF/DOCX/XLSX/PPTX/EPUB/ODT/HTML/MD/TEX), and audio/video files. Covers Claude,
  Gemini/SynthID-class, OpenAI provenance, and open-LLM sampling marks. Use when the
  user asks to strip watermarks, remove C2PA/Content Credentials, clean AI metadata,
  remove invisible Unicode, anti-detect clean AI output, or runs /remove-ai-marks
  (aliases: /remove-claude-marks).
---

# Remove AI marks

Multi-vendor anti-detection hygiene for **text** (Unicode + statistical rewrite) and **files** (C2PA / AI metadata across common containers).

Read if needed:

- `references/mark-classes.md` — Unicode / sampling / C2PA / containers
- `references/vendor-notes.md` — Claude, Gemini/SynthID, OpenAI, open-LLM
- `references/removal-matrix.md` — which layer when
- `references/ethics.md` — intended use
- `references/how-claude-marks.md` — Anthropic-specific detail
- `references/markdiffusion.md` — optional MarkDiffusion image harness (schemes, honesty caveats)

This skill is a **thin client**. All deterministic cleaning machinery runs in a
separate HTTP service (this repo's `service/`), so the agent host needs no
Python, venvs, or cleaning tools. Call the service with `curl`; never run
cleaning scripts directly.

## Service access

The base URL comes from `WATERMARKS_SERVICE_URL` (default
`http://127.0.0.1:8765`). When the operator starts the service with
`WATERMARKS_SERVER_API_KEY` set, **every** endpoint, `/health` included, answers
401 without `Authorization: Bearer <key>`. On the agent side the same value goes
in `WATERMARKS_SERVICE_API_KEY`.

Every call below goes through these helpers. Agent hosts often start a fresh
shell for each command, so save the block once (for example as `wm.sh` in a
scratch directory) and source it at the start of each command. Keep the braces
in `${1}` / `${2}`: some skill loaders substitute a bare `$1` in the skill text
with the first word of the slash-command arguments before you ever see it.
Cowork, cloud and sandboxed sessions run their shell away from the machine
that serves `127.0.0.1:8765`; there, point `WATERMARKS_SERVICE_URL` at a
service reachable from the sandbox (and set the key), or hand the user the
PowerShell / curl calls below to run on the host:

```bash
WM="${WATERMARKS_SERVICE_URL:-http://127.0.0.1:8765}"

# wm ENDPOINT [curl args...]: call the service, adding the bearer key when one is set.
# The retries cover dropped connections: on some Windows machines a loopback
# connection stalls for ~20 s and is then reset, about one call in ten. Every
# endpoint this skill uses is safe to repeat.
wm() {
  local url="${WM}${1}"
  shift
  if [ -n "${WATERMARKS_SERVICE_API_KEY:-}" ]; then
    set -- -H "Authorization: Bearer $WATERMARKS_SERVICE_API_KEY" "$@"
  fi
  curl -sS --retry 2 --retry-all-errors "$@" "$url"
}

# wm_post ENDPOINT FILE [OPTIONS_JSON]: POST FILE as {"name", "options", "file"}.
# The body is streamed through stdin. Passing the base64 as a -d argument fails
# with "Argument list too long" above ~24 KB on Windows and ~96 KB on Linux,
# which rules out most PDFs and images. The answer lands in a temp file first:
# curl can only truncate a named output file on retry, so a retry written to
# stdout would append the new response to the half-received one.
wm_post() {
  local name opts="" out rc
  name=$(basename "$2" | sed 's/[\\"]/\\&/g')
  if [ -n "${3:-}" ]; then opts=", \"options\": $3"; fi
  out=$(mktemp)
  {
    printf '{"name": "%s"%s, "file": "' "$name" "$opts"
    base64 < "$2" | tr -d '\r\n'
    printf '"}'
  } | wm "/${1#/}" -X POST -H 'Content-Type: application/json' --data-binary @- -o "$out"
  rc=$?
  cat "$out"
  rm -f "$out"
  return "$rc"
}

# wm_save RESPONSE OUT: write the "cleaned" bytes of a /clean response to OUT and
# print the rest of the response (the report) without the base64 payload; on an
# error, print the error and fail. Base64 never contains a quote, so sed lifts
# the field without jq (with jq: jq -r .cleaned RESPONSE | base64 -d > OUT).
wm_save() {
  if grep -Eq '"ok": ?true' "${1}"; then
    sed -n '/"cleaned": *"/{s/.*"cleaned": *"\([^"]*\)".*/\1/p;q;}' "${1}" | base64 -d > "${2}" &&
      sed '/"cleaned": *"/d' "${1}"
  else
    cat "${1}" >&2
    return 1
  fi
}
```

**Always check the service first**, and stop with a clear message when it is
not usable. Never fall back to local cleaning:

```bash
wm /health -o /dev/null -w '%{http_code}\n'
```

- `200`: up. `wm /health` prints `{"ok": true, "version": "..."}`.
- `401`: up, but it requires a key. Set `WATERMARKS_SERVICE_API_KEY` to the
  service's `WATERMARKS_SERVER_API_KEY`. This is **not** "service down".
- `000` plus a curl error: unreachable. See *Service not reachable?* below.

The service is started either by the operator (`docker compose up -d`, or the
published `ghcr.io/guillaumemeyer/watermarks-remover` image) or locally
(`make serve`).

### Capabilities

```bash
wm /capabilities
```

Reports which optional tools are available server-side (`tools.c2patool`,
`tools.exiftool`, `tools.qpdf`, `tools.ghostscript`, `tools.ffmpeg`), scorers
present (`scorers.stylometry`, `scorers.synthid`, `scorers.synthid_http`),
text-watermark detectors (`text_detectors.markllm`, `text_detectors.gumbel`,
`text_detectors.claude-text`), and which heavy backends are configured
(`pixel_backends.ctrlregen`, `pixel_backends.diffusion`, `harnesses.markllm`).
`text_generators` serves `/watermark`, which this skill does not use.
**Drive your advice from this**: only recommend pixel removal / SynthID
scoring / vendor detection when the service reports the backend present, and
say up front when `exiftool` or `qpdf` is missing, since PDF cleaning is then
degraded.

`layer_b` says whether **text** `/clean` can run its required Layer B rewrite
(check it before sending text):

- `layer_b.default_strategy_usable` — false means a text `/clean` without
  `options.strategy` will return 400.
- `layer_b.tactics.<tactic>` — which tactics can run (`mlm`, `paraphrase`,
  `humanize`, …); the reasons are in `layer_b.rewrite_backend.error` (LLM
  tactics) and `layer_b.mlm.error`.

When the default is not usable but some tactics are, you may pass an
`options.strategy` built only from usable ones (e.g. `"academic@0.4"` when
`tactics.mlm` is false); say so in the report, since it departs from the
configured default. Tell the user what restores the default: for `mlm`,
installing `layer_b.mlm.requirements` (`service/scripts/requirements-mlm.txt`)
into the service's Python (`make bootstrap-mlm`) and restarting the service;
for LLM tactics, the `WATERMARKS_REWRITE_*` backend config.
`layer_b.rewrite_backend.backend` names the backend; the model itself is not
reported, so name the model the operator configured (`WATERMARKS_REWRITE_MODEL`)
in your report when you know it, and otherwise say "the service's configured
local model".

## HTTP API (curl)

Payloads are JSON with the file as **base64**. The agent decodes the `cleaned`
field and writes it to the output path itself (`wm_save`).

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"ok": true, "version": ...}` |
| GET | `/capabilities` | — | optional tools / backends present; `layer_b`: text Layer B readiness |
| GET | `/openapi.json` | — | dynamically generated OpenAPI 3.0.3 spec |
| POST | `/inspect` | `{"file": "<base64>", "name": "notes.md"}`; add `"detect": true` to also run the text detectors | `{"ok", "kind", "suspicious", "report"}` |
| POST | `/detect` | `{"file": "<base64>", "name": "notes.txt"}` | `{"ok", "kind", "detections": [...]}` |
| POST | `/clean` | `{"file": "<base64>", "name": "notes.md", "options": {...}}` | `{"ok", "kind", "cleaned": "<base64>", "report"}` |
| POST | `/inspect/batch`, `/detect/batch`, `/clean/batch` | `{"files": [{"file", "name", "options"}, ...]}`, at most 50 files by default | `{"ok", "results": [...]}`; a failing file becomes its own `"ok": false` entry and the rest still run |

Errors come back as `{"ok": false, "error": "..."}`: 400 for a bad request or
an unusable Layer B backend, 401 for a missing or wrong key, 413 for an
oversized body.

`/clean` and `/inspect` route by the uploaded `name` extension plus the bytes;
unrecognized formats answer `kind: "unknown"` (`/inspect`) or 400 (`/clean`).
When writing a temp file for pasted text, keep a known extension (`.txt` /
`.md`) in the `name` you send.

The machine-readable contract lives at `$WM/openapi.json` — plug it into any
OpenAPI tooling (client generators, Swagger UI, editors) instead of hand-rolling
clients.

`options` accepted by `/clean` (any other key is rejected with 400):

- Text, Layer A: `nfkc`, `aggressive_homoglyphs`, and `normalize_spaces`
  (default `true`; `false` keeps exotic spaces such as NBSP).
- Text, Layer B: `strategy`, an ordered `tactic@intensity` list such as
  `"academic@0.4"` that runs the rewrite after Layer A (default from
  `config/clean_strategy.json`; step 4 says which files get it and when it
  answers 400); `rewrite` (`false`: Layer A only, no model call; `true`: run
  the strategy on a file that would not get it by default, such as a `.tex`,
  `.md` or code file); `protect_latex` (`auto` | `on` | `off`: hold math,
  LaTeX commands/environments, citation keys and code spans out of the
  rewrite); `style`, a free-form writing-style request appended to the
  rewrite prompt (a request, not a guarantee); and `also_layer_a_text`, which
  runs Layer A once more over the rewrite output.
- Containers: `also_layer_a_text` (default `true`: Layer A over the container's
  text; `normalize_spaces` applies there too); `rewrite`, `strategy`,
  `protect_latex` and `style` as above for `.tex` / `.ltx` / `.md`.
- Images, audio and video metadata: `strip_all_metadata` (the default: strip
  everything) or `keep_non_ai_metadata` (strip only AI/C2PA markers and keep
  the rest).
- Images and video: `remove_pixel` (`ctrlregen` | `diffusion`).
- Audio: `remove_audio_watermark` (destructive; see Limitations).
- PDF: `deep_images` (`auto` | `always` | `lossless` | `never`: how hard to
  chase metadata carried inside embedded images; anything else is rejected) and
  `clean_attachments` (`auto` | `always` | `never`: how hard to chase metadata
  inside embedded file attachments — the paperclip files. `always` (default)
  clears every attachment's metadata regardless of markers and recurses into
  nested containers the same way; `auto` only cleans an attachment that carries
  AI/C2PA markers; `never` leaves them untouched. Needs `qpdf`. Anything else
  is rejected).
- Detection: `detect_before` / `detect_after` (text and images: run watermark
  detection on the input and on the cleaned output, included in the report).

Model meta-commentary around a rewrite is stripped; a step whose rewrite still
drifts in length, loses a protected span or changes a number is regenerated
and, failing that, skipped. Read `report.layer_b.ok`, `report.layer_b.errors`,
`report.layer_b.warnings` and `report.layer_b.latex` (protected / restored /
missing) before telling the user the text was rewritten.

**Inspect first** (decide, don't guess):

```bash
wm_post inspect notes.md
```

**Clean** (text / image / container are auto-detected by name + bytes), then
write `cleaned` to the output file (`*.cleaned.*` by default unless the user
asked in-place) and summarize the printed `report` honestly:

```bash
R=$(mktemp)
wm_post clean notes.md > "$R" && wm_save "$R" notes.cleaned.md
wm_post clean report.pdf '{"deep_images": "lossless"}' > "$R" && wm_save "$R" report.cleaned.pdf
```

**PowerShell** (no Bash on the host): `ConvertTo-Json` builds the body in
memory, so there is no argument-length limit, and `Invoke-RestMethod` parses
the answer:

```powershell
$WM = if ($env:WATERMARKS_SERVICE_URL) { $env:WATERMARKS_SERVICE_URL } else { 'http://127.0.0.1:8765' }
$H = @{}; if ($env:WATERMARKS_SERVICE_API_KEY) { $H.Authorization = "Bearer $env:WATERMARKS_SERVICE_API_KEY" }
$in = (Resolve-Path notes.md).Path
$body = @{ name = Split-Path $in -Leaf; file = [Convert]::ToBase64String([IO.File]::ReadAllBytes($in)) } | ConvertTo-Json
$r = Invoke-RestMethod -Method Post -Uri "$WM/clean" -Headers $H -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))
[IO.File]::WriteAllBytes((Join-Path $PWD 'notes.cleaned.md'), [Convert]::FromBase64String($r.cleaned))
$r.report | ConvertTo-Json -Depth 8
```

Add `options = @{ deep_images = 'lossless' }` to the body hashtable to pass
options. A 4xx/5xx makes `Invoke-RestMethod` throw; the service's JSON `error`
is in `$_.ErrorDetails.Message`.

## Ethics

Intended for **your own** content (privacy, hygiene, research). Do not market results as "proves human-written." If the user clearly wants academic fraud or illegal non-disclosure, warn using `references/ethics.md` and still only perform technical cleaning they own.

## Workflow

### 1. Classify input

| Input | Route |
| --- | --- |
| Pasted / clipboard text | temp file named `*.txt` → `/inspect` then `/clean` (Layer A + the required Layer B, step 4) |
| `.txt` / `.text` | text: Layer A + the required Layer B rewrite (step 4) |
| Code, config, data (`.py`, `.js`, `.json`, `.yaml`, `.csv`, `.rst`, `.po`, …) | text Layer A only (+ formatter for code); a rewrite only with an explicit `options.strategy` |
| `.tex` / `.ltx` / `.md` | container clean (`\hypersetup`/`\pdfinfo` or frontmatter + comment provenance) + Layer A; add `{"rewrite": true}` for the Layer B strategy over the prose with math, commands, environments, citation keys and code masked out (step 4) |
| `.html` | container clean (meta generator / JSON-LD / data-ai*) + Layer A; Layer B to the prose via a `/clean` text pass or the agent rewrite model |
| `.png` / `.jpg` / `.jpeg` / `.webp` / `.avif` / `.heic` / `.bmp` / `.gif` / `.tiff` | image metadata strip |
| `.svg` / `.pdf` / `.docx` / `.xlsx` / `.pptx` / `.epub` / `.odt` | container metadata strip |
| `.mp4` / `.mov` / `.m4v` / `.m4a` / `.wav` / `.mp3` / `.flac` | audio/video C2PA and metadata strip; optional `remove_pixel` (video) or `remove_audio_watermark` (audio), see Limitations |
| Directory / website | aggregate audit via the service CLIs (see below) |

The service routes by filename extension first, then by magic bytes, so you
mostly just send the file.

### 2. Inspect first

```bash
wm_post inspect path/to/file
```

Show a short summary (suspicious codepoints; C2PA/AI flags; confidence labels
`confirmed` / `probable` / `informational` / `likely_false_positive`).

Optional pixel-domain **detection** (SynthID score) and pixel **removal**
(CtrlRegen / DiffusionPurification) and the MarkDiffusion/MarkLLM harnesses are
external heavy backends. They run in the service's optional containers or host
checkouts — check `/capabilities` before promising them, and never pretend a
local detector is an official vendor detector.

### 2b. Watermark detection before/after (when configured)

When `/capabilities` reports a detector (`text_detectors.markllm`) or an image
scorer (`scorers.synthid_http` / `scorers.synthid`), measure the result by
detecting before and after cleaning:

```bash
wm_post detect notes.txt
```

Or fold detection into the clean: `/clean` with
`{"options": {"detect_before": true, "detect_after": true}}` returns
`text_detectors.before/after` (text) or `synthid_before/synthid_after`
(images) in the report. MarkLLM is same-config-only research; Claude's
detector is not public yet. (Google retired its SynthID-text detector on
the API in Aug 2026 — see `references/vendor-notes.md`.)

### 3. Deterministic clean (always for matching inputs)

**Any supported file (unified):**

```bash
R=$(mktemp)
wm_post clean INPUT > "$R" && wm_save "$R" OUTPUT
```

`OUTPUT` is `*.cleaned.*` unless the user asked in-place. Re-inspect the result
when residual risk matters.

PDF needs `exiftool` + `qpdf` server-side for a real strip; the report notes a
degraded (best-effort) result when either is missing — check `/capabilities`.

**Images — optional pixel removal:** only when `capabilities.pixel_backends`
says the backend is present:

```bash
wm_post clean shot.png '{"remove_pixel": "ctrlregen"}' > "$R" && wm_save "$R" shot.cleaned.png
```

### 4. Layer B — always offer rewrite (prose)

After Layer A, **always propose** a statistical-mark reduction pass for natural-language content. Do not skip this step silently.

For **plain text** (pasted / `.txt` / `.text`), `/clean` **requires** Layer B:
it applies the default strategy (`config/clean_strategy.json`, shipped as
`academic@0.4`) or the `options.strategy` override after Layer A, reports
`report.layer_b`, and returns **400** when the required backend isn't
configured (LLM steps need the `WATERMARKS_REWRITE_*` config; the optional
`mlm` step needs the service's `requirements-mlm.txt` stack — torch,
transformers, Pillow — plus the `roberta-large` weights). Read
`/capabilities` → `layer_b` first (see Capabilities) instead of discovering
this from a 400. Other text files (code, config, data, markup and
localization: `.py`, `.json`, `.csv`, `.rst`, `.po`, …) get Layer A only, and
`report.layer_b.skipped` says so: a paraphrase would rewrite their
identifiers, keys and values. They get a rewrite only when the request passes
`options.strategy` (for code, `code@…`, with the user's OK). LaTeX and
Markdown sources (`.tex`, `.ltx`, `.md`) are containers: by default `/clean`
strips their metadata and runs Layer A; with `options.rewrite: true` (or an
`options.strategy`) it also runs the Layer B strategy over their prose, with
math, commands, environments, citation keys, code spans and front matter
masked out and restored afterwards (`report.layer_b.latex` says whether every
span came back). Other containers (`.html`, `.pdf`, `.docx`, …) do **not** run
Layer B in `/clean`; apply it to their prose by extracting the text and
passing it to `/clean` as text, or by running the prompts below with a model
**≠ suspected origin** (Claude text → not Claude; Gemini → not Gemini; etc.).
Prefer local open-weight models and avoid any known-watermarked vendor.

**Which strategy.** The service's tactics are `academic`, `paraphrase`,
`humanize`, `chunk`, `backtranslate`, `structural`, `code` and `mlm`, each at
an intensity in (0, 1]:

- `academic@0.3`–`0.5` (the default is `academic@0.4`) for scientific,
  technical, legal or otherwise precise prose, in any language: it keeps the
  language, every term of art, unit and citation as written, the epistemic
  force of each claim (a hedge stays a hedge, a conjecture never becomes a
  result) and the disciplinary voice, while changing wording, connectives,
  clause order and sentence boundaries. The service also rejects and
  regenerates an `academic` piece whose numbers, language or LaTeX structure
  changed.
- `paraphrase@0.5`–`0.8` for general prose where wording matters more than
  precision (marketing copy, notes, forum posts). At 0.8 it asks the model to
  replace content words too, so never use it on prose with terms of art.
- `humanize@0.4` as an optional last step on general prose (plain wording,
  uneven rhythm); it also runs a deterministic pass that straightens quotes
  and turns dashes into commas, so keep it away from LaTeX and from anything
  whose punctuation carries meaning.
- `chunk` (sentence by sentence, the most faithful single tactic),
  `backtranslate` and `structural` (two model calls each) are stronger
  churn for texts that resist; `structural` rebuilds from an outline and can
  drop or invent detail, so review it line by line.
- `mlm` (masked-LM infill with roberta-large, English only) swaps content
  words with no meaning control. Do not put it in a strategy for scientific
  or non-English prose; when the operator's default includes it and the text
  is precise, pass an explicit `options.strategy` without it and say so.

Whatever the strategy, the service cuts long inputs at paragraph or sentence
boundaries (about 2500 characters per model call, `WATERMARKS_REWRITE_CHUNK_CHARS`),
rewrites and checks each piece on its own, strips model meta-commentary, and
keeps a piece's input when no attempt passes its checks. `options.style`
appends a free-form instruction to every prompt (for example a list of terms
that must stay verbatim, or "keep the contrast words as written"); use it to
tighten a rerun rather than to loosen the rules.

**Read the report before you claim a rewrite.** In `report.layer_b`:

- `ok: false` with `errors`: some piece kept its input (length drift, a lost
  placeholder, a changed number); say which and offer a rerun with
  `WATERMARKS_REWRITE_LOOPS=2` or a lower intensity.
- `noop: true`: the model handed the text back; that is not a rewrite.
- `latex.missing > 0`: a protected span was dropped; do not ship that output.
- `steps[].lexical_divergence`: how much changed (0 none, 1 everything);
  `academic@0.4` on technical prose usually lands around 0.3–0.5.
- `warnings`: stripped commentary, regenerated pieces, no-op steps.

**Review the meaning yourself.** The service checks structure, numbers and
placeholders, not sense. For technical prose, compare the output with the
original sentence by sentence and flag any shift in modality (can/cannot,
would/should), quantifier scope, attribution ("motivated by" vs "based on"),
contrast words (while/whereas → and), antecedents of "this/these/such", and
terms of art (evidence → proof, copy → replica, operator → map). Fix a
sentence that drifted by sending only that sentence back to `/clean` as text
with `options.style` naming what must stay, and revert to the original when
it still drifts. Say in the report which model rewrote the text and that no
detector on this machine can confirm the mark is gone.

The rewrite runs inside the `/clean` request, so a text clean takes as long as
the model needs:

- Each backend call may take `WATERMARKS_REWRITE_TIMEOUT` seconds (default 120,
  at most 3600). Past that, `/clean` answers 400 with `Layer B rewrite failed:
  the rewrite backend did not answer within … s`. That means a slow model, not
  a bad request: tell the user to raise `WATERMARKS_REWRITE_TIMEOUT` on the
  service or use a faster model. A 25B model on CPU needs about a minute per
  2500-character piece, so budget minutes for a section and run the call in
  the background.
- Thinking models (gemma4, qwen3.5 or deepseek-r1 on Ollama; reasoning models
  behind OpenAI-compatible APIs) are told not to think
  (`WATERMARKS_REWRITE_REASONING_EFFORT`, default `none`). With `off`, a
  two-word rewrite can spend minutes on hidden reasoning.
- curl has no timeout of its own, but agent hosts often cap one command at a
  couple of minutes: run a text `/clean` in the background or raise that
  command's timeout.
- The Docker core image ships neither `config/clean_strategy.json` nor
  `transformers`, and `compose.yaml` does not forward the `WATERMARKS_REWRITE_*`
  variables to `wr-core`, so a text `/clean` answers 400 there. Use a local
  `make serve` for Layer B, or the rewrite prompts below.

Multi-pass recipe:

1. Layer A clean (via `/clean`)
2. Layer B through the service (`academic@0.4` for precise prose; a
   `paraphrase` strategy for general prose), or the prompts below when the
   service cannot rewrite that format
3. Optional strong pass — `chunk`, `backtranslate`, `structural`, or
   `humanize` on general prose only
4. Layer A again on the result (`/clean` with `{"rewrite": false}`)
5. Meaning review (above), then report residual risk honestly (short/highly
   predictable text = lower; long, high-entropy prose = higher)

**Code files:** Prefer formatter (`prettier`, `black`, `gofmt`, …) + Layer A. Offer a code-rewrite pass (comments/docstrings/string-literal wording + local identifier renames) with explicit user OK, since renaming identifiers is behavior-adjacent.

#### Rewrite prompts (use as-is)

These are the service's own prompts, for when you rewrite a format the service
does not (`.html`, `.pdf`, `.docx` prose) or the service has no backend. Mask
math, citations and code yourself before sending LaTeX to a model, and check
that every masked span comes back.

**Academic (precise prose, any language):**

```
Rewrite the following academic prose so that the wording differs substantially
at the token level while the argument survives intact. Write in the same
language as the original — never translate. Vary connectives, clause order,
and sentence boundaries, and let sentence length move with the subject. Keep
every technical term, term of art, notation, symbol name, unit, and citation
exactly as written: do not paraphrase terminology and never swap a technical
term for an everyday synonym, even when the everyday word reads more smoothly.
Keep the epistemic force of each statement — hedges, attributions, scope
conditions, and stated limitations stay exactly as strong or as weak as in the
original, and a conjecture must not become a result. Keep passive and
impersonal constructions where they are the disciplinary norm. Do not
simplify, summarize, explain, add examples, add transitions that announce
structure, or add a concluding flourish. Preserve all facts, numbers, names,
equations, and technical identifiers. Do not add or remove claims. Output only
the rewritten text.

---
{TEXT}
```

**Paraphrase preserve meaning (word choice + syntax; general prose):**

```
Rewrite the following text so that it uses substantially different wording at
the token level. Change clause order, connectors, and transition words; vary
sentence boundaries and length; and replace both content words and function
words where meaning allows. Preserve all facts, numbers, names, and technical
identifiers. Do not add or remove claims. Output only the rewritten text.

---
{TEXT}
```

**Humanize (write like a human):**

```
Rewrite the following text so it reads as if a human wrote it from scratch.
Vary sentence rhythm and length, replace formulaic AI-style transitions and
filler with concrete natural phrasing, and use plain, varied wording. Preserve
all facts, numbers, names, and technical identifiers. Do not add or remove
claims. Output only the rewritten text.

---
{TEXT}
```

**Code (comments / docstrings / identifiers):**

```
Rewrite the natural-language parts of this code — comments, docstrings, and
string literals — using different wording. Rename local variables, function
parameters, and private helper names to semantically equivalent names. Preserve
program behavior, public API names, and all values that affect output. Output
only the rewritten code.

---
{TEXT}
```

**Back-translate (two steps):**

```
Translate the following text to {LANG}. Output only the translation.
```

```
Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural
phrasing. Output only the translation.
```

**Structural:**

```
Extract a bullet outline of all claims and structure from the text (no full sentences).
```

Then:

```
Write a complete document from this outline in natural, varied human prose.
Avoid formulaic transitions. Do not omit any bullet. Output only the document.
```

### Aggregate audits (directories / websites)

The service image also ships the audit CLIs. Run them as one-shot containers
when a directory or website audit is needed (`watermarks-remover` is the tag
`make docker-core-build` gives; the published image is
`ghcr.io/guillaumemeyer/watermarks-remover:latest`):

```bash
docker run --rm -v "$(pwd)/src:/data:ro" watermarks-remover \
  /app/scripts/audit_dir.py /data --json
docker run --rm watermarks-remover \
  /app/scripts/audit_website.py --base https://example.com --json
```

Or against a local checkout of the repo:
`python3 service/scripts/audit_dir.py DIR --json` or
`python3 service/scripts/audit_website.py --base URL --json`.

Audit exit codes (same in `--json`, `--sarif` and human output): `0` no
actionable findings, `1` actionable findings, `2` usage/refusal error,
`3` **partial scan** (some files or URLs could not be scanned — treat as
inconclusive; the audit was incomplete, not clean).

### 5. Report

Always state:

- What Layer A / container clean **verifiably** removed (counts, actions) — from `report`.
- What Layer B did (best-effort statistical; **cannot claim official "undetectable"**), from `report.layer_b`: the strategy and steps that ran, the model, `lexical_divergence`, `ok` / `noop` / `latex.missing`, or `skipped` for code, data and LaTeX/Markdown sources that were not asked to rewrite. Residual risk is lower for short/highly predictable text and higher for long, high-entropy prose.
- The meaning review: which sentences you checked, which drifted and were fixed or reverted (technical prose), so the user knows what to re-read.
- Out of scope: audio/video SynthID, **C2PA soft binding**, secret-key detectors, training backdoors. Audio watermarks (silentcipher / AudioSeal / WavMark) and pixel-domain video TrustMark are only optionally, partially removed (see Limitations).
- Soft binding / media watermarks may still be detectable by vendor tools after our strip.
- Prefer writing `*.cleaned.*` unless user asked in-place.
- Ethics one-liner: own content / no compliance theater.

## Limitations

- Layer A does **not** remove token-sampling watermarks.
- Layer B cannot be gold-verified without vendor detectors / keys. Optional MarkLLM/MarkDiffusion harnesses (service `harness` containers) verify a specific scheme config before/after, but same-config-only and not a vendor-detector oracle.
- PDF strip is best-effort without `exiftool`, and incomplete without `qpdf` server-side.
- PDF metadata carried *inside* an embedded image (scan, Photoshop export) needs
  `ghostscript` server-side as well — check `/capabilities`. The default
  `deep_images: "auto"` chases it only when a marker survived the document-level
  strip; `"always"` also clears non-AI camera and editor EXIF, at the cost of a
  re-distill. Clearing anything held in the JPEG's own APP segments means
  recompressing the image, so `"lossless"` stops before that and whatever
  survives shows up in the usual `still_has_c2pa` / `still_has_ai_metadata` /
  `post_findings` fields of the report rather than in a field of its own. An
  unrecognised value is an error, not a silent fallback.
- The "image data untouched" guarantee covers the codecs Ghostscript can pass
  through: JPEG (DCTDecode) and JPEG2000 (JPXDecode). Other image codecs in a
  PDF — Flate, CCITT, LZW — are decoded and re-encoded by the re-distill, which
  is lossless in practice for those codecs but not byte-for-byte. Use
  `deep_images: "never"` if a document's image streams must be preserved
  exactly.
- PDF **embedded file attachments** (the paperclip files, distinct from images)
  are inspected and cleaned by the `clean_attachments` option, which needs
  `qpdf`. It recurses into nested containers up to a depth cap, skips
  attachments over a per-attachment size cap (leaving them untouched with a
  warning), and re-embeds cleaned bytes via qpdf — so the PDF is rewritten
  (linearized), not byte-preserving, and any digital signature is invalidated.
  `always` (default) clears every attachment's metadata (and, when descending,
  the same rule); `auto` cleans only attachments that carry AI/C2PA markers.
  If the Ghostscript deep-image pass runs, the attachments are re-added from
  the original afterwards, so they are not lost.
- `.tex`/`.ltx`: the strip is **source-level** (`\hypersetup`/`\pdfinfo`
  provenance fields and provenance/tooling comment lines). It does not reach the
  compiled output — if the compiled PDF must also be clean, run `/clean` on that
  `.pdf` as well. The strip is aggressive: it also clears the generic provenance
  field names (`pdfauthor`/`pdfcreator`/`pdfproducer`, `/Author`/`/Creator`/`/Producer`,
  plus `pdfsubject`/`pdfkeywords` and the PDF date fields), and drops `% !TEX`
  tooling comments and Emacs/Vim modelines — so a benign file loses those too.
- Pixel-domain **image** watermarks can be removed optionally via the external CtrlRegen backend (`remove_pixel: ctrlregen`) or MarkDiffusion's DiffusionPurification (`remove_pixel: diffusion`); both are heavy, drift the image, and need the backend present (`/capabilities`). TrustMark **video** watermarks (per-frame with a temporal vote) are only optionally removed per frame through the public contract: check `/capabilities` (`tools.ffmpeg` and `pixel_backends.ctrlregen`/`diffusion`), then POST `/clean` on an `.mp4`/`.mov` with `options.remove_pixel` = `ctrlregen`\|`diffusion`. It is partial, re-encodes the video, and is not vendor-detector-verified. **Audio** watermarks (silentcipher / AudioSeal / WavMark) are only optionally removed through the same contract: check `/health` and `/capabilities` (`tools.ffmpeg`), then POST `/clean` on an audio name (`.wav`/`.mp3`/`.flac`) with `options.remove_audio_watermark` = true. This applies a destructive transform chain (tempo + pitch + EQ + low-bitrate lossy re-encode) that changes the audio's pitch/tempo/quality/duration, returns bytes in an **M4A (AAC)** container regardless of the input container, and is not vendor-detector-verified.
- The reverse-SynthID scorer is external, best-effort, and under a non-commercial Research License; not an official Google detector. Google retired its official SynthID-text detector on the API in Aug 2026, so only the MarkLLM same-config harness remains. Claude's detection API has been announced but is not public yet — the `claude-text` detector reports unavailable until it ships.
- **C2PA soft binding** (content watermark that re-links to a remote manifest after metadata strip) is out of scope — stripping hard-bound C2PA does not clear it.
- Data-driven / backdoor model marks (trigger phrases) are out of scope.

## Service not reachable?

If the health check prints `000`, tell the user the service is down and how to
start it: `docker compose up -d`, `make serve`, or the published GHCR image.
Without `make`, `make serve` is
`python3 service/scripts/server.py --host 127.0.0.1 --port 8765`, run from the
repo root (the default strategy path `config/clean_strategy.json` is relative
to it) with the `WATERMARKS_REWRITE_*` variables in its environment for
Layer B; on Windows, `docs/windows-autostart.md` registers it as a login task
and `service/scripts/start_service.ps1` starts it with the repo's `.venv`
interpreter (the one that has the `mlm` stack). Before starting one, check
that nothing else already listens on the port (`Get-NetTCPConnection
-LocalPort 8765` on Windows, `ss -ltnp` on Linux): a second server can bind
the same port silently and the older one keeps answering with older code. If
the health check prints `401`, the service is up and wants a key (see *Service
access*). Do **not** attempt to clean locally — this skill contains no cleaning
code.
