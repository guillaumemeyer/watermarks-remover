# Removal matrix

The *Service side* column names the implementation behind `/inspect` and
`/clean`. The agent reaches it over HTTP and never runs these scripts itself
(see `SKILL.md`).

| Target | Method | Service side | Side effects | Verifiable today? |
| --- | --- | --- | --- | --- |
| Invisible Unicode / exotic spaces / bidi / tags | Strip / normalize | `inspect_text.py`, `clean_text.py`, `clean_file.py` | Minimal | Yes (codepoint report) |
| Stylometric AI cadence / burstiness / n-grams (zero-LLM) | Statistical variance & cadence scoring | `score_stylometry.py`, `inspect_text.py --stylometry`, `audit_dir.py --check-stylometry` | None (detection only) | Yes (calibrated score + phrase spans) |
| Statistical text watermark (SynthID-class / Kirchenbauer) | Multi-pass paraphrase / humanize / back-translate / structural | `/clean` Layer B strategy on plain text (`rewrite_text.py`), or the agent's own rewrite model | Meaning/style drift | No without vendor key/detector; **MarkLLM harness** (`detect_text_watermark.py`) verifies a specific scheme config before/after |
| C2PA on PNG/JPEG/WebP/AVIF/HEIC | Drop APP11 / PNG `caBX` / RIFF `C2PA` / ISOBMFF `jumb` & `uuid` / exiftool | `clean_image.py` | Loses provenance metadata | Yes |
| C2PA on WAV/MP3/MP4/MOV (ISOBMFF `jumb`/`c2pa`/`uuid` incl. the C2PA content-provenance `uuid` box, RIFF `C2PA`, ID3v2) | Drop the C2PA box/chunk/frame (offset-preserving `free` box in ISOBMFF) | `av_meta.py` (`clean_av`) | Loses provenance metadata | Yes (re-inspect) |
| GIF comment/XMP extensions | Drop 0xFE / XMP application extensions (keep `NETSCAPE2.0`) | `clean_image.py` | Loses GIF comments/XMP | Yes (re-inspect) |
| TIFF XMP/EXIF/GPS/IPTC/MakerNote (classic + BigTIFF) | Drop IFD tags, zero payloads, keep strip offsets | `clean_image.py` | Loses TIFF metadata | Yes (re-inspect) |
| BMP trailing metadata | Truncate non-image trailing bytes, fix file-size field | `clean_image.py` | Removes appended metadata | Yes (re-inspect) |
| EPUB OPF metadata / XHTML meta / embedded media | Scrub OPF, strip XHTML meta/JSON-LD, clean embedded media, Layer A (skip encrypted parts) | `clean_file.py` | Loses book metadata; rewrites archive | Yes (re-inspect) |
| SVG metadata / XMP / embedded data URIs | Drop `<metadata>`, xmpmeta; clean embedded data URIs | `clean_file.py` | Loses SVG metadata; cleans embedded rasters | Yes (re-inspect) |
| PDF XMP / info | exiftool `-all=` preferred | `clean_file.py` | Loses PDF metadata; degraded without exiftool | Partial |
| DOCX / XLSX / PPTX props / customXml / embedded media | Rewrite OOXML zip, scrub text runs, clean media/ | `clean_file.py` | Loses doc properties; cleans embedded rasters | Yes |
| ODT meta:generator | Scrub `meta.xml` | `clean_file.py` | Loses generator tag | Yes |
| HTML generator / JSON-LD / provenance comments (AI tool name, AI-generated, C2PA, content credential) / embedded data URIs | Strip tags and comments; clean embedded data URIs | `clean_file.py` | Loses meta; cleans embedded rasters; `content` prose under a non-naming `<meta name>` is left alone | Yes |
| Markdown AI frontmatter keys / provenance comments (AI tool name, AI-generated, C2PA, content credential) / embedded data URIs | Drop keys and comments outside code fences; clean embedded data URIs | `clean_file.py` | Loses YAML keys; cleans embedded rasters; prose values under a non-naming key are left alone | Yes |
| Pixel image watermark (SynthID-media / StegaStamp / Tree-Ring / StableSignature) | CtrlRegen regeneration (external backend) | `clean_ctrlregen.py` / `clean_image.py --remove-pixel ctrlregen` | Regenerates pixels; heavy compute; detail drift at higher intensity | No without official detector; reverse-SynthID score is a local surrogate; **MarkDiffusion same-scheme harness** (`markdiffusion_harness.py detect`) verifies a Tree-Ring-class scheme config before/after |
| Pixel image watermark (Tree-Ring-class) | DiffusionPurification regeneration (external MarkDiffusion backend) | `clean_image.py --remove-pixel diffusion` | Blind regeneration; more drift than CtrlRegen; heavy compute | Same-scheme only via the MarkDiffusion harness (not a vendor-detector oracle) |
| TrustMark video watermark (per-frame + temporal vote) | Per-frame pixel purification (CtrlRegen / DiffusionPurification) + ffmpeg demux/remux, guided by a vote-collapse frame planner | `/clean` (kind=av, `options.remove_pixel` = `ctrlregen`\|`diffusion`) | Re-encodes video (lossy); heavy compute; needs `tools.ffmpeg` + `pixel_backends` present; raises the purge count to the minimum that crosses `vote_threshold` | Model-based (`vote_threshold`); not vendor-detector-verified |
| SynthID audio/video watermark | — | Out of scope | — | — |
| Audio watermarks (silentcipher / AudioSeal / WavMark) | Destructive transform chain (tempo + pitch + EQ + low-bitrate lossy re-encode) | `/clean` (kind=av, `options.remove_audio_watermark` = true) | Alters pitch/tempo/quality/duration; returns M4A/AAC; needs `tools.ffmpeg` | Not vendor-detector-verified (we do not ship silentcipher/AudioSeal/WavMark decoders) |
| C2PA soft binding (in-content link to manifest) | — | Out of scope (survives our metadata strip) | — | Vendor detector only |
| Data-driven model backdoors | — | Out of scope | — | — |

## Default pipeline

1. **Inspect** (`/inspect`).
2. **Deterministic clean** (`/clean`) — Layer A text and/or container/image metadata; for images, optionally add pixel removal (`options.remove_pixel` = `ctrlregen`) after the metadata strip.
3. **Always offer Layer B** rewrite for prose (paraphrase → optional strong pass: `humanize` / back-translate / structural).
4. Prefer a **non-origin, open-weight** rewrite model when available (avoid re-stamping).
5. Layer A again after rewrite.
6. Report: Layer B is best-effort; residual risk remains.
7. **Optional verification:** when `/capabilities` reports `text_detectors.markllm`, `/detect` or `/clean` with `detect_before` / `detect_after` runs a MarkLLM before/after detection (service-side `detect_text_watermark.py` harness) to show a specific scheme config clears. Same-config-only; not a vendor-detector oracle.

## Code vs prose

- **Plain-text prose (pasted / `.txt` / `.text`):** full A + B — `/clean` (kind `text`) runs the Layer B strategy and requires the rewrite backend configured (rejects with 400 otherwise).
- **Markdown / HTML body (container):** `/clean` runs container/metadata clean + Layer A; the Layer B rewrite applies to the prose when it is processed as a text pass (extract the prose or send the content to `/clean` as text), or via the agent rewrite model.
- **Code, config and data** (`.py`, `.json`, `.csv`, `.po`, …): `/clean` runs Layer A only and reports `layer_b.skipped`, because the default rewrite would change identifiers, keys and values. Add a formatter; statistical marks are weak. Offer a `code` rewrite (comments/docstrings/string-literal wording + local identifier renames, `options.strategy` = `code@…`) only with the user's OK.

## Layer B tactics

| Tactic | When |
| --- | --- |
| `paraphrase` | Default; explicit word-choice + syntax churn |
| `humanize` | Zero-shot "write like a human" token reshuffle |
| `backtranslate` | Stronger token reshuffle via pivot language |
| `structural` | Strongest; most drift (outline → human prose) |
| `code` | Comments/docstrings/string-literal wording + local identifier renames |

Frontier production watermarks are currently **token-by-token** (streaming
constraint); paragraph-level robust methods (SemStamp / PostMark) are not yet
deployed, so paraphrase-class attacks remain effective today.
