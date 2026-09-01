# Removal matrix

| Target | Method | Script / action | Side effects | Verifiable today? |
| --- | --- | --- | --- | --- |
| Invisible Unicode / exotic spaces / bidi / tags | Strip / normalize | `inspect_text.py`, `clean_text.py`, `clean_file.py` | Minimal | Yes (codepoint report) |
| Stylometric AI cadence / burstiness / n-grams (zero-LLM) | Statistical variance & cadence scoring | `score_stylometry.py`, `inspect_text.py --stylometry`, `audit_dir.py --check-stylometry` | None (detection only) | Yes (calibrated score + phrase spans) |
| Statistical text watermark (SynthID-class / Kirchenbauer) | Multi-pass paraphrase / humanize / back-translate / structural | Agent Layer B + optional `rewrite_text.py` | Meaning/style drift | No without vendor key/detector; **MarkLLM harness** (`detect_text_watermark.py`) verifies a specific scheme config before/after |
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
| HTML generator / JSON-LD / embedded data URIs | Strip tags; clean embedded data URIs | `clean_file.py` | Loses meta; cleans embedded rasters | Yes |
| Markdown AI frontmatter keys / embedded data URIs | Drop keys; clean embedded data URIs | `clean_file.py` | Loses YAML keys; cleans embedded rasters | Yes |
| Pixel image watermark (SynthID-media / StegaStamp / Tree-Ring / StableSignature) | CtrlRegen regeneration (external backend) | `clean_ctrlregen.py` / `clean_image.py --remove-pixel ctrlregen` | Regenerates pixels; heavy compute; detail drift at higher intensity | No without official detector; reverse-SynthID score is a local surrogate; **MarkDiffusion same-scheme harness** (`markdiffusion_harness.py detect`) verifies a Tree-Ring-class scheme config before/after |
| Pixel image watermark (Tree-Ring-class) | DiffusionPurification regeneration (external MarkDiffusion backend) | `clean_image.py --remove-pixel diffusion` | Blind regeneration; more drift than CtrlRegen; heavy compute | Same-scheme only via the MarkDiffusion harness (not a vendor-detector oracle) |
| TrustMark video watermark (per-frame + temporal vote) | Per-frame pixel purification (CtrlRegen / DiffusionPurification) + ffmpeg demux/remux, guided by a vote-collapse frame planner | `/clean` (kind=av, `options.remove_pixel` = `ctrlregen`\|`diffusion`) | Re-encodes video (lossy); heavy compute; needs `tools.ffmpeg` + `pixel_backends` present; raises the purge count to the minimum that crosses `vote_threshold` | Model-based (`vote_threshold`); not vendor-detector-verified |
| SynthID audio/video watermark | — | Out of scope | — | — |
| Audio watermarks (silentcipher / AudioSeal / WavMark) | Destructive transform chain (tempo + pitch + EQ + low-bitrate lossy re-encode) | `/clean` (kind=av, `options.remove_audio_watermark` = true) | Alters pitch/tempo/quality/duration; returns M4A/AAC; needs `tools.ffmpeg` | Not vendor-detector-verified (we do not ship silentcipher/AudioSeal/WavMark decoders) |
| C2PA soft binding (in-content link to manifest) | — | Out of scope (survives our metadata strip) | — | Vendor detector only |
| Data-driven model backdoors | — | Out of scope | — | — |

## Default pipeline

1. **Inspect** (`inspect_file.py` or specific inspect_*).
2. **Deterministic clean** — Layer A text and/or container/image metadata; for images, optionally add pixel removal (`--remove-pixel ctrlregen`) after the metadata strip.
3. **Always offer Layer B** rewrite for prose (paraphrase → optional strong pass: `humanize` / back-translate / structural).
4. Prefer a **non-origin, open-weight** rewrite model when available (avoid re-stamping).
5. Layer A again after rewrite.
6. Report: Layer B is best-effort; residual risk remains.
7. **Optional verification:** `rewrite_text.py --markllm-scheme kgw|synthid` runs a MarkLLM before/after detection (external `detect_text_watermark.py` harness) to show a specific scheme config clears. Same-config-only; not a vendor-detector oracle.

## Code vs prose

- **Prose / Markdown / HTML body:** full A + B.
- **Code:** Layer A + formatter; statistical marks are weak; offer `code` rewrite (comments/docstrings/string-literal wording + local identifier renames) with user OK.

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
