# Mark classes

## 1. Edit-based text (Unicode / rules)

Invisible or near-invisible characters, exotic spaces, bidi controls, tag characters, synonym tables.

| Inspect kinds (Layer A) | Examples |
| --- | --- |
| `zwj_family` | ZWSP, ZWNJ, ZWJ, WJ, BOM |
| `bidi` | LRE/RLO/LRI/… |
| `tag_chars` | U+E0001–U+E007F |
| `variation_selector` | VS1–VS256 |
| `private_use` | U+E000–F8FF, U+F0000–FFFFD, U+100000–10FFFD |
| `noncharacter` | U+FDD0–FDEF, U+FFFE/U+FFFF at the end of every plane |
| `reserved_ignorable` | U+2065, U+FFF0–FFF8, U+E0000, U+E0080–E00FF, U+E01F0–E0FFF (unassigned Default_Ignorable) |
| `space` | NBSP, em space, ideographic space |
| `confusable` | Cyrillic/fullwidth Latin (aggressive) |

**Removal:** Layer A via `/clean` — deterministic, verifiable.

Load-bearing invisibles are preserved by default so real text is not corrupted: emoji glue (ZWJ/VS after an emoji base), script joiners (ZWNJ/ZWJ inside complex scripts like Persian or Devanagari), flag tag-char sequences, same-script fillers/selectors (Mongolian free variation selectors after a Mongolian letter, Khmer inherent vowels after a Khmer consonant, Hangul jamo fillers in a partial syllable), orthographic Arabic/Syriac `Cf` marks, and visible-layout format controls next to their own script (Egyptian hieroglyph quadrat controls `U+13430`–`U+1343F`, Duployan shorthand controls `U+1BCA0`–`U+1BCA3`, musical beam/tie/slur/phrase controls `U+1D173`–`U+1D17A`). The same characters between plain ASCII stay carriers and are still stripped. Paranoid mode (strip all of them) exists only as the service-side `clean_text.py --strip-emoji-glue` CLI; `/clean` has no such option.

Maps to Nature paper “edit-based watermarking.”

## 2. Generative / statistical text (token sampling)

Bias next-token sampling toward a pseudo-random green list / score (Kirchenbauer, SynthID-Text / Tournament sampling, etc.). Signal lives in **word choice**, not metadata.

**Removal:** Layer B rewrite (paraphrase → back-translate → structural). Best-effort; no gold cert without vendor detector/key.

Maps to Nature paper primary method (SynthID-Text).

## 3. Data-driven / backdoor

Model trained or fine-tuned so trigger prompts produce marked or identifiable behavior.

**Out of scope** for this skill (model-side).

## 4. File provenance metadata (C2PA / EXIF / XMP / props)

Signed Content Credentials and AI generator tags in containers (hard-bound to the file: JUMBF/APP11, PNG chunks, XMP packets, OOXML props, etc.).

Industry framing (C2PA + SynthID two-layer model; see Institute of AI PM guide in README references):

| Layer | Mechanism | Survives metadata strip? | This project |
| --- | --- | --- | --- |
| **Hard-bound C2PA** | Signed manifest *in* the file | No — strip/re-encode drops it | **In scope** — `/clean` |
| **Soft binding** | Imperceptible watermark *in content* that can resolve to a remote manifest | Yes (by design) | **Out of scope** — pixel/audio/video signal |
| **Standalone SynthID-class** | Pixel / waveform / token watermark without needing C2PA | Yes for media; text is weaker | Media: optional, partial pixel/audio removal (see `removal-matrix.md`); text → Layer B best-effort |

| Format | Support |
| --- | --- |
| PNG / JPEG / WebP / AVIF / HEIC / GIF / TIFF / BMP | Full strip (stdlib + optional exiftool) |
| SVG | Drop metadata/XMP blocks |
| PDF | Prefer exiftool + qpdf; degraded stdlib XMP strip |
| DOCX / XLSX / PPTX / ODT / EPUB | Scrub zip XML props / customXml / OPF |
| HTML | Meta generator / JSON-LD / data-ai* |
| Markdown | YAML frontmatter AI keys |
| LaTeX (`.tex` / `.ltx`) | `\hypersetup` / `\pdfinfo` provenance fields and comments |
| WAV / MP3 / FLAC / MP4 / MOV | C2PA boxes / chunks / ID3 frames |

A metadata *value* (frontmatter value, `<meta content="...">`) is only matched
against the full vendor vocabulary when its name is a naming field
(`generator`, `created_with`, `tool`, ...). Under any other name the value is
free prose and only unambiguous markers (C2PA, SynthID, AIGC,
digitalSourceType) count — "a static site generator" and "Claude Monet" are
ordinary writing, not provenance.

**Removal:** `/clean` (service-side `clean_file.py` / `clean_image.py`) — usually verifiable by re-inspect.

**Honest report:** after a successful C2PA strip, soft-bound / pixel SynthID (if the generator used them) may still be detectable by vendor tools (e.g. SynthID Detector, Content Credentials verify sites).

## 5. Pixel-domain image (and audio/video) watermarks

Invisible media marks live in the signal, not the metadata. SynthID for audio/video and C2PA **soft binding** are **out of scope**. Pixel-domain image marks (CtrlRegen / DiffusionPurification), TrustMark video and silentcipher / AudioSeal / WavMark audio marks are only optionally and partially removed through `/clean` when the backend is present (see `removal-matrix.md`).
