# How Claude marks AI-generated content

Primary source: [Anthropic Help Center](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content) (EU AI Act Article 50(2) Code of Practice).

## Policy snapshot

| Topic | Anthropic position |
| --- | --- |
| New models | Marking for models launched on/after **2026-08-02** |
| Older models | Transition; “in progress” |
| Surfaces | API, Claude, Claude Code, Cowork, Tag |
| Regions | **Worldwide** |
| Detection | Third-party detection promised; docs **forthcoming** |

## Mechanism 1 — embedded text watermarks

- Applied at the **model level** into the text itself (not file metadata).
- Imperceptible; survives copy-paste; may survive light editing.
- Weakened by paraphrase, translation, heavy edit, mixing, short text.

**Likely technical class** (Anthropic has not published the algorithm): statistical **token-sampling** watermarks (Kirchenbauer / SynthID-style). See `vendor-notes.md` and `mark-classes.md`.

Layer A only removes **Unicode / homoglyph** carriers. Layer B (rewrite) targets statistical marks.

## Mechanism 2 — C2PA on files

- Signed **Content Credentials** on supported types (examples: `.png`, `.jpg`, `.svg`).
- Tamper-evident while present; stripped by re-encode, metadata scrub, or many upload pipelines.
- Inspect and strip through the service (`/inspect`, `/clean`), which uses `c2patool` and ExifTool server-side when they are installed (check `/capabilities`).

## Caveats (Anthropic)

- Detected mark ⇒ content **may have been processed** by Claude — not proof of sole authorship.
- No mark ≠ human-only origin.
- Proofreading / translate / summarize can stamp human material.
