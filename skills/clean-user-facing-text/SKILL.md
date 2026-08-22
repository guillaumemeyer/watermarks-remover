---
name: clean-user-facing-text
description: Clean and finalize authorized natural-language text intended for readers by auditing suspicious invisible Unicode and rewriting prose while preserving facts, meaning, and the writer's voice. Use in Cursor when the user asks to clean, humanize, polish, or finalize articles, manuscripts, reports, documentation, emails, product copy, UI text, Markdown, or HTML prose, or when an installed Cursor Rule explicitly requires this workflow. Don't use for code-only tasks or undisclosed authorship evasion; leave code, commands, identifiers, paths, APIs, formulas, citations, required disclosures, and verbatim quotations unchanged.
---

# Clean user-facing text

Apply a final text-hygiene pass to prose the user owns or is authorized to process. Treat Unicode cleanup as deterministic and statistical-watermark reduction as best-effort; never claim that a rewrite proves human authorship or is undetectable. Preserve required academic, legal, platform, and regulatory disclosures.

## Workflow

1. Identify the prose that readers will see.
2. Protect non-prose spans:
   - fenced and inline code
   - commands, paths, URLs, identifiers, API names, and exact values
   - formulas, citations, and text the user asks to quote verbatim
3. Preserve every claim, fact, number, name, citation, and requirement.
4. Establish the writing brief before changing prose:
   - use a voice sample only when the user owns it or is authorised to use it; don't imitate another named person
   - when there is no sample, make the prose clear and natural without pretending to imitate a particular person
   - keep required disclosures, uncertainty, and the writer's actual point of view
5. Rewrite the remaining prose once:
   - vary clause order, sentence boundaries, rhythm, connectors, and function words
   - replace formulaic transitions and filler with direct, natural wording
   - keep the concrete details and judgement that make the text recognisable as the writer's
   - treat unusual grammar, repetition, directness, and phrasing as possible voice or accessibility choices; change them only when the user asks or when they create a clear reading problem
   - preserve the requested language, tone, structure, and formatting; never translate unless asked
   - for non-English text, use fluent constructions native to that language rather than English sentence patterns
   - do not add or remove claims merely to increase variation
6. For text artifacts or supplied text files, run the deterministic Unicode pass after rewriting.
7. Return only the polished result unless the user asks for an audit or explanation.

For practical guidance on preserving a writer's voice and removing formulaic prose,
read `references/writing-in-your-voice.md` whenever the user asks to retain or adjust voice.

## Deterministic Unicode pass

Resolve `SCRIPTS` to this skill's `scripts/` directory.
Use the available Python 3 launcher for the platform. Replace `PYTHON` below
with `python3` on most macOS/Linux systems, `py` on Windows, or another verified
Python 3 command.

Inspect first when editing an existing file:

```bash
PYTHON "$SCRIPTS/inspect_text.py" --json INPUT
PYTHON "$SCRIPTS/clean_text.py" INPUT -o OUTPUT --stats --no-normalize-spaces
PYTHON "$SCRIPTS/inspect_text.py" --json OUTPUT
```

Use `-` for stdin. Prefer a new `*.cleaned.*` output unless the user explicitly requests in-place editing.

Use `--no-normalize-spaces` by default so NBSP, narrow no-break spaces, figure spaces, and CJK ideographic spaces retain their layout semantics. Normalize spaces only when the user requests it.

Do not use `--aggressive-homoglyphs`, `--nfkc`, or `--strip-emoji-glue` unless the user requests aggressive normalization and accepts possible changes to multilingual text, emoji, directionality, or typography.

The scripts support plain text, source text, Markdown, and HTML source as text. For mixed Markdown or HTML, inspect hit positions first. If a hit falls inside protected code, attributes, or another non-prose span, do not run whole-file cleanup; clean only the prose segments or leave that hit unchanged. Do not pass binary containers such as PDF, DOCX, images, or archives.

For a chat-only response that is not written to a file, perform the rewrite workflow directly. Do not claim that the chat response received a deterministic post-send Unicode filter.

## Code boundary

When prose and code are mixed, rewrite prose only. Never rename variables, alter string literals, reformat code, or change executable output as part of this skill. If a Markdown or HTML file contains executable snippets, preserve those spans byte-for-byte whenever practical.

## Reporting

When the user asks for an audit, distinguish:

- **Verifiable:** Unicode characters removed or replaced, with script counts.
- **Best-effort:** prose rewritten to alter token and syntax patterns.
- **Not established:** official detector evasion, human authorship, or removal of a vendor's secret-key watermark.

For technical background, read `references/watermark-notes.md`. For misuse or disclosure questions, read `references/responsible-use.md`.
