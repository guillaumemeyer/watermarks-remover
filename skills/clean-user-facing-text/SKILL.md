---
name: clean-user-facing-text
description: Clean and finalize authorized natural-language text intended for readers by auditing suspicious invisible Unicode and rewriting prose while preserving facts, meaning, and the writer's voice. Use when the user asks to clean, humanize, polish, or finalize articles, manuscripts, reports, documentation, emails, product copy, UI text, Markdown, or HTML prose, or when a project rule or instruction file explicitly requires this workflow. Don't use for code-only tasks or undisclosed authorship evasion; leave code, commands, identifiers, paths, APIs, formulas, citations, required disclosures, and verbatim quotations unchanged.
---

# Clean user-facing text

Apply a final text-hygiene pass to prose the user owns or is authorized to process. Treat Unicode cleanup as deterministic and statistical-watermark reduction as best-effort; never claim that a rewrite proves human authorship or is undetectable. Preserve required academic, legal, platform, and regulatory disclosures.

## Workflow

1. Identify the prose that readers will see.
2. Protect non-prose spans:
   - fenced and inline code
   - commands, paths, URLs, identifiers, API names, and exact values
   - formulas, citations, and text the user asks to quote verbatim
3. Preserve every claim, fact, number, name, citation, and requirement. Never invent a detail, name, number, quote, or source to make the prose easier to write or more varied: if a fact is missing, flag the gap rather than fill it. The rewrite may sharpen, compress, or reorder, but it may not add or remove claims.
4. **Measure before.** Inspect and score the input with the vendored zero-LLM stylometry estimator (see Scoring) and record the score. Read the report's `density_tier`: rewrite only when it is `high`; for `low` or `medium`, verify the text and otherwise leave the text unchanged. For a flag-only audit that never rewrites, use `--audit`:

   ```bash
   PYTHON "$SCRIPTS/inspect_text.py" --stylometry --json INPUT
   PYTHON "$SCRIPTS/inspect_text.py" --audit INPUT  # detect-only: lists flagged spans, no rewrite
   ```

5. Establish the writing brief before changing prose:
   - use a voice sample only when the user owns it or is authorised to use it; don't imitate another named person
   - when there is no sample, make the prose clear and natural without pretending to imitate a particular person
   - never inject a voice the source lacks: no fake first person ("I've seen this"), invented specifics, forced contrarianism, performed candor, or added stance and personality. Preserve the writer's deliberate rough edges and domain terms rather than scrubbing them
   - keep required disclosures, uncertainty, and the writer's actual point of view
   - pick the voice and domain preset the text fits (see Voice and domain presets); the default is general prose
6. **Layer A — strip artifacts first.** For text artifacts or supplied text files, run the deterministic Unicode pass before rewriting, so the rewrite operates on clean, marker-free text:

   ```bash
   PYTHON "$SCRIPTS/clean_text.py" INPUT -o OUTPUT --stats --no-normalize-spaces
   ```

7. **Layer B — rewrite once.** Rewrite the remaining prose once, applying the detector levers (see Detector levers) in order:
   - vary clause order, sentence boundaries, rhythm, connectors, and function words
   - replace formulaic transitions and filler with direct, natural wording
   - keep the concrete details and judgement that make the text recognisable as the writer's
   - treat unusual grammar, repetition, directness, and phrasing as possible voice or accessibility choices; change them only when the user asks or when they create a clear reading problem
   - preserve the requested language, tone, structure, and formatting; never translate unless asked
   - for non-English text, use fluent constructions native to that language rather than English sentence patterns
   - do not add or remove claims merely to increase variation
8. **Layer A again.** Run the deterministic Unicode pass on the rewritten result to catch any artifacts the rewrite introduced (smart quotes, em dashes, homoglyphs):

   ```bash
   PYTHON "$SCRIPTS/clean_text.py" OUTPUT -o FINAL --stats --no-normalize-spaces
   ```

9. **Measure after.** Score the rewritten text the same way. Report scores and confidence levels when available; otherwise report `status: insufficient_length`. A lower after-score means the measurable signals moved; it is not a verdict from any detector, and it never overrides the fact and voice rules above.
10. Return only the polished result unless the user asks for an audit or explanation.

For practical guidance on preserving a writer's voice and removing formulaic prose,
read `references/writing-in-your-voice.md` whenever the user asks to retain or adjust voice.
For what detectors really measure and which claims are legitimate, read `references/detectors.md`.

## Scoring

`scripts/inspect_text.py --stylometry` (or the standalone `scripts/score_stylometry.py`,
which also accepts `--explain`) runs the zero-LLM estimator vendored from the service
pipeline: sentence-length burstiness (coefficient of variation), weighted AI-cadence
phrase density per 100 words, lexical diversity (MATTR), and a dampened composite
score from 0 to 1. Exit code 1 means the score is at or above the threshold
(default 0.65). The report also carries a `density_tier` (`low` / `medium` /
`high`, or `uncalibrated` when not scored) that re-labels where the score sits
so the rewrite pass engages only for `high`. `inspect_text.py --audit` produces
the same scoring plus the detect-only flagged-span list but never rewrites.
Under 30 words the estimator reports `status: insufficient_length` instead of a
score. Nothing here calls the network; the skill stays self-contained.

Limits: the estimator is calibrated for the statistical detector family
(perplexity and burstiness style signals). It is not the output of trained neural
classifiers such as GPTZero, Turnitin, Originality, or Pangram, it does not detect
secret-key watermarks, and it says nothing about authenticity. A low after-score
means the measured signals moved; it does not prove the text reads as human or that
any particular detector would accept it. When the user asks for an audit, report
the numbers as a gauge, not as a verdict (see Reporting).

## Deterministic Unicode pass

Resolve `SCRIPTS` to this skill's `scripts/` directory.
Use the available Python 3 launcher for the platform. Replace `PYTHON` below
with `python3` on most macOS/Linux systems, `py` on Windows, or another verified
Python 3 command.

This skill is self-contained and runs its vendored scripts directly; it has no
service or network dependency. Deterministic Layer A cleaning below is invoked
by script, intentionally, so the skill works where the HTTP service is absent.

Inspect first when editing an existing file (include `--stylometry` to record
the before score):

```bash
PYTHON "$SCRIPTS/inspect_text.py" --stylometry --json INPUT
PYTHON "$SCRIPTS/clean_text.py" INPUT -o OUTPUT --stats --no-normalize-spaces
PYTHON "$SCRIPTS/inspect_text.py" --stylometry --json OUTPUT
```

Use `-` for stdin. Prefer a new `*.cleaned.*` output unless the user explicitly requests in-place editing.

Use `--no-normalize-spaces` by default so NBSP, narrow no-break spaces, figure spaces, and CJK ideographic spaces retain their layout semantics. Normalize spaces only when the user requests it.

Do not use `--aggressive-homoglyphs`, `--nfkc`, or `--strip-emoji-glue` unless the user requests aggressive normalization and accepts possible changes to multilingual text, emoji, directionality, or typography.

The scripts support plain text, source text, Markdown, and HTML source as text. For mixed Markdown or HTML, inspect hit positions first. If a hit falls inside protected code, attributes, or another non-prose span, do not run whole-file cleanup; clean only the prose segments or leave that hit unchanged. Do not pass binary containers such as PDF, DOCX, images, or archives.

For a chat-only response that is not written to a file, perform the rewrite workflow directly. Do not claim that the chat response received a deterministic post-send Unicode filter.

## Detector levers

Statistical detectors score probability patterns: AI prose is too predictable (low
perplexity), too even (low burstiness), and too full of stock phrases. The levers
below target those signals, most effective first. Levers 1, 2, and 6 are
deterministic or near-deterministic; 3 to 5 are aims, not guarantees. Engage the
rewrite levers only when the measure-before `density_tier` is `high`; for `low`
or `medium`, the measurable AI-density signals are weak, so verify and otherwise
leave the text unchanged. The pass ordering below follows the pattern
catalogs in `references/detectors.md`.

1. **Strip artifacts first (always).** Run the deterministic Unicode pass
   (`clean_text.py`) before anything else: invisible characters and homoglyphs are
   mechanical markers that hurt with every detector family and are cheap to remove.
2. **Kill the stock vocabulary (perplexity).** Replace AI-stock words with plain,
   concrete, specific ones: delve, tapestry, testament, underscore, foster,
   seamless, multifaceted, myriad, paradigm shift, harness the power of, plays a
   crucial role, in today's fast-paced world, it is important to note. The vendored
   scorer's phrase list is the canonical list to check; the rewrite must clear it.
3. **Inject burstiness.** Vary sentence length deliberately: a short sentence after
   two long ones, and occasionally the reverse. Uniform mid-length cadence
   (sentence-length CV below 0.35) is one of the strongest AI signals.
4. **Flatten the structure.** Break formulaic sections into the prose itself:
   "despite X, the future looks bright" closers, forced groups of three,
   "challenges and opportunities" templates, and announcement-style headers.
5. **Match a real voice and keep specifics.** Prefer concrete detail from the
   source over generic phrasing, and let the writer's sample set the rhythm
   (see `references/writing-in-your-voice.md`). Never invent a fact to raise
   variance; a lower score with a fabricated detail is still a failed rewrite.
6. **Normalize punctuation and RLHF voice.** Straight quotes, no em dashes, no
   bolded mini-headers, no "I hope this helps", no "as an AI", no hedged
   perfectionism or over-cautious disclaimers beyond what the user must keep.
   Required disclosures stay.
7. **Optional recursive paraphrase.** For keyed statistical watermarks (KGW,
   SynthID-Text style), any meaning-preserving rewrite that changes token order
   degrades detection, and back-translation or repeat paraphrase is the strongest
   known pass. Do this only when the user asks for watermark reduction: each extra
   pass risks meaning drift, and no vendor secret key can be verified locally.

Honest caveat: these levers are aimed at statistical detectors. Trained neural
classifiers are adversarially trained against paraphrase-style edits (see
`references/detectors.md`); against those, the only robust lever is matching a real
human distribution, and even that cannot be guaranteed.

## Voice and domain presets

Presets tune how hard to apply the levers and how much personality to allow. They
never override the fact rules (fiction excepted, see step 3 above) and they are
not guarantees of any detector result.

| Preset | Personality | Rhythm and phrasing | What to eliminate |
| --- | --- | --- | --- |
| General prose (default) | Author's voice first, no injected stance | Mild variation, natural connectors | Stock AI vocabulary, uniform cadence |
| Essay / blog | Stance, asides, mixed feelings welcome | Strong length variation, uneven rhythm | Significance hype, aphorism formulas, rule of three |
| Technical / documentation | Neutral, precise | Moderate variation, short declaratives | Promo language, em dashes, bolded mini-headers; keep code and identifiers intact |
| Academic / professional | Formal, evidence-first | Restrained variation, controlled hedging | Over-claiming verbs, novelty padding, citation dumps; keep required discipline |
| Business / product copy | Plain claims, concrete value | Direct sentences | "Seamless", "empower", vague benefits, rule of three, required disclaimers kept |
| Fiction | Invented detail allowed | Variation to fit the narrator | Uniform cadence, editorial clichés; preserve dialect and quirks |

### Plain-language sub-mode

For procedures, runbooks, errors, and other engineer-facing text, the user may
request a plain-language sub-mode: short common words, one instruction per
sentence, imperative verbs for steps, one meaning per term, and no marketing
adjectives or unbounded hedging. This sub-mode is a clarity floor, not a new
personality: it strips voice deliberately, keeps every claim and requirement,
and is not a detector-evasion tool. Use the voice-preserving presets above for
essays, posts, and personal prose instead.

## Code boundary

When prose and code are mixed, rewrite prose only. Never rename variables, alter string literals, reformat code, or change executable output as part of this skill. If a Markdown or HTML file contains executable snippets, preserve those spans byte-for-byte whenever practical.

## Reporting

When the user asks for an audit, distinguish:

- **Verifiable:** Unicode characters removed or replaced, with script counts; stylometry scores before and after, with confidence levels (as a gauge, not a verdict).
- **Best-effort:** prose rewritten to alter token and syntax patterns.
- **Not established:** official detector evasion, human authorship, or removal of a vendor's secret-key watermark.

For technical background, read `references/watermark-notes.md`. For misuse or disclosure questions, read `references/responsible-use.md`.
