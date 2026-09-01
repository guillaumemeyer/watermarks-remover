# SynthID-text removal benchmark

bench_synthid_text.py measures how well the Layer B rewrite
(rewrite_text.py) removes SynthID-text-class watermarks, and at what
cost. It generates a controlled corpus with the MarkLLM SynthID scheme, runs
removal variants, and emits a shareable report.

## What it measures

| Metric | Meaning |
| --- | --- |
| Clear rate | % of watermarked samples that flip to not-watermarked after removal (MarkLLM same-config detection) |
| Removal margin | mean (threshold - score) after removal. The robust objective: a "clear" that crosses the threshold by a hair (margin ≈ 0) is not a real removal |
| Score suppression | mean/median drop in detector score (before - after) |
| Quality | lexical divergence (bigram Jaccard distance), semantic divergence (1 - cosine similarity via sentence-transformers), length drift, number/URL survival |
| Cost | estimated tokens in/out, wall time per document, optional USD at your prices |
| Efficiency | clears per million output tokens - removal rate per unit of rewrite cost |
| Attempts | mean rewrite attempts per document (the Layer B loop stops early on pass) |
| Controls | Layer A only (expect ~0% - Unicode scrub must not clear a statistical mark), sanity-gate exclusions, optional re-stamp check |

**Semantic divergence** is `1 - cosine(embed(original), embed(candidate))`
using a SentenceTransformer (default `sentence-transformers/all-MiniLM-L6-v2`).
It is opt-in: if `sentence-transformers` is not installed the metric is
`None` and renders as `—` in the report/CSV. Install it into the MarkLLM venv
to enable the column:

    ~/MarkLLM/.venv/bin/pip install -r service/scripts/requirements-semantic.txt

Model caches must be writable: set HF_HOME to a repo-local cache (e.g.
`$ROOT/.hf-cache`, as benchmarks/benchmark-full.sh does) - a root-owned
`~/.cache/huggingface` silently breaks both the MarkLLM model and the
embedding model.

It is `1 - cosine`, so it measures meaning drift independently of surface
wording: lexical divergence can be high (many different words) while semantic
divergence is low (same meaning), and vice versa.

## How to run

Prerequisites (all external, matching the repo's optional-harness model):

1. A MarkLLM checkout: run service/scripts/setup_markllm.sh (clones
   THU-BPM/MarkLLM at a pinned commit and creates ~/MarkLLM/.venv).
2. A rewrite backend: Ollama (default, loopback) or any
   OpenAI-compatible endpoint. The rewrite model must be a real model.

    # minimal: 3 docs, 1 seed, paraphrase with up to 3 attempts (default, Ollama)
    MARKLLM_DIR=~/MarkLLM \
    python3 service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --rewrite-backend ollama --rewrite-model llama3.2 \
      --out-dir out/bench-2026-06-01

    # recommended full run: more docs/seeds, backtranslate variant, re-stamp control
    python3 service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --docs 10 --seeds 3 \
      --variants "paraphrase:3,backtranslate:3" \
      --restamp-control \
      --rewrite-backend openai-compatible \
      --rewrite-model deepseek-v4-flash \
      --rewrite-base-url https://api.deepseek.com \
      --rewrite-allow-remote \
      --out-dir out/bench-deepseek \
      --tag deepseek-v4-flash

API keys are read from the environment only (WATERMARKS_REWRITE_API_KEY),
never argv. Non-loopback rewrite endpoints require --rewrite-allow-remote.

No vendor tier: Google retired SynthID text watermarking on its API in
Aug 2026 (DETECT_TEXT_WATERMARK is rejected on current models), so detection
here is MarkLLM same-config only. A vendor tier can be re-added if Google
exposes detection again (e.g. via Vertex AI).

**How variants map to rewrites:** each <tactic>:<candidates> variant runs
the Layer B rewrite with candidates as the **variants per evaluation round**;
`--rewrite-loops` (default 1, mirrors `--max-loops` /
`WATERMARKS_REWRITE_LOOPS`) sets how many rounds run before the best-effort
variant is returned. The rewrite is iterative: it generates a round's
candidates, scores each one, and selects by the margin-aware policy below —
there is no "first pass wins" early stop, so a variant evaluates every
candidate in a round and costs at most its candidate count per round (raise
`--rewrite-loops` to run more rounds). The report's att column (and
mean_attempts in results.json / attempts in results.csv) records the actual
attempts per document.

**Selection is margin-aware, not first-pass.** The rewrite evaluates every
candidate in a round (no early stop on the first "not watermarked"), treats a
candidate as a pass only when its after-score sits at least `--target-margin`
below the threshold (default 0.0 = any not-watermarked verdict), and then picks
the passing candidate that changed the least by default (`--select
min-divergence`, content-preserving) or the one with the largest margin
(`--select max-margin`, robustness-first). Raising `--target-margin` is how you
ask for a removal that survives a stricter/vendor detector rather than a
hair-thin threshold crossing.

**Tactics:** `paraphrase` (same language, rephrase), `backtranslate` (via
another language), `structural`, `humanize`, `code`, and `chunk`. `chunk`
splits the document into sentence/paragraph fragments, rewrites each with a
fresh context (new per-token watermark keys), and reassembles them. It is the
strongest removal at a given rewrite cost because every fragment re-keys
independently; use `--tactic chunk` for that, optionally with
`--chunk-shuffle` to also shuffle the rewritten fragments (which breaks
paragraph/line order). Without `--chunk-shuffle`, `chunk` keeps the original
separators so layout is preserved. `chunk` is accepted wherever a variant
tactic is expected, including `parse_variants` (e.g. `--variants
"chunk:2,paraphrase:3"`).

## Minimal-rewrite-level mode (`--mode minimal`)

The named-tactic variants above answer "does this rewrite remove the mark?"
The minimal mode answers **"what is the smallest rewrite that removes the
mark?"** for a given sample, then aggregates that minimum across samples.

- It uses a **numeric rewrite level** instead of a named tactic. The level is
  a request in `(0, 1]`: 0 (the unchanged original) is excluded, 1 means
  "rewrite everything". The actual lexical/semantic divergence of the output is
  *measured*, not guaranteed — the level is a prompt, not a contract.
- For each watermarked sample it starts at `--rewrite-level-start` (0.1),
  tries up to `--level-attempts` (3) rewrites at that level, and if none
  clears the mark it raises the level by `--rewrite-level-step` (0.1) and
  repeats, up to `--rewrite-level-max` (1.0).
- At the first level where at least one rewrite clears (same-config MarkLLM
  detection), it keeps the rewrite with the **smallest semantic divergence**
  and records that level. One row per sample records the chosen level, its
  lexical/semantic divergence, its removal margin, and the attempts spent.
- Pass `--target-margin` (e.g. 0.5) to require a rewrite to sit at least that
  far below the threshold before a level counts as cleared. Without it a level
  that crosses the threshold by a hair — the failing "100% clear, Δ≈0.03" case
  — is reported as cleared even though it is not a robust removal.
- `aggregate_minimal` reports across samples: clear rate, mean/median minimal
  level, mean/median minimal semantic divergence, mean minimal lexical
  divergence, and a level-usage histogram.

    python3 service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --mode minimal \
      --docs 10 --seeds 3 \
      --rewrite-level-start 0.1 --rewrite-level-step 0.1 --rewrite-level-max 1.0 \
      --level-attempts 3 \
      --rewrite-backend ollama --rewrite-model llama3.2 \
      --out-dir out/bench-minimal --tag minimal

**Verdict semantics per sample:** a row is `cleared` (True) when a rewrite at
some level is no longer detected watermarked; `cleared=False` when no level up
to the max cleared it (the sample counts in the clear-rate denominator but is
excluded from the divergence averages); `cleared=None` when the rewrite failed
or MarkLLM verification was unavailable (also excluded from averages). Only
cleared samples contribute to the mean/median minimal level and divergence.

Cost warning: with MarkLLM as the evaluator, each attempt also costs one
MarkLLM detection — up to (candidates x loops) detections per input. The
persistent serve worker (default) keeps the model loaded so detections are
cheap; the --no-worker one-shot path re-loads the model per detection.

Slow machines: MarkLLM SynthID generation on CPU can take ~40 min per
sample (opt-1.3b, 2x300 tokens). The default generation timeout is 900s
(WATERMARKS_BENCH_WATERMARK_TIMEOUT) and the resident worker's op timeout is
600s (--markllm-timeout); on such a box raise both (e.g. 5400 / 7200) or
samples are excluded as "watermark generation timed out" and each failed
worker op falls back to slow one-shot subprocesses.

## Choosing a default rewrite level

The minimal mode measures, per sample, the smallest level whose rewrite clears
the mark. A default rewrite level should then be chosen by how the pipeline
uses it:

- **Adaptive pipeline (recommended):** default the *starting* level to the
  typical minimal level and escalate by 0.1 on failure. Use the median
  (commonly 0.3) as the start, cap escalation at your content-churn budget,
  and require a margin floor (--target-margin, e.g. 0.02-0.05) so a
  hair-thin threshold crossing does not count as a robust removal. Escalation
  is what makes the tail (samples needing 0.7-0.8) reachable at all.
- **Fixed single-shot level:** pick a coverage quantile, not the mean of
  minimal levels. In the 10-doc/3-seed deepseek-v4-flash run the coverage
  curve was roughly 62% at 0.3 (median), 71% at 0.4, 81% at 0.5 and 100% at
  0.8, while measured lexical divergence stayed ~0.5-0.7 across levels (the
  level is a prompt, not a contract). 0.5 is the usual compromise: it covers
  the bulk at near-equal measured churn; the remaining ~20% needs 0.7-0.8 and
  should go through the adaptive path instead of being the default.
- **Never default to 0.8+:** it buys only the last ~20% of samples at the
  highest measured divergence.
- **Quality claims need semantic divergence.** Lexical divergence reacts
  mostly to the rewrite model, not the level. Install
  sentence-transformers (above), and rerun with a margin arm if a report
  shows all '—' in the sem div column: a semantic-less minimal run cannot
  rank levels by meaning drift.
- **Report admission, not just clears.** Sanity-gate exclusions
  (watermarked sample not detected) can silently eat 10-15% of the sample
  set; the report now shows exclusion counts and duplicate-generation
  warnings so a default is not tuned on a biased subset.

Cost modeling: --cost-per-mtok-in 0.30 --cost-per-mtok-out 1.20 (example
prices) attaches an estimated USD figure per row; token counts are
chars / --chars-per-token estimates (default 4.0).

## Outputs (in --out-dir)

- report.md - self-contained Markdown you can paste anywhere: methodology,
  config, results table, controls, caveats, exact reproduction command.
- results.json - full per-sample/per-row data + aggregates.
- results.csv - one row per (doc, seed, variant) for plotting.
- work/ - generated watermarked/unwatermarked samples (kept for inspection). In strategy mode it also holds `work/strategies/<strategy>/` - one directory per evaluated strategy candidate with `input_<doc>_seed<seed>.txt` and `output_<doc>_seed<seed>.txt` side by side, so each combination's rewritten result can be inspected against its input. Controlled by `--write-strategy-outputs` (default on; `--no-write-strategy-outputs` to disable).

## Running from Docker (compose)

The `wr-markllm` service in compose.yaml can run the benchmark end-to-end
(image: pinned MarkLLM checkout at /opt/markllm + all scripts). The image
installs CPU torch by design, so use it for portability/CI, not for GPU
throughput on this machine — for GPU runs use the host `setup_markllm.sh`
venv instead (see README).

```bash
docker compose --profile harness build wr-markllm
docker compose run --rm wr-markllm \
  /app/bench_synthid_text.py --markllm-dir /opt/markllm \
  --corpus /bench-corpus --out-dir /data --tag docker-run \
  --docs 10 --seeds 3 --variants "paraphrase:3,backtranslate:3" \
  --restamp-control
```

Env (rewrite backend) is wired from your .env via compose interpolation;
results land in the `bench-out` volume (/data); the bundled
corpus is mounted read-only at /bench-corpus. The image runs the
persistent MarkLLM serve worker by default, so the ~2-4h one-shot runs
are not a constraint inside the container either.

## What it can and cannot claim

- Can claim: under the MarkLLM SynthID scheme config the benchmark
  controls, at these seeds/docs, with this rewrite backend, this clear rate and
  cost were observed. Same-config-only detection is deterministic and
  reproducible (fixed seeds, pinned MarkLLM commit, recorded commands).
- Cannot claim: that Google's production SynthID-Text detector will fail.
  MarkLLM's SynthID is a research reimplementation with a different keying,
  and Google retired text watermark detection on its API (Aug 2026), so no
  vendor tier exists to verify against. Rewriting with a watermarked model
  can also re-stamp the text - run --restamp-control to check.

## Humanizer and cross-model hygiene

Two things are easy to mistake for removal but are not, or can silently undo it:

- **The humanizer is style polish, not removal.** It targets deterministic
  writing tells (em dashes, rule of three, AI vocabulary, passive voice) and
  intentionally preserves facts, numbers, and names — the opposite of what a
  watermark attack wants. It changes few tokens per pass and optimizes "sounds
  like a human", which is a different objective from "green-list bias gone".
  Keep it as an optional final style polish, but measure removal with the
  detector score (and `--target-margin`), not with how natural the text reads.
  In the variant table it is a moderate-intensity pass, not the removal step.
- **Cross-model hygiene (correctness).** Rewriting with a model that is the
  generator or is itself watermarked re-stamps the text you just removed. Use a
  rewrite backend that is neither, and run `--restamp-control` to detect when
  the backend re-stamps the unwatermarked control (after-positive > 0).

## Sharing a run

Share the --out-dir directory. report.md embeds the reproduction command,
the MarkLLM commit, the watermarks-remover commit, and the caveats, so a reader
can (a) trust what was measured and (b) rerun it. Keep work/ out of archives
unless you want the raw samples.

## Notes on statistical power

- A single document tells you nothing - the watermark is probabilistic. Use
  several documents (--docs 10+) and several seeds per document
  (--seeds 3+) so clear-rate differences are distinguishable.
- Longer text carries more watermark signal: default --max-new-tokens 300.
  Very short samples are excluded by the sanity gate automatically.
- Compare variants (tactic x candidates) within one run, not across runs
  with different backends - the rewrite model dominates the outcome.

## Measurements you should read together

- **clear % vs robust %.** clear % is the per-sample, single-threshold verdict;
  robust % only counts samples whose after-score sits at least `--target-margin`
  below the threshold. At the default `--target-margin 0.0` the two are equal, so
  a hair-thin crossing still reads as "cleared" - that is why a default-level
  decision should run with `--target-margin 0.03`.
- **noop.** A rewrite that changed fewer than `--noop-lex-floor` (default 0.05)
  of bigrams is reported as a no-op and excluded from the clear-rate denominator.
  A `backtranslate` row showing ~0% clear with near-zero lex divergence is a
  **backend no-op**, not evidence that backtranslation is weak - check the noop
  column before believing a 0% clear.
- **AUROC (post).** Area Under the ROC curve between the rewritten-watermarked
  and rewritten-unwatermarked after-scores. 1.0 = perfectly separable, 0.5 =
  indistinguishable. It is threshold-independent and population-level, so it
  complements clear %: a row that clears by a hair still moves AUROC little,
  while a row that erases the mark's distributional signal drives AUROC toward
  0.5. Requires `--restamp-control`; degrades to `—` otherwise.
- **human_like ↑** (`1 − AI-likeness`) under `--human-backend`: `stylometry`
  (default, stdlib), `lastde`/`binoculars` (offline `ai_human.py` checkout), or
  `pangram` (Pangram Labs async **bulk** API; key in `PANGRAM_API_KEY`, model via
  `--human-pangram-model`). In the variants table the same backend score is
  shown as raw `AI-likeness ↓`. A gauge, not a proof of human authorship; a
  missing key/backend degrades to stylometry.

## Strategy search mode (`--mode strategy`)

Instead of a per-variant table, this mode answers "what is the best combination
of tactics, and at what intensity for each?" A strategy is an ordered list of
`tactic@intensity` steps, each a Layer B rewrite with a numeric intensity that
modulates that tactic's prompt (e.g. `chunk@0.6,paraphrase@0.3,humanize@1.0`),
applied sequentially - each step's output feeds the next.

- Phase 1 sweeps each tactic over `--intensity-grid` (single-step strategies) and
  produces the per-tactic intensity curves (robust %, sem div, human_like vs
  intensity).
- Phase 2 runs a beam search (`--beam`, `--max-passes`) once per weight vector in
  `--weight-grid`, combining an order of tactics with the top
  `--phase2-levels-per-tactic` intensities for that weight, so both step order
  and intensity are explored.
- **Cross-input scoring.** Every candidate is scored on the aggregate **population**
  (all docs × seeds), so the primary axis is **coverage** = robust clear rate, not a
  single-sample verdict. A single input tells you almost nothing; use `--docs 10+`
  and `--seeds 3+` for meaningful coverage.
- The report's **Pareto frontier** is computed by dominance (no weights) over the
  union of all candidates, so it is weight-independent. The "recommended" strategy
  is the frontier point best matching `--recommend-weight` (a
  w_removal/w_semantic/w_human triple summing to 1.0, default `0.5/0.3/0.2`).
- **A strategy is only recommended if it clears enough inputs.** `--coverage-floor`
  (default `0.5`) is the minimum population robust clear rate a candidate needs
  before it can be put forward. If nothing clears the floor, the report says so and
  shows **no recommended strategy** (the frontier is diagnostics only) - it never
  recommends a non-removing strategy.
- **Hold-out validation.** `--eval-split 0.8` keeps 80% of documents for the search
  and holds the rest out; the frontier candidates are then re-measured on the held-out
  documents (reported as an extra `holdout %` column and used for the recommendation),
  so a strategy that only overfits the search subset is not recommended.
- **`humanize` always runs last.** Any strategy containing `humanize` is ordered so
  it is the final step, and the recommended strategy is auto-finished with
  `humanize@--humanize-intensity` (default `0.4`) as the user-facing polish. Its
  reported axes reflect that final output.
- **Adaptive escalate-on-resist.** `--adaptive` takes the recommended strategy as a mild
  default and, for any input that resists it, re-runs with every step's intensity raised
  by `--escalation-step` (capped at `--escalation-max`, up to `--escalation-attempts`
  rounds) until the mark is removed. The report shows the default vs. escalated clear rate
  and the escalation-level distribution, so the resistant tail is reachable without
  over-rewriting easy inputs.
- `--strategies "chunk@0.6,paraphrase@0.3"` composes and scores one explicit strategy
  instead of searching.
- `--layer-a-after` re-runs the Unicode scrub on the final output; default **off**
  because the rewrite backend is assumed watermark-safe.
- **Every evaluated strategy is written for inspection.** Each candidate directory
  (`work/strategies/<strategy>/`) contains the per-sample `input_*.txt` and
  `output_*.txt` rewritten text, so you can eyeball what each combination actually
  produced relative to its watermarked input. `results.json` keeps the text out
  and records the `output_dir` / `output_files` paths instead, so the JSON stays
  readable. Disable with `--no-write-strategy-outputs`.

    ~/MarkLLM/.venv/bin/python service/scripts/bench_synthid_text.py \
      --markllm-dir ~/MarkLLM \
      --corpus benchmarks/corpus-large --docs 20 --seeds 3 --max-new-tokens 300 \
      --mode strategy --target-margin 0.03 --coverage-floor 0.5 \
      --eval-split 0.8 --humanize-intensity 0.4 --adaptive \
      --escalation-step 0.1 --escalation-attempts 3 \
      --rewrite-backend openai-compatible --rewrite-model <model> \
      --rewrite-base-url <url> --rewrite-allow-remote --tag <backend>

A strategy search is expensive (each candidate = a full rewrite chain per sample).
Run Phase 1 coarsely first with fewer docs/seeds, then confirm the winning
strategies on a larger run with adequate statistical power. Lower
`--phase2-levels-per-tactic` (or cut docs/seeds, `--beam`, `--max-passes`) to
keep a run inside a tight wall-clock budget.
