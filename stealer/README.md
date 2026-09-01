# stealer

Black-box watermark-stealing module for SynthID-class text watermarks, and the
prompt-corpus downloader it runs on.  It implements the one-time **steal** step
of the attack from the shared conversation (and the corresponding ETH SRI line
of work on SynthID-Text detection): collect a large corpus of watermarked model
replies, estimate how much each next token is boosted after each short context
relative to a non-watermarked baseline, and emit that estimate as
`s*(token | context)`.  A downstream cleaner can consume `s*` to demote the
"green" tokens the watermark favors.

This is research / robustness tooling for content and models you run or are
authorized to test.  It does not recover the secret key, and `s*` is an
estimator, not the true green set — see [the honest-use caveats](#honest-use).

## Layout

```text
stealer/
  download_prompts.py   fetch a prompt corpus (default 30k C4 RealNewsLike)
  tokens.py             tokenizer + (context, next-token) counting
  scorer.py             build s* and apply it (score / demote / promote)
  steal.py              CLI: query -> build -> detect
  prompts/              where the downloaded corpus lands (prompts.jsonl)
```

## Steal pipeline

The conversation describes three steps:

1. **Query** the watermarked model with many benign prompts; collect long
   watermarked replies.  (`steal.py query`)
2. **Count** how often each next token follows each short context in the
   watermarked replies, vs a non-watermarked baseline (a different model or
   old unwatermarked dumps — the baseline model need not match the watermarked
   one).  (`steal.py build`)
3. Convert the ratios into a score `s*(token | context)`: high ⇒ "likely green
   / watermark-boosted."  The output is **not** the secret key, but a reusable
   table/function.  (`steal.py build --out s_star.json`)

Then use `s*` to scrub: run a paraphraser on the text to clean and, at each
step, subtract `delta * s*` from the candidate logits to demote green tokens;
keep only paraphrases that stay similar and fail the detector.  `s*` can be
applied directly by a caller via `scorer.apply_delta`.

### The three models

The full pipeline involves up to three models, but **`steal.py` only ever queries
two of them**:

| # | Role | Which model | Used by |
| --- | --- | --- | --- |
| 1 | **Target** — the watermarked model whose mark we extract | the model you want to cleanse | `steal.py query` (first run) |
| 2 | **Baseline** — non-watermarked reference distribution | a *different* model (or old unwatermarked dumps) | `steal.py query` (second run) |
| 3 | **Scrubber / paraphraser** — applies `s*` to logits at cleanup time | e.g. DIPPER | downstream; **not** called by `steal.py` |

`build` and `detect` are **model-free**: they derive `s*` and score text with
deterministic counting, so there is no third "analysis" LLM.  The baseline `#2`
may be a different model from the target `#1` — the question the shared
conversation started from.

## Quick start

```bash
# 1. Corpus of prompts (30k C4 RealNewsLike passages) -> stealer/prompts/prompts.jsonl
python3 stealer/download_prompts.py --count 30000

# 2. Collect watermarked replies (dry-run by default; point at a model for real)
python3 stealer/steal.py query --prompts stealer/prompts/prompts.jsonl \
  --out stealer/replies.jsonl --backend openai-compatible --model <model>

# 2b. Non-watermarked baseline replies from a different model (optional but
#     recommended; without it build() falls back to unigram statistics)
python3 stealer/steal.py query --prompts stealer/prompts/prompts.jsonl \
  --out stealer/baseline.jsonl --backend openai-compatible --model <baseline-model>

# 3. Derive s* (scorer table)
python3 stealer/steal.py build --replies stealer/replies.jsonl \
  --baseline stealer/baseline.jsonl --ctx 8 --topk 50 --out stealer/s_star.json

# 4. Score a candidate text against the stolen scorer
python3 stealer/steal.py detect --text "some text" --s-star stealer/s_star.json --ctx 8
```

### Configuring the target and baseline models

Each `query` run talks to **one** model, so the target and baseline are each
configured by their own `query` invocation (see the quick start above).  Point
the target run at the model you want to cleanse and the baseline run at a
different model:

```bash
# 1. Target — the watermarked model whose mark we're extracting.
WATERMARKS_STEAL_BASE_URL=<url> WATERMARKS_STEAL_API_KEY=<key> \
WATERMARKS_STEAL_MODEL=<target-model> \
  python3 stealer/steal.py query --prompts stealer/prompts/prompts.jsonl \
  --out stealer/replies.jsonl

# 2. Baseline — a *different*, non-watermarked model.
#    A loopback endpoint (e.g. local Ollama) needs no --allow-remote.
WATERMARKS_STEAL_BASE_URL=<url> WATERMARKS_STEAL_MODEL=<baseline-model> \
  python3 stealer/steal.py query --prompts stealer/prompts/prompts.jsonl \
  --out stealer/baseline.jsonl
```

`query` reads these vars from the environment.  They mirror the repo's
`WATERMARKS_REWRITE_*` naming under their own `WATERMARKS_STEAL_*` prefix, so the
rewrite backend's settings never leak into a steal run:

| Env var | Flag | Meaning |
| --- | --- | --- |
| `WATERMARKS_STEAL_BASE_URL` | `--base-url` | OpenAI-compatible base (default `https://api.openai.com/v1`) |
| `WATERMARKS_STEAL_API_KEY` | — | Key — read from the environment only, never on argv |
| `WATERMARKS_STEAL_MODEL` | `--model` | Model this run queries (target on run 1, baseline on run 2) |
| `WATERMARKS_STEAL_ALLOW_REMOTE=1` | `--allow-remote` | Required to hit a non-loopback endpoint |

Keys are read from the environment only.  `dry-run` writes deterministic
placeholder replies so the rest of the pipeline runs without a model.  A
baseline is optional but recommended — without it `build()` falls back to
unigram statistics.

## Downloader

`download_prompts.py` uses the Hugging Face datasets-server `rows` API
(`allenai/c4`, config `realnewslike`), so it needs only the stdlib.  It applies
`--min-chars` / `--max-chars` filtering, writes `prompts.jsonl` line-buffered,
and checkpoints after each complete page in `prompts/.download-state.json`, so a
big run can be paused and resumed without duplicating or dropping rows — an
interrupted page is rolled back to the last committed boundary.

C4 RealNewsLike is news-article prose, not literally prompts; the module treats
each passage as a query you send to the model.

To lift the unauthenticated datasets-server rate limit, export `HF_TOKEN` and
the downloader sends it as a bearer header (env-only, never printed):

```bash
HF_TOKEN=$(cat ~/.hf_token) python3 stealer/download_prompts.py --count 30000
```

## Honest use

- `s*` is a black-box estimate of the watermark's green-set prior.  It is valid
  for the model family / context-length it was built against, and becomes stale
  if the vendor changes the scheme.
- This is the same honesty caveat the repo applies to its other same-scheme
  harnesses: "same-config-only, not a vendor oracle."  A high `s*` removes the
  *stolen* prior, not necessarily the mark a vendor's detector checks.
- Remove provenance marks only on content **you own or are authorized to
  process**; see [`skills/remove-ai-marks/references/ethics.md`](../skills/remove-ai-marks/references/ethics.md).
