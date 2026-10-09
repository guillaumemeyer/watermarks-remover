# DeepSeek configuration and sidecar smoke test

Use DeepSeek through the existing OpenAI-compatible client, without adding an
SDK or dependencies to `wr-core`. The SynthID sidecar generates controlled
watermarked samples; DeepSeek rewrites them. This does not assume DeepSeek
exposes a public watermark or detector. Use only content and models you own or
are authorized to test.

## Configure DeepSeek

Supply `WATERMARKS_REWRITE_API_KEY` in the environment. Never put keys in
arguments, fixtures, reports, or Git. Calls consume provider credits and send
generated text to DeepSeek; the runner requires explicit `--allow-remote`.

```bash
export WATERMARKS_REWRITE_BASE_URL=https://api.deepseek.com
export WATERMARKS_REWRITE_MODEL=deepseek-flash
```

The [official API guide](https://api-docs.deepseek.com/) lists current models.
As checked on 2026-10-09, `deepseek-flash` is the recommended Flash name;
`deepseek-v4-flash` is a legacy alias served by a newer model. Recheck before
running. The report records the requested model and time, not a pinned provider
implementation. The reused client returns text only, so response model IDs and
provider token usage are not recorded.

| Client | Example base URL | Appended path |
| --- | --- | --- |
| `rewrite_text.py` / smoke runner | `https://api.deepseek.com` | `/v1/chat/completions` |
| `stealer/steal.py query` | `https://api.deepseek.com` | `/chat/completions` |

Do not append `/v1` to the rewrite base URL. Flags `--model` and
`--rewrite-base-url` override the corresponding rewrite environment variables.
The fixture supplies the model when neither a flag nor the environment does.
The runner defaults to `--reasoning-effort none`, matching the rewrite CLI.
A short real `deepseek-flash` request accepted that setting on 2026-10-09 and
returned no reasoning content. `--reasoning-effort off` omits the field for
models that reject it; omission does not guarantee reasoning is disabled.

## Start the existing sidecar

From the repository root, with Docker Compose available:

```bash
export WATERMARKS_SYNTHID_TEXT_URL=http://wr-synthid-text:8767
export WATERMARKS_SYNTHID_TEXT_TIMEOUT=600
docker compose --profile harness up --build -d wr-core wr-synthid-text
```

The existing optional MarkLLM image may download its model on first use.
Generation can be slow on CPU. The core timeout is a **whole-batch budget**,
clamped to 600 seconds. Warm the model and retry, or use a one-sample fixture
if a cold start exceeds it. Raising only the runner timeout does not raise the
core budget. The sidecar needs no published host port.

Authentication stays separate at each hop:

| Environment variable | Used by |
| --- | --- |
| `WATERMARKS_SERVER_API_KEY` | Runner -> core; must match core configuration |
| `WATERMARKS_SYNTHID_TEXT_API_KEY` | Core -> sidecar; existing Compose wiring |
| `WATERMARKS_REWRITE_API_KEY` | Runner -> DeepSeek only |

The first two keys are optional for the existing local setup. The provider key
is required. Keep the keys distinct.

## Run the fixture

```bash
python3 service/scripts/bench_deepseek_smoke.py \
  --allow-remote --timeout 610 --out out/deepseek-smoke.json
```

The runner loads `benchmarks/providers/deepseek.json`, sends its two fixed
prompts to `POST /watermark/batch`, and makes one paraphrase request for each
successful sample using the existing rewrite client. There are no provider
retries. No detector runs; unchanged rewrites are recorded with `changed: false`.

Use `--fixture path/to/fixture.json` for a different small corpus or generation
options, keeping the same JSON structure and unique sample IDs. Use `--core-url`
for another existing core. `--allow-remote` permits remote core and rewrite
URLs; HTTP(S) URLs cannot contain credentials, queries, or fragments. Redirects
are refused by the existing HTTP client.

The report includes configuration, total/batch timing, per-sample rewrite
timing, generated/rewritten text, and failures. It always records
`watermark_verification: "not_performed"` and contains **no detector scores or
clear rate**. HTTP failures retain status/error categories, not remote error
bodies. Configured credentials are redacted even if echoed by a service.
Inspect generated text before sharing reports.

Exit status is zero only when all samples have nonempty generation and rewrite
outputs. Batch protocol errors and individual failures produce a nonzero exit
and a report; invalid configuration fails before requests and creates no report.
Setup/output errors also return nonzero. Success establishes HTTP
interoperability, not watermark removal or output quality.

For actual removal measurements, use the existing
[full SynthID benchmark](synthid-text-benchmark.md) with matching generation and
detection configuration, controls, and adequate sample size. It currently uses
its MarkLLM worker directly; this HTTP smoke test does not replace it.

## Optional target/baseline collection

For a small owned `small-prompts.jsonl` corpus (one `{"text": "..."}` per line):

```bash
export WATERMARKS_STEAL_BASE_URL=https://api.deepseek.com
export WATERMARKS_STEAL_MODEL=deepseek-flash
# Supply WATERMARKS_STEAL_API_KEY separately in the environment.
python3 stealer/steal.py query \
  --backend openai-compatible --allow-remote \
  --prompts small-prompts.jsonl --out out/deepseek-replies.jsonl \
  --concurrency 1 --max-new-tokens 256
```

The backend flag is required: `query` defaults to `dry-run`. Its `WATERMARKS_STEAL_*`
environment is independent of `WATERMARKS_REWRITE_*`. A comparison baseline is
not proof of watermark-free output. Distribution differences between models
must not be interpreted as evidence of a DeepSeek watermark. See
[stealer's caveats](../stealer/README.md#honest-use).

## Offline verification

```bash
python3 -m pytest -q tests/test_bench_deepseek_smoke.py
```

Tests exercise the real core handler and rewrite client with local stub
services. They need no Docker, ML dependencies, or real keys, and never call
DeepSeek. They validate wiring/failures, not real SynthID generation.
