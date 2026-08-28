# SynthID-text benchmark corpus

Seed documents for `bench_synthid_text.py`. Each file is a factual,
neutral prompt; the benchmark extends it with MarkLLM's `facebook/opt-1.3b`
generator (300 new tokens by default) and uses the full prompt+continuation
as the watermarked artifact.

- Keep seeds short (50-90 words) so the generated document is mostly
  model output — that is where the token-sampling watermark lives.
- Vary domains and style so results are not an artifact of one topic.
- Add your own files for a custom corpus; pass `--corpus /path/to/dir`.

`corpus-large/` is the 30-doc superset (the 8 corpus/ docs plus 22 more
domains) used for powered runs: `--corpus benchmarks/corpus-large --docs 30`.

Run metadata that affects interpretation: sanity-gate exclusions
(watermarked sample not detected) and duplicate generations across seeds are
reported by the benchmark, so a sample set's effective size is never hidden.

Seeds are deterministic inputs only — the watermark comes from the
generation step, not from these files.
