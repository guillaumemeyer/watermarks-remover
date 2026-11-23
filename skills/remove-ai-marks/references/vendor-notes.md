# Vendor notes (public / class-level)

This skill targets **mark classes**, not reverse-engineered private detectors. Details below are from public docs and the research literature. Algorithms may change.

## Industry two-layer model (context)

Product and regulatory guidance often frames AI disclosure as:

1. **C2PA Content Credentials** — signed, hard-bound metadata (easy to strip; what this skill removes).
2. **Imperceptible watermark** (SynthID-class) — survives strip/re-upload; includes **soft binding** that can re-attach a remote C2PA manifest.

See: [Institute of AI PM — C2PA and SynthID guide](https://www.institutepm.com/knowledge-hub/ai-content-provenance-watermarking) (SB 942 / EU AI Act Art. 50 framing). This project only implements the **hard-bound / Unicode / rewrite** side of that stack.

## Anthropic / Claude

- **Embedded text watermarks** at model level (imperceptible; survive copy-paste). Public description matches **statistical token-sampling** class, not only Unicode.
- **C2PA Content Credentials** on supported files (e.g. PNG, JPEG, SVG).
- Models launched on/after **2026-08-02**: marking at launch; older models in transition; **worldwide**.
- Detection APIs for third parties: described as forthcoming.
- Caveats: mark ⇒ may have been processed by Claude; no mark ≠ human-only; proofreading can stamp human text.

**Skill mapping:** Layer A (Unicode hygiene) + Layer B (rewrite) + container/image C2PA strip.

Source: [How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content).

## Google Gemini / SynthID-Text

- Nature 2024 paper: *Scalable watermarking for identifying large language model outputs* (SynthID-Text).
- **Generative watermarking**: modifies next-token sampling (Tournament sampling); detection uses a scoring function + key; no need for the LLM at detect time.
- Paper also taxonomizes:
  - **Edit-based** (Unicode / synonym rules) → our Layer A (+ Layer B for synonyms)
  - **Data-driven / backdoor** (trigger phrases) → **out of scope**
  - **Generative** (sampling) → Layer B best-effort
- Productionized in Gemini-scale systems; open research code exists, but **production keys are not public** — this skill does **not** ship a SynthID detector.
- **Confirmed watermarked (Aug 2026), no public text detection:** Google staff confirmed on the AI Developers Forum that "text generated via the Gemini API IS actually SynthID-watermarked", correcting their own earlier Aug 5 statement that it was not; the confirmation extends to Google AI Studio and Antigravity ([post 4, the correction](https://discuss.ai.google.dev/t/does-gemini-api-text-output-carry-synthid-watermarking-gemini-2-5-flash-lite-gemini-3-1-flash-lite-eu-ai-act-art-50-2/177241/4); [post 2, retracted](https://discuss.ai.google.dev/t/does-gemini-api-text-output-carry-synthid-watermarking-gemini-2-5-flash-lite-gemini-3-1-flash-lite-eu-ai-act-art-50-2/177241/2)). No public text-detection route exists: the project reported in August that its `DETECT_TEXT_WATERMARK` request was rejected, and Google staff confirm there is no public or self-service tool to verify SynthID watermarks on text from hosted Gemini API endpoints — the public [SynthID Detector](https://synthid.com/) portal accepts images, video, and audio only ([post 9, Google staff, Nov 16](https://discuss.ai.google.dev/t/does-gemini-api-text-output-carry-synthid-watermarking-gemini-2-5-flash-lite-gemini-3-1-flash-lite-eu-ai-act-art-50-2/177241/9)) — so the gemini-synthid-text detector stays **removed**; a vendor detector can be re-added (e.g. via Vertex AI) if Google offers a public text-detection endpoint.
- Optional external verification harness: [`THU-BPM/MarkLLM`](https://github.com/THU-BPM/MarkLLM) (Apache-2.0) reimplements SynthID-Text among other schemes with configurable keys; wired as `detect_text_watermark.py` / `rewrite_text.py --markllm-scheme`. Same-config-only — it verifies a mark you generated under a known config, not Google's production keying.
- Current frontier production watermarks are **token-by-token** (streaming constraint); paragraph-level robust methods (SemStamp / PostMark) are not deployed yet, which keeps paraphrase-class attacks effective today.
- Optional external reference: [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID) provides a reverse-engineered pixel-domain scorer. It is **not bundled** here, is best-effort, and is under a non-commercial Research License; it is not the official Google detector.
- Optional pixel-domain removal: [`mertizci/noai-watermark`](https://github.com/mertizci/noai-watermark)'s CtrlRegen profile is wired through `clean_image.py --remove-pixel ctrlregen` / `clean_ctrlregen.py`. It is **not bundled** (no LICENSE file → all-rights-reserved), and no local detector certifies the result; the official Google check is the final authority. For Tree-Ring-class marks, the optional MarkDiffusion harness (`markdiffusion_harness.py`, Apache-2.0) adds a same-scheme detector and a blind-regeneration removal engine (`--remove-pixel diffusion`) — see `references/markdiffusion.md`.

**Skill mapping:** same Layer B rewrite attacks (paraphrase / back-translate / structural) used in the literature against sampling watermarks.

## OpenAI / ChatGPT

- Public provenance often surfaces as **labels**, **C2PA / Content Credentials** on some media exports, and product UI disclosure — not a fully public text-sampling watermark spec comparable to SynthID-Text.
- Treat **file metadata / C2PA** as in-scope when present; treat any **unpublished** text watermark as the same statistical class → Layer B only, best-effort.
- Do not invent algorithm claims.

**Skill mapping:** container/image metadata strip + Layer A/B on text.

## Open-weight / open-LLM (Kirchenbauer-style)

- Classic green-list / red-list sampling bias (Kirchenbauer et al.) and variants.
- Detectable with the **key** and tokenizer; removal still relies on heavy paraphrase or regeneration.
- Optional external harness: `MarkLLM` (`detect_text_watermark.py --scheme kgw`) reproduces KGW detection under a config you control, for controlled before/after experiments.

**Skill mapping:** Layer B multi-pass; prefer rewrite with a **different** model family when possible.

## Cross-vendor hygiene rule

| Suspected origin | Prefer rewrite backend |
| --- | --- |
| Claude | Non-Claude (local Ollama, other API) |
| Gemini | Non-Gemini |
| OpenAI | Non-OpenAI |
| Unknown | Local open-weight if available |

Prefer local open-weight backends and avoid any known-watermarked vendor, not
just the suspected origin. Then re-run Layer A on the rewritten text.
