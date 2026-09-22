# Frontmatter scanning — why keys and values use different patterns

Markdown frontmatter is scanned for provenance marks by `inspect_markdown()` and stripped by
`clean_markdown()` in `service/scripts/container_meta.py`. The two functions are contracted to
agree: clean must never remove something inspect called fine.

Both examine every top-level key **and** its value, but **they use different patterns on purpose**,
and getting that wrong caused real data loss.

## The asymmetry

| Position | Pattern | Rationale |
|---|---|---|
| **Key** | `AI_FRONTMATTER_KEYS` + `AI_META_NAME_RE` | A key is a *claim about the document*. A key literally called `claude:`, `generator:` or `made_with:` is provenance whatever its value. Bare vendor names belong here. |
| **Value** | `named_value_is_ai()` (`GENERATOR_NAME_KEYS` + `AI_FREE_TEXT_MARKER_RE`) | A value is *content*. Naming a vendor in it says nothing about how the file was made unless under a generator naming key. Only provenance-shaped phrasing counts in free text: a watermark scheme (`synthid`, `c2pa`), or an explicit `generated with <vendor>`. |

## The bug this prevents

A Claude Code subagent definition carries a tool grant:

```yaml
---
name: inbox-router
description: "Triages the inbox and routes each mail to its project."
tools: Read, Write, Edit, mcp__claude_ai_acme__mail__get-message
model: opus
---
```

Before 2026-09-14 the value check used `AI_META_NAME_RE`, so the substring **"claude"** inside
`mcp__claude_ai_…` made `tools` a value hit. Because `clean_markdown()` drops the **whole key** on a
value hit, cleaning such a file **deleted the agent's tool grant outright** — turning a cosmetic
scan into silent, hard-to-diagnose breakage across every `.claude/agents/*.md` in a repository.
`model:` had the same problem through `AI_FRONTMATTER_KEYS`.

Both were reported by the PostToolUse hook as a genuine finding, with a suggested remedy that would
have caused the damage.

## The agent-shape exemption

`model` and `tools` are legitimate provenance key names in an ordinary document (`model: gpt-4` in a
generated report *is* provenance). They are configuration only in a Claude Code agent or skill
definition, which is recognised by its **shape** rather than its path — `inspect_container()` never
receives one:

```
name  AND  description  AND  (tools OR allowed-tools)
```

All three must be present. Ordinary prose frontmatter does not carry the set, so this is not a way
for a real watermark to exempt itself — and **values are still scanned even inside an agent
definition**, so `description: Generated with Claude Code` is still caught and still stripped.

## Closing the key-side gap

Narrowing the value pattern in free text requires keys like `generated-with:` and `made_with:` to be
recognized as generator naming keys (`GENERATOR_NAME_KEYS`), where vendor names in the value are
evaluated as provenance marks, while free text uses `AI_FREE_TEXT_MARKER_RE`.

## Tests

`tests/test_frontmatter_false_positives.py` covers both directions in one file, deliberately: the
false positives that must stop firing, and every real watermark that must keep firing — including
one hiding inside an agent definition, which is exactly where an exemption could be abused.
