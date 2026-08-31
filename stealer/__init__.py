"""Black-box watermark-stealing module for SynthID-class text watermarks.

This package implements the one-time "steal" step of the attack described in
the shared conversation and in the ETH SRI line of work on SynthID-Text
detection: query a watermarked model with a benign prompt corpus, count how
often each next token follows each short context, compare those counts against
a non-watermarked baseline, and derive a reusable scorer ``s*(token | context)``
that ranks which tokens the watermark boosted ("green").  The scorer can then
drive scrubbing (demote green tokens) or, sign-flipped, spoofing.

Everything here is stdlib-only so it fits the rest of the repository.  The
model-query step is pluggable and defaults to a dry-run so the pipeline runs
offline; a real run points ``query`` at an OpenAI-compatible endpoint.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
