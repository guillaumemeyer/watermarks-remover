#!/usr/bin/env python3
"""Build a human-written baseline corpus from Project Gutenberg.

Downloads a fixed list of public-domain books, strips the Gutenberg
header/footer, and cuts deterministic excerpts at several target lengths.
The excerpts are the negative class for bench_human_fpr.py: text known to be
written by people, long before LLMs, so every "AI" verdict on it is a false
positive.

Output (default benchmarks/corpus-human/):
  gb<id>-<words>w-<k>.txt   one excerpt per file, paragraphs joined by blank lines
  SOURCES.tsv               file -> book id, title, author, year, word count, URL

Limitation: this is 19th/early-20th century prose. It measures how detectors
treat human text in general, not modern student or forum writing; drop your own
pre-2022 human texts into the same directory to extend it.

Exit codes:
  0  corpus written
  2  usage error or every download failed
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import eprint  # noqa: E402

try:
    DEFAULT_OUT = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus-human"
except IndexError:
    DEFAULT_OUT = Path(__file__).resolve().parent / "benchmarks" / "corpus-human"

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# (gutenberg id, title, author, first published). Varied genres on purpose:
# travel, memoir, essay, comic novel, detective fiction, short story, translation.
BOOKS: tuple[tuple[int, str, str, int], ...] = (
    (245, "Life on the Mississippi", "Mark Twain", 1883),
    (308, "Three Men in a Boat", "Jerome K. Jerome", 1889),
    (205, "Walden", "Henry David Thoreau", 1854),
    (944, "The Voyage of the Beagle", "Charles Darwin", 1839),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", 1892),
    (20203, "Autobiography of Benjamin Franklin", "Benjamin Franklin", 1791),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman", 1892),
    (5200, "Metamorphosis", "Franz Kafka (tr. David Wyllie)", 1915),
)

DEFAULT_LENGTHS = (60, 150, 400)
MIN_PARAGRAPH_WORDS = 25

_START_RE = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_END_RE = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_HEADING_RE = re.compile(r"^(chapter|book|part|section|letter)\b|^[IVXLC]+\.?$", re.I)


def strip_gutenberg(raw: str) -> str:
    """Return the book body between the Gutenberg START and END markers."""
    text = raw.replace("\r\n", "\n").lstrip("\ufeff")
    start = _START_RE.search(text)
    end = _END_RE.search(text)
    lo = start.end() if start else 0
    hi = end.start() if end and (not start or end.start() > lo) else len(text)
    return text[lo:hi]


def paragraphs(body: str) -> list[str]:
    """Split a hard-wrapped body into prose paragraphs, dropping headings and fragments."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", body):
        para = " ".join(line.strip() for line in block.splitlines() if line.strip())
        para = para.replace("_", "")  # Gutenberg marks italics as _word_
        if not para or _HEADING_RE.match(para) or para.isupper():
            continue
        if len(para.split()) < MIN_PARAGRAPH_WORDS:
            continue
        out.append(para)
    return out


def excerpts(paras: list[str], target_words: int, count: int, rng: random.Random) -> list[str]:
    """Pick ``count`` non-overlapping runs of consecutive paragraphs of >= target_words.

    A single paragraph longer than the target is cut at the last sentence end
    within it, so short excerpts stay short.
    """
    taken: set[int] = set()
    found: list[str] = []
    starts = list(range(len(paras)))
    rng.shuffle(starts)
    for start in starts:
        if len(found) >= count:
            break
        if start in taken:
            continue
        chunk: list[str] = []
        words = 0
        i = start
        while i < len(paras) and words < target_words and i not in taken:
            chunk.append(paras[i])
            words += len(paras[i].split())
            i += 1
        if words < target_words:
            continue
        text = "\n\n".join(chunk)
        if words > target_words * 1.5:
            text = _trim_to_sentence(text, target_words)
            if text is None:
                continue
        taken.update(range(start, i))
        found.append(text)
    return found


def _trim_to_sentence(text: str, target_words: int) -> str | None:
    """Cut text at the first sentence end at or after target_words; None if none fits."""
    words = text.split(" ")
    for n in range(target_words, min(len(words), int(target_words * 1.5)) + 1):
        if re.search(r"[.!?][\"')\]]?$", words[n - 1]):
            return " ".join(words[:n])
    return None


def download(book_id: int, cache_dir: Path | None, timeout: float) -> str:
    """Fetch one book's plain text, reusing cache_dir when given."""
    if cache_dir is not None:
        cached = cache_dir / f"pg{book_id}.txt"
        if cached.is_file():
            return cached.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(  # noqa: S310 - fixed https URL
        GUTENBERG_URL.format(id=book_id),
        headers={"User-Agent": "watermarks-remover human-corpus builder"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        raw = resp.read().decode("utf-8", errors="replace")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"pg{book_id}.txt").write_text(raw, encoding="utf-8")
    return raw


def build(
    out_dir: Path,
    lengths: list[int],
    per_length: int,
    seed: int,
    cache_dir: Path | None,
    timeout: float,
) -> int:
    """Download every book and write excerpts + SOURCES.tsv. Returns files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = ["file\tgutenberg_id\ttitle\tauthor\tyear\twords\turl"]
    written = 0
    for book_id, title, author, year in BOOKS:
        try:
            raw = download(book_id, cache_dir, timeout)
        except Exception as e:  # one bad book should not sink the corpus
            eprint(f"skip pg{book_id} ({title}): {e}")
            continue
        if title.split()[0].lower() not in raw[:3000].lower():
            eprint(f"warning: pg{book_id} header does not mention '{title}'")
        paras = paragraphs(strip_gutenberg(raw))
        rng = random.Random(f"{seed}:{book_id}")  # noqa: S311 - sampling, not crypto
        for target in lengths:
            for k, text in enumerate(excerpts(paras, target, per_length, rng), 1):
                name = f"gb{book_id}-{target}w-{k}.txt"
                (out_dir / name).write_text(text + "\n", encoding="utf-8")
                rows.append(
                    f"{name}\t{book_id}\t{title}\t{author}\t{year}\t{len(text.split())}\t"
                    f"{GUTENBERG_URL.format(id=book_id)}"
                )
                written += 1
    (out_dir / "SOURCES.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--lengths",
        default=",".join(map(str, DEFAULT_LENGTHS)),
        help="comma-separated target excerpt lengths in words (default: %(default)s)",
    )
    p.add_argument("--per-length", type=int, default=3, help="excerpts per book per length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cache-dir", type=Path, default=None, help="keep raw downloads here to avoid refetching"
    )
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    try:
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    except ValueError:
        eprint(f"error: bad --lengths {args.lengths!r}")
        return 2
    if not lengths or min(lengths) < 1 or args.per_length < 1:
        eprint("error: --lengths and --per-length must be positive")
        return 2
    n = build(args.out_dir, lengths, args.per_length, args.seed, args.cache_dir, args.timeout)
    if n == 0:
        eprint("error: no excerpts written (all downloads failed?)")
        return 2
    print(f"wrote {n} excerpts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
