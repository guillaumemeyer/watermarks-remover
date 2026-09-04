# macOS app

SwiftUI front-end for this repository's **existing** `service/scripts`. It does
not reimplement inspect/clean. Drop files or paste text; Layer A always runs;
Layer B rewrite is off until Ollama or an OpenAI-compatible endpoint is set.

This is a GUI for [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover),
not a separate product. Signing and GitHub Releases stay with the maintainer.

## Build

Needs **Xcode** (SwiftUI). From the repo root:

```bash
make mac-app
open mac/build/WatermarksRemover.app
```

Or `./mac/Scripts/build_app.sh`. The bundle copies `inspect_file.py`,
`clean_file.py`, `rewrite_text.py`, and their stdlib import closure from
`service/scripts` at this commit.

Python 3.10+ on the Mac (`/opt/homebrew/bin/python3` or `/usr/bin/python3`).
Optional: `exiftool`, `qpdf`, `c2patool` for a deeper PDF strip.

Optional signed local build (your Developer ID, for testing only):

```bash
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
  ./mac/Scripts/build_app.sh
```

## Use

1. Drop files (or Open…) → Inspect → Clean. Writes `name.cleaned.ext` beside the original. Never overwrites.
2. Text tab: paste → Inspect / Clean.
3. Settings: rewrite Off / Ollama / OpenAI-compatible. Keys go to Keychain, never argv.

A clean is not a claim that vendor detectors will fail. Pixel regen and
SynthID on audio/video are out of scope here.
