# Watermarker

A Mac app for the Layer B text tools in this repository. Paste text or import a
Markdown or Word file, pick an OpenRouter model, and run the statistical
watermark rewrite — the same `service/scripts/rewrite_text.py` the CLI and the
`/clean` service run, invoked verbatim rather than reimplemented in Swift.

<img src="Icon/appicon-1024.png" width="128" alt="The Watermarker icon: a strainer over a stack of documents, catching watermark glyphs.">

## Building

Xcode or the Xcode command line tools are the only requirement.

```sh
make -C mac app        # build mac/build/Watermarker.app
make -C mac run        # build it and launch it
make -C mac install    # copy it to /Applications
```

That build is ad-hoc signed, which is enough to run locally. Ad-hoc signatures
cannot carry iCloud entitlements, so settings stay on the one Mac; the settings
sheet says so plainly rather than pretending to sync. To get iCloud sync, sign
with an identity whose team owns the container:

```sh
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
TEAM_ID=TEAMID make -C mac app
```

`mac/Resources/Watermarker.entitlements` declares the `iCloud.<bundle id>`
CloudKit container and the matching keychain access group. Create the container
in the Apple Developer portal under the same identifier before signing.

## Using it

1. **Settings ▸ Model** — paste an OpenRouter API key and name a model
   (`vendor/model`). Prefer a model from a *different* vendor than the one that
   wrote the text: rewriting Claude output with Claude can re-stamp the marks
   you are removing.
2. Type into the left pane, or **Open File…** (or drop a file on the window) to
   import `.md`, `.txt`, or `.docx`. A `.docx` is unzipped and its body
   flattened to plain text — the rewrite works on wording, so formatting is
   dropped rather than round-tripped.
3. **Remove Watermarks** (⌘↵) runs the strategy. The cleaned text lands on the
   right, ready to copy, save, or feed back in for another pass.

### Strategies

Layer B removes statistical (token-sampling) watermarks by rewriting, so every
strategy costs some of your wording. Settings ▸ Strategy offers:

| Preset | Spec | Notes |
| --- | --- | --- |
| Paraphrase | `paraphrase@0.8` | The default. One model call, nothing but the standard library needed. |
| Humanize | `humanize@0.8` | Rewrites for human cadence, then runs the deterministic humanizer pass. |
| Paraphrase, then humanize | `paraphrase@0.8,humanize@0.6` | Two model calls. Diverges furthest, costs the most. |
| Benchmark default | `paraphrase@0.8,mlm@0.2` | The repository's benchmark-tuned default. The `mlm` step needs `transformers` and `torch` installed for the Python the app runs. |

The custom field takes any `tactic@intensity` list `rewrite_text.py --strategy`
accepts, with intensity in `(0,1]`.

Rewriting is **best-effort**, the same caveat the CLI carries. The app reports
what the script reports; it cannot certify that a vendor detector will fail. See
the root [README's disclaimer](../README.md#disclaimer-what-removing-a-text-watermark-costs).

## Keeping the tools current

Settings ▸ Tools checks GitHub for newer Layer B scripts and installs them into
`~/Library/Application Support/Watermarker/PythonScripts`, which takes
precedence over the copy inside the app bundle. That is what lets the tool track
upstream without a new build.

Downloads are staged, compiled with `python3 -m py_compile`, and only then swap
into place, with the previous copy kept until the new one lands — a truncated
download or an upstream syntax error can never take out the Run button.
**Revert to Bundled** throws the downloaded copy away.

The update source defaults to the upstream project,
`guillaumemeyer/watermarks-remover` on `main`. Point it at your own fork to
track that instead.

## Where things are stored

| What | Where | Syncs via |
| --- | --- | --- |
| OpenRouter API key | Login keychain, `kSecAttrSynchronizable` | iCloud Keychain |
| Model, strategy, endpoint, update source | `NSUbiquitousKeyValueStore`, mirrored to `UserDefaults` | CloudKit key-value store |
| Downloaded scripts | `~/Library/Application Support/Watermarker` | not synced |

The key never reaches the key-value store, and it reaches the rewrite script
through the environment rather than on the command line — a key on `argv` shows
up in `ps` and shell history. Log output shown in the window is redacted for the
key before it is displayed.

Without the iCloud entitlements every write still lands in `UserDefaults` and a
local keychain item, so the app works unsigned; it just does not sync.

## The icon

`Icon/render_icon.py` draws the artwork and writes the committed
`Icon/appicon-1024.png`, so the icon is source rather than an opaque binary
(`make -C mac icon` regenerates it; it is the one thing here that needs Pillow).

To use your own artwork instead, drop a square PNG at
`Icon/appicon-source.png` — the build prefers it, and
`Scripts/make_icon.swift` trims any uniform border (white, black, or
transparent) off it before generating the `.icns`. That crop matters: an icon
with baked-in whitespace renders visibly smaller than every other icon in the
Dock.

## Layout

```
mac/
  Package.swift                    SwiftPM executable; build_app.sh wraps it in a bundle
  Makefile                         app / run / install / icon / clean
  Icon/render_icon.py              the artwork, as code
  Resources/Info.plist             bundle metadata, with build-time placeholders
  Resources/Watermarker.entitlements  CloudKit container and keychain group
  Scripts/build_app.sh             assembles Watermarker.app
  Scripts/make_icon.swift          auto-crop + .icns, using only ImageIO
  Sources/Watermarker/
    WatermarkerApp.swift           @main App, menu commands
    Theme.swift                    the palette, taken from the icon
    Models/AppModel.swift          window state: input, output, status
    Models/SettingsStore.swift     CloudKit key-value store, with local fallback
    Models/KeychainStore.swift     the API key, in iCloud Keychain
    Services/RewriteService.swift  builds the rewrite_text.py invocation
    Services/PythonRunner.swift    finds python3 and runs it out of process
    Services/ScriptBundle.swift    which copy of the scripts is live
    Services/ScriptUpdater.swift   fetch, verify, and swap in newer scripts
    Services/DocumentImporter.swift  .md / .txt / .docx to plain text
    Services/ZipArchive.swift      the little of ZIP a .docx needs
    Views/ContentView.swift        the window
    Views/SettingsView.swift       the settings sheet
```

`tests/test_mac_app.py` keeps the three places that name the bundled Python
files in agreement with what `rewrite_text.py` actually imports, and checks that
every one of them is standard-library-only — the app runs the system `python3`,
which has no third-party packages.
