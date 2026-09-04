#!/bin/bash
# Assemble WatermarksRemover.app from the SwiftPM binary + in-tree service/scripts.
# Ad-hoc signed by default. Optional:
#   CODESIGN_IDENTITY="Developer ID Application: Name (TEAMID)" ./mac/Scripts/build_app.sh
set -euo pipefail

MAC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$MAC_DIR/.." && pwd)"
CONFIGURATION="${CONFIGURATION:-release}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

if [[ -d /Applications/Xcode-beta.app ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode-beta.app/Contents/Developer}"
elif [[ -d /Applications/Xcode.app ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
fi

command -v swift >/dev/null 2>&1 || {
  echo "build_app: swift not found. Install Xcode (SwiftUI is not buildable with Command Line Tools alone)." >&2
  exit 1
}

echo "==> Building WatermarksMac ($CONFIGURATION)"
swift build --package-path "$MAC_DIR" -c "$CONFIGURATION"
BINARY="$(swift build --package-path "$MAC_DIR" -c "$CONFIGURATION" --show-bin-path)/WatermarksMac"
[ -x "$BINARY" ] || { echo "build_app: no binary at $BINARY" >&2; exit 1; }

BUILD_DIR="$MAC_DIR/build"
APP="$BUILD_DIR/WatermarksRemover.app"
CONTENTS="$APP/Contents"
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources/PythonScripts"
cp "$BINARY" "$CONTENTS/MacOS/WatermarksMac"
cp "$MAC_DIR/Resources/Info.plist" "$CONTENTS/Info.plist"
printf 'APPL????' > "$CONTENTS/PkgInfo"

# Copied from service/scripts at this commit — not a vendored fork.
# Keep this list in lockstep with ScriptBundle.requiredScripts.
for script in \
  av_meta.py clean_file.py common.py container_meta.py detect_gumbel.py \
  format_dispatch.py humanize_pass.py image_meta.py inspect_file.py \
  rewrite_text.py text_detectors.py text_unicode.py
do
  src="$REPO_DIR/service/scripts/$script"
  [ -f "$src" ] || { echo "build_app: missing $src" >&2; exit 1; }
  cp "$src" "$CONTENTS/Resources/PythonScripts/$script"
done

echo "==> Signing ($CODESIGN_IDENTITY)"
codesign --force --sign "$CODESIGN_IDENTITY" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"
echo "==> Built $APP"
