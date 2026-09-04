#!/bin/bash
#
# build_app.sh — assemble Watermarker.app from the SwiftPM build product.
#
#   ./mac/Scripts/build_app.sh                 # ad-hoc signed, local-only settings
#   CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
#   TEAM_ID=TEAMID ./mac/Scripts/build_app.sh  # signed, iCloud sync live
#
# Environment:
#   CONFIGURATION      debug | release          (default: release)
#   BUNDLE_ID          reverse-DNS identifier   (default: com.symbiola.Watermarker)
#   VERSION            CFBundleShortVersionString (default: 1.0.0)
#   BUILD_NUMBER       CFBundleVersion          (default: git commit count)
#   CODESIGN_IDENTITY  signing identity         (default: - , i.e. ad hoc)
#   TEAM_ID            Apple Developer team id  (required for iCloud entitlements)
#   ICON_SOURCE        square PNG for the icon  (default: mac/Icon/appicon-1024.png)
#
set -euo pipefail

MAC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$MAC_DIR/.." && pwd)"

CONFIGURATION="${CONFIGURATION:-release}"
BUNDLE_ID="${BUNDLE_ID:-com.symbiola.Watermarker}"
VERSION="${VERSION:-1.0.0}"
BUILD_NUMBER="${BUILD_NUMBER:-$(git -C "$REPO_DIR" rev-list --count HEAD 2>/dev/null || echo 1)}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
TEAM_ID="${TEAM_ID:-}"

BUILD_DIR="$MAC_DIR/build"
APP="$BUILD_DIR/Watermarker.app"
CONTENTS="$APP/Contents"

# The user's own artwork wins when it is present; otherwise the rendered icon
# committed alongside this script is used. Either way make_icon.swift trims any
# uniform border before generating the .icns.
if [ -n "${ICON_SOURCE:-}" ]; then
	ICON_PNG="$ICON_SOURCE"
elif [ -f "$MAC_DIR/Icon/appicon-source.png" ]; then
	ICON_PNG="$MAC_DIR/Icon/appicon-source.png"
else
	ICON_PNG="$MAC_DIR/Icon/appicon-1024.png"
fi

command -v swift >/dev/null 2>&1 || {
	echo "build_app: swift not found. Install Xcode or the command line tools." >&2
	exit 1
}

echo "==> Building Watermarker ($CONFIGURATION)"
swift build --package-path "$MAC_DIR" -c "$CONFIGURATION"
BINARY="$(swift build --package-path "$MAC_DIR" -c "$CONFIGURATION" --show-bin-path)/Watermarker"
[ -x "$BINARY" ] || { echo "build_app: no binary at $BINARY" >&2; exit 1; }

echo "==> Assembling the bundle"
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"
cp "$BINARY" "$CONTENTS/MacOS/Watermarker"
printf 'APPL????' > "$CONTENTS/PkgInfo"

sed -e "s|__BUNDLE_ID__|$BUNDLE_ID|g" \
    -e "s|__SHORT_VERSION__|$VERSION|g" \
    -e "s|__BUILD_VERSION__|$BUILD_NUMBER|g" \
    "$MAC_DIR/Resources/Info.plist" > "$CONTENTS/Info.plist"

echo "==> Rendering the app icon from $(basename "$ICON_PNG")"
swift "$MAC_DIR/Scripts/make_icon.swift" "$ICON_PNG" "$CONTENTS/Resources/AppIcon.icns"

# The Layer B scripts ship inside the bundle so a fresh install works offline.
# Settings ▸ Tools can replace them later from GitHub.
echo "==> Copying the Layer B scripts"
mkdir -p "$CONTENTS/Resources/PythonScripts"
for script in rewrite_text.py common.py humanize_pass.py text_detectors.py \
              text_unicode.py detect_gumbel.py; do
	src="$REPO_DIR/service/scripts/$script"
	[ -f "$src" ] || { echo "build_app: missing $src" >&2; exit 1; }
	cp "$src" "$CONTENTS/Resources/PythonScripts/$script"
done

echo "==> Signing ($CODESIGN_IDENTITY)"
if [ "$CODESIGN_IDENTITY" = "-" ]; then
	# Ad-hoc signing cannot carry iCloud entitlements, so the app falls back to
	# storing settings on this Mac only. That is expected for a local build.
	codesign --force --sign - --timestamp=none "$APP"
	echo "    ad-hoc signed: settings will be stored locally, not in iCloud."
else
	[ -n "$TEAM_ID" ] || {
		echo "build_app: set TEAM_ID alongside CODESIGN_IDENTITY so the iCloud" >&2
		echo "           entitlements name a real container." >&2
		exit 1
	}
	ENTITLEMENTS="$BUILD_DIR/Watermarker.entitlements"
	sed -e "s|__BUNDLE_ID__|$BUNDLE_ID|g" \
	    -e "s|\$(TeamIdentifierPrefix)|$TEAM_ID.|g" \
	    -e "s|\$(AppIdentifierPrefix)|$TEAM_ID.|g" \
	    "$MAC_DIR/Resources/Watermarker.entitlements" > "$ENTITLEMENTS"
	codesign --force --options runtime --timestamp \
	         --entitlements "$ENTITLEMENTS" \
	         --sign "$CODESIGN_IDENTITY" "$APP"
fi

codesign --verify --deep --strict "$APP"
echo "==> Built $APP"
