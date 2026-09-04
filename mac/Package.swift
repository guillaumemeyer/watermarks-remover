// swift-tools-version: 5.9
import PackageDescription

// The executable this package builds is not the shipping product on its own:
// `make -C mac app` drops the binary into a Watermarker.app bundle alongside
// Info.plist, the .icns, and the Layer B Python scripts. Building with SwiftPM
// rather than an .xcodeproj keeps the whole app buildable from a checkout with
// nothing but the Xcode command line tools installed.
let package = Package(
    name: "Watermarker",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "Watermarker",
            path: "Sources/Watermarker"
        )
    ]
)
