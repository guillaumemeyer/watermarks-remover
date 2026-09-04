// swift-tools-version: 6.0
import PackageDescription

// Front-end only: `./mac/Scripts/build_app.sh` wraps this binary with Info.plist
// and copies service/scripts into the bundle. Cleaning logic stays in Python.
let package = Package(
    name: "WatermarksMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "WatermarksMac", targets: ["WatermarksMac"])
    ],
    targets: [
        .executableTarget(
            name: "WatermarksMac",
            path: "Sources/WatermarksMac"
        )
    ]
)
