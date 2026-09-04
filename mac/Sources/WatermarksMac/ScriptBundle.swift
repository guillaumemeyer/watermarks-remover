import Foundation

enum ScriptBundle {
    /// Keep in lockstep with `mac/Scripts/build_app.sh` and `tests/test_mac_app.py`.
    /// Union of the import closures of inspect_file.py, clean_file.py, rewrite_text.py.
    static let requiredScripts = [
        "av_meta.py",
        "clean_file.py",
        "common.py",
        "container_meta.py",
        "detect_gumbel.py",
        "format_dispatch.py",
        "humanize_pass.py",
        "image_meta.py",
        "inspect_file.py",
        "rewrite_text.py",
        "text_detectors.py",
        "text_unicode.py",
    ]

    static let inspectEntry = "inspect_file.py"
    static let cleanEntry = "clean_file.py"
    static let rewriteEntry = "rewrite_text.py"

    static func directory() throws -> URL {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["WATERMARKS_SCRIPTS_DIR"], !env.isEmpty {
            let url = URL(fileURLWithPath: env)
            if fm.fileExists(atPath: url.appendingPathComponent(inspectEntry).path) {
                return url
            }
        }

        var candidates: [URL] = []
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent("PythonScripts"))
            candidates.append(resources.appendingPathComponent("scripts"))
        }
        candidates.append(contentsOf: repoScriptCandidates())
        if let exe = Bundle.main.executableURL {
            let macos = exe.deletingLastPathComponent()
            candidates.append(macos.deletingLastPathComponent().appendingPathComponent("Resources/PythonScripts"))
        }

        for candidate in candidates {
            if fm.fileExists(atPath: candidate.appendingPathComponent(inspectEntry).path) {
                return candidate
            }
        }
        throw RunnerError.missingScripts(candidates.map(\.path).joined(separator: ", "))
    }

    /// Checkout layout: mac/Sources/WatermarksMac/*.swift → repo/service/scripts
    private static func repoScriptCandidates() -> [URL] {
        var urls: [URL] = []
        let fromFile = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("service/scripts")
        urls.append(fromFile)

        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<8 {
            urls.append(dir.appendingPathComponent("service/scripts"))
            dir.deleteLastPathComponent()
        }
        return urls
    }
}
