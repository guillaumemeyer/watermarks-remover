import Foundation

/// Where the Layer B Python scripts live and which copy is in use.
///
/// A copy ships inside the app bundle so a fresh install works offline. Updates
/// pulled from GitHub land in Application Support and take precedence, which is
/// what lets the tool track upstream without a new build of the app.
enum ScriptBundle {
    /// The import closure of `rewrite_text.py`, computed from its imports.
    /// Every one of these is standard-library-only Python.
    static let requiredScripts = [
        "rewrite_text.py",
        "common.py",
        "humanize_pass.py",
        "text_detectors.py",
        "text_unicode.py",
        "detect_gumbel.py",
    ]

    static let entryPoint = "rewrite_text.py"

    enum Source: Equatable, Sendable {
        case bundled
        case updated(version: String)

        var label: String {
            switch self {
            case .bundled: return "Bundled with the app"
            case .updated(let version): return "Updated — \(version)"
            }
        }

        var isUpdated: Bool {
            if case .updated = self { return true }
            return false
        }
    }

    /// `~/Library/Application Support/Watermarker`
    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("Watermarker", isDirectory: true)
    }

    static var updatedDirectory: URL {
        supportDirectory.appendingPathComponent("PythonScripts", isDirectory: true)
    }

    static var manifestURL: URL {
        supportDirectory.appendingPathComponent("scripts-manifest.json")
    }

    static var bundledDirectory: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("PythonScripts", isDirectory: true)
    }

    /// A directory counts only when every required script is present; a partial
    /// update must never shadow a complete bundled copy.
    static func isComplete(_ directory: URL) -> Bool {
        let fm = FileManager.default
        return requiredScripts.allSatisfy {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    struct Manifest: Codable, Sendable {
        var repository: String
        var ref: String
        var commit: String
        var shortCommit: String
        var installedAt: Date

        var displayVersion: String {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            return "\(repository)@\(shortCommit), \(formatter.string(from: installedAt))"
        }
    }

    static func loadManifest() -> Manifest? {
        guard let data = try? Data(contentsOf: manifestURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(Manifest.self, from: data)
    }

    static func saveManifest(_ manifest: Manifest) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: supportDirectory,
                                                withIntermediateDirectories: true)
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
    }

    /// The directory the app should actually run, and where it came from.
    static func active() -> (directory: URL, source: Source)? {
        if isComplete(updatedDirectory), let manifest = loadManifest() {
            return (updatedDirectory, .updated(version: manifest.displayVersion))
        }
        if let bundled = bundledDirectory, isComplete(bundled) {
            return (bundled, .bundled)
        }
        // A `swift run` build has no bundle resources; fall back to the scripts
        // in the checkout so the app is usable straight from the repository.
        let checkout = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("service/scripts", isDirectory: true)
        if isComplete(checkout) { return (checkout, .bundled) }
        return nil
    }
}
