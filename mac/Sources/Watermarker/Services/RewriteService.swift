import Foundation

/// Drives the Layer B rewrite: `service/scripts/rewrite_text.py` against an
/// OpenRouter model.
///
/// The script is treated as the source of truth. This type only assembles its
/// arguments and environment, then reads back the `--json-stats` report it
/// writes to stderr.
struct RewriteService: Sendable {
    struct Request: Sendable {
        var text: String
        var model: String
        var baseURL: String
        var apiKey: String
        var strategy: String
        var temperature: Double
        var timeout: Double
        /// Passed through to `--reasoning-effort`. Always sent explicitly: the
        /// script's own default is `none`, and `WATERMARKS_REWRITE_REASONING_EFFORT`
        /// in the user's shell would otherwise decide it.
        var reasoningEffort: String
    }

    struct Outcome: Sendable {
        var text: String
        /// The parsed `--json-stats` object, when the script emitted one.
        var stats: Stats?
        /// The script's stderr, minus the stats JSON, for the log pane.
        var log: String
    }

    struct Stats: Decodable, Sendable {
        struct Step: Decodable, Sendable {
            var tactic: String
            var intensity: Double
            var inChars: Int
            var outChars: Int

            enum CodingKeys: String, CodingKey {
                case tactic
                case intensity
                case inChars = "in_chars"
                case outChars = "out_chars"
            }
        }

        var backend: String
        var tactic: String
        var mode: String?
        var strategy: [String]?
        var steps: [Step]?
        var inputChars: Int
        var outputChars: Int?

        enum CodingKeys: String, CodingKey {
            case backend, tactic, mode, strategy, steps
            case inputChars = "input_chars"
            case outputChars = "output_chars"
        }
    }

    enum RewriteError: LocalizedError {
        case missingAPIKey
        case missingModel
        case scriptsUnavailable
        case scriptFailed(status: Int32, message: String)
        case emptyResult

        var errorDescription: String? {
            switch self {
            case .missingAPIKey:
                return "Add an OpenRouter API key in Settings before running the tool."
            case .missingModel:
                return "Choose a model in Settings before running the tool."
            case .scriptsUnavailable:
                return "The Layer B scripts are missing from the app. Reinstall "
                    + "Watermarker, or use Settings ▸ Update Tools to fetch them."
            case .scriptFailed(let status, let message):
                let detail = message.trimmingCharacters(in: .whitespacesAndNewlines)
                return detail.isEmpty
                    ? "The rewrite failed (exit code \(status))."
                    : "The rewrite failed: \(detail)"
            case .emptyResult:
                return "The model returned nothing. Try a different model, or a "
                    + "shorter passage."
            }
        }
    }

    /// Run the rewrite. Blocking — callers dispatch it off the main actor.
    func run(_ request: Request) throws -> Outcome {
        let key = request.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw RewriteError.missingAPIKey }
        let model = request.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { throw RewriteError.missingModel }
        guard let scripts = ScriptBundle.active() else { throw RewriteError.scriptsUnavailable }

        let arguments = [
            "-",                       // read the text from stdin
            "--backend", "openai-compatible",
            "--model", model,
            "--base-url", request.baseURL,
            "--strategy", request.strategy,
            "--temperature", String(format: "%.2f", request.temperature),
            "--timeout", String(format: "%.0f", request.timeout),
            "--reasoning-effort", request.reasoningEffort,
            "--allow-remote",          // OpenRouter is, by definition, not loopback
            "--json-stats",
            "--output", "-",
        ]

        let environment = [
            // The script reads the key from the environment only; passing it on
            // argv would leak it into `ps` and shell history.
            "WATERMARKS_REWRITE_API_KEY": key,
            "WATERMARKS_REWRITE_ALLOW_REMOTE": "1",
        ]

        let result = try PythonRunner.run(
            script: scripts.directory.appendingPathComponent(ScriptBundle.entryPoint),
            arguments: arguments,
            input: request.text,
            environment: environment,
            workingDirectory: scripts.directory,
            // Each strategy step is its own model call with its own --timeout,
            // so the outer limit has to cover all of them plus start-up.
            timeout: request.timeout * Double(max(1, Self.stepCount(request.strategy))) + 30
        )

        let (stats, log) = Self.splitStats(from: result.standardError)
        guard result.succeeded else {
            throw RewriteError.scriptFailed(status: result.exitCode,
                                            message: Self.redact(Self.failureLines(log), key: key))
        }
        let text = result.standardOutput
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RewriteError.emptyResult
        }
        return Outcome(text: text, stats: stats, log: Self.redact(log, key: key))
    }

    /// How many `tactic@intensity` steps a strategy string describes.
    static func stepCount(_ strategy: String) -> Int {
        strategy.split(separator: ",")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .count
    }

    /// `--json-stats` prints `json.dumps(..., indent=2)` to stderr after any log
    /// lines. Indentation is what identifies the top-level object: its opening
    /// brace is the only one at column zero, so nested step objects cannot be
    /// mistaken for the start of the report.
    static func splitStats(from stderr: String) -> (Stats?, String) {
        let lines = stderr.components(separatedBy: "\n")
        guard let start = lines.lastIndex(where: { $0 == "{" }) else {
            return (nil, stderr)
        }
        let candidate = lines[start...].joined(separator: "\n")
        guard let data = candidate.data(using: .utf8),
              let stats = try? JSONDecoder().decode(Stats.self, from: data)
        else { return (nil, stderr) }
        let log = lines[..<start].joined(separator: "\n")
        return (stats, log)
    }

    /// The failure, without the advisory lines that share the same stream.
    ///
    /// The script warns on stderr that a non-loopback endpoint means the text
    /// leaves the machine -- true and worth saying, but it is not why a run
    /// failed, and folding it into the error banner buried the real reason
    /// underneath it. Warnings stay in the log pane; only the rest becomes the
    /// message. If nothing is left, the warnings are better than nothing.
    static func failureLines(_ log: String) -> String {
        let lines = log.components(separatedBy: "\n")
        let kept = lines.filter {
            let line = $0.trimmingCharacters(in: .whitespaces)
            return !line.isEmpty && !line.lowercased().hasPrefix("warning:")
        }
        return (kept.isEmpty ? lines : kept)
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Belt and braces: the script never echoes the key, but the log is shown
    /// in the window and copied into bug reports.
    static func redact(_ text: String, key: String) -> String {
        guard key.count > 4 else { return text }
        return text.replacingOccurrences(of: key, with: "sk-or-…redacted")
    }
}
