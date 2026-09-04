import Foundation

struct Hit: Sendable, Identifiable {
    var id: String { "\(codepoint)-\(kind)-\(count)" }
    var codepoint: String
    var label: String
    var count: Int
    var kind: String
    var confidence: String
}

struct InspectReport: Sendable {
    var kind: String
    var path: String?
    var note: String?
    var format: String?
    var length: Int?
    var suspiciousTotal: Int
    var hits: [Hit]
    var hasC2PA: Bool
    var hasAIMetadata: Bool
    var findings: [String]
    var notes: [String]
    var rawJSON: String

    var isActionable: Bool {
        suspiciousTotal > 0 || hasC2PA || hasAIMetadata || !findings.isEmpty
    }

    var summary: String {
        if kind == "unknown" {
            return note ?? "Unrecognized format"
        }
        var parts: [String] = [kind]
        if let format, !format.isEmpty { parts.append(format) }
        if suspiciousTotal > 0 { parts.append("\(suspiciousTotal) Unicode marks") }
        if hasC2PA { parts.append("C2PA") }
        if hasAIMetadata { parts.append("AI metadata") }
        if !isActionable { parts.append("clean") }
        return parts.joined(separator: " · ")
    }

    static func parse(json: String, fallbackKind: String = "unknown") -> InspectReport {
        var report = InspectReport(
            kind: fallbackKind,
            path: nil,
            note: nil,
            format: nil,
            length: nil,
            suspiciousTotal: 0,
            hits: [],
            hasC2PA: false,
            hasAIMetadata: false,
            findings: [],
            notes: [],
            rawJSON: json
        )
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            if !json.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                report.note = json
            }
            return report
        }
        report.kind = obj["kind"] as? String ?? fallbackKind
        report.path = obj["path"] as? String
        report.note = obj["note"] as? String
        report.format = obj["format"] as? String
        report.length = obj["length"] as? Int
        report.suspiciousTotal = obj["suspicious_total"] as? Int ?? 0
        report.hasC2PA = obj["has_c2pa"] as? Bool ?? false
        report.hasAIMetadata = obj["has_ai_metadata"] as? Bool ?? false
        report.findings = obj["findings"] as? [String] ?? []
        report.notes = obj["notes"] as? [String] ?? []
        if let hits = (obj["hits"] as? [[String: Any]]) ?? (obj["layer_a_hits"] as? [[String: Any]]) {
            report.hits = hits.map { hit in
                Hit(
                    codepoint: hit["codepoint"] as? String ?? "",
                    label: hit["label"] as? String ?? "",
                    count: hit["count"] as? Int ?? 0,
                    kind: hit["kind"] as? String ?? "",
                    confidence: hit["confidence"] as? String ?? ""
                )
            }
        }
        return report
    }
}

struct CleanReport: Sendable {
    var kind: String
    var changed: Bool
    var output: String?
    var actions: [String]
    var stillHasC2PA: Bool
    var stillHasAIMetadata: Bool
    var removedCount: Int
    var replacedCount: Int
    var degraded: Bool
    var rawJSON: String

    var residual: Bool { stillHasC2PA || stillHasAIMetadata }

    static func parse(json: String) -> CleanReport {
        var report = CleanReport(
            kind: "unknown",
            changed: false,
            output: nil,
            actions: [],
            stillHasC2PA: false,
            stillHasAIMetadata: false,
            removedCount: 0,
            replacedCount: 0,
            degraded: false,
            rawJSON: json
        )
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return report }
        report.kind = obj["kind"] as? String ?? "unknown"
        report.changed = obj["changed"] as? Bool ?? false
        report.output = obj["output"] as? String
        report.actions = obj["actions"] as? [String] ?? []
        report.stillHasC2PA = obj["still_has_c2pa"] as? Bool ?? false
        report.stillHasAIMetadata = obj["still_has_ai_metadata"] as? Bool ?? false
        if let stats = obj["stats"] as? [String: Any] {
            report.removedCount = stats["removed_count"] as? Int ?? 0
            report.replacedCount = stats["replaced_count"] as? Int ?? 0
        }
        if let meta = obj["meta"] as? [String: Any] {
            report.degraded = meta["degraded"] as? Bool ?? false
        }
        return report
    }
}

enum WatermarkService {
    static func siblingCleanedURL(for url: URL) -> URL {
        let ext = url.pathExtension
        let stem = url.deletingPathExtension().lastPathComponent
        let name = ext.isEmpty ? "\(stem).cleaned" : "\(stem).cleaned.\(ext)"
        return url.deletingLastPathComponent().appendingPathComponent(name)
    }

    static func inspect(file: URL, pythonPath: String) async throws -> (InspectReport, ProcessResult) {
        let result = try await PythonRunner.run(
            script: "inspect_file.py",
            arguments: [file.path, "--json"],
            pythonPath: pythonPath,
            timeout: 60
        )
        if result.exitCode == 2 {
            throw RunnerError.launchFailed(result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty ?? "Inspect refused this file.")
        }
        let report = InspectReport.parse(json: result.stdout)
        return (report, result)
    }

    static func clean(file: URL, output: URL, pythonPath: String) async throws -> (CleanReport, ProcessResult) {
        let result = try await PythonRunner.run(
            script: "clean_file.py",
            arguments: [file.path, "-o", output.path, "--json"],
            pythonPath: pythonPath,
            timeout: 60
        )
        if result.exitCode == 2 {
            throw RunnerError.launchFailed(result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty ?? "Clean refused this file.")
        }
        let report = CleanReport.parse(json: result.stdout)
        return (report, result)
    }

    static func rewrite(
        file: URL,
        output: URL,
        settings: SettingsSnapshot
    ) async throws -> ProcessResult {
        var args = [
            file.path,
            "-o", output.path,
            "--backend", settings.backendFlag,
            "--tactic", settings.tactic,
            "--reasoning-effort", "off",
            "--json-stats",
        ]
        if !settings.model.isEmpty {
            args += ["--model", settings.model]
        }
        if !settings.baseURL.isEmpty {
            args += ["--base-url", settings.baseURL]
        }
        var env: [String: String] = [:]
        if !settings.apiKey.isEmpty {
            env["WATERMARKS_REWRITE_API_KEY"] = settings.apiKey
        }
        if settings.allowRemote {
            env["WATERMARKS_REWRITE_ALLOW_REMOTE"] = "1"
            args.append("--allow-remote")
        }
        let result = try await PythonRunner.run(
            script: "rewrite_text.py",
            arguments: args,
            extraEnv: env,
            pythonPath: settings.pythonPath,
            timeout: 180
        )
        if result.exitCode != 0 {
            throw RunnerError.launchFailed(
                result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
                    ?? "Rewrite failed (exit \(result.exitCode))."
            )
        }
        return result
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}
