import Foundation
import Security

enum RewriteBackend: String, CaseIterable, Identifiable {
    case off
    case ollama
    case openaiCompatible = "openai"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .off: "Off (Layer A only)"
        case .ollama: "Ollama (local)"
        case .openaiCompatible: "OpenAI-compatible"
        }
    }
}

enum RewriteTactic: String, CaseIterable, Identifiable {
    case paraphrase
    case humanize

    var id: String { rawValue }
}

struct SettingsSnapshot: Sendable {
    var pythonPath: String
    var backendFlag: String
    var model: String
    var baseURL: String
    var apiKey: String
    var tactic: String
    var allowRemote: Bool
    var rewriteEnabled: Bool
}

@MainActor
@Observable
final class SettingsStore {
    var pythonPath: String {
        didSet { UserDefaults.standard.set(pythonPath, forKey: "pythonPath") }
    }
    var backend: RewriteBackend {
        didSet { UserDefaults.standard.set(backend.rawValue, forKey: "rewriteBackend") }
    }
    var ollamaModel: String {
        didSet { UserDefaults.standard.set(ollamaModel, forKey: "ollamaModel") }
    }
    var ollamaBaseURL: String {
        didSet { UserDefaults.standard.set(ollamaBaseURL, forKey: "ollamaBaseURL") }
    }
    var openaiModel: String {
        didSet { UserDefaults.standard.set(openaiModel, forKey: "openaiModel") }
    }
    var openaiBaseURL: String {
        didSet { UserDefaults.standard.set(openaiBaseURL, forKey: "openaiBaseURL") }
    }
    var tactic: RewriteTactic {
        didSet { UserDefaults.standard.set(tactic.rawValue, forKey: "rewriteTactic") }
    }
    var apiKey: String

    init() {
        let defaults = UserDefaults.standard
        pythonPath = defaults.string(forKey: "pythonPath") ?? ""
        backend = RewriteBackend(rawValue: defaults.string(forKey: "rewriteBackend") ?? "off") ?? .off
        ollamaModel = defaults.string(forKey: "ollamaModel") ?? "llama3.2"
        ollamaBaseURL = defaults.string(forKey: "ollamaBaseURL") ?? "http://127.0.0.1:11434"
        openaiModel = defaults.string(forKey: "openaiModel") ?? "gpt-4o-mini"
        openaiBaseURL = defaults.string(forKey: "openaiBaseURL") ?? "https://api.openai.com/v1"
        tactic = RewriteTactic(rawValue: defaults.string(forKey: "rewriteTactic") ?? "paraphrase") ?? .paraphrase
        apiKey = KeychainStore.get() ?? ""
    }

    func saveAPIKey() {
        KeychainStore.set(apiKey)
    }

    func snapshot() -> SettingsSnapshot {
        let rewriteOn = backend != .off
        let model: String
        let baseURL: String
        switch backend {
        case .off:
            model = ""
            baseURL = ""
        case .ollama:
            model = ollamaModel
            baseURL = ollamaBaseURL
        case .openaiCompatible:
            model = openaiModel
            baseURL = openaiBaseURL
        }
        let allowRemote: Bool = {
            guard let host = URL(string: baseURL)?.host else { return false }
            return !["127.0.0.1", "localhost", "::1"].contains(host)
        }()
        return SettingsSnapshot(
            pythonPath: pythonPath,
            backendFlag: backend == .openaiCompatible ? "openai-compatible" : "ollama",
            model: model,
            baseURL: baseURL,
            apiKey: apiKey,
            tactic: tactic.rawValue,
            allowRemote: allowRemote,
            rewriteEnabled: rewriteOn
        )
    }
}

enum KeychainStore {
    private static let service = "io.github.guillaumemeyer.watermarksremover"
    private static let account = "rewrite-api-key"

    static func get() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func set(_ value: String) {
        let payload = Data(value.utf8)
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(base as CFDictionary)
        guard !value.isEmpty else { return }
        var add = base
        add[kSecValueData as String] = payload
        add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
        SecItemAdd(add as CFDictionary, nil)
    }
}
