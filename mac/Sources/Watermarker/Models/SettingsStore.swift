import Combine
import Foundation

/// Everything the user configures, mirrored into CloudKit.
///
/// Preferences (model, strategy, endpoint, update source) go to
/// `NSUbiquitousKeyValueStore` — the CloudKit-backed key-value store, so the
/// same choices show up on the user's other Macs. The API key goes to iCloud
/// Keychain instead (see `KeychainStore`); a secret does not belong in a
/// key-value store that syncs as a plist.
///
/// A build signed without the iCloud entitlements gets neither, so every write
/// also lands in `UserDefaults` and `syncStatus` says plainly which store is
/// live. Nothing in the app depends on iCloud being available.
@MainActor
final class SettingsStore: ObservableObject {
    enum SyncStatus: Equatable {
        case cloud
        case localOnly(String)

        var isCloud: Bool { self == .cloud }

        var label: String {
            switch self {
            case .cloud: return "Synced with iCloud"
            case .localOnly(let reason): return "Stored on this Mac — \(reason)"
            }
        }
    }

    /// What to send as the OpenAI `reasoning_effort` parameter, if anything.
    ///
    /// OpenRouter rejects `reasoning_effort` outright for models that have no
    /// reasoning mode -- which is most of them -- so the default omits the
    /// parameter and the app works with any slug. Sending `none` is worth it
    /// only on a reasoning model that accepts it: without it, a model like
    /// deepseek-v4-flash spends thousands of chain-of-thought tokens on a
    /// one-line rewrite.
    /// The case for `reasoning_effort: "none"` is spelled `skip` rather than
    /// `none`, so it can never be confused with `Optional.none` at a use site.
    enum ReasoningEffort: String, CaseIterable, Identifiable, Sendable {
        case off = "off"
        case skip = "none"
        case low = "low"
        case medium = "medium"
        case high = "high"

        var id: String { rawValue }

        var label: String {
            switch self {
            case .off: return "Omit — works with every model"
            case .skip: return "none — skip chain-of-thought"
            case .low: return "low"
            case .medium: return "medium"
            case .high: return "high"
            }
        }
    }

    /// A named Layer B strategy, in the `tactic@intensity,...` form
    /// `rewrite_text.py --strategy` parses.
    struct StrategyPreset: Identifiable, Hashable {
        let id: String
        let name: String
        let spec: String
        let detail: String
        /// True when the strategy needs Python packages beyond the standard
        /// library (the `mlm` tactic pulls in transformers and torch).
        let needsExtraDependencies: Bool
    }

    static let presets: [StrategyPreset] = [
        StrategyPreset(
            id: "paraphrase",
            name: "Paraphrase",
            spec: "paraphrase@0.8",
            detail: "One heavy paraphrase pass. The default, and the only one "
                + "that needs nothing but a model.",
            needsExtraDependencies: false
        ),
        StrategyPreset(
            id: "humanize",
            name: "Humanize",
            spec: "humanize@0.8",
            detail: "Rewrites for human cadence and strips AI tells, then runs "
                + "the deterministic humanizer pass.",
            needsExtraDependencies: false
        ),
        StrategyPreset(
            id: "paraphrase-humanize",
            name: "Paraphrase, then humanize",
            spec: "paraphrase@0.8,humanize@0.6",
            detail: "Two model passes. Slower and costlier, and it moves the "
                + "wording furthest from the original.",
            needsExtraDependencies: false
        ),
        StrategyPreset(
            id: "benchmark",
            name: "Benchmark default (paraphrase + masked-LM)",
            spec: "paraphrase@0.8,mlm@0.2",
            detail: "The repository's benchmark-tuned default. The masked-LM "
                + "step needs transformers and torch installed for the Python "
                + "the app runs.",
            needsExtraDependencies: true
        ),
    ]

    // MARK: Keys

    private enum Key {
        static let model = "openrouter.model"
        static let baseURL = "openrouter.baseURL"
        static let strategy = "rewrite.strategy"
        static let temperature = "rewrite.temperature"
        static let timeout = "rewrite.timeout"
        static let reasoningEffort = "rewrite.reasoningEffort"
        static let updateRepository = "updates.repository"
        static let updateRef = "updates.ref"
        static let recentModels = "openrouter.recentModels"
    }

    static let keychainService = "com.symbiola.Watermarker.openrouter"
    static let keychainAccount = "api-key"

    // MARK: Defaults

    /// `rewrite_text.py` appends `/v1/chat/completions`, so the configured base
    /// URL stops at OpenRouter's `/api`.
    static let defaultBaseURL = "https://openrouter.ai/api"
    static let defaultModel = "openai/gpt-4o-mini"
    static let defaultRepository = "guillaumemeyer/watermarks-remover"
    static let defaultRef = "main"

    // MARK: Published state

    @Published var model: String { didSet { put(Key.model, model) } }
    @Published var baseURL: String { didSet { put(Key.baseURL, baseURL) } }
    @Published var strategy: String { didSet { put(Key.strategy, strategy) } }
    @Published var temperature: Double { didSet { put(Key.temperature, temperature) } }
    @Published var timeoutSeconds: Double { didSet { put(Key.timeout, timeoutSeconds) } }
    @Published var reasoningEffort: ReasoningEffort {
        didSet { put(Key.reasoningEffort, reasoningEffort.rawValue) }
    }
    @Published var updateRepository: String { didSet { put(Key.updateRepository, updateRepository) } }
    @Published var updateRef: String { didSet { put(Key.updateRef, updateRef) } }
    @Published private(set) var recentModels: [String] { didSet { put(Key.recentModels, recentModels) } }

    /// Held in memory only for the lifetime of the window; the durable copy
    /// lives in the keychain.
    @Published var apiKey: String
    @Published private(set) var syncStatus: SyncStatus
    @Published private(set) var keychainStatus: SyncStatus

    private let cloud = NSUbiquitousKeyValueStore.default
    private let local = UserDefaults.standard
    private let cloudAvailable: Bool
    // Held for the life of the app; the store outlives every window, so there
    // is no teardown to do.
    private var observer: NSObjectProtocol?

    var hasAPIKey: Bool { !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    /// The preset matching the current strategy string, if any.
    var matchingPreset: StrategyPreset? {
        Self.presets.first { $0.spec == strategy }
    }

    init() {
        // `synchronize()` returning false means the process has no usable
        // ubiquity KVS: no entitlement, or the user is not signed into iCloud.
        let signedIn = FileManager.default.ubiquityIdentityToken != nil
        let kvsUsable = NSUbiquitousKeyValueStore.default.synchronize()
        cloudAvailable = signedIn && kvsUsable
        if cloudAvailable {
            syncStatus = .cloud
        } else if !signedIn {
            syncStatus = .localOnly("not signed in to iCloud")
        } else {
            syncStatus = .localOnly("this build is not signed for iCloud")
        }

        let kvs = NSUbiquitousKeyValueStore.default
        let defaults = UserDefaults.standard
        func string(_ key: String, _ fallback: String) -> String {
            let value = (kvs.string(forKey: key) ?? defaults.string(forKey: key)) ?? fallback
            return value.isEmpty ? fallback : value
        }
        func number(_ key: String, _ fallback: Double) -> Double {
            if kvs.object(forKey: key) != nil { return kvs.double(forKey: key) }
            if defaults.object(forKey: key) != nil { return defaults.double(forKey: key) }
            return fallback
        }

        model = string(Key.model, Self.defaultModel)
        baseURL = string(Key.baseURL, Self.defaultBaseURL)
        strategy = string(Key.strategy, Self.presets[0].spec)
        temperature = number(Key.temperature, 0.9)
        timeoutSeconds = number(Key.timeout, 180)
        reasoningEffort = ReasoningEffort(rawValue: string(Key.reasoningEffort, "off")) ?? .off
        updateRepository = string(Key.updateRepository, Self.defaultRepository)
        updateRef = string(Key.updateRef, Self.defaultRef)
        recentModels = (kvs.array(forKey: Key.recentModels) as? [String])
            ?? (defaults.array(forKey: Key.recentModels) as? [String])
            ?? []

        if let stored = KeychainStore.read(service: Self.keychainService,
                                           account: Self.keychainAccount) {
            apiKey = stored.value
            keychainStatus = stored.placement == .synchronized
                ? .cloud
                : .localOnly("iCloud Keychain unavailable")
        } else {
            apiKey = ""
            keychainStatus = .localOnly("no key saved yet")
        }

        observer = NotificationCenter.default.addObserver(
            forName: NSUbiquitousKeyValueStore.didChangeExternallyNotification,
            object: kvs,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.adoptCloudChanges() }
        }
    }

    // MARK: Writes

    private func put(_ key: String, _ value: Any) {
        local.set(value, forKey: key)
        guard cloudAvailable else { return }
        cloud.set(value, forKey: key)
        cloud.synchronize()
    }

    /// Persist the key to the keychain. Called explicitly from the settings
    /// sheet rather than on every keystroke.
    func commitAPIKey() {
        let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        apiKey = trimmed
        do {
            let placement = try KeychainStore.write(trimmed,
                                                    service: Self.keychainService,
                                                    account: Self.keychainAccount)
            if trimmed.isEmpty {
                keychainStatus = .localOnly("no key saved yet")
            } else {
                keychainStatus = placement == .synchronized
                    ? .cloud
                    : .localOnly("iCloud Keychain unavailable")
            }
        } catch {
            keychainStatus = .localOnly(error.localizedDescription)
        }
    }

    func rememberCurrentModel() {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var updated = recentModels.filter { $0 != trimmed }
        updated.insert(trimmed, at: 0)
        recentModels = Array(updated.prefix(8))
    }

    /// Adopt values another Mac wrote. Only fields the user is not mid-edit on
    /// matter here, so a straight overwrite is fine.
    private func adoptCloudChanges() {
        if let value = cloud.string(forKey: Key.model), value != model { model = value }
        if let value = cloud.string(forKey: Key.baseURL), value != baseURL { baseURL = value }
        if let value = cloud.string(forKey: Key.strategy), value != strategy { strategy = value }
        if let value = cloud.string(forKey: Key.updateRepository),
           value != updateRepository { updateRepository = value }
        if let value = cloud.string(forKey: Key.updateRef), value != updateRef { updateRef = value }
        if cloud.object(forKey: Key.temperature) != nil {
            let value = cloud.double(forKey: Key.temperature)
            if value != temperature { temperature = value }
        }
        if cloud.object(forKey: Key.timeout) != nil {
            let value = cloud.double(forKey: Key.timeout)
            if value != timeoutSeconds { timeoutSeconds = value }
        }
        if let raw = cloud.string(forKey: Key.reasoningEffort),
           let value = ReasoningEffort(rawValue: raw), value != reasoningEffort {
            reasoningEffort = value
        }
        if let value = cloud.array(forKey: Key.recentModels) as? [String],
           value != recentModels { recentModels = value }
        // A key written on another Mac arrives through iCloud Keychain, not the
        // KVS, so re-read it whenever the KVS wakes us up.
        if let stored = KeychainStore.read(service: Self.keychainService,
                                           account: Self.keychainAccount),
           stored.value != apiKey {
            apiKey = stored.value
            keychainStatus = stored.placement == .synchronized
                ? .cloud
                : .localOnly("iCloud Keychain unavailable")
        }
    }
}
