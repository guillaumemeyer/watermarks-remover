import SwiftUI

/// Settings: the OpenRouter credentials and model, the Layer B strategy, and
/// the updater that keeps the Python scripts level with upstream.
@MainActor
struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var selection: Tab = .model

    enum Tab: String, CaseIterable, Identifiable {
        case model = "Model"
        case strategy = "Strategy"
        case tools = "Tools"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .model: return "key.horizontal"
            case .strategy: return "slider.horizontal.3"
            case .tools: return "arrow.triangle.2.circlepath"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $selection) {
                ForEach(Tab.allCases) { tab in
                    // Segmented pickers on macOS render text far more reliably
                    // than a Label's icon-plus-title pair.
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 20)
            .padding(.top, 18)
            .padding(.bottom, 14)

            ScrollView {
                Group {
                    switch selection {
                    case .model: ModelSettings()
                    case .strategy: StrategySettings()
                    case .tools: ToolsSettings()
                    }
                }
                .padding(.horizontal, 22)
                .padding(.bottom, 20)
            }

            Divider().overlay(Theme.steel.opacity(0.18))

            HStack {
                SyncBadge(status: settings.syncStatus)
                Spacer()
                Button("Done") {
                    settings.commitAPIKey()
                    dismiss()
                }
                .buttonStyle(PrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .frame(width: 560, height: 520)
        .background(Theme.windowBackground)
        .foregroundStyle(Theme.ink)
    }
}

// MARK: - Model

@MainActor
private struct ModelSettings: View {
    @EnvironmentObject private var settings: SettingsStore
    @State private var revealKey = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsSection(
                title: "OpenRouter API key",
                caption: "Stored in your keychain and synced through iCloud Keychain, "
                    + "never in a preferences file. It is passed to the rewrite script "
                    + "through the environment, never on the command line."
            ) {
                HStack(spacing: 8) {
                    Group {
                        if revealKey {
                            TextField("sk-or-v1-…", text: $settings.apiKey)
                        } else {
                            SecureField("sk-or-v1-…", text: $settings.apiKey)
                        }
                    }
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()

                    Button {
                        revealKey.toggle()
                    } label: {
                        Image(systemName: revealKey ? "eye.slash" : "eye")
                    }
                    .buttonStyle(SecondaryButtonStyle())
                    .help(revealKey ? "Hide the key" : "Show the key")
                }

                HStack(spacing: 10) {
                    Button("Save Key") { settings.commitAPIKey() }
                        .buttonStyle(SecondaryButtonStyle())
                    SyncBadge(status: settings.keychainStatus)
                }
            }

            SettingsSection(
                title: "Model",
                caption: "Any OpenRouter model slug. Prefer a model from a different "
                    + "vendor than the one that wrote the text — rewriting Claude "
                    + "output with Claude can re-stamp the marks you are removing."
            ) {
                TextField("vendor/model", text: $settings.model)
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()

                if !settings.recentModels.isEmpty {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("RECENT")
                            .font(.system(size: 9, weight: .semibold))
                            .tracking(0.8)
                            .foregroundStyle(Theme.inkDim.opacity(0.8))
                        WrappingChips(items: settings.recentModels) { item in
                            settings.model = item
                        }
                    }
                }
            }

            SettingsSection(
                title: "Endpoint",
                caption: "The rewrite script appends /v1/chat/completions, so this "
                    + "stops at /api. Change it only to point at a different "
                    + "OpenAI-compatible gateway."
            ) {
                TextField(SettingsStore.defaultBaseURL, text: $settings.baseURL)
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()
            }
        }
    }
}

// MARK: - Strategy

@MainActor
private struct StrategySettings: View {
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsSection(
                title: "Layer B strategy",
                caption: "Statistical text watermarks live in the wording itself, so "
                    + "removing them means rewriting. A heavier strategy diverges "
                    + "further from the mark and further from your voice."
            ) {
                VStack(spacing: 8) {
                    ForEach(SettingsStore.presets) { preset in
                        PresetRow(preset: preset,
                                  isSelected: settings.strategy == preset.spec) {
                            settings.strategy = preset.spec
                        }
                    }
                }
            }

            SettingsSection(
                title: "Custom strategy",
                caption: "The tactic@intensity form rewrite_text.py --strategy parses, "
                    + "for example paraphrase@0.8,humanize@0.4. Intensity is in (0,1]."
            ) {
                TextField("paraphrase@0.8", text: $settings.strategy)
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()
            }

            SettingsSection(title: "Sampling", caption: nil) {
                LabeledSlider(label: "Temperature",
                              value: $settings.temperature,
                              range: 0.1...1.5,
                              format: "%.2f")
                LabeledSlider(label: "Timeout per step",
                              value: $settings.timeoutSeconds,
                              range: 30...900,
                              format: "%.0f s")
            }
        }
    }
}

private struct PresetRow: View {
    let preset: SettingsStore.StrategyPreset
    let isSelected: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(isSelected ? Theme.accent : Theme.inkDim.opacity(0.6))
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(preset.name).font(.system(size: 12, weight: .medium))
                        if preset.needsExtraDependencies {
                            Text("extra deps")
                                .font(.system(size: 9, weight: .semibold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Theme.slate))
                                .foregroundStyle(Theme.ink)
                        }
                    }
                    Text(preset.detail)
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.inkDim)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(preset.spec)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.accent.opacity(0.9))
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(isSelected ? 0.10 : 0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(isSelected ? Theme.accent.opacity(0.7) : Theme.steel.opacity(0.2),
                            lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Tools

@MainActor
private struct ToolsSettings: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var model: AppModel

    @State private var checking = false
    @State private var installing = false
    @State private var available: ScriptUpdater.Availability?
    @State private var message: String?
    @State private var isError = false

    private var updater: ScriptUpdater {
        ScriptUpdater(repository: settings.updateRepository, ref: settings.updateRef)
    }

    private var interpreter: URL? { PythonRunner.findInterpreter() }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsSection(
                title: "Layer B scripts",
                caption: "Watermarker runs the repository's own Python rather than a "
                    + "reimplementation, so improvements upstream arrive as a script "
                    + "update instead of a new build of the app."
            ) {
                HStack(spacing: 8) {
                    Image(systemName: "shippingbox")
                        .foregroundStyle(Theme.inkDim)
                    Text(model.scriptSource?.label ?? "No scripts found")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.inkDim)
                }

                HStack(spacing: 10) {
                    Button(checking ? "Checking…" : "Check for Updates") {
                        Task { await check() }
                    }
                    .buttonStyle(SecondaryButtonStyle())
                    .disabled(checking || installing)

                    if let available, available.isNewer {
                        Button(installing ? "Installing…" : "Install \(available.shortCommit)") {
                            Task { await install(available) }
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(installing)
                    }

                    if model.scriptSource?.isUpdated == true {
                        Button("Revert to Bundled") { revert() }
                            .buttonStyle(SecondaryButtonStyle())
                            .disabled(installing)
                    }
                }

                if let available {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(available.isNewer
                             ? "Update available: \(available.shortCommit)"
                             : "Already up to date at \(available.shortCommit)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(available.isNewer ? Theme.accent : Theme.success)
                        Text(available.message)
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.inkDim)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if let message {
                    Text(message)
                        .font(.system(size: 11))
                        .foregroundStyle(isError ? Theme.danger : Theme.success)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }

            SettingsSection(
                title: "Update source",
                caption: "Defaults to the upstream project. Point it at your own fork "
                    + "to track that instead."
            ) {
                TextField(SettingsStore.defaultRepository, text: $settings.updateRepository)
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()
                TextField(SettingsStore.defaultRef, text: $settings.updateRef)
                    .textFieldStyle(.plain)
                    .font(Theme.monoFont)
                    .padding(7)
                    .watermarkerWell()
            }

            SettingsSection(title: "Python", caption: nil) {
                HStack(spacing: 8) {
                    Image(systemName: interpreter == nil
                          ? "exclamationmark.triangle" : "checkmark.seal")
                        .foregroundStyle(interpreter == nil ? Theme.danger : Theme.success)
                    Text(interpreter?.path
                         ?? "No python3 found — run xcode-select --install")
                        .font(Theme.monoFont)
                        .foregroundStyle(Theme.inkDim)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private func check() async {
        checking = true
        message = nil
        defer { checking = false }
        do {
            available = try await updater.check()
        } catch {
            isError = true
            message = error.localizedDescription
        }
    }

    private func install(_ target: ScriptUpdater.Availability) async {
        installing = true
        message = nil
        defer { installing = false }
        do {
            let manifest = try await updater.install(commit: target.commit)
            model.refreshScriptSource()
            available = ScriptUpdater.Availability(commit: manifest.commit,
                                                   shortCommit: manifest.shortCommit,
                                                   message: target.message,
                                                   isNewer: false)
            isError = false
            message = "Installed \(manifest.shortCommit) and verified it compiles."
        } catch {
            isError = true
            message = error.localizedDescription
        }
    }

    private func revert() {
        do {
            try ScriptUpdater.revertToBundled()
            model.refreshScriptSource()
            available = nil
            isError = false
            message = "Back to the scripts that shipped with the app."
        } catch {
            isError = true
            message = error.localizedDescription
        }
    }
}

// MARK: - Shared pieces

private struct SettingsSection<Content: View>: View {
    let title: String
    let caption: String?
    let content: Content

    init(title: String, caption: String?, @ViewBuilder content: () -> Content) {
        self.title = title
        self.caption = caption
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.9)
                .foregroundStyle(Theme.inkDim)
            if let caption {
                Text(caption)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.inkDim.opacity(0.8))
                    .fixedSize(horizontal: false, vertical: true)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct LabeledSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.system(size: 11))
                .frame(width: 118, alignment: .leading)
            Slider(value: $value, in: range)
                .tint(Theme.accent)
            Text(String(format: format, value))
                .font(Theme.monoFont)
                .foregroundStyle(Theme.inkDim)
                .frame(width: 52, alignment: .trailing)
        }
    }
}

private struct WrappingChips: View {
    let items: [String]
    let select: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 5) {
                    ForEach(row, id: \.self) { item in
                        Button(item) { select(item) }
                            .buttonStyle(.plain)
                            .font(.system(size: 10, design: .monospaced))
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(Capsule().fill(Color.white.opacity(0.08)))
                            .overlay(Capsule().stroke(Theme.steel.opacity(0.28), lineWidth: 1))
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }

    /// Chips are laid out in fixed rows of two: model slugs are long enough
    /// that a flow layout would rarely fit more, and this keeps the settings
    /// sheet free of a layout dependency.
    private var rows: [[String]] {
        stride(from: 0, to: items.count, by: 2).map {
            Array(items[$0..<min($0 + 2, items.count)])
        }
    }
}

struct SyncBadge: View {
    let status: SettingsStore.SyncStatus

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: status.isCloud ? "icloud.fill" : "internaldrive")
                .font(.system(size: 10))
            Text(status.label)
                .font(.system(size: 10))
        }
        .foregroundStyle(status.isCloud ? Theme.accent : Theme.inkDim)
    }
}
