import SwiftUI
import UniformTypeIdentifiers

/// The one window: type or import text on the left, read the cleaned text on
/// the right, and run the Layer B tools with the button between them.
@MainActor
struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var settings: SettingsStore
    @State private var isTargetedForDrop = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Theme.steel.opacity(0.18))
            editors
            Divider().overlay(Theme.steel.opacity(0.18))
            footer
        }
        .background(Theme.windowBackground)
        .foregroundStyle(Theme.ink)
        .frame(minWidth: 860, minHeight: 560)
        .sheet(isPresented: $model.isShowingSettings) {
            SettingsView().environmentObject(settings).environmentObject(model)
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargetedForDrop) { providers in
            handleDrop(providers)
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            StrainerMark()
                .frame(width: 30, height: 30)

            VStack(alignment: .leading, spacing: 1) {
                Text("Watermarker")
                    .font(.system(size: 16, weight: .semibold))
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.inkDim)
            }

            Spacer(minLength: 12)

            Button {
                model.presentOpenPanel()
            } label: {
                Label("Open File…", systemImage: "doc.badge.plus")
            }
            .buttonStyle(SecondaryButtonStyle())
            .help("Import a .md, .txt, or .docx file")

            Button {
                model.isShowingSettings = true
            } label: {
                Label("Settings", systemImage: "gearshape")
            }
            .buttonStyle(SecondaryButtonStyle())
            .help("OpenRouter key, model, strategy, and tool updates")
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(Color.black.opacity(0.16))
    }

    private var subtitle: String {
        if let name = model.sourceName {
            return model.sourceWasConverted
                ? "\(name) — imported as plain text"
                : name
        }
        return "Layer B watermark removal for text"
    }

    // MARK: Editors

    private var editors: some View {
        HSplitView {
            pane(
                title: "Text to clean",
                count: model.inputWordCount,
                trailing: AnyView(
                    Button("Clear") { model.clearInput() }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.inkDim)
                        .disabled(model.inputText.isEmpty)
                )
            ) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $model.inputText)
                        .font(Theme.editorFont)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                        .foregroundStyle(Theme.ink)
                        .padding(8)
                        .disabled(model.isRunning)
                    if model.inputText.isEmpty {
                        Text("Paste text here, or drop a .md or .docx file onto the window.")
                            .font(Theme.editorFont)
                            .foregroundStyle(Theme.inkDim.opacity(0.7))
                            .padding(.horizontal, 13)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                }
            }
            .frame(minWidth: 320)

            pane(
                title: "Cleaned text",
                count: model.outputWordCount,
                trailing: AnyView(
                    HStack(spacing: 12) {
                        Button("Copy") { model.copyOutput() }
                        Button("Save…") { model.saveOutput() }
                        Button("Reuse") { model.useOutputAsInput() }
                            .help("Move the result back into the editor for another pass")
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(model.outputText.isEmpty ? Theme.inkDim.opacity(0.4) : Theme.accent)
                    .disabled(model.outputText.isEmpty)
                )
            ) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: .constant(model.outputText))
                        .font(Theme.editorFont)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                        .foregroundStyle(Theme.ink)
                        .padding(8)
                    if model.outputText.isEmpty && !model.isRunning {
                        Text("The rewritten text appears here.")
                            .font(Theme.editorFont)
                            .foregroundStyle(Theme.inkDim.opacity(0.7))
                            .padding(.horizontal, 13)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                    if model.isRunning {
                        RunningOverlay(model: settings.model, strategy: settings.strategy)
                    }
                }
            }
            .frame(minWidth: 320)
        }
        .padding(16)
        .overlay(alignment: .center) {
            if isTargetedForDrop {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Theme.accent, style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Theme.accent.opacity(0.10))
                    )
                    .overlay(Text("Drop a .md, .txt, or .docx file")
                        .font(.system(size: 14, weight: .medium)))
                    .padding(10)
                    .allowsHitTesting(false)
            }
        }
    }

    private func pane<Content: View>(title: String, count: Int, trailing: AnyView,
                                     @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                Text(title.uppercased())
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(0.9)
                    .foregroundStyle(Theme.inkDim)
                Text("\(count) words")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.inkDim.opacity(0.7))
                Spacer()
                trailing
            }
            content().watermarkerWell()
        }
        .padding(.horizontal, 4)
    }

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: 0) {
            if let message = model.errorMessage {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(Theme.danger)
                    Text(message)
                        .font(.system(size: 11))
                        .textSelection(.enabled)
                    Spacer(minLength: 8)
                    Button("Dismiss") { model.errorMessage = nil }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.inkDim)
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 9)
                .background(Theme.danger.opacity(0.12))
            }

            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.status)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.inkDim)
                    Text(configurationLine)
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.inkDim.opacity(0.65))
                }

                Spacer(minLength: 8)

                if model.isRunning {
                    ProgressView()
                        .controlSize(.small)
                        .tint(Theme.steelBright)
                }

                Button {
                    Task { await model.run(settings: settings) }
                } label: {
                    Label(model.isRunning ? "Removing…" : "Remove Watermarks",
                          systemImage: "wand.and.sparkles")
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!model.canRun || !settings.hasAPIKey)
                .help(settings.hasAPIKey
                      ? "Run the Layer B rewrite on the text on the left"
                      : "Add an OpenRouter API key in Settings first")
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
        }
        .background(Color.black.opacity(0.16))
    }

    private var configurationLine: String {
        let preset = settings.matchingPreset?.name ?? settings.strategy
        let scripts = model.scriptSource?.label ?? "Scripts missing"
        return "\(settings.model) · \(preset) · \(scripts)"
    }

    // MARK: Drag and drop

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            Task { @MainActor in model.importFile(at: url) }
        }
        return true
    }
}

/// The spinner shown over the result pane while the model is working.
private struct RunningOverlay: View {
    let model: String
    let strategy: String

    var body: some View {
        VStack(spacing: 10) {
            ProgressView()
                .controlSize(.large)
                .tint(Theme.steelBright)
            Text("Straining \(strategy) through \(model)")
                .font(.system(size: 11))
                .foregroundStyle(Theme.inkDim)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.deepNavy.opacity(0.55))
    }
}

/// A small strainer drawn from shapes — the icon's motif, at toolbar size.
struct StrainerMark: View {
    var body: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)
            let rim = size * 0.86
            ZStack {
                Circle()
                    .trim(from: 0, to: 0.5)
                    .fill(Theme.steel)
                    .frame(width: rim, height: rim)
                    .rotationEffect(.degrees(180))
                    .offset(y: size * 0.10)
                Ellipse()
                    .stroke(Theme.steelBright, lineWidth: max(1.4, size * 0.075))
                    .frame(width: rim, height: rim * 0.42)
                    .offset(y: -size * 0.09)
                Ellipse()
                    .fill(Theme.navy)
                    .frame(width: rim * 0.86, height: rim * 0.30)
                    .offset(y: -size * 0.09)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}
