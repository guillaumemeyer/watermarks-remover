import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model

        VStack(spacing: 0) {
            header
            Divider()
            Group {
                if model.tab == .files {
                    filesLayout
                } else {
                    textLayout
                }
            }
            Divider()
            statusBar
        }
        .frame(minWidth: 820, minHeight: 520)
        .background(Color(nsColor: .windowBackgroundColor))
        .sheet(isPresented: $model.showSettings) {
            SettingsView()
                .environment(model)
                .frame(width: 520, height: 460)
        }
    }

    private var header: some View {
        @Bindable var model = model
        return HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Watermarks Remover")
                    .font(.title2.weight(.semibold))
                Text("Strip AI provenance from files you own")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Picker("Mode", selection: $model.tab) {
                Text("Files").tag(AppTab.files)
                Text("Text").tag(AppTab.text)
            }
            .pickerStyle(.segmented)
            .frame(width: 160)
            .labelsHidden()

            Button("Inspect") {
                Task {
                    if model.tab == .text { await model.inspectText() }
                    else { await model.inspectSelected() }
                }
            }
            .disabled(model.isBusy || (model.tab == .files && model.jobs.isEmpty) || (model.tab == .text && model.textInput.isEmpty))

            Button("Clean") {
                Task {
                    if model.tab == .text { await model.cleanText() }
                    else { await model.cleanSelected() }
                }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.isBusy || (model.tab == .files && model.jobs.isEmpty) || (model.tab == .text && model.textInput.isEmpty))

            Button {
                model.showSettings = true
            } label: {
                Image(systemName: "gearshape")
            }
            .help("Settings")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var filesLayout: some View {
        HSplitView {
            filesSidebar
                .frame(minWidth: 240, idealWidth: 280)
            reportPane
                .frame(minWidth: 420)
        }
    }

    private var filesSidebar: some View {
        @Bindable var model = model
        return VStack(spacing: 0) {
            dropZone
                .padding(12)
            Divider()
            if model.jobs.isEmpty {
                Spacer()
                Text("No files yet")
                    .foregroundStyle(.tertiary)
                Spacer()
            } else {
                List(selection: $model.selectedJobID) {
                    ForEach(model.jobs) { job in
                        jobRow(job)
                            .tag(Optional(job.id))
                    }
                }
                .listStyle(.sidebar)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard !model.isBusy else { return false }
            return handleDrop(providers)
        }
    }

    private var dropZone: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.badge.gearshape")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(.secondary)
            Text("Drop files here")
                .font(.headline)
            Text("Originals are never overwritten")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Open…") { model.openFiles() }
                .disabled(model.isBusy)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.2, dash: [6, 4]))
                .foregroundStyle(.separator)
        )
    }

    private func jobRow(_ job: FileJob) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(job.name)
                    .font(.body)
                    .lineLimit(1)
                Spacer()
                phaseBadge(job.phase)
            }
            if let inspect = job.inspect {
                Text(inspect.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else if let error = job.error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }

    private func phaseBadge(_ phase: JobPhase) -> some View {
        Text(phase.rawValue)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color.secondary.opacity(0.15)))
    }

    private var reportPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                honestyBanner
                if let error = model.errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
                if let job = model.selectedJob {
                    fileDetail(job)
                } else {
                    Text("Select a file to see the inspect report.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    private func fileDetail(_ job: FileJob) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(job.name)
                    .font(.title3.weight(.semibold))
                Spacer()
                if job.cleanedURL != nil {
                    Button("Reveal in Finder") { model.revealCleaned() }
                }
            }
            if let error = job.error {
                Text(error).foregroundStyle(.red).textSelection(.enabled)
            }
            if let inspect = job.inspect {
                section("Verifiable") {
                    labeled("Kind", inspect.kind)
                    if let format = inspect.format { labeled("Format", format) }
                    labeled("Unicode marks", "\(inspect.suspiciousTotal)")
                    labeled("C2PA", inspect.hasC2PA ? "yes" : "no")
                    labeled("AI metadata", inspect.hasAIMetadata ? "yes" : "no")
                    if inspect.hits.isEmpty && inspect.findings.isEmpty && !inspect.isActionable {
                        Text("No Layer A marks found.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(inspect.hits) { hit in
                        Text("\(hit.codepoint) \(hit.label) ×\(hit.count)")
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                    }
                    ForEach(inspect.findings, id: \.self) { finding in
                        Text("• \(finding)")
                            .textSelection(.enabled)
                    }
                }
            }
            if let clean = job.clean {
                section("Clean result") {
                    labeled("Changed", clean.changed ? "yes" : "no")
                    if clean.removedCount + clean.replacedCount > 0 {
                        labeled("Removed / replaced", "\(clean.removedCount) / \(clean.replacedCount)")
                    }
                    ForEach(clean.actions, id: \.self) { action in
                        Text("• \(action)").textSelection(.enabled)
                    }
                    if clean.residual {
                        Text("Residual C2PA or AI metadata may remain. Soft-bound and pixel marks are out of scope.")
                            .foregroundStyle(.orange)
                    }
                    if clean.degraded {
                        Text("Degraded PDF clean — install qpdf / Ghostscript for a structural strip.")
                            .foregroundStyle(.orange)
                    }
                }
            }
            if !job.log.isEmpty {
                DisclosureGroup("Tool log") {
                    Text(job.log)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var textLayout: some View {
        @Bindable var model = model
        return HSplitView {
            VStack(alignment: .leading, spacing: 8) {
                Text("Input")
                    .font(.headline)
                TextEditor(text: $model.textInput)
                    .font(.body)
                    .disabled(model.isBusy)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.separator))
            }
            .padding(16)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Cleaned")
                        .font(.headline)
                    Spacer()
                    Button("Copy") { model.copyOutput() }
                        .disabled(model.textOutput.isEmpty)
                    Button("Save…") { model.saveTextOutput() }
                        .disabled(model.textOutput.isEmpty)
                }
                if let banner = model.rewriteSkippedMessage {
                    Text(banner)
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                ScrollView {
                    Text(model.textOutput.isEmpty ? "Cleaned text appears here." : model.textOutput)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .foregroundStyle(model.textOutput.isEmpty ? .secondary : .primary)
                }
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.separator))

                if let inspect = model.textInspect {
                    section("Verifiable") {
                        labeled("Unicode marks", "\(inspect.suspiciousTotal)")
                        ForEach(inspect.hits) { hit in
                            Text("\(hit.codepoint) \(hit.label) ×\(hit.count)")
                                .font(.system(.caption, design: .monospaced))
                        }
                    }
                }
            }
            .padding(16)
        }
    }

    private var honestyBanner: some View {
        Text("Reports split verifiable removals (Unicode, C2PA, metadata) from best-effort rewrite. A clean file does not mean vendor detectors will fail.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }

    private var statusBar: some View {
        HStack {
            Text(model.status)
                .foregroundStyle(.secondary)
            Spacer()
            if model.isBusy {
                ProgressView().controlSize(.small)
            }
            if model.tab == .files, !model.jobs.isEmpty {
                Button("Clear list", action: model.clearFiles)
                    .disabled(model.isBusy)
            }
        }
        .font(.caption)
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func labeled(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            Text(value)
                .textSelection(.enabled)
            Spacer()
        }
        .font(.callout)
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                let dropped: URL?
                if let url = item as? URL {
                    dropped = url
                } else if let data = item as? Data {
                    dropped = URL(dataRepresentation: data, relativeTo: nil)
                } else {
                    dropped = nil
                }
                guard let dropped else { return }
                Task { @MainActor in
                    model.addFiles([dropped])
                }
            }
        }
        return true
    }
}
