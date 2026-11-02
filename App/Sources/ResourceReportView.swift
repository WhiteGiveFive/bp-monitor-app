import SwiftUI
import UniformTypeIdentifiers

struct ResourceReportView: View {
    let report: SessionResourceReport
    @Environment(\.dismiss) private var dismiss
    @State private var exportDocument: ResourceReportDocument?
    @State private var showExporter = false
    @State private var exportError: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    overview
                    batterySection
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Model resources")
                            .font(.system(size: 20, weight: .bold, design: .rounded))
                        Text("Measured across all \(report.recordingCount) recordings in this session.")
                            .font(.caption).foregroundStyle(Palette.muted)
                        ForEach(report.models) { model in
                            ModelResourceCard(summary: model)
                        }
                    }
                    measurementNotes
                }
                .padding(20)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .background(Palette.canvas)
            .foregroundStyle(Palette.ink)
            .navigationTitle("Resource report")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("closeResourceReport")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        do {
                            exportDocument = ResourceReportDocument(data: try report.jsonData())
                            showExporter = true
                        } catch { exportError = error.localizedDescription }
                    } label: { Label("Export JSON", systemImage: "square.and.arrow.up") }
                        .accessibilityIdentifier("exportResourceReport")
                }
            }
            .fileExporter(isPresented: $showExporter, document: exportDocument,
                          contentType: .json, defaultFilename: "bp-resources-\(report.id.uuidString.lowercased())") { result in
                if case .failure(let error) = result { exportError = error.localizedDescription }
            }
            .alert("Unable to export report", isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })) {
                Button("OK", role: .cancel) { exportError = nil }
            } message: { Text(exportError ?? "") }
        }
        .tint(Palette.teal)
        .presentationDragIndicator(.visible)
    }

    private var overview: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(report.environment.isSimulator ? "SIMULATOR RUN" : "DEVICE RUN")
                        .font(.system(size: 10, weight: .bold)).tracking(1.4).foregroundStyle(Palette.teal)
                    Text("\(report.environment.buildConfiguration) · \(report.recordingCount) recordings")
                        .font(.system(size: 19, weight: .bold, design: .rounded))
                }
                Spacer(minLength: 8)
                Image(systemName: report.environment.isSimulator ? "desktopcomputer" : "iphone")
                    .font(.title2).foregroundStyle(Palette.teal)
            }
            Text("\(report.environment.deviceModel) · \(report.environment.platform) \(report.environment.systemVersion) · \(report.environment.processorCount) CPU cores")
                .font(.caption).foregroundStyle(Palette.muted)
            ResourceMetric(label: "Complete session", value: resourceTime(report.sessionWallMilliseconds), emphasized: true)
            ResourceMetric(label: "Started", value: report.startedAt.formatted(date: .abbreviated, time: .standard))
            ResourceMetric(label: "Finished", value: report.finishedAt.formatted(date: .abbreviated, time: .standard))
            Text(report.modelsWereAlreadyLoaded
                 ? "Models were already loaded for this run. Initial load measurements below come from their first load in this app process."
                 : "This run loaded the models. Initial load measurements are shown separately from each model's inference time.")
                .font(.caption).foregroundStyle(Palette.muted)
            if report.environment.isSimulator || report.environment.buildConfiguration.lowercased() == "debug" {
                Text("Simulator and Debug measurements do not represent Release performance on an iPhone.")
                    .font(.caption).foregroundStyle(Palette.muted)
            }
        }
        .padding(18)
        .background(.white, in: RoundedRectangle(cornerRadius: 20))
        .accessibilityIdentifier("resourceRunContext")
    }

    private var batterySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Battery & device state", systemImage: "battery.75percent")
                .font(.system(size: 17, weight: .bold, design: .rounded))
            Text(report.batterySummary).font(.subheadline)
            Text("Per-model battery consumption is unavailable")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Palette.muted)
                .accessibilityIdentifier("perModelBatteryUnavailable")
            Divider().overlay(Palette.line)
            ResourceMetric(label: "Charge level", value: "\(batteryLevel(report.start.batteryLevel)) → \(batteryLevel(report.end.batteryLevel))")
            ResourceMetric(label: "Battery state", value: stateChangeText(report.start.batteryState, report.end.batteryState))
            ResourceMetric(label: "Thermal state", value: stateChangeText(report.start.thermalState, report.end.thermalState))
            ResourceMetric(label: "Low Power Mode", value: stateChangeText(report.start.lowPowerMode ? "On" : "Off", report.end.lowPowerMode ? "On" : "Off"))
            Text("Arrows show the start and end of the session. Device charge levels cannot isolate the energy used by one model.")
                .font(.caption).foregroundStyle(Palette.muted)
        }
        .padding(18)
        .background(.white, in: RoundedRectangle(cornerRadius: 20))
    }

    private var measurementNotes: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("How to read these measurements")
                .font(.system(size: 17, weight: .bold, design: .rounded))
            Text("Memory measurements cover the whole app, including every loaded model. They are not a model's isolated memory footprint. The observed peak is the largest measurement at a checkpoint and can miss brief spikes.")
            Text("CPU time is process CPU time during each measured stage; other app work can contribute. CPU time can exceed wall time when multiple cores are active.")
            Text("Hybrid includes LightGBM, Mean Beats, blending and session adjustment. It shares that inference work; its total is not an additional cost to add to the other cards. FusionNet is not part of Hybrid.")
            ForEach(Array(report.notes.enumerated()), id: \.offset) { _, note in
                Text(note)
            }
        }
        .font(.caption)
        .foregroundStyle(Palette.muted)
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct ModelResourceCard: View {
    let summary: ModelResourceSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(summary.model.title).font(.system(size: 18, weight: .bold, design: .rounded))
                Spacer(minLength: 8)
                if summary.model == .hybrid {
                    Text("SHARED WORK")
                        .font(.system(size: 9, weight: .bold)).tracking(0.7)
                        .foregroundStyle(Palette.teal)
                }
            }
            ResourceMetric(label: "Session wall time", value: resourceTime(summary.total.wallMilliseconds), emphasized: true)
            ResourceMetric(label: "Per recording · amortized", value: resourceTime(summary.meanWallMilliseconds))
            ResourceMetric(label: "Process CPU time", value: resourceTime(summary.total.cpuMilliseconds))
            ResourceMetric(label: "Observed peak app memory", value: memorySize(summary.total.observedPeakMemoryBytes))
            Text("\(summary.total.memorySampleCount) memory checkpoints · whole app with all models resident")
                .font(.system(size: 11)).foregroundStyle(Palette.muted)
            Divider().overlay(Palette.line)
            if let load = summary.load {
                ResourceMetric(label: "Model assets on disk", value: memorySize(load.diskBytes))
                ResourceMetric(label: "Initial model load", value: resourceTime(load.resources.wallMilliseconds))
                ResourceMetric(label: "App memory change on load", value: signedMemorySize(load.resources.memoryDeltaBytes))
            } else {
                ResourceMetric(label: "Model assets on disk", value: summary.model == .hybrid ? "Shared component assets" : "Unavailable")
                ResourceMetric(label: "Initial model load", value: summary.model == .hybrid ? "No separate model" : "Unavailable")
                ResourceMetric(label: "App memory change on load", value: "Not separately measured")
            }
            if summary.model == .hybrid {
                Text("LightGBM + Mean Beats + blend + session adjustment. Shares work already shown above; excludes FusionNet.")
                    .font(.caption).foregroundStyle(Palette.muted)
            }
            ForEach(Array(summary.notes.enumerated()), id: \.offset) { _, note in
                Text(note).font(.caption).foregroundStyle(Palette.muted)
            }
        }
        .padding(18)
        .background(.white, in: RoundedRectangle(cornerRadius: 20))
        .overlay {
            if summary.model == .hybrid {
                RoundedRectangle(cornerRadius: 20).stroke(Palette.teal.opacity(0.25), lineWidth: 1)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("resource_\(summary.model.rawValue)")
    }
}

private struct ResourceMetric: View {
    let label: String
    let value: String
    var emphasized = false

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(label).foregroundStyle(Palette.muted).fixedSize()
                Spacer(minLength: 0)
                metricValue.fixedSize()
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(label).foregroundStyle(Palette.muted)
                metricValue
            }
        }
        .font(.caption)
        .accessibilityElement(children: .combine)
    }

    private var metricValue: some View {
        Text(value)
            .fontWeight(emphasized ? .bold : .medium)
            .foregroundStyle(emphasized ? Palette.teal : Palette.ink)
            .monospacedDigit()
    }
}

private struct ResourceReportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.data = data
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

private func resourceTime(_ milliseconds: Double?) -> String {
    guard let milliseconds, milliseconds.isFinite else { return "Unavailable" }
    if milliseconds >= 1000 { return String(format: "%.2f s", milliseconds / 1000) }
    if milliseconds < 1 { return String(format: "%.2f ms", milliseconds) }
    return String(format: "%.1f ms", milliseconds)
}

private func memorySize(_ bytes: UInt64?) -> String {
    guard let bytes else { return "Unavailable" }
    return String(format: "%.2f MiB", Double(bytes) / 1_048_576)
}

private func signedMemorySize(_ bytes: Int64?) -> String {
    guard let bytes else { return "Unavailable" }
    return String(format: "%+.2f MiB", Double(bytes) / 1_048_576)
}

private func batteryLevel(_ level: Double?) -> String {
    guard let level, level.isFinite, (0...1).contains(level) else { return "Unavailable" }
    return String(format: "%.0f%%", level * 100)
}

private func stateChangeText(_ start: String, _ end: String) -> String {
    "\(start.capitalized) → \(end.capitalized)"
}
