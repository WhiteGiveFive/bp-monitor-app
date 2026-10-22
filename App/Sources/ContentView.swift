import SwiftUI
import UniformTypeIdentifiers

private enum Palette {
    static let ink = Color(red: 0.12, green: 0.20, blue: 0.26)
    static let muted = Color(red: 0.40, green: 0.47, blue: 0.52)
    static let teal = Color(red: 0.08, green: 0.47, blue: 0.44)
    static let canvas = Color(red: 0.96, green: 0.97, blue: 0.975)
    static let line = Color(red: 0.89, green: 0.92, blue: 0.93)
}

struct ContentView: View {
    @ObservedObject var store: DemoStore
    @State private var showImporter = false
    @State private var showAbout = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    header
                    sessionSection
                    if let record = store.selectedRecord {
                        SignalCard(record: record)
                        runSection
                        resultsSection
                    } else {
                        ContentUnavailableView("Your recordings, on device", systemImage: "waveform.path.ecg",
                            description: Text("Open the sample session or import prepared recordings to compare the models."))
                    }
                    footer
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 28)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .background(Palette.canvas)
            .foregroundStyle(Palette.ink)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 6) {
                        Circle().fill(Palette.teal).frame(width: 6, height: 6)
                        Text("ON DEVICE").font(.system(size: 10, weight: .bold, design: .rounded)).tracking(1.5)
                    }.foregroundStyle(Palette.teal)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showAbout = true } label: { Image(systemName: "info.circle").foregroundStyle(Palette.muted) }
                        .accessibilityLabel("About this demonstrator")
                }
            }
            .task { await store.loadBundled() }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.json], allowsMultipleSelection: false) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first { Task { await store.importFile(url) } }
                case .failure(let error): store.errorMessage = error.localizedDescription
                }
            }
            .alert("Unable to continue", isPresented: Binding(get: { store.errorMessage != nil }, set: { if !$0 { store.errorMessage = nil } })) {
                Button("OK", role: .cancel) { store.errorMessage = nil }
            } message: { Text(store.errorMessage ?? "") }
            .sheet(isPresented: $showAbout) { aboutSheet }
        }
        .tint(Palette.teal)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("BP Monitor").font(.system(size: 32, weight: .bold, design: .rounded))
            Text("Offline research inference")
                .font(.subheadline).foregroundStyle(Palette.muted)
        }
        .accessibilityElement(children: .combine)
    }

    private var sessionSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("SESSION").font(.system(size: 10, weight: .bold)).tracking(1.5).foregroundStyle(Palette.muted)
                    Text(store.sourceName).font(.headline).lineLimit(2)
                    if !store.records.isEmpty {
                        Text("\(store.records.count) recordings · \(store.subjectCount) \(store.subjectCount == 1 ? "subject" : "subjects")")
                            .font(.caption).foregroundStyle(Palette.muted)
                    }
                }
                Spacer(minLength: 8)
                Menu {
                    Button("Import prepared JSON", systemImage: "square.and.arrow.down") { showImporter = true }
                    Button("Open sample session", systemImage: "waveform.path") { Task { await store.restoreSample() } }
                } label: {
                    Label("Open", systemImage: "folder")
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 12).padding(.vertical, 9)
                        .background(Palette.teal.opacity(0.08), in: Capsule())
                }
                .disabled(store.isRunning || store.isLoading)
                .accessibilityIdentifier("openSession")
            }
            if store.isLoading {
                ProgressView("Reading prepared inputs…").font(.caption)
            } else if !store.records.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(store.records) { record in
                            Button { store.selectedID = record.id } label: {
                                HStack(spacing: 5) {
                                    VStack(alignment: .leading, spacing: 3) {
                                        if store.subjectCount > 1 {
                                            Text(record.subjectID)
                                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                                .opacity(0.75)
                                        }
                                        Text("Round \(record.roundIndex)")
                                    }
                                    if store.predictions[record.id]?.hybrid != nil { Image(systemName: "checkmark.circle.fill").font(.caption2) }
                                }
                                .font(.system(size: 13, weight: .semibold))
                                .padding(.horizontal, 14).padding(.vertical, 10)
                                .foregroundStyle(store.selectedRecord?.id == record.id ? .white : Palette.muted)
                                .background(store.selectedRecord?.id == record.id ? Palette.ink : Palette.canvas, in: Capsule())
                            }
                            .accessibilityLabel("\(record.subjectID), round \(record.roundIndex)")
                            .accessibilityAddTraits(store.selectedRecord?.id == record.id ? .isSelected : [])
                            .accessibilityIdentifier("record_\(record.id)")
                        }
                    }
                }
            }
        }
        .padding(18)
        .background(.white, in: RoundedRectangle(cornerRadius: 20))
    }

    private var runSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button {
                if store.isRunning { store.cancel() } else { store.runSession() }
            } label: {
                HStack(spacing: 10) {
                    if store.isRunning { ProgressView().tint(.white) }
                    else { Image(systemName: store.isComplete ? "arrow.clockwise" : "play.fill").font(.system(size: 13, weight: .bold)) }
                    Text(store.isRunning ? "Stop after this recording" : store.isComplete ? "Run session again" : "Run session")
                        .font(.system(size: 15, weight: .semibold))
                    Spacer()
                    if !store.isRunning { Image(systemName: "arrow.right").font(.system(size: 14, weight: .semibold)) }
                }
                .padding(.horizontal, 20).padding(.vertical, 17)
                .foregroundStyle(.white)
                .background(Palette.teal, in: RoundedRectangle(cornerRadius: 16))
            }
            .disabled(store.isLoading)
            .accessibilityIdentifier("runSession")
            if store.isRunning {
                ProgressView(value: Double(store.completedCount), total: Double(max(store.records.count, 1)))
                    .tint(Palette.teal)
            }
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: store.isComplete ? "checkmark.circle" : "cpu")
                    .foregroundStyle(store.isComplete ? Palette.teal : Palette.muted)
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.progressMessage)
                    if let milliseconds = store.sessionMilliseconds {
                        Text("Session \(formatTime(milliseconds)) · initial load \(formatTime(store.modelLoadMilliseconds ?? 0))")
                            .monospacedDigit()
                    }
                }
                .font(.caption).foregroundStyle(Palette.muted)
            }
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Model comparison").font(.system(size: 19, weight: .bold, design: .rounded))
                Spacer()
                Text("mmHg").font(.caption).foregroundStyle(Palette.muted)
            }
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 12) {
                    predictionCards(firstColumn: true).frame(minWidth: 145)
                    predictionCards(firstColumn: false).frame(minWidth: 145)
                }
                VStack(spacing: 12) {
                    predictionCards(firstColumn: true)
                    predictionCards(firstColumn: false)
                }
            }
            if let result = store.selectedPrediction {
                if let count = result.hybridRoundCount {
                    Label("Hybrid uses all \(count) imported \(count == 1 ? "round" : "rounds") for this subject.", systemImage: "square.stack.3d.up")
                        .font(.caption).foregroundStyle(Palette.muted)
                } else {
                    Text("Hybrid currently shows the raw blend. Its session adjustment is applied after every recording finishes.")
                        .font(.caption).foregroundStyle(Palette.muted)
                }
            } else {
                Text("The models run locally on the prepared inputs. Run the full session to calculate the Hybrid adjustment.")
                    .font(.caption).foregroundStyle(Palette.muted)
            }
        }
    }

    @ViewBuilder private func predictionCards(firstColumn: Bool) -> some View {
        let result = store.selectedPrediction
        VStack(spacing: 12) {
            if firstColumn {
                PredictionCard(title: "LightGBM", detail: "Engineered features", symbol: "point.3.connected.trianglepath.dotted",
                    pressure: result?.lightGBM.pressure, time: result.map { formatTime($0.lightGBM.milliseconds) }, highlighted: false)
                PredictionCard(title: "Mean Beats", detail: "Beat fusion ensemble", symbol: "waveform.path",
                    pressure: result?.meanBeats.pressure, time: result.map { formatTime($0.meanBeats.milliseconds) }, highlighted: false)
            } else {
                PredictionCard(title: "FusionNet", detail: "Basic fusion ensemble", symbol: "square.stack.3d.down.right",
                    pressure: result?.fusionNet.pressure, time: result.map { formatTime($0.fusionNet.milliseconds) }, highlighted: false)
                PredictionCard(title: "Hybrid", detail: result?.hybrid != nil ? "Session adjusted" : result != nil ? "Raw blend · provisional" : "Blend + session adjustment", symbol: "circle.hexagongrid",
                    pressure: result?.hybrid ?? result?.rawBlend, time: nil, highlighted: true)
            }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider().overlay(Palette.line)
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "lock.shield").font(.caption)
                Text("Prepared recording replay · no network required")
                    .font(.caption)
            }
            Text("Research demonstrator. These estimates are not a clinical measurement or a basis for treatment decisions.")
                .font(.system(size: 11)).fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(Palette.muted)
    }

    private var aboutSheet: some View {
        NavigationStack {
            List {
                Section("On-device inference") {
                    Text("This demonstrator runs the exported LightGBM, FusionNet and Mean Beats models locally. It uses five waveform views and five neural seeds. Hybrid blends LightGBM and Mean Beats and applies the saved subject/round adjustment.")
                    Text("CPU execution and float32 neural tensors preserve the reference pipeline's numerical contract. Timings are measured on this device, including input packing. Model loading is reported separately.")
                }
                Section("Prepared recordings") {
                    Text("Import a single prepared recording or a session JSON file. Raw MAT files and camera measurements need the frozen preprocessing pipeline before import.")
                    Text("Include all rounds for each subject to reproduce that session's Hybrid results. Previews are for display only; inference uses the prepared tensors and feature vector.")
                }
                Section("Research use") {
                    Text("The displayed results are research estimates from recorded data. This app does not collect live sensor data or establish clinical accuracy.")
                }
            }
            .font(.subheadline)
            .navigationTitle("About BP Monitor")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showAbout = false } } }
        }
        .presentationDragIndicator(.visible)
    }
}

private struct PredictionCard: View {
    let title: String
    let detail: String
    let symbol: String
    let pressure: BloodPressure?
    let time: String?
    let highlighted: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Image(systemName: symbol).font(.system(size: 16, weight: .medium)).foregroundStyle(highlighted ? Palette.teal : Palette.muted)
                Spacer(minLength: 0)
                if let time { Text(time).font(.system(size: 10, weight: .medium, design: .monospaced)).foregroundStyle(Palette.muted) }
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 15, weight: .bold))
                Text(detail).font(.system(size: 10)).foregroundStyle(Palette.muted).lineLimit(2).frame(minHeight: 24, alignment: .top)
            }
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                pressureValue(pressure?.sbp, label: "SYS")
                Text("/").font(.system(size: 20, weight: .light)).foregroundStyle(Palette.muted.opacity(0.5))
                pressureValue(pressure?.dbp, label: "DIA")
            }
            .minimumScaleFactor(0.65)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(highlighted ? Palette.teal.opacity(0.055) : .white, in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(highlighted ? Palette.teal.opacity(0.18) : .clear, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(pressure.map { "Systolic \(String(format: "%.1f", $0.sbp)), diastolic \(String(format: "%.1f", $0.dbp)) millimeters of mercury. \(detail)." } ?? "Not yet calculated")
        .accessibilityIdentifier("prediction_\(title)")
    }

    private func pressureValue(_ value: Double?, label: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(value.map { String(format: "%.1f", $0) } ?? "—")
                .font(.system(size: 25, weight: .semibold, design: .rounded)).monospacedDigit().lineLimit(1)
            Text(label).font(.system(size: 9, weight: .semibold)).tracking(1).foregroundStyle(Palette.muted)
        }
    }
}

private struct SignalCard: View {
    let record: PreparedRecording

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Recording \(record.roundIndex)").font(.system(size: 17, weight: .bold, design: .rounded))
                    Text(record.id).font(.system(size: 10, design: .monospaced)).foregroundStyle(Palette.muted).lineLimit(1)
                }
                Spacer(minLength: 8)
                Text("\(Int(record.display.durationSeconds)) s").font(.system(size: 12, weight: .semibold, design: .rounded))
                    .padding(.horizontal, 9).padding(.vertical, 6).background(Palette.canvas, in: Capsule())
            }
            VStack(spacing: 12) {
                waveform(record.display.ppg, label: "PPG", color: Palette.teal)
                waveform(record.display.ecg, label: "ECG", color: Color(red: 0.33, green: 0.43, blue: 0.61))
            }
            HStack(spacing: 12) {
                if let age = record.display.age { Label("\(Int(age)) years", systemImage: "person") }
                if let sex = record.display.sex, !sex.isEmpty { Text(sex.capitalized) }
                Spacer(minLength: 0)
                Text("\(record.beatValid.filter { $0 }.count) valid beats")
            }
            .font(.system(size: 10)).foregroundStyle(Palette.muted)
        }
        .padding(18)
        .background(.white, in: RoundedRectangle(cornerRadius: 20))
    }

    private func waveform(_ values: [Double], label: String, color: Color) -> some View {
        HStack(spacing: 10) {
            Text(label).font(.system(size: 9, weight: .bold)).tracking(0.5).foregroundStyle(color).frame(width: 28, alignment: .leading)
            WaveformView(values: values, color: color).frame(height: 46)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label) recording preview")
    }
}

private struct WaveformView: View {
    let values: [Double]
    let color: Color

    var body: some View {
        Canvas { context, size in
            var guide = Path()
            for fraction in [0.25, 0.5, 0.75] {
                guide.move(to: CGPoint(x: 0, y: size.height * fraction))
                guide.addLine(to: CGPoint(x: size.width, y: size.height * fraction))
            }
            context.stroke(guide, with: .color(Palette.line.opacity(0.8)), lineWidth: 0.5)
            guard values.count > 1, let minimum = values.min(), let maximum = values.max() else { return }
            // Normalize before subtracting so even very large finite imported
            // preview values cannot overflow into invalid drawing coordinates.
            let magnitude = max(abs(minimum), abs(maximum), 1e-10)
            let scaledMinimum = minimum / magnitude
            let extent = max(maximum / magnitude - scaledMinimum, 1e-10)
            let step = max(1, values.count / 1200)
            let samples = stride(from: 0, to: values.count, by: step).map { $0 }
            var path = Path()
            for (point, index) in samples.enumerated() {
                let x = Double(index) / Double(values.count - 1) * size.width
                let y = (1 - (values[index] / magnitude - scaledMinimum) / extent) * (size.height - 8) + 4
                if point == 0 { path.move(to: CGPoint(x: x, y: y)) }
                else { path.addLine(to: CGPoint(x: x, y: y)) }
            }
            context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 1.35, lineCap: .round, lineJoin: .round))
        }
    }
}

private func formatTime(_ milliseconds: Double) -> String {
    if milliseconds >= 1000 { return String(format: "%.2f s", milliseconds / 1000) }
    if milliseconds < 1 { return String(format: "%.2f ms", milliseconds) }
    return String(format: "%.1f ms", milliseconds)
}
