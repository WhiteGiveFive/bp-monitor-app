import Foundation
import SwiftUI

actor InferenceService {
    private var engine: InferenceEngine?

    func loadSession(url: URL) throws -> PreparedSession { try PreparedSession.load(url: url) }

    func prepare(modelsURL: URL) throws -> Double {
        if engine == nil { engine = try InferenceEngine(modelsURL: modelsURL) }
        return engine?.loadMilliseconds ?? 0
    }

    func predict(_ record: PreparedRecording) throws -> RecordPrediction {
        guard let engine else { throw DemonstratorError.resource("Models have not loaded.") }
        return try engine.predict(record: record)
    }

    func finalize(_ records: [PreparedRecording], predictions: [RecordPrediction]) throws -> [RecordPrediction] {
        guard let engine else { throw DemonstratorError.resource("Models have not loaded.") }
        return try engine.finalizeSession(records: records, predictions: predictions)
    }
}

@MainActor
final class DemoStore: ObservableObject {
    @Published private(set) var records: [PreparedRecording] = []
    @Published var selectedID: String?
    @Published private(set) var predictions: [String: RecordPrediction] = [:]
    @Published private(set) var isRunning = false
    @Published private(set) var isLoading = false
    @Published private(set) var completedCount = 0
    @Published private(set) var progressMessage = "Choose a prepared session to begin."
    @Published private(set) var sourceName = "Sample session"
    @Published private(set) var sessionMilliseconds: Double?
    @Published private(set) var modelLoadMilliseconds: Double?
    @Published var errorMessage: String?
    private let service = InferenceService()
    private var runTask: Task<Void, Never>?

    var selectedRecord: PreparedRecording? { records.first { $0.id == selectedID } ?? records.first }
    var selectedPrediction: RecordPrediction? { selectedRecord.flatMap { predictions[$0.id] } }
    var subjectCount: Int { Set(records.map(\.subjectID)).count }
    var isComplete: Bool { !records.isEmpty && predictions.count == records.count && predictions.values.allSatisfy { $0.hybrid != nil } }

    func loadBundled() async {
        guard records.isEmpty, !isLoading else { return }
        guard let url = Bundle.main.url(forResource: "sample_session", withExtension: "json", subdirectory: "Data") else {
            errorMessage = "The sample session is missing from this app build. Import a prepared JSON session to continue."
            return
        }
        await load(url: url, title: "Sample session")
    }

    func restoreSample() async {
        guard !isRunning else { return }
        guard let url = Bundle.main.url(forResource: "sample_session", withExtension: "json", subdirectory: "Data") else {
            errorMessage = "The bundled sample session could not be found."
            return
        }
        await load(url: url, title: "Sample session")
    }

    func importFile(_ url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        await load(url: url, title: url.deletingPathExtension().lastPathComponent)
    }

    private func load(url: URL, title: String) async {
        guard !isRunning else { return }
        isLoading = true
        defer { isLoading = false }
        do {
            let session = try await service.loadSession(url: url)
            records = session.records
            selectedID = session.records.first?.id
            predictions = [:]
            sourceName = title
            completedCount = 0
            sessionMilliseconds = nil
            progressMessage = "Ready to compare four models."
            errorMessage = nil
        } catch {
            errorMessage = "Could not open this session. \(error.localizedDescription)"
        }
    }

    func runSession() {
        guard !isRunning, !isLoading, !records.isEmpty else { return }
        let sessionRecords = records
        predictions = [:]
        completedCount = 0
        sessionMilliseconds = nil
        errorMessage = nil
        isRunning = true
        progressMessage = "Loading models on this device…"
        runTask = Task {
            defer { isRunning = false; runTask = nil }
            do {
                guard let modelsURL = Bundle.main.url(forResource: "Models", withExtension: nil) else {
                    throw DemonstratorError.resource("The Models folder is missing from this app build.")
                }
                modelLoadMilliseconds = try await service.prepare(modelsURL: modelsURL)
                let start = ProcessInfo.processInfo.systemUptime
                var results = [RecordPrediction]()
                for (index, record) in sessionRecords.enumerated() {
                    try Task.checkCancellation()
                    progressMessage = "Running round \(record.roundIndex) · recording \(index + 1) of \(sessionRecords.count)"
                    let result = try await service.predict(record)
                    try Task.checkCancellation()
                    results.append(result)
                    predictions[record.id] = result
                    completedCount = index + 1
                }
                progressMessage = "Applying the Hybrid session adjustment…"
                let final = try await service.finalize(sessionRecords, predictions: results)
                try Task.checkCancellation()
                predictions = Dictionary(uniqueKeysWithValues: final.map { ($0.recordID, $0) })
                sessionMilliseconds = (ProcessInfo.processInfo.systemUptime - start) * 1000
                progressMessage = "All \(sessionRecords.count) recordings processed on this device."
            } catch is CancellationError {
                progressMessage = "Stopped. Run the session to complete the Hybrid adjustment."
            } catch {
                errorMessage = "Inference stopped. \(error.localizedDescription)"
                progressMessage = "The session could not finish. Completed records remain available."
            }
        }
    }

    func cancel() {
        runTask?.cancel()
        progressMessage = "Stopping after the current recording…"
    }
}
