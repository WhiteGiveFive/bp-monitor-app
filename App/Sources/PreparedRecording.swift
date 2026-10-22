import Foundation

enum DemonstratorError: LocalizedError {
    case invalidInput(String)
    case resource(String)
    case inference(String)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let message), .resource(let message), .inference(let message): return message
        }
    }
}

struct PreparedView: Codable, Sendable {
    let ppg: [Float]
    let ecg: [Float]
    let acc: [Float]
}

struct RecordingDisplay: Codable, Sendable {
    let age: Double?
    let sex: String?
    let ppg: [Double]
    let ecg: [Double]
    let durationSeconds: Double
}

/// Inputs are already prepared with the frozen Python signal-processing pipeline.
/// Waveform previews are display-only and never become inference inputs.
struct PreparedRecording: Codable, Identifiable, Sendable {
    let schemaVersion: Int
    let id: String
    let subjectID: String
    let roundIndex: Int
    let display: RecordingDisplay
    let gbmFeatures: [Double?]
    let tab: [Float]
    let accSc: [Float]
    let ecgOK: Bool
    let views: [PreparedView]
    let beats: [Float]
    let beatValid: [Bool]
    let beatPresent: Bool

    func validate() throws {
        guard schemaVersion == 1 else { throw DemonstratorError.invalidInput("Unsupported recording schema version \(schemaVersion).") }
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !subjectID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              id.count <= 256, subjectID.count <= 256 else {
            throw DemonstratorError.invalidInput("Each recording needs a nonempty ID and subject ID, at most 256 characters long.")
        }
        guard (1...5).contains(roundIndex) else { throw DemonstratorError.invalidInput("\(id): roundIndex must be between 1 and 5.") }
        guard gbmFeatures.count == 161, gbmFeatures.allSatisfy({ $0?.isFinite ?? true }) else {
            throw DemonstratorError.invalidInput("\(id): LightGBM expects exactly 161 finite values or nulls.")
        }
        try requireArray(tab, count: 309, name: "tab")
        try requireArray(accSc, count: 7, name: "accSc")
        guard views.count == 5 else { throw DemonstratorError.invalidInput("\(id): exactly five prepared waveform views are required.") }
        for (index, view) in views.enumerated() {
            try requireArray(view.ppg, count: 16 * 120, name: "views[\(index)].ppg")
            try requireArray(view.ecg, count: 5 * 120, name: "views[\(index)].ecg")
            try requireArray(view.acc, count: 3 * 120, name: "views[\(index)].acc")
        }
        try requireArray(beats, count: 32 * 5 * 120, name: "beats")
        guard beatValid.count == 32 else { throw DemonstratorError.invalidInput("\(id): beatValid must contain 32 booleans.") }
        guard !beatPresent || beatValid.contains(true) else {
            throw DemonstratorError.invalidInput("\(id): present beats require at least one valid beat.")
        }
        guard !beatPresent || ecgOK else {
            throw DemonstratorError.invalidInput("\(id): aligned beats require a usable ECG reference.")
        }
        guard display.durationSeconds.isFinite, display.durationSeconds > 0,
              display.durationSeconds <= 3600,
              display.age.map({ $0.isFinite && $0 > 0 && $0 < 150 }) ?? true,
              display.sex.map({ $0.count <= 64 }) ?? true else {
            throw DemonstratorError.invalidInput("\(id): invalid display metadata.")
        }
        for (name, values) in [("PPG", display.ppg), ("ECG", display.ecg)] {
            guard (2...30000).contains(values.count), values.allSatisfy(\.isFinite) else {
                throw DemonstratorError.invalidInput("\(id): the \(name) preview must contain 2–30,000 finite samples.")
            }
        }
    }

    private func requireArray(_ values: [Float], count: Int, name: String) throws {
        guard values.count == count, values.allSatisfy(\.isFinite) else {
            throw DemonstratorError.invalidInput("\(id): \(name) must contain exactly \(count) finite values.")
        }
    }
}

struct PreparedSession: Codable, Sendable {
    let schemaVersion: Int
    let records: [PreparedRecording]

    static func load(url: URL) throws -> PreparedSession {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let maximumBytes = 64 * 1024 * 1024
        let data = try handle.read(upToCount: maximumBytes + 1) ?? Data()
        guard !data.isEmpty, data.count <= maximumBytes else {
            throw DemonstratorError.invalidInput("Choose a nonempty prepared JSON file smaller than 64 MB.")
        }
        return try decode(data)
    }

    static func decode(_ data: Data) throws -> PreparedSession {
        let decoder = JSONDecoder()
        let envelope = try decoder.decode(Envelope.self, from: data)
        let session: PreparedSession
        if envelope.records != nil {
            session = try decoder.decode(PreparedSession.self, from: data)
        } else {
            session = PreparedSession(schemaVersion: 1, records: [try decoder.decode(PreparedRecording.self, from: data)])
        }
        try session.validate()
        return session
    }

    func validate() throws {
        guard schemaVersion == 1 else { throw DemonstratorError.invalidInput("Unsupported session schema version \(schemaVersion).") }
        guard (1...100).contains(records.count) else { throw DemonstratorError.invalidInput("A prepared session must contain 1–100 recordings.") }
        guard Set(records.map(\.id)).count == records.count else { throw DemonstratorError.invalidInput("The session contains duplicate recording IDs.") }
        var rounds = Set<String>()
        for record in records {
            try record.validate()
            let key = "\(record.subjectID)\u{001f}\(record.roundIndex)"
            guard rounds.insert(key).inserted else {
                throw DemonstratorError.invalidInput("The session contains more than one recording for \(record.subjectID), round \(record.roundIndex).")
            }
        }
    }

    private struct Envelope: Decodable { let records: [IgnoredValue]? }
    private struct IgnoredValue: Decodable {}
}
