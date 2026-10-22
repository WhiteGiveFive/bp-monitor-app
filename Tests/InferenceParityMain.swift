import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

private enum ParityFailure: Error, LocalizedError {
    case failed(String)
    var errorDescription: String? {
        switch self { case .failed(let reason): return reason }
    }
}

@main
struct InferenceParityMain {
    private static let tolerance = 0.005
    private static let modelNames = ["lightGBM", "fusionNet", "meanBeats", "hybrid"]

    static func main() {
        guard CommandLine.arguments.count == 4 else {
            fail("Usage: inference-parity ModelsFolder SampleSessionJSON OutputJSON")
        }
        let modelsURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let sessionURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
        do {
            let report = try run(modelsURL: modelsURL, sessionURL: sessionURL)
            try write(report, to: outputURL)
            let status = report["status"] as? String ?? "failed"
            print("\(status): native inference parity report -> \(outputURL.path)")
            if status != "passed" {
                fail(report["failure"] as? String ?? "Native predictions exceeded the parity tolerance")
            }
        } catch {
            let report: [String: Any] = ["status": "failed", "failure": error.localizedDescription,
                                         "absoluteToleranceMMHg": tolerance]
            do { try write(report, to: outputURL) }
            catch { fail("Could not write failure report: \(error.localizedDescription)") }
            fail(error.localizedDescription)
        }
    }

    private static func run(modelsURL: URL, sessionURL: URL) throws -> [String: Any] {
        let start = ProcessInfo.processInfo.systemUptime
        let session = try PreparedSession.load(url: sessionURL)
        // Read historical references independently; PreparedRecording deliberately
        // ignores them and the production inference engine never receives them.
        let sourceObjects = try recordingObjects(from: Data(contentsOf: sessionURL))
        let references = try parseReferences(sourceObjects)
        guard Set(references.keys) == Set(session.records.map(\.id)) else {
            throw ParityFailure.failed("Historical reference IDs do not exactly match the imported recordings")
        }
        let validationChecks = try checkValidation(record: session.records[0], source: sourceObjects[0])
        let engine = try InferenceEngine(modelsURL: modelsURL)
        let mathChecks = try checkSessionMath(engine: engine, template: session.records[0])
        let raw = try session.records.map { try engine.predict(record: $0) }
        let final = try engine.finalizeSession(records: session.records, predictions: raw)
        var maximumErrors = Dictionary(uniqueKeysWithValues: modelNames.map { ($0, 0.0) })
        var comparisons = [[String: Any]]()
        for result in final {
            guard let reference = references[result.recordID], let hybrid = result.hybrid else {
                throw ParityFailure.failed("Missing reference or finalized Hybrid value for \(result.recordID)")
            }
            let actual = ["lightGBM": result.lightGBM.pressure.values,
                          "fusionNet": result.fusionNet.pressure.values,
                          "meanBeats": result.meanBeats.pressure.values,
                          "hybrid": hybrid.values]
            var errors = [String: [Double]]()
            for name in modelNames {
                guard let predicted = actual[name], let expected = reference[name] else {
                    throw ParityFailure.failed("Missing \(name) comparison for \(result.recordID)")
                }
                let differences = zip(predicted, expected).map { abs($0.0 - $0.1) }
                errors[name] = differences
                maximumErrors[name] = max(maximumErrors[name] ?? 0, differences.max() ?? 0)
            }
            comparisons.append(["recordID": result.recordID, "reference": reference,
                                "absoluteErrorMMHg": errors])
        }
        let encoder = JSONEncoder()
        let encodedPredictions = try JSONSerialization.jsonObject(with: encoder.encode(final))
        let passing = maximumErrors.values.allSatisfy { $0 <= tolerance }
        var report: [String: Any] = [
            "status": passing ? "passed" : "failed",
            "records": session.records.count,
            "subjects": Set(session.records.map(\.subjectID)).count,
            "missingECGRecords": session.records.filter { !$0.ecgOK }.count,
            "absoluteToleranceMMHg": tolerance,
            "maximumAbsoluteErrorMMHg": maximumErrors,
            "referenceKind": "historical_saved_predictions_not_ground_truth",
            "computeUnits": "cpuOnly", "modelLoadMilliseconds": engine.loadMilliseconds,
            "totalMilliseconds": (ProcessInfo.processInfo.systemUptime - start) * 1000,
            "inputValidationChecks": validationChecks, "sessionMathChecks": mathChecks,
            "predictions": encodedPredictions, "comparisons": comparisons,
        ]
        if !passing {
            let failures = maximumErrors.filter { $0.value > tolerance }
                .sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value) mmHg" }.joined(separator: ", ")
            report["failure"] = "Historical reference tolerance exceeded: \(failures)"
        }
        return report
    }

    private static func recordingObjects(from data: Data) throws -> [[String: Any]] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ParityFailure.failed("The sample JSON root must be an object")
        }
        if object["records"] != nil {
            guard let records = object["records"] as? [[String: Any]], !records.isEmpty else {
                throw ParityFailure.failed("The sample session records must be a nonempty object array")
            }
            return records
        }
        return [object]
    }

    private static func parseReferences(_ objects: [[String: Any]]) throws -> [String: [String: [Double]]] {
        var result = [String: [String: [Double]]]()
        for object in objects {
            guard let id = object["id"] as? String, result[id] == nil,
                  let reference = object["reference"] as? [String: Any] else {
                throw ParityFailure.failed("Every sample requires a unique ID and historical reference object")
            }
            var modelReferences = [String: [Double]]()
            for name in modelNames {
                guard let numbers = reference[name] as? [NSNumber], numbers.count == 2 else {
                    throw ParityFailure.failed("\(id): missing or invalid \(name) [SBP, DBP] reference")
                }
                let pair = numbers.map(\.doubleValue)
                guard pair.allSatisfy(\.isFinite) else {
                    throw ParityFailure.failed("\(id): nonfinite \(name) historical reference")
                }
                modelReferences[name] = pair
            }
            result[id] = modelReferences
        }
        return result
    }

    private static func checkValidation(record: PreparedRecording, source: [String: Any]) throws -> [String] {
        var checks = [String]()
        func rejects(_ name: String, _ body: () throws -> Void) throws {
            do { try body() }
            catch DemonstratorError.invalidInput {
                checks.append(name)
                return
            }
            catch { throw ParityFailure.failed("\(name): unexpected validation error: \(error.localizedDescription)") }
            throw ParityFailure.failed("\(name): invalid input was accepted")
        }
        try rejects("duplicate_recording_ids") {
            let data = try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "records": [source, source]])
            _ = try PreparedSession.decode(data)
        }
        try rejects("duplicate_subject_rounds") {
            var other = source
            other["id"] = record.id + "_duplicate_round"
            let data = try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "records": [source, other]])
            _ = try PreparedSession.decode(data)
        }
        try rejects("wrong_feature_count") {
            var other = source
            other["gbmFeatures"] = Array(record.gbmFeatures.dropLast()).map { value -> Any in
                if let value { return value }
                return NSNull()
            }
            let data = try JSONSerialization.data(withJSONObject: other)
            _ = try PreparedSession.decode(data)
        }
        try rejects("wrong_tabular_count") {
            try copy(record, tab: Array(record.tab.dropLast())).validate()
        }
        try rejects("nonfinite_neural_input") {
            var tab = record.tab
            tab[0] = .nan
            try copy(record, tab: tab).validate()
        }
        try rejects("nonfinite_gbm_recording_input") {
            var features = record.gbmFeatures
            features[0] = Double.nan
            try copy(record, gbmFeatures: features).validate()
        }
        try rejects("unsupported_recording_version") {
            var other = source
            other["schemaVersion"] = 999
            _ = try PreparedSession.decode(JSONSerialization.data(withJSONObject: other))
        }
        try rejects("beats_without_ecg") {
            var other = source
            other["ecgOK"] = false
            other["beatPresent"] = true
            other["beatValid"] = Array(repeating: true, count: 32)
            _ = try PreparedSession.decode(JSONSerialization.data(withJSONObject: other))
        }
        return checks
    }

    private static func checkSessionMath(engine: InferenceEngine, template: PreparedRecording) throws -> [String] {
        func prediction(_ id: String, _ pair: [Double]) throws -> RecordPrediction {
            let pressure = try BloodPressure(pair)
            let timed = TimedPrediction(pressure: pressure, milliseconds: 0)
            return RecordPrediction(recordID: id, lightGBM: timed, fusionNet: timed, meanBeats: timed,
                rawBlend: pressure, hybrid: nil, hybridRoundCount: nil, totalMilliseconds: 0)
        }
        func equal(_ actual: [Double], _ expected: [Double], _ label: String) throws {
            guard actual.count == 2, zip(actual, expected).allSatisfy({ abs($0.0 - $0.1) <= 1e-12 }) else {
                throw ParityFailure.failed("\(label): expected \(expected), received \(actual)")
            }
        }
        let one = copy(template, id: "analytic_singleton", subjectID: "analytic_a", roundIndex: 3)
        let single = try engine.finalizeSession(records: [one], predictions: [prediction(one.id, [123, 78])])
        guard let singleHybrid = single[0].hybrid, single[0].hybridRoundCount == 1 else {
            throw ParityFailure.failed("Singleton Hybrid did not preserve its session size")
        }
        try equal(singleHybrid.values, [123, 78], "singleton round-offset cancellation")

        let first = copy(template, id: "analytic_a1", subjectID: "analytic_a", roundIndex: 1)
        let third = copy(template, id: "analytic_a3", subjectID: "analytic_a", roundIndex: 3)
        let other = copy(template, id: "analytic_b5", subjectID: "analytic_b", roundIndex: 5)
        let records = [third, other, first]
        let predictions = try [prediction(first.id, [110, 70]), prediction(third.id, [130, 90]),
                               prediction(other.id, [140, 95])]
        let final = try engine.finalizeSession(records: records, predictions: predictions)
        guard final.map(\.recordID) == records.map(\.id),
              let offsetOne = engine.configuration.roundOffsets["1"],
              let offsetThree = engine.configuration.roundOffsets["3"] else {
            throw ParityFailure.failed("Incomplete-session output order or round-offset metadata is invalid")
        }
        let values = Dictionary(uniqueKeysWithValues: final.map { ($0.recordID, $0) })
        for (record, offset) in [(first, offsetOne), (third, offsetThree)] {
            guard let output = values[record.id], let pressure = output.hybrid, output.hybridRoundCount == 2 else {
                throw ParityFailure.failed("Missing two-round Hybrid result for \(record.id)")
            }
            let base: [Double] = [120.0, 80.0]
            var expected = [Double]()
            for target in 0..<2 {
                let centered = offset[target] - (offsetOne[target] + offsetThree[target]) / 2.0
                expected.append(base[target] + centered)
            }
            try equal(pressure.values, expected, "incomplete-session recentered offset")
        }
        guard let otherHybrid = values[other.id]?.hybrid else {
            throw ParityFailure.failed("Missing second-subject Hybrid result")
        }
        try equal(otherHybrid.values, [140, 95], "subject isolation")
        do {
            _ = try engine.finalizeSession(records: records, predictions: Array(predictions.dropLast()))
            throw ParityFailure.failed("Finalization accepted an incomplete prediction inventory")
        } catch DemonstratorError.inference { }
        return ["singleton_offset_cancellation", "incomplete_rounds_recenter_offsets",
                "subject_isolation", "record_order_preserved", "missing_predictions_rejected"]
    }

    private static func copy(_ record: PreparedRecording, id: String? = nil, subjectID: String? = nil,
                             roundIndex: Int? = nil, tab: [Float]? = nil,
                             gbmFeatures: [Double?]? = nil) -> PreparedRecording {
        PreparedRecording(schemaVersion: record.schemaVersion, id: id ?? record.id,
            subjectID: subjectID ?? record.subjectID, roundIndex: roundIndex ?? record.roundIndex,
            display: record.display, gbmFeatures: gbmFeatures ?? record.gbmFeatures,
            tab: tab ?? record.tab, accSc: record.accSc, ecgOK: record.ecgOK, views: record.views,
            beats: record.beats, beatValid: record.beatValid, beatPresent: record.beatPresent)
    }

    private static func write(_ report: [String: Any], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        exit(1)
    }
}
