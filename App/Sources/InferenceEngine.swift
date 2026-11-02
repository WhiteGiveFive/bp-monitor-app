import Foundation
import CoreML

struct InferenceConfiguration: Decodable, Sendable {
    let bpCenter: [Float]
    let bpScale: Float
    let neuralWeight: Double
    let roundOffsets: [String: [Double]]

    func validate() throws {
        guard bpCenter.count == 2, bpCenter.allSatisfy(\.isFinite),
              bpScale.isFinite, bpScale > 0,
              neuralWeight.isFinite, (0...1).contains(neuralWeight),
              Set(roundOffsets.keys) == Set((1...5).map(String.init)),
              roundOffsets.values.allSatisfy({ $0.count == 2 && $0.allSatisfy(\.isFinite) }) else {
            throw DemonstratorError.resource("The bundled inference configuration is invalid.")
        }
    }
}

struct BloodPressure: Codable, Sendable, Equatable {
    let sbp: Double
    let dbp: Double

    init(_ pair: [Double]) throws {
        guard pair.count == 2, pair.allSatisfy(\.isFinite) else {
            throw DemonstratorError.inference("The model returned an invalid blood-pressure pair.")
        }
        sbp = pair[0]
        dbp = pair[1]
    }

    var values: [Double] { [sbp, dbp] }
}

struct TimedPrediction: Codable, Sendable {
    let pressure: BloodPressure
    let milliseconds: Double
    var resources: ResourceUsage? = nil
}

struct RecordPrediction: Codable, Sendable {
    let recordID: String
    let lightGBM: TimedPrediction
    let fusionNet: TimedPrediction
    let meanBeats: TimedPrediction
    let rawBlend: BloodPressure
    var hybrid: BloodPressure?
    var hybridRoundCount: Int?
    let totalMilliseconds: Double
    var blendResources: ResourceUsage? = nil
}

/// Pure Foundation/Core ML runtime shared by the iPhone app and macOS parity CLI.
/// The prepared arrays use C-order layout; every neural operation is float32.
/// Core ML uses CPU only initially, to validate parity before accelerator tuning.
final class InferenceEngine {
    let configuration: InferenceConfiguration
    let loadMilliseconds: Double
    let loadProfiles: [ModelLoadProfile]
    private(set) var sessionAdjustmentResources: ResourceUsage?
    private let trees: TreeEnsemble
    private let fusionModels: [MLModel]
    private let meanModels: [MLModel]

    init(modelsURL: URL) throws {
        let start = ProcessInfo.processInfo.systemUptime
        configuration = try JSONDecoder().decode(InferenceConfiguration.self,
            from: Data(contentsOf: modelsURL.appendingPathComponent("inference_config.json")))
        try configuration.validate()
        let treesURL = modelsURL.appendingPathComponent("trees.json")
        let (loadedTrees, treeResources) = try ResourceProbe.measure(label: "Load LightGBM") { probe in
            let loaded = try TreeEnsemble(url: treesURL)
            probe.checkpoint()
            return loaded
        }
        trees = loadedTrees
        let options = MLModelConfiguration()
        options.computeUnits = .cpuOnly
        func loadModels(_ prefix: String, probe: ResourceProbe) throws -> [MLModel] {
            try (1000...1004).map { seed in
                let url = modelsURL.appendingPathComponent("\(prefix)_seed\(seed).mlmodelc")
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw DemonstratorError.resource("Missing model: \(url.lastPathComponent).")
                }
                let model = try MLModel(contentsOf: url, configuration: options)
                probe.checkpoint()
                return model
            }
        }
        let (loadedFusion, fusionResources) = try ResourceProbe.measure(label: "Load FusionNet") { probe in
            try loadModels("fusion", probe: probe)
        }
        fusionModels = loadedFusion
        let (loadedMean, meanResources) = try ResourceProbe.measure(label: "Load Mean Beats") { probe in
            try loadModels("meanbeats", probe: probe)
        }
        meanModels = loadedMean
        loadMilliseconds = (ProcessInfo.processInfo.systemUptime - start) * 1000
        loadProfiles = [
            ModelLoadProfile(model: .lightGBM, resources: treeResources,
                diskBytes: Self.logicalDiskBytes(at: [treesURL])),
            ModelLoadProfile(model: .fusionNet, resources: fusionResources,
                diskBytes: Self.logicalDiskBytes(at: (1000...1004).map {
                    modelsURL.appendingPathComponent("fusion_seed\($0).mlmodelc")
                })),
            ModelLoadProfile(model: .meanBeats, resources: meanResources,
                diskBytes: Self.logicalDiskBytes(at: (1000...1004).map {
                    modelsURL.appendingPathComponent("meanbeats_seed\($0).mlmodelc")
                })),
        ]
    }

    func predict(record: PreparedRecording) throws -> RecordPrediction {
        sessionAdjustmentResources = nil
        try record.validate()
        let start = ProcessInfo.processInfo.systemUptime
        let (gbm, gbmResources) = try ResourceProbe.measure(label: "LightGBM inference") { _ in
            try BloodPressure(trees.predict(features: record.gbmFeatures))
        }
        let (fusion, fusionResources) = try ResourceProbe.measure(label: "FusionNet inference") { probe in
            try neuralPrediction(record: record, models: fusionModels, meanBeats: false, probe: probe)
        }
        let (mean, meanResources) = try ResourceProbe.measure(label: "Mean Beats inference") { probe in
            try neuralPrediction(record: record, models: meanModels, meanBeats: true, probe: probe)
        }
        let (blend, blendResources) = try ResourceProbe.measure(label: "Hybrid blend") { _ in
            try BloodPressure(zip(gbm.values, mean.values).map {
                (1 - configuration.neuralWeight) * $0.0 + configuration.neuralWeight * $0.1
            })
        }
        return RecordPrediction(recordID: record.id,
            lightGBM: TimedPrediction(pressure: gbm, milliseconds: gbmResources.wallMilliseconds, resources: gbmResources),
            fusionNet: TimedPrediction(pressure: fusion, milliseconds: fusionResources.wallMilliseconds, resources: fusionResources),
            meanBeats: TimedPrediction(pressure: mean, milliseconds: meanResources.wallMilliseconds, resources: meanResources),
            rawBlend: blend, hybrid: nil, hybridRoundCount: nil,
            totalMilliseconds: elapsed(start), blendResources: blendResources)
    }

    /// Apply the frozen subject-level correction only after all imported rounds
    /// have predictions. Each record gets its subject's mean blend plus its
    /// round offset, centered across that subject's imported rounds.
    func finalizeSession(records: [PreparedRecording], predictions: [RecordPrediction]) throws -> [RecordPrediction] {
        sessionAdjustmentResources = nil
        let (result, resources) = try ResourceProbe.measure(label: "Hybrid session adjustment") { probe in
            try finalizeSessionUnmeasured(records: records, predictions: predictions, probe: probe)
        }
        sessionAdjustmentResources = resources
        return result
    }

    private func finalizeSessionUnmeasured(records: [PreparedRecording], predictions: [RecordPrediction],
                                          probe: ResourceProbe) throws -> [RecordPrediction] {
        guard predictions.count == records.count,
              Set(predictions.map(\.recordID)).count == predictions.count,
              Set(predictions.map(\.recordID)) == Set(records.map(\.id)) else {
            throw DemonstratorError.inference("Hybrid needs a prediction for every imported recording.")
        }
        let lookup = Dictionary(uniqueKeysWithValues: predictions.map { ($0.recordID, $0) })
        let subjects = Dictionary(grouping: records, by: \.subjectID)
        var final: [String: RecordPrediction] = [:]
        for subjectRecords in subjects.values {
            var blendSum = [Double](repeating: 0, count: 2)
            var offsetSum = [Double](repeating: 0, count: 2)
            for record in subjectRecords {
                guard let prediction = lookup[record.id], let offset = configuration.roundOffsets[String(record.roundIndex)] else {
                    throw DemonstratorError.inference("The session contains an unsupported recording or round.")
                }
                for target in 0..<2 {
                    blendSum[target] += prediction.rawBlend.values[target]
                    offsetSum[target] += offset[target]
                }
            }
            let count = Double(subjectRecords.count)
            for record in subjectRecords {
                guard var prediction = lookup[record.id], let offset = configuration.roundOffsets[String(record.roundIndex)] else {
                    throw DemonstratorError.inference("Unable to match a Hybrid round.")
                }
                prediction.hybrid = try BloodPressure((0..<2).map {
                    blendSum[$0] / count + offset[$0] - offsetSum[$0] / count
                })
                prediction.hybridRoundCount = subjectRecords.count
                final[record.id] = prediction
            }
        }
        let result = try records.map {
            guard let result = final[$0.id] else { throw DemonstratorError.inference("Missing final Hybrid prediction.") }
            return result
        }
        probe.checkpoint()
        return result
    }

    private func neuralPrediction(record: PreparedRecording, models: [MLModel], meanBeats: Bool,
                                  probe: ResourceProbe) throws -> BloodPressure {
        let inputs = try record.views.map { try featureProvider(record: record, view: $0, meanBeats: meanBeats) }
        probe.checkpoint()
        var seedPredictions = [[Float]]()
        for model in models {
            var viewSum: [Float] = [0, 0]
            for input in inputs {
                let output = try model.prediction(from: input)
                probe.checkpoint()
                guard let array = output.featureValue(for: "prediction")?.multiArrayValue,
                      array.shape.map(\.intValue) == [1, 2] else {
                    throw DemonstratorError.inference("The neural model returned an unexpected output shape.")
                }
                for target in 0..<2 {
                    let value = array[[0, NSNumber(value: target)]].floatValue
                    guard value.isFinite else { throw DemonstratorError.inference("The neural model returned a nonfinite value.") }
                    viewSum[target] += value
                }
            }
            seedPredictions.append((0..<2).map {
                (viewSum[$0] / Float(inputs.count)) * configuration.bpScale + configuration.bpCenter[$0]
            })
        }
        // Both pipelines average views and unscale per seed in float32.
        // Mean Beats then explicitly promotes the across-seed mean to float64;
        // basic FusionNet's NumPy stack/mean retains its float32 dtype.
        return try BloodPressure((0..<2).map { target in
            if meanBeats {
                return seedPredictions.reduce(0.0) { $0 + Double($1[target]) } / Double(seedPredictions.count)
            }
            let total: Float = seedPredictions.reduce(Float(0)) { $0 + $1[target] }
            return Double(total / Float(seedPredictions.count))
        })
    }

    private func featureProvider(record: PreparedRecording, view: PreparedView, meanBeats: Bool) throws -> MLDictionaryFeatureProvider {
        var arrays: [String: MLMultiArray] = [
            "tab": try array(record.tab, shape: [1, 309]),
            "ppg": try array(view.ppg, shape: [1, 16, 120]),
            "ecg": try array(view.ecg, shape: [1, 5, 120]),
            "acc": try array(view.acc, shape: [1, 3, 120]),
            "acc_sc": try array(record.accSc, shape: [1, 7]),
        ]
        let available: Float = record.ecgOK ? 1 : 0
        if meanBeats {
            arrays["present"] = try array([1, 1, available, available, available], shape: [1, 5])
            arrays["beats"] = try array(record.beats, shape: [1, 32, 5, 120])
            arrays["valid"] = try array(record.beatValid.map { $0 ? 1 : 0 }, shape: [1, 32])
            arrays["beat_present"] = try array([record.beatPresent ? 1 : 0], shape: [1])
        } else {
            arrays["ecg_ok"] = try array([available], shape: [1, 1])
        }
        return try MLDictionaryFeatureProvider(dictionary: arrays)
    }

    private func array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        guard shape.reduce(1, *) == values.count else { throw DemonstratorError.inference("Invalid tensor dimensions.") }
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let destination = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { source in
            if let address = source.baseAddress { destination.update(from: address, count: source.count) }
        }
        return array
    }

    private func elapsed(_ start: Double) -> Double { (ProcessInfo.processInfo.systemUptime - start) * 1000 }

    /// Sum regular-file logical sizes, including every file inside compiled
    /// model directories. This is distinct from allocated disk space or RAM.
    private static func logicalDiskBytes(at urls: [URL]) -> UInt64? {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey, .fileSizeKey]
        var total: UInt64 = 0
        func addFile(_ url: URL) throws -> Bool {
            let values = try url.resourceValues(forKeys: keys)
            guard values.isRegularFile == true else { return true }
            guard let size = values.fileSize, size >= 0 else { return false }
            let (sum, overflow) = total.addingReportingOverflow(UInt64(size))
            guard !overflow else { return false }
            total = sum
            return true
        }
        do {
            for url in urls {
                let values = try url.resourceValues(forKeys: keys)
                if values.isDirectory == true {
                    var failed = false
                    guard let files = FileManager.default.enumerator(at: url,
                        includingPropertiesForKeys: Array(keys), options: [], errorHandler: { _, _ in
                            failed = true
                            return false
                        }) else { return nil }
                    for case let fileURL as URL in files {
                        guard try addFile(fileURL) else { return nil }
                    }
                    guard !failed else { return nil }
                } else {
                    guard values.isRegularFile == true, try addFile(url) else { return nil }
                }
            }
            return total
        } catch {
            return nil
        }
    }
}
