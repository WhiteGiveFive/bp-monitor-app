import Foundation

private struct CheckFailure: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

@main struct ResourceReportMain {
    static func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() { throw CheckFailure(message: message) }
    }

    static func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) < 1e-7 }

    static func usage(wall: Double = 1, cpu: Double? = 2, before: UInt64? = 100,
                      after: UInt64? = 80, peak: UInt64? = 120) -> ResourceUsage {
        ResourceUsage(wallMilliseconds: wall, cpuMilliseconds: cpu, memoryBeforeBytes: before,
            memoryAfterBytes: after, observedPeakMemoryBytes: peak, memorySampleCount: 2)
    }

    static func checkProbeAndCombination() throws {
        let a = usage(), b = usage(wall: 3, cpu: 4, before: 80, after: 150, peak: 160)
        let combined = ResourceUsage.combined([a, b])
        try require(combined.wallMilliseconds == 4 && combined.cpuMilliseconds == 6, "Durations must add")
        try require(combined.observedPeakMemoryBytes == 160, "Memory peaks must use maximum, not sum")
        try require(a.memoryDeltaBytes == -20, "Memory decreases must remain signed")
        try require(ResourceUsage.combined([a, usage(cpu: nil)]).cpuMilliseconds == nil,
                    "Unavailable CPU must not become zero")
        try require(usage(before: nil).memoryDeltaBytes == nil, "Unavailable memory must remain unavailable")
        try require(usage(before: UInt64.max, after: 0).memoryDeltaBytes == nil, "Overflow must not trap")
        let probe = ResourceProbe(label: "Resource report test")
        probe.checkpoint()
        let first = probe.finish()
        probe.checkpoint()
        let second = probe.finish()
        try require(first.wallMilliseconds == second.wallMilliseconds && first.memorySampleCount == second.memorySampleCount,
                    "Finishing a probe must be idempotent")
        try require(first.cpuMilliseconds != nil && first.memoryBeforeBytes != nil,
                    "Darwin CPU and footprint APIs must produce readings on this host")
        do {
            _ = try ResourceProbe.measure(label: "Throwing operation") { _ -> Int in
                throw CheckFailure(message: "expected")
            }
            throw CheckFailure(message: "Measured errors must propagate")
        } catch let error as CheckFailure {
            try require(error.message == "expected", "Measured errors must preserve their cause")
        }
    }

    static func main() throws {
        guard CommandLine.arguments.count == 4 else {
            throw CheckFailure(message: "Usage: resource-report ModelsFolder SampleSessionJSON OutputJSON")
        }
        try checkProbeAndCombination()
        let records = try PreparedSession.load(url: URL(fileURLWithPath: CommandLine.arguments[2])).records
        let clock = ProcessInfo.processInfo.systemUptime
        let started = Date()
        let engine = try InferenceEngine(modelsURL: URL(fileURLWithPath: CommandLine.arguments[1]))
        let raw = try records.map { try engine.predict(record: $0) }
        let final = try engine.finalizeSession(records: records, predictions: raw)
        let wall = (ProcessInfo.processInfo.systemUptime - clock) * 1000
        let environment = ResourceEnvironment(platform: "macOS validation", deviceModel: "test host",
            systemVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            processorCount: ProcessInfo.processInfo.processorCount, isSimulator: false, buildConfiguration: "Release")
        func snapshot(_ level: Double?, _ state: String) -> DeviceResourceSnapshot {
            DeviceResourceSnapshot(capturedAt: Date(), batteryLevel: level, batteryState: state,
                                   thermalState: "Nominal", lowPowerMode: false)
        }
        func report(start: DeviceResourceSnapshot, end: DeviceResourceSnapshot,
                    env: ResourceEnvironment = environment) throws -> SessionResourceReport {
            try SessionResourceReport.make(predictions: final, loadProfiles: engine.loadProfiles,
                adjustment: engine.sessionAdjustmentResources, startedAt: started, finishedAt: Date(),
                sessionWallMilliseconds: wall, modelsWereAlreadyLoaded: false,
                environment: env, start: start, end: end)
        }
        let measured = try report(start: snapshot(nil, "Unavailable"), end: snapshot(nil, "Unavailable"))
        try require(measured.models.count == 4 && measured.samples.count == records.count, "Every model and recording must be reported")
        try require(!measured.perModelEnergyAvailable, "Do not claim per-model energy measurement")
        for summary in measured.models {
            try require(summary.total.wallMilliseconds > 0, "Wall time must be measured")
            try require((summary.total.cpuMilliseconds ?? -1) >= 0, "CPU time must be measured")
            try require((summary.total.observedPeakMemoryBytes ?? 0) > 0, "Memory footprint must be measured")
            try require((summary.load?.diskBytes ?? 0) > 0, "Model disk sizes must be measured")
            try require(close(summary.meanWallMilliseconds * Double(records.count), summary.total.wallMilliseconds),
                        "Amortized per-record wall time must match total")
        }
        for sample in measured.samples {
            try require(sample.fusionNet.memorySampleCount >= 28 && sample.meanBeats.memorySampleCount >= 28,
                        "Neural calls must sample input buffers and all 25 predictions")
        }
        let models = Dictionary(uniqueKeysWithValues: measured.models.map { ($0.model, $0) })
        guard let hybrid = models[.hybrid], let gbm = models[.lightGBM], let mean = models[.meanBeats] else {
            throw CheckFailure(message: "Missing model summaries")
        }
        let blendWall = measured.samples.reduce(0) { $0 + $1.hybridBlend.wallMilliseconds }
        let expectedWall = gbm.total.wallMilliseconds + mean.total.wallMilliseconds + blendWall + measured.hybridAdjustment.wallMilliseconds
        try require(close(hybrid.total.wallMilliseconds, expectedWall), "Hybrid must include dependencies and exactly one correction, without FusionNet")
        let expectedPeak = (measured.samples.flatMap { [$0.lightGBM, $0.meanBeats, $0.hybridBlend] } + [measured.hybridAdjustment])
            .compactMap(\.observedPeakMemoryBytes).max()
        try require(hybrid.total.observedPeakMemoryBytes == expectedPeak, "Hybrid peak must not sum memory")
        try require(hybrid.load?.resources.memoryDeltaBytes == nil, "Disjoint Hybrid loads must not include the intervening FusionNet memory delta")
        try require(hybrid.load?.diskBytes == (gbm.load?.diskBytes ?? 0) + (mean.load?.diskBytes ?? 0), "Hybrid assets must sum only dependencies")
        let same = try report(start: snapshot(0.8, "Unplugged"), end: snapshot(0.8, "Unplugged"))
        try require(same.batterySummary.contains("does not mean zero"), "Unchanged battery is not zero consumption")
        let charging = try report(start: snapshot(0.8, "Charging"), end: snapshot(0.9, "Charging"))
        try require(charging.batterySummary.contains("cannot describe"), "Charging changes must not be reported as consumption")
        let falling = try report(start: snapshot(0.8, "Unplugged"), end: snapshot(0.79, "Unplugged"))
        try require(falling.batterySummary.contains("1.0 percentage points"), "Battery fraction must convert to percentage points")
        let simulator = ResourceEnvironment(platform: "iOS Simulator", deviceModel: "test", systemVersion: "18.6",
            processorCount: 1, isSimulator: true, buildConfiguration: "Debug")
        let simulated = try report(start: snapshot(1, "Full"), end: snapshot(1, "Full"), env: simulator)
        try require(simulated.batterySummary.contains("unavailable in the simulator"), "Simulator must not claim battery observations")
        do {
            _ = try SessionResourceReport.make(predictions: raw, loadProfiles: engine.loadProfiles, adjustment: nil,
                startedAt: started, finishedAt: Date(), sessionWallMilliseconds: wall, modelsWereAlreadyLoaded: true,
                environment: environment, start: snapshot(nil, "Unknown"), end: snapshot(nil, "Unknown"))
            throw CheckFailure(message: "Incomplete sessions must not produce full reports")
        } catch is DemonstratorError { }
        let data = try measured.jsonData()
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(SessionResourceReport.self, from: data)
        try require(decoded.models.count == 4 && decoded.schemaVersion == 1, "Export must round-trip")
        let text = String(decoding: data, as: UTF8.self)
        try require(!text.contains("subject_") && !text.contains("sbp") && !text.contains("ppg"), "Resource export must omit clinical inputs and predictions")
        try data.write(to: URL(fileURLWithPath: CommandLine.arguments[3]), options: .atomic)
        print("passed: live resource readings, Hybrid accounting, battery interpretation, validation, and JSON export")
    }
}
