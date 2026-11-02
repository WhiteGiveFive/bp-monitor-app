import Foundation
import Darwin
#if canImport(UIKit)
import UIKit
#endif

struct ResourceEnvironment: Codable, Sendable {
    let platform: String
    let deviceModel: String
    let systemVersion: String
    let processorCount: Int
    let isSimulator: Bool
    let buildConfiguration: String

    @MainActor static func current() -> Self {
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif
        #if DEBUG
        let configuration = "Debug"
        #else
        let configuration = "Release"
        #endif
        #if canImport(UIKit)
        let device = UIDevice.current
        let platform = simulator ? "iOS Simulator" : "iOS device"
        let model = simulator ? (ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "Simulator") : hardwareModel()
        let version = device.systemVersion
        #else
        let platform = "macOS validation"
        let model = hardwareModel()
        let version = ProcessInfo.processInfo.operatingSystemVersionString
        #endif
        return Self(platform: platform, deviceModel: model, systemVersion: version,
                    processorCount: ProcessInfo.processInfo.processorCount,
                    isSimulator: simulator, buildConfiguration: configuration)
    }

    private static func hardwareModel() -> String {
        var size = 0
        guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.machine", &bytes, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: bytes)
    }
}

struct DeviceResourceSnapshot: Codable, Sendable {
    let capturedAt: Date
    /// Fraction of device charge, never an attribution to a model.
    let batteryLevel: Double?
    let batteryState: String
    let thermalState: String
    let lowPowerMode: Bool

    @MainActor static func current() -> Self {
        let process = ProcessInfo.processInfo
        let thermal: String
        switch process.thermalState {
        case .nominal: thermal = "Nominal"
        case .fair: thermal = "Fair"
        case .serious: thermal = "Serious"
        case .critical: thermal = "Critical"
        @unknown default: thermal = "Unknown"
        }
        var charge: Double?
        var state = "Unavailable"
        #if canImport(UIKit) && !targetEnvironment(simulator)
        let device = UIDevice.current
        let level = Double(device.batteryLevel)
        if (0...1).contains(level) { charge = level }
        switch device.batteryState {
        case .unplugged: state = "Unplugged"
        case .charging: state = "Charging"
        case .full: state = "Full"
        case .unknown: state = "Unknown"
        @unknown default: state = "Unknown"
        }
        #endif
        return Self(capturedAt: Date(), batteryLevel: charge, batteryState: state,
                    thermalState: thermal, lowPowerMode: process.isLowPowerModeEnabled)
    }
}

/// Monitoring is active only for a run; restore the preexisting setting on every exit path.
@MainActor final class DeviceResourceMonitor {
    #if canImport(UIKit)
    private var previousMonitoring: Bool?
    #endif

    func start() -> DeviceResourceSnapshot {
        #if canImport(UIKit) && !targetEnvironment(simulator)
        previousMonitoring = UIDevice.current.isBatteryMonitoringEnabled
        UIDevice.current.isBatteryMonitoringEnabled = true
        #endif
        return .current()
    }

    func stop() {
        #if canImport(UIKit)
        if let previousMonitoring {
            UIDevice.current.isBatteryMonitoringEnabled = previousMonitoring
            self.previousMonitoring = nil
        }
        #endif
    }
}

struct ModelResourceSummary: Codable, Sendable, Identifiable {
    var id: String { model.rawValue }
    let model: ModelKind
    let total: ResourceUsage
    let meanWallMilliseconds: Double
    let load: ModelLoadProfile?
    let notes: [String]
}

struct RecordResourceSample: Codable, Sendable {
    /// Report indexes deliberately omit recording IDs, waveforms and BP predictions.
    let recordingIndex: Int
    let lightGBM: ResourceUsage
    let fusionNet: ResourceUsage
    let meanBeats: ResourceUsage
    let hybridBlend: ResourceUsage
}

struct SessionResourceReport: Codable, Sendable, Identifiable {
    let schemaVersion: Int
    let id: UUID
    let startedAt: Date
    let finishedAt: Date
    let recordingCount: Int
    let sessionWallMilliseconds: Double
    let modelsWereAlreadyLoaded: Bool
    let computeUnits: String
    let environment: ResourceEnvironment
    let start: DeviceResourceSnapshot
    let end: DeviceResourceSnapshot
    let models: [ModelResourceSummary]
    let samples: [RecordResourceSample]
    let hybridAdjustment: ResourceUsage
    let perModelEnergyAvailable: Bool
    let notes: [String]

    var batterySummary: String {
        if environment.isSimulator { return "Battery measurement is unavailable in the simulator." }
        guard let before = start.batteryLevel, let after = end.batteryLevel else {
            return "Device battery readings are unavailable for this run."
        }
        guard start.batteryState == "Unplugged", end.batteryState == "Unplugged" else {
            return "Charging or unknown power state: charge change cannot describe battery consumption."
        }
        let drop = (before - after) * 100
        if abs(drop) < 0.01 {
            return "No charge-level change was detected. This does not mean zero energy was consumed."
        }
        if drop < 0 { return "The reported charge increased; battery consumption cannot be inferred from this run." }
        return String(format: "Device charge fell by %.1f percentage points across the whole session. This includes other device activity and cannot be assigned to individual models.", drop)
    }

    static func make(predictions: [RecordPrediction], loadProfiles: [ModelLoadProfile],
                     adjustment: ResourceUsage?, startedAt: Date, finishedAt: Date,
                     sessionWallMilliseconds: Double, modelsWereAlreadyLoaded: Bool,
                     environment: ResourceEnvironment, start: DeviceResourceSnapshot,
                     end: DeviceResourceSnapshot) throws -> Self {
        guard !predictions.isEmpty, let adjustment,
              predictions.allSatisfy({ $0.hybrid != nil }) else {
            throw DemonstratorError.inference("A resource report requires a complete session.")
        }
        let samples = try predictions.enumerated().map { index, prediction -> RecordResourceSample in
            guard let gbm = prediction.lightGBM.resources, let fusion = prediction.fusionNet.resources,
                  let mean = prediction.meanBeats.resources, let blend = prediction.blendResources else {
                throw DemonstratorError.inference("A model resource measurement is missing.")
            }
            return RecordResourceSample(recordingIndex: index + 1, lightGBM: gbm,
                                        fusionNet: fusion, meanBeats: mean, hybridBlend: blend)
        }
        let loads = Dictionary(uniqueKeysWithValues: loadProfiles.map { ($0.model, $0) })
        var summaries = [ModelResourceSummary]()
        for kind in ModelKind.allCases where kind != .hybrid {
            let usages = samples.map { sample -> ResourceUsage in
                switch kind {
                case .lightGBM: return sample.lightGBM
                case .fusionNet: return sample.fusionNet
                case .meanBeats: return sample.meanBeats
                case .hybrid: return sample.hybridBlend
                }
            }
            let total = ResourceUsage.combined(usages)
            summaries.append(ModelResourceSummary(model: kind, total: total,
                meanWallMilliseconds: total.wallMilliseconds / Double(samples.count),
                load: loads[kind], notes: ["Includes input packing and ensemble evaluation. Model loading is separate."]))
        }
        // Hybrid reuses the already measured GBM and Mean Beats passes. Include
        // correction once, not once per record, and never include FusionNet.
        let hybridParts = samples.flatMap { [$0.lightGBM, $0.meanBeats, $0.hybridBlend] } + [adjustment]
        let hybridTotal = ResourceUsage.combined(hybridParts)
        let hybridLoads = [loads[.lightGBM], loads[.meanBeats]].compactMap { $0 }
        let hybridDisk = hybridLoads.count == 2 && hybridLoads.allSatisfy({ $0.diskBytes != nil })
            ? hybridLoads.reduce(UInt64(0)) { $0 + ($1.diskBytes ?? 0) } : nil
        let combinedLoad = ResourceUsage.combined(hybridLoads.map(\.resources))
        // The two dependencies load in disjoint intervals with FusionNet in
        // between. A before-to-after delta would falsely include that load.
        let hybridLoad = hybridLoads.count == 2 ? ModelLoadProfile(model: .hybrid,
            resources: ResourceUsage(wallMilliseconds: combinedLoad.wallMilliseconds,
                cpuMilliseconds: combinedLoad.cpuMilliseconds, memoryBeforeBytes: nil, memoryAfterBytes: nil,
                observedPeakMemoryBytes: combinedLoad.observedPeakMemoryBytes,
                memorySampleCount: combinedLoad.memorySampleCount), diskBytes: hybridDisk) : nil
        summaries.append(ModelResourceSummary(model: .hybrid, total: hybridTotal,
            meanWallMilliseconds: hybridTotal.wallMilliseconds / Double(samples.count), load: hybridLoad,
            notes: ["Composed cost: LightGBM + Mean Beats + blend + one session adjustment. Shares the measured base-model work; it is not additional execution.",
                    "Per-record time amortizes the session adjustment. Memory is the maximum observed app footprint across these phases, with all models resident."]))
        return Self(schemaVersion: 1, id: UUID(), startedAt: startedAt, finishedAt: finishedAt,
            recordingCount: samples.count, sessionWallMilliseconds: sessionWallMilliseconds,
            modelsWereAlreadyLoaded: modelsWereAlreadyLoaded, computeUnits: "cpuOnly", environment: environment,
            start: start, end: end, models: summaries, samples: samples, hybridAdjustment: adjustment,
            perModelEnergyAvailable: false,
            notes: ["Wall time uses a monotonic clock. CPU time is process user + system time across threads; other app activity and instrumentation may contribute.",
                    "Memory is whole-app physical footprint, not exclusive model RAM. All models remain loaded. Checkpoints can miss transient peaks; peaks must not be summed.",
                    "Load metrics describe the first load in this app process and are not repeated on a warm run. Core ML may lazily allocate more memory on first prediction.",
                    "Per-model battery consumption and joules are unavailable through the public APIs used here. Battery readings describe coarse whole-device charge, including display and other activity.",
                    "Session wall time and battery observation include model preparation and progress updates. Model totals exclude loading. Hybrid overlaps its dependencies; do not sum all four rows.",
                    "For device comparisons use Release builds on the same unplugged physical iPhone, consistent thermal/power state, and repeated warm runs. Simulator results are Mac measurements."])
    }

    func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }
}
