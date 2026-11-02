import Foundation
import Darwin
import os

public enum ModelKind: String, Codable, CaseIterable, Sendable {
    case lightGBM
    case fusionNet
    case meanBeats
    case hybrid

    public var title: String {
        switch self {
        case .lightGBM: return "LightGBM"
        case .fusionNet: return "FusionNet"
        case .meanBeats: return "Mean Beats"
        case .hybrid: return "Hybrid"
        }
    }
}

/// Measurements describe the whole app process during an interval. CPU time
/// includes user and system time across all its threads, so it can exceed wall
/// time. Memory is the process physical footprint, not exclusive model memory.
public struct ResourceUsage: Codable, Sendable {
    public let wallMilliseconds: Double
    public let cpuMilliseconds: Double?
    public let memoryBeforeBytes: UInt64?
    public let memoryAfterBytes: UInt64?
    /// Maximum successful sample at start, explicit checkpoints and finish.
    /// This is not an operating-system high-water mark; transient peaks can be missed.
    public let observedPeakMemoryBytes: UInt64?
    public let memorySampleCount: Int

    public init(wallMilliseconds: Double, cpuMilliseconds: Double?,
                memoryBeforeBytes: UInt64?, memoryAfterBytes: UInt64?,
                observedPeakMemoryBytes: UInt64?, memorySampleCount: Int) {
        self.wallMilliseconds = wallMilliseconds
        self.cpuMilliseconds = cpuMilliseconds
        self.memoryBeforeBytes = memoryBeforeBytes
        self.memoryAfterBytes = memoryAfterBytes
        self.observedPeakMemoryBytes = observedPeakMemoryBytes
        self.memorySampleCount = memorySampleCount
    }

    /// A signed process-footprint change, which may be negative. Nil means a
    /// boundary was unavailable or the difference cannot be represented as Int64.
    public var memoryDeltaBytes: Int64? {
        guard let before = memoryBeforeBytes, let after = memoryAfterBytes else { return nil }
        if after >= before { return Int64(exactly: after - before) }
        let magnitude = before - after
        if magnitude == UInt64(Int64.max) + 1 { return Int64.min }
        guard let signed = Int64(exactly: magnitude) else { return nil }
        return -signed
    }

    /// Combine measurements in execution order. Durations are summed; the
    /// observed process peak is their maximum, never a sum of model RAM.
    /// CPU is unavailable if any component is unavailable. An empty list has
    /// zero durations, no memory observations and no samples.
    public static func combined(_ measurements: [ResourceUsage]) -> ResourceUsage {
        let cpu: Double?
        if measurements.allSatisfy({ $0.cpuMilliseconds != nil }) {
            cpu = measurements.reduce(0) { $0 + ($1.cpuMilliseconds ?? 0) }
        } else {
            cpu = nil
        }
        return ResourceUsage(
            wallMilliseconds: measurements.reduce(0) { $0 + $1.wallMilliseconds },
            cpuMilliseconds: cpu,
            memoryBeforeBytes: measurements.first?.memoryBeforeBytes,
            memoryAfterBytes: measurements.last?.memoryAfterBytes,
            observedPeakMemoryBytes: measurements.compactMap(\.observedPeakMemoryBytes).max(),
            memorySampleCount: measurements.reduce(0) { $0 + $1.memorySampleCount }
        )
    }
}

public struct ModelLoadProfile: Codable, Sendable {
    public let model: ModelKind
    public let resources: ResourceUsage
    /// Logical model-file bytes on disk; this is distinct from process memory.
    public let diskBytes: UInt64?

    public init(model: ModelKind, resources: ResourceUsage, diskBytes: UInt64?) {
        self.model = model
        self.resources = resources
        self.diskBytes = diskBytes
    }
}

/// A synchronous, thread-confined probe with no polling timer. Call checkpoint
/// at useful operation boundaries while allocations are still alive. All
/// counters are process-wide, include instrumentation overhead and can include
/// unrelated work in other threads. Do not share one probe across concurrent tasks.
public final class ResourceProbe {
    private static let log = OSLog(subsystem: "com.bpmonitor.research", category: "ResourceMetrics")

    private let startWall: TimeInterval
    private let startCPU: Double?
    private let memoryBefore: UInt64?
    private let signpostID: OSSignpostID
    private var peakMemory: UInt64?
    private var sampleCount: Int
    private var completed: ResourceUsage?

    public init(label: StaticString) {
        startWall = ProcessInfo.processInfo.systemUptime
        startCPU = Self.processCPUMilliseconds()
        let initialMemory = Self.processFootprintBytes()
        memoryBefore = initialMemory
        peakMemory = initialMemory
        sampleCount = initialMemory == nil ? 0 : 1
        signpostID = OSSignpostID(log: Self.log)
        os_signpost(.begin, log: Self.log, name: "InferenceResource", signpostID: signpostID,
                    "%{public}@", String(describing: label))
    }

    deinit {
        // A forgotten finish must not leave an open Instruments interval.
        if completed == nil {
            os_signpost(.end, log: Self.log, name: "InferenceResource", signpostID: signpostID)
        }
    }

    public func checkpoint() {
        guard completed == nil else { return }
        _ = sampleMemory()
    }

    /// Finish once; repeated calls return the same measurement and do not sample
    /// again. Checkpoints after finish are ignored.
    public func finish() -> ResourceUsage {
        if let completed { return completed }
        let memoryAfter = sampleMemory()
        let endCPU = Self.processCPUMilliseconds()
        let wall = max(0, (ProcessInfo.processInfo.systemUptime - startWall) * 1000)
        let cpu: Double?
        if let startCPU, let endCPU, endCPU >= startCPU {
            cpu = endCPU - startCPU
        } else {
            cpu = nil
        }
        let usage = ResourceUsage(wallMilliseconds: wall, cpuMilliseconds: cpu,
            memoryBeforeBytes: memoryBefore, memoryAfterBytes: memoryAfter,
            observedPeakMemoryBytes: peakMemory, memorySampleCount: sampleCount)
        completed = usage
        os_signpost(.end, log: Self.log, name: "InferenceResource", signpostID: signpostID)
        return usage
    }

    /// Close the probe and its signpost even if the measured operation throws.
    public static func measure<T>(label: StaticString, _ body: (ResourceProbe) throws -> T) throws -> (T, ResourceUsage) {
        let probe = ResourceProbe(label: label)
        defer { _ = probe.finish() }
        let result = try body(probe)
        return (result, probe.finish())
    }

    @discardableResult
    private func sampleMemory() -> UInt64? {
        guard let value = Self.processFootprintBytes() else { return nil }
        peakMemory = max(peakMemory ?? value, value)
        sampleCount += 1
        return value
    }

    private static func processCPUMilliseconds() -> Double? {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return nil }
        let user = Double(usage.ru_utime.tv_sec) * 1000 + Double(usage.ru_utime.tv_usec) / 1000
        let system = Double(usage.ru_stime.tv_sec) * 1000 + Double(usage.ru_stime.tv_usec) / 1000
        let total = user + system
        return total.isFinite && total >= 0 ? total : nil
    }

    private static func processFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        let wordSize = MemoryLayout<integer_t>.size
        let capacity = MemoryLayout<task_vm_info_data_t>.size / wordSize
        var count = mach_msg_type_number_t(capacity)
        // TASK_VM_INFO_COUNT is a sizeof-based C macro. Compute its equivalent
        // from the imported struct, and accept older supported struct revisions
        // only when the returned byte count includes phys_footprint (rev1).
        guard let footprintOffset = MemoryLayout<task_vm_info_data_t>.offset(of: \.phys_footprint) else { return nil }
        let requiredBytes = footprintOffset + MemoryLayout.size(ofValue: info.phys_footprint)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) { words in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), words, &count)
            }
        }
        guard result == KERN_SUCCESS, Int(count) * wordSize >= requiredBytes else { return nil }
        return UInt64(info.phys_footprint)
    }
}
