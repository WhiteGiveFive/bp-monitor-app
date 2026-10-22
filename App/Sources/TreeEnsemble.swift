import Foundation

/// The frozen 161-feature LightGBM model, evaluated in Double precision.
/// Tree leaf values already include shrinkage and the model's initial score.
/// Input nil and NaN both represent missing values; infinity is invalid input.
public struct TreeEnsemble: Sendable {
    public let featureNames: [String]
    private let targets: [Target]
    // LightGBM's native dense predictor uses this Float32-defined threshold
    // when populating its feature buffer, including when zero_as_missing=false.
    private static let inputZeroThreshold = Double(Float(1e-35))

    private struct Document: Decodable {
        let schemaVersion: Int
        let featureCount: Int
        let featureNames: [String]
        let inputZeroThreshold: Double
        let targetOrder: [String]
        let targets: [Target]
    }

    private struct Target: Decodable, Sendable {
        let name: String
        let models: [SeedModel]
    }

    private struct SeedModel: Decodable, Sendable {
        let seed: Int
        let trees: [Tree]
    }

    private struct Tree: Decodable, Sendable {
        let f: [Int]
        let t: [Double]
        let l: [Int]
        let r: [Int]
        let m: [Int]
        let d: [Bool]
        let v: [Double]

        func validate(featureCount: Int, label: String) throws {
            let count = f.count
            guard [t.count, l.count, r.count, m.count, d.count].allSatisfy({ $0 == count }),
                  v.count == count + 1 else {
                throw TreeEnsembleError.invalidModel("\(label): inconsistent tree array lengths")
            }
            guard f.allSatisfy({ (0..<featureCount).contains($0) }),
                  t.allSatisfy(\.isFinite), v.allSatisfy(\.isFinite),
                  m.allSatisfy({ $0 == 0 || $0 == 2 }) else {
                throw TreeEnsembleError.invalidModel("\(label): invalid feature, number or missing mode")
            }
            for child in l + r {
                let valid = child >= 0 ? child < count : -(child + 1) < v.count
                guard valid else {
                    throw TreeEnsembleError.invalidModel("\(label): child index is out of bounds")
                }
            }
            // Validate reachability and acyclicity once, before any predictions.
            var visited = Set<Int>()
            var leaves = Set<Int>()
            var pending = count == 0 ? [-1] : [0]
            while let index = pending.popLast() {
                if index < 0 {
                    guard leaves.insert(-(index + 1)).inserted else {
                        throw TreeEnsembleError.invalidModel("\(label): duplicate leaf reference")
                    }
                } else {
                    guard visited.insert(index).inserted else {
                        throw TreeEnsembleError.invalidModel("\(label): cyclic or shared branch")
                    }
                    pending.append(l[index])
                    pending.append(r[index])
                }
            }
            guard visited.count == count, leaves.count == v.count else {
                throw TreeEnsembleError.invalidModel("\(label): unreachable tree nodes")
            }
        }

        func predict(_ features: [Double?]) -> Double {
            if f.isEmpty { return v[0] }
            var node = 0
            while node >= 0 {
                let input = features[f[node]]
                let missing = input == nil || input!.isNaN
                let goLeft: Bool
                if missing && m[node] == 2 {
                    goLeft = d[node]
                } else {
                    // LightGBM maps NaN to zero for a numeric split whose
                    // missing_type is None; its default_left flag is unused.
                    let value = missing ? 0.0 : input!
                    goLeft = value <= t[node]
                }
                node = goLeft ? l[node] : r[node]
            }
            return v[-(node + 1)]
        }
    }

    public init(url: URL) throws {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: url))
        } catch {
            throw TreeEnsembleError.invalidModel("Could not read \(url.lastPathComponent): \(error.localizedDescription)")
        }
        guard document.schemaVersion == 1 else {
            throw TreeEnsembleError.invalidModel("Unsupported tree schema version \(document.schemaVersion)")
        }
        guard document.featureCount == 161, document.featureNames.count == 161,
              Set(document.featureNames).count == 161 else {
            throw TreeEnsembleError.invalidModel("Expected 161 uniquely named features")
        }
        guard document.inputZeroThreshold == Self.inputZeroThreshold else {
            throw TreeEnsembleError.invalidModel("Unsupported native input zero threshold")
        }
        guard document.targetOrder == ["sbp", "dbp"],
              document.targets.map(\.name) == ["sbp", "dbp"] else {
            throw TreeEnsembleError.invalidModel("Expected target order [sbp, dbp]")
        }
        for target in document.targets {
            guard target.models.map(\.seed) == Array(0..<5) else {
                throw TreeEnsembleError.invalidModel("\(target.name): expected seeds 0 through 4 in order")
            }
            for model in target.models {
                guard model.trees.count == 500 else {
                    throw TreeEnsembleError.invalidModel("\(target.name) seed \(model.seed): expected 500 trees")
                }
                for (index, tree) in model.trees.enumerated() {
                    try tree.validate(featureCount: 161, label: "\(target.name) seed \(model.seed) tree \(index)")
                }
            }
        }
        self.featureNames = document.featureNames
        self.targets = document.targets
    }

    /// Returns [SBP, DBP] in mmHg, averaging all five seeds per target.
    public func predict(features: [Double?]) throws -> [Double] {
        try predictPerSeed(features: features).map { $0.reduce(0, +) / Double($0.count) }
    }

    /// Internal parity-test hook: [target][seed], SBP first and seeds 0...4.
    func predictPerSeed(features: [Double?]) throws -> [[Double]] {
        guard features.count == featureNames.count else {
            throw TreeEnsembleError.invalidInput("Expected \(featureNames.count) features, received \(features.count)")
        }
        if let index = features.firstIndex(where: { $0?.isInfinite == true }) {
            throw TreeEnsembleError.invalidInput("Feature \(featureNames[index]) is infinite; use nil for missing values")
        }
        let normalized = features.map { value -> Double? in
            guard let value else { return nil }
            return abs(value) <= Self.inputZeroThreshold ? 0.0 : value
        }
        let predictions = targets.map { target in
            target.models.map { model in
                model.trees.reduce(0.0) { $0 + $1.predict(normalized) }
            }
        }
        guard predictions.joined().allSatisfy(\.isFinite) else {
            throw TreeEnsembleError.invalidModel("The tree ensemble produced nonfinite output")
        }
        return predictions
    }
}

public enum TreeEnsembleError: Error, LocalizedError, Sendable {
    case invalidModel(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let reason): return "Tree model error: \(reason)"
        case .invalidInput(let reason): return "Tree input error: \(reason)"
        }
    }
}
