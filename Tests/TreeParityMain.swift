import Foundation

private struct FixtureFile: Decodable {
    let schemaVersion: Int
    let absoluteTolerance: Double
    let lightgbmVersion: String
    let preparedRecordCount: Int?
    let preparedReferenceMaxAbsoluteError: Double?
    let cases: [Fixture]
}

private struct Fixture: Decodable {
    let name: String
    let features: [Double?]
    let perSeed: [[Double]]
    let prediction: [Double]
}

private enum TestError: Error, LocalizedError {
    case failure(String)
    var errorDescription: String? {
        switch self { case .failure(let reason): return reason }
    }
}

@main
struct TreeParityMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            throw TestError.failure("Usage: tree-parity trees.json tree_parity_fixtures.json")
        }
        let ensemble = try TreeEnsemble(url: URL(fileURLWithPath: CommandLine.arguments[1]))
        let fixtureFile = try JSONDecoder().decode(FixtureFile.self, from: Data(contentsOf:
            URL(fileURLWithPath: CommandLine.arguments[2])))
        guard fixtureFile.schemaVersion == 1, !fixtureFile.cases.isEmpty else {
            throw TestError.failure("Unsupported or empty parity fixture file")
        }
        var maximumSeedError = 0.0
        var maximumMeanError = 0.0
        for fixture in fixtureFile.cases {
            let seeded = try ensemble.predictPerSeed(features: fixture.features)
            let average = try ensemble.predict(features: fixture.features)
            guard fixture.perSeed.count == 2, fixture.perSeed.allSatisfy({ $0.count == 5 }),
                  fixture.prediction.count == 2 else {
                throw TestError.failure("Invalid expected prediction shape: \(fixture.name)")
            }
            for target in 0..<2 {
                for seed in 0..<5 {
                    let error = abs(seeded[target][seed] - fixture.perSeed[target][seed])
                    maximumSeedError = max(maximumSeedError, error)
                    guard error <= fixtureFile.absoluteTolerance else {
                        throw TestError.failure("\(fixture.name), target \(target), seed \(seed): error \(error)")
                    }
                }
                let error = abs(average[target] - fixture.prediction[target])
                maximumMeanError = max(maximumMeanError, error)
                guard error <= fixtureFile.absoluteTolerance else {
                    throw TestError.failure("\(fixture.name), target \(target): ensemble error \(error)")
                }
            }
            // JSON null and explicit Double.nan must have identical semantics.
            let explicitNaNs = fixture.features.map { $0 ?? Double.nan }.map(Optional.some)
            guard try ensemble.predict(features: explicitNaNs) == average else {
                throw TestError.failure("NaN/null semantics differ: \(fixture.name)")
            }
        }
        do {
            _ = try ensemble.predict(features: [0.0])
            throw TestError.failure("Wrong feature count was accepted")
        } catch TreeEnsembleError.invalidInput { }
        do {
            _ = try ensemble.predict(features: Array(repeating: Double.infinity, count: 161))
            throw TestError.failure("Infinite features were accepted")
        } catch TreeEnsembleError.invalidInput { }
        let report: [String: Any] = [
            "status": "passed", "cases": fixtureFile.cases.count,
            "preparedRecordCount": fixtureFile.preparedRecordCount ?? 0,
            "preparedReferenceMaxAbsoluteError": fixtureFile.preparedReferenceMaxAbsoluteError ?? 0,
            "lightgbmVersion": fixtureFile.lightgbmVersion,
            "seedPredictionsChecked": fixtureFile.cases.count * 10,
            "maxSeedAbsoluteError": maximumSeedError,
            "maxEnsembleAbsoluteError": maximumMeanError,
            "absoluteTolerance": fixtureFile.absoluteTolerance,
            "nullNaNParity": "passed", "invalidInputChecks": "passed"
        ]
        let reportData = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: reportData, as: UTF8.self))
    }
}
