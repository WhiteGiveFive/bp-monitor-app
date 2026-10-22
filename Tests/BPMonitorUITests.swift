import XCTest

@MainActor
final class BPMonitorUITests: XCTestCase {
    func testBundledSessionRunsOfflineAndShowsActualPredictions() throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launch()
        let run = app.buttons["runSession"]
        XCTAssertTrue(run.waitForExistence(timeout: 20), "The real bundled session must load.")
        attachScreenshot(app, named: "01-prepared-session")
        run.tap()
        let completed = app.staticTexts["All 8 recordings processed on this device."]
        XCTAssertTrue(completed.waitForExistence(timeout: 180), "All eight bundled recordings must complete real inference.")
        XCTAssertEqual(app.alerts.count, 0, "Inference errors must never be silently replaced with predictions.")
        let hybrid = app.descendants(matching: .any).matching(identifier: "prediction_Hybrid").firstMatch
        XCTAssertTrue(hybrid.exists)
        XCTAssertTrue((hybrid.value as? String)?.contains("Session adjusted") == true,
                      "Hybrid must use the finished session adjustment.")
        for name in ["LightGBM", "FusionNet", "Mean Beats", "Hybrid"] {
            let card = app.descendants(matching: .any).matching(identifier: "prediction_\(name)").firstMatch
            XCTAssertTrue((card.value as? String)?.contains("Systolic") == true, "\(name) must show a calculated value.")
        }
        for _ in 0..<4 where !hybrid.isHittable { app.scrollViews.firstMatch.swipeUp() }
        attachScreenshot(app, named: "02-on-device-model-comparison")
    }

    private func attachScreenshot(_ app: XCUIApplication, named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
