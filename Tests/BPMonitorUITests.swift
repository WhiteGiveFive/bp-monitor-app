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

    func testCompletedSessionShowsMeasuredResourceReport() throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launch()
        let run = app.buttons["runSession"]
        XCTAssertTrue(run.waitForExistence(timeout: 20))
        XCTAssertFalse(app.buttons["resourceReportToolbar"].exists,
                       "A resource report must not be presented before a measured run.")
        run.tap()
        XCTAssertTrue(app.staticTexts["All 8 recordings processed on this device."].waitForExistence(timeout: 180))
        XCTAssertEqual(app.alerts.count, 0)
        let openReport = app.buttons["resourceReportToolbar"]
        XCTAssertTrue(openReport.waitForExistence(timeout: 5))
        openReport.tap()
        XCTAssertTrue(app.navigationBars["Resource report"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["exportResourceReport"].exists)
        let batteryNotice = app.staticTexts["perModelBatteryUnavailable"]
        XCTAssertTrue(batteryNotice.exists)
        XCTAssertEqual(batteryNotice.label, "Per-model battery consumption is unavailable")
        attachScreenshot(app, named: "03-resource-report-context")
        for model in ["lightGBM", "fusionNet", "meanBeats", "hybrid"] {
            let card = app.descendants(matching: .any).matching(identifier: "resource_\(model)").firstMatch
            for _ in 0..<10 where !card.isHittable { app.scrollViews.firstMatch.swipeUp() }
            XCTAssertTrue(card.exists, "The resource report must include \(model).")
        }
        attachScreenshot(app, named: "04-resource-report-models")
        app.buttons["closeResourceReport"].tap()
        XCTAssertTrue(app.buttons["runSession"].exists)
    }

    private func attachScreenshot(_ app: XCUIApplication, named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
