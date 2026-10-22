import SwiftUI

@main
struct BPMonitorApp: App {
    @StateObject private var store = DemoStore()

    var body: some Scene {
        WindowGroup {
            ContentView(store: store)
                .preferredColorScheme(.light)
        }
    }
}
