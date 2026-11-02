#!/usr/bin/env python3
"""Generate a dependency-free Xcode project with folder-preserved model assets."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BPMonitor.xcodeproj"


def identifier(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24].upper()


def main():
    sources = ["BPMonitorApp.swift", "ContentView.swift", "DemoStore.swift", "PreparedRecording.swift", "InferenceEngine.swift", "TreeEnsemble.swift", "ResourceMetrics.swift", "ResourceReport.swift", "ResourceReportView.swift"]
    ids = {name: identifier(name) for name in ["project", "mainGroup", "sourcesGroup", "resourcesGroup", "productGroup", "product", "target", "sourcesPhase", "resourcesPhase", "frameworksPhase", "projectConfig", "targetConfig", "projectDebug", "projectRelease", "targetDebug", "targetRelease", "Models", "Data", "testTarget", "testProduct", "testGroup", "testSource", "testSourcesPhase", "testFrameworksPhase", "testConfig", "testDebug", "testRelease", "testDependency", "testProxy"] + sources}
    def uid(name): return ids[name]
    lines = ["// !$*UTF8*$!", "{", "archiveVersion = 1;", "classes = {};", "objectVersion = 56;", "objects = {"]
    for name in sources:
        lines += [f'{uid(name)} = {{ isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = "{name}"; sourceTree = "<group>"; }};',
                  f'{identifier("build-" + name)} = {{ isa = PBXBuildFile; fileRef = {uid(name)}; }};']
    for name in ["Models", "Data"]:
        lines += [f'{uid(name)} = {{ isa = PBXFileReference; lastKnownFileType = folder; path = "{name}"; sourceTree = "<group>"; }};',
                  f'{identifier("build-" + name)} = {{ isa = PBXBuildFile; fileRef = {uid(name)}; }};']
    lines += [
        f'{uid("product")} = {{ isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = BPMonitor.app; sourceTree = BUILT_PRODUCTS_DIR; }};',
        f'{uid("mainGroup")} = {{ isa = PBXGroup; children = ({uid("sourcesGroup")}, {uid("resourcesGroup")}, {uid("testGroup")}, {uid("productGroup")}); sourceTree = "<group>"; }};',
        f'{uid("sourcesGroup")} = {{ isa = PBXGroup; children = ({", ".join(uid(name) for name in sources)}); path = App/Sources; sourceTree = "<group>"; }};',
        f'{uid("resourcesGroup")} = {{ isa = PBXGroup; children = ({uid("Models")}, {uid("Data")}); path = App/Resources; sourceTree = "<group>"; }};',
        f'{uid("productGroup")} = {{ isa = PBXGroup; children = ({uid("product")}, {uid("testProduct")}); name = Products; sourceTree = "<group>"; }};',
        f'{uid("sourcesPhase")} = {{ isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({", ".join(identifier("build-" + name) for name in sources)}); runOnlyForDeploymentPostprocessing = 0; }};',
        f'{uid("resourcesPhase")} = {{ isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({identifier("build-Models")}, {identifier("build-Data")}); runOnlyForDeploymentPostprocessing = 0; }};',
        f'{uid("frameworksPhase")} = {{ isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; }};',
        f'{uid("target")} = {{ isa = PBXNativeTarget; buildConfigurationList = {uid("targetConfig")}; buildPhases = ({uid("sourcesPhase")}, {uid("frameworksPhase")}, {uid("resourcesPhase")}); buildRules = (); dependencies = (); name = BPMonitor; productName = BPMonitor; productReference = {uid("product")}; productType = "com.apple.product-type.application"; }};',
        f'{uid("project")} = {{ isa = PBXProject; attributes = {{ BuildIndependentTargetsInParallel = YES; LastSwiftUpdateCheck = 2630; LastUpgradeCheck = 2630; TargetAttributes = {{ {uid("target")} = {{ CreatedOnToolsVersion = 26.3; }}; {uid("testTarget")} = {{ CreatedOnToolsVersion = 26.3; TestTargetID = {uid("target")}; }}; }}; }}; buildConfigurationList = {uid("projectConfig")}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = {uid("mainGroup")}; productRefGroup = {uid("productGroup")}; projectDirPath = ""; projectRoot = ""; targets = ({uid("target")}, {uid("testTarget")}); }};',
        f'{uid("testSource")} = {{ isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = BPMonitorUITests.swift; sourceTree = "<group>"; }};',
        f'{identifier("build-testSource")} = {{ isa = PBXBuildFile; fileRef = {uid("testSource")}; }};',
        f'{uid("testProduct")} = {{ isa = PBXFileReference; explicitFileType = wrapper.cfbundle; includeInIndex = 0; path = BPMonitorUITests.xctest; sourceTree = BUILT_PRODUCTS_DIR; }};',
        f'{uid("testGroup")} = {{ isa = PBXGroup; children = ({uid("testSource")}); path = Tests; sourceTree = "<group>"; }};',
        f'{uid("testSourcesPhase")} = {{ isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({identifier("build-testSource")}); runOnlyForDeploymentPostprocessing = 0; }};',
        f'{uid("testFrameworksPhase")} = {{ isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; }};',
        f'{uid("testProxy")} = {{ isa = PBXContainerItemProxy; containerPortal = {uid("project")}; proxyType = 1; remoteGlobalIDString = {uid("target")}; remoteInfo = BPMonitor; }};',
        f'{uid("testDependency")} = {{ isa = PBXTargetDependency; target = {uid("target")}; targetProxy = {uid("testProxy")}; }};',
        f'{uid("testTarget")} = {{ isa = PBXNativeTarget; buildConfigurationList = {uid("testConfig")}; buildPhases = ({uid("testSourcesPhase")}, {uid("testFrameworksPhase")}); buildRules = (); dependencies = ({uid("testDependency")}); name = BPMonitorUITests; productName = BPMonitorUITests; productReference = {uid("testProduct")}; productType = "com.apple.product-type.bundle.ui-testing"; }};',
    ]
    common = {
        "ALWAYS_SEARCH_USER_PATHS": "NO", "CLANG_ENABLE_MODULES": "YES", "CLANG_ENABLE_OBJC_ARC": "YES",
        "IPHONEOS_DEPLOYMENT_TARGET": "17.0", "SDKROOT": "iphoneos", "SWIFT_VERSION": "5.0",
    }
    target = {
        "CODE_SIGN_STYLE": "Automatic", "CURRENT_PROJECT_VERSION": "1", "DEVELOPMENT_TEAM": '""',
        "GENERATE_INFOPLIST_FILE": "NO", "INFOPLIST_FILE": "App/Info.plist", "MARKETING_VERSION": "1.0",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.bpmonitor.research", "PRODUCT_NAME": '"$(TARGET_NAME)"',
        "SUPPORTED_PLATFORMS": '"iphoneos iphonesimulator"', "SUPPORTS_MACCATALYST": "NO",
        "TARGETED_DEVICE_FAMILY": '"1"', "SWIFT_EMIT_LOC_STRINGS": "YES",
        "LD_RUNPATH_SEARCH_PATHS": '"$(inherited) @executable_path/Frameworks"',
    }
    test = {
        "CODE_SIGN_STYLE": "Automatic", "DEVELOPMENT_TEAM": '""', "GENERATE_INFOPLIST_FILE": "YES",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.bpmonitor.research.uitests", "PRODUCT_NAME": '"$(TARGET_NAME)"',
        "TARGETED_DEVICE_FAMILY": '"1"', "TEST_TARGET_NAME": "BPMonitor",
        "LD_RUNPATH_SEARCH_PATHS": '"$(inherited) @executable_path/Frameworks @loader_path/Frameworks"',
    }
    for category in ["project", "target", "test"]:
        for configuration in ["Debug", "Release"]:
            settings = dict(common if category == "project" else target if category == "target" else test)
            if category == "project":
                settings.update({"DEBUG_INFORMATION_FORMAT": '"dwarf"' if configuration == "Debug" else '"dwarf-with-dsym"',
                    "SWIFT_OPTIMIZATION_LEVEL": '"-Onone"' if configuration == "Debug" else '"-O"',
                    "SWIFT_COMPILATION_MODE": "singlefile" if configuration == "Debug" else "wholemodule"})
                if configuration == "Debug": settings["SWIFT_ACTIVE_COMPILATION_CONDITIONS"] = '"DEBUG $(inherited)"'
            payload = " ".join(f"{key} = {value};" for key, value in settings.items())
            lines.append(f'{uid(category + configuration)} = {{ isa = XCBuildConfiguration; buildSettings = {{ {payload} }}; name = {configuration}; }};')
        lines.append(f'{uid(category + "Config")} = {{ isa = XCConfigurationList; buildConfigurations = ({uid(category + "Debug")}, {uid(category + "Release")}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }};')
    lines += ["};", f'rootObject = {uid("project")};', "}"]
    PROJECT.mkdir(exist_ok=True)
    (PROJECT / "project.pbxproj").write_text("\n".join(lines) + "\n")
    scheme_dir = PROJECT / "xcshareddata/xcschemes"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    ref = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{uid("target")}" BuildableName="BPMonitor.app" BlueprintName="BPMonitor" ReferencedContainer="container:BPMonitor.xcodeproj"/>'
    test_ref = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{uid("testTarget")}" BuildableName="BPMonitorUITests.xctest" BlueprintName="BPMonitorUITests" ReferencedContainer="container:BPMonitor.xcodeproj"/>'
    (scheme_dir / "BPMonitor.xcscheme").write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2630" version="1.3">
<BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{ref}</BuildActionEntry></BuildActionEntries></BuildAction>
<TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables><TestableReference skipped="NO">{test_ref}</TestableReference></Testables></TestAction>
<LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{ref}</BuildableProductRunnable></LaunchAction>
<ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{ref}</BuildableProductRunnable></ProfileAction>
<AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>\n''')
    print(PROJECT)


if __name__ == "__main__": main()
