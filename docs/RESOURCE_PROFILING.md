# Resource profiling

Run a prepared session to completion, then open **Resource report** from the result card or toolbar. The report shows the current run's device context, start/end battery and thermal state, and resource measurements for LightGBM, FusionNet, Mean Beats and Hybrid. A cancelled or failed run does not produce a completed report. Importing another session or starting another run clears the previous report, so export it first if you want to retain it.

Use **Export JSON** to save through Files. The suggested filename is `bp-resources-<report UUID>.json`. Export is a manual action; no remote service is used for inference or report creation.

## What the measurements mean

| Measurement | Scope and interpretation |
|---|---|
| Complete session | Wall time from model preparation through inference and final session adjustment, including progress updates and dispatch overhead. Initial loading contributes only when the models need loading. |
| Model session wall time | Sum of measured inference intervals for that model across the imported recordings. Includes input packing and ensemble evaluation; excludes model loading. |
| Per recording · amortized | Model session wall time divided by the number of recordings. Hybrid includes its share of the single session adjustment. |
| Process CPU time | Difference in process user + system CPU time from `getrusage(RUSAGE_SELF)`, across all app threads. Other app activity and instrumentation can contribute. Parallel work can make CPU time exceed wall time. |
| Observed peak app memory | Largest successful `TASK_VM_INFO.phys_footprint` sample during the measured intervals. This is the whole app's physical footprint with all models resident, not RAM owned exclusively by one model. |
| App memory change on load | Signed difference between process footprint immediately before and after an initial model load. It may be negative and depends on allocations, caches and other app activity. |
| Model assets on disk | Recursive sum of regular-file logical sizes, displayed in MiB (1,048,576 bytes). It does not measure allocated disk blocks or runtime memory. |

Wall time uses a monotonic clock. Unavailable CPU or memory readings remain unavailable rather than being replaced with zero. The export retains memory before/after values, observed peaks and successful sample counts; the displayed load-memory change is calculated from the two boundary values.

Memory samples occur at the start and end of each measured operation, after all neural input arrays are constructed, and after each of the 25 Core ML predictions for a neural ensemble. Loading also samples after each seed model loads. There is no polling timer. These checkpoints help observe retained working buffers but can miss allocations that are created and freed within a Core ML call. The observed peak is not a guaranteed true peak or an operating-system high-water mark. Peaks across phases are combined by maximum, never by addition.

The models load in this order: LightGBM, five FusionNet models, then five Mean Beats models. They remain resident for subsequent runs. Each report retains the initial load measurements from that app process and records `modelsWereAlreadyLoaded` to distinguish a warm run. Core ML can allocate additional resources on the first prediction, so initial loading alone does not describe the full memory requirement. Fixed evaluation order, caches and the preceding work can affect comparisons.

## Hybrid accounting

Hybrid reuses the LightGBM and Mean Beats predictions already calculated for the comparison. Its resource total is:

```text
sum of LightGBM inference intervals
+ sum of Mean Beats inference intervals
+ sum of raw-blend arithmetic intervals
+ one final session-adjustment interval
```

FusionNet is not a Hybrid dependency. The correction is measured once for the entire imported session, including grouping and all subject/round adjustments. Hybrid's per-record value amortizes that correction over the recording count. The four cards therefore describe overlapping work; adding all four would double-count the base models used by Hybrid.

Hybrid asset size and initial-load durations combine the LightGBM and Mean Beats components. Hybrid has no separate model file or independent load. Its load-memory delta is intentionally unavailable: the component loads occupy disjoint intervals with FusionNet loading between them, so a single before-to-after difference would incorrectly include FusionNet. All reported peaks still describe the whole app, including every model currently resident.

## Battery and device state

The app enables battery monitoring for a run and restores the prior setting afterward. It records the charge fraction and battery state at the beginning and end, together with thermal state and Low Power Mode. Charge fractions are displayed as percentages; a decrease from 0.80 to 0.79 is one percentage point. Apple's [`UIDevice.batteryLevel`](https://developer.apple.com/documentation/uikit/uidevice/batterylevel) reports charge level, not energy consumed by a model.

These observations cover the whole device, including the app, display, operating system and other activity. They cannot provide joules or battery consumption for individual models. No detected charge-level change does not mean zero energy consumption. Charging, unknown state or an increased reading prevents a consumption interpretation. Simulator battery readings are deliberately unavailable.

The start/end [`ProcessInfo.thermalState`](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.property) values provide context for performance changes. They are snapshots, not a history of every thermal transition during the session. The complete-session timer and battery observation include model preparation and UI progress updates; model inference totals exclude loading.

## Compare runs on an iPhone

Use a Release build on the same physical iPhone, the same prepared recordings and consistent display, background activity, thermal state and Low Power Mode. Unplug the device when comparing charge observations. Run once to initialize models and caches, then export several warm runs. Keep a separate first-run report if loading behavior matters. The report identifies the platform, hardware model, OS version, build configuration and whether models were already loaded.

Simulator and Mac CLI measurements describe the host Mac. Debug builds, a connected debugger, changing thermal state, execution order and cache state can alter results. The current implementation always uses Core ML `cpuOnly`; these reports do not benchmark the GPU or Neural Engine. Measurements include the probes' own overhead, which is proportionally more noticeable for short operations such as blend arithmetic.

For deeper power investigation, profile the Release app on a physical iPhone with Instruments. Apple's [Power Profiler guide](https://developer.apple.com/documentation/xcode/measuring-your-app-s-power-use-with-power-profiler) requires a supported physical iPhone running iOS 26 or later. Follow its device and Xcode requirements; the app's own iOS 17 minimum does not imply Power Profiler availability on every supported device. Use its power observations together with the app's signposts, without treating coarse charge changes as precise per-model energy measurements.

The probes emit Instruments intervals with subsystem `com.bpmonitor.research`, category `ResourceMetrics`, and name `InferenceResource`. Their labels identify `Load LightGBM`, `Load FusionNet`, `Load Mean Beats`, `LightGBM inference`, `FusionNet inference`, `Mean Beats inference`, `Hybrid blend`, and `Hybrid session adjustment`. Correlate those intervals with the Instruments timeline to inspect loading, input preparation and inference phases. The reported model intervals include input packing; the signposts do not claim that all work observed by Instruments belongs exclusively to a model.

## Export format and validation

The JSON report has `schemaVersion: 1`. It includes a report UUID, timestamps, recording count, complete-session duration, `cpuOnly` execution setting, environment and device-state snapshots, four model summaries, per-record resource samples indexed from 1, and the single Hybrid adjustment measurement. Each resource measurement contains wall/CPU time and available memory samples. `perModelEnergyAvailable` is always `false`.

Recording IDs, subject IDs, waveform arrays, model input tensors and blood-pressure predictions are excluded. Device metadata, timing data, battery/thermal observations and the number of recordings remain in the export. It is a resource report, not a prepared recording file, so it cannot be imported as inference input.

The [README verification commands](../README.md#verify) compile the shared native engine with `ResourceMetrics.swift` and build the resource CLI with `ResourceReport.swift`. The resource CLI uses live Mac CPU/memory probes, validates shared Hybrid accounting, tests battery interpretations and round-trips the JSON export. Its device-state snapshots are test inputs, so they do not represent a physical iPhone's battery or thermal observations.

The instrumented native engine retained the previous maximum historical-prediction difference of **0.0000938416 mmHg**, below the 0.005 mmHg tolerance. Live probe, Hybrid arithmetic, battery interpretation and CLI export checks passed. See the [instrumented parity report](../artifacts/validation/resource-feature-parity.json), [Mac resource report](../artifacts/validation/resource-report-macos.json) and [validation evidence](VALIDATION.md) for recorded results. No physical-iPhone resource measurements are claimed by these Mac checks.
