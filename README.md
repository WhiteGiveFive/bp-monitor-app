# BP Monitor — iPhone inference demonstrator

Compare LightGBM, basic FusionNet, Mean Beats and their Hybrid on prepared recordings. All four predictions run locally on the iPhone through Swift and Core ML. Python is used only on the Mac to prepare inputs and convert the research models.

The bundled sample contains eight recordings from two complete test sessions, including one recording without a usable ECG reference. It contains input arrays and waveform previews only; saved predictions are excluded from the app bundle.

## Run the app

1. Open `BPMonitor.xcodeproj` in Xcode.
2. Choose the **BPMonitor** scheme and an iPhone simulator running iOS 17 or later, then Run.
3. Tap **Run session**. Select a recording to inspect its waveforms, four model outputs and measured inference times.
4. To use an iPhone, select the connected device and choose your own development **Team** under **Signing & Capabilities**. Xcode manages the signing configuration. No Team is embedded in the project.

The supplied local project includes generated models and sample data. A source-only checkout needs the reproduction steps below before it can run inference. The project has no third-party Swift package dependencies.

Use **Open → Import prepared JSON** to open a prepared recording or session JSON from Files. The format and Mac preparation workflow are described in [PREPARED_RECORDINGS.md](docs/PREPARED_RECORDINGS.md). Predictions are computed after import; reference values in an imported file are ignored.

## Models and output meaning

| App output | Frozen implementation |
|---|---|
| LightGBM | Five SBP and five DBP boosters, seeds 0–4; native Swift tree evaluation in Double precision |
| FusionNet | Five original basic FusionNet models, seeds 1000–1004; five context views per model |
| Mean Beats | Five `mean_beat_embeddings` models, seeds 1000–1004; five context views and up to 32 jointly aligned ECG/PPG beats |
| Hybrid | 0.3 LightGBM + 0.7 Mean Beats, followed by subject averaging and centered recording-round offsets |

The neural models run in float32 with Core ML's CPU-only setting. Each seed averages its five standardized view predictions, then applies target scale 12 and center `[115.95759582519531, 75.6153335571289]`. FusionNet averages its five seed predictions in float32; Mean Beats averages them in float64. ECG availability also controls the R-aligned PPG and acceleration context streams. Empty beat pools use the trained missing-beat embedding.

Hybrid becomes **Session adjusted** after every imported recording has a prediction. For each subject, it averages the raw blends and adds the round offset minus that subject's mean imported-round offset. With one recording, the offset cancels and Hybrid equals the raw blend. Import the complete session to reproduce the historical session-level result. The app presents the standard Hybrid output; the separate research high-BP stretch is not applied.

The model sources are the requested original upstream run and Mean Beats v2 run in the sibling `../bp-monitor` repository:

- `results/track1/hybrid_upstream/20261012_holdout_v1/final/models/`
- `results/track1/att_fusion/m6_mean_beat_final_v2/20261016_full_refit_r2/fits/`
- `results/track1/att_fusion/m6_mean_beat_final_v2/20261016_full_refit_r2/final/`

The last directory supplies preprocessing, calibration and saved comparison outputs; it is not a separate Hybrid neural network. This app retains its 0.7 Mean Beats weight, not the different weight from the later updated-GBM study.

## Reproduce the local assets

Run these commands from the `bp-monitor-app` folder, with the original `bp-monitor` repository beside it. Conversion was performed with Python 3.12 and the exact packages in `requirements-conversion.lock.txt`. A full Xcode installation is needed for Core ML compilation and Swift validation.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-conversion.lock.txt
mkdir -p artifacts/models artifacts/validation build/ModuleCache

python tools/prepare_recordings.py \
  --source-root ../bp-monitor \
  --data-root ../bp-monitor/data/bhi-2026-data-challenge-competition-track1 \
  --subjects 0013 0030 \
  --output artifacts/prepared

python tools/export_trees.py \
  --model-dir ../bp-monitor/results/track1/hybrid_upstream/20261012_holdout_v1/final/models \
  --output artifacts/models/trees.json \
  --fixtures-output artifacts/validation/tree_parity_fixtures.json \
  --prepared-dir artifacts/prepared

xcrun swiftc -O -parse-as-library -module-cache-path build/ModuleCache \
  App/Sources/TreeEnsemble.swift Tests/TreeParityMain.swift \
  -o artifacts/tree-parity
artifacts/tree-parity artifacts/models/trees.json \
  artifacts/validation/tree_parity_fixtures.json \
  > artifacts/validation/tree_parity_report.json

python tools/export_neural.py \
  --source-root ../bp-monitor \
  --prepared artifacts/prepared \
  --output artifacts/models \
  --seeds 1000 1001 1002 1003 1004

python tools/package_resources.py --project-root .
```

Packaging requires successful reports for all ten neural models and the native tree evaluator. It compiles the Core ML models into `App/Resources/Models`, copies the tree/configuration files, and creates `App/Resources/Data/sample_session.json` without reference predictions. Research checkpoints are never trained or modified.

The generated `.xcodeproj` is already supplied. If its source/resource inventory needs regeneration, run `python tools/generate_xcode_project.py` before making any personal signing changes.

## Verify

The recorded neural conversion check compares the saved Core ML packages with the original PyTorch models across eight real recordings × five views and five synthetic availability cases per model. All ten models passed the 0.005 mmHg tolerance; maximum observed conversion difference was **0.0000085831 mmHg**. [Conversion report](artifacts/models/neural_parity.json).

The Swift tree evaluator matched all ten LightGBM boosters exactly over **918 inputs**, including the eight prepared recordings, NaNs, zeros and split boundaries. [Tree report](artifacts/validation/tree_parity_report.json).

Run the complete native pipeline against historical saved outputs, including session correction and input-validation checks:

```sh
xcrun swiftc -O -parse-as-library -module-cache-path build/ModuleCache \
  App/Sources/PreparedRecording.swift App/Sources/TreeEnsemble.swift \
  App/Sources/InferenceEngine.swift Tests/InferenceParityMain.swift \
  -framework CoreML -o artifacts/inference-parity
artifacts/inference-parity App/Resources/Models \
  artifacts/validation/reference_session.json \
  artifacts/validation/inference_parity_report.json
```

The recorded Mac run of this shared native inference engine passed all eight historical comparisons, including the missing-ECG recording, input-validation checks and session arithmetic. Maximum absolute error across the four outputs was **0.0000938416 mmHg**. See the [native report](artifacts/validation/inference_parity_report.json) for individual results. The comparison tolerance is 0.005 mmHg; reference predictions are regression checks, not reference blood-pressure measurements.

For the simulator UI test, select an installed iPhone simulator in Xcode and choose **Product → Test**. `BPMonitorUITests` runs the bundled eight-record session, checks that all four calculated outputs appear, and saves screenshots in the test result. To choose the destination from the command line:

```sh
xcodebuild -project BPMonitor.xcodeproj -scheme BPMonitor -showdestinations
```

Use an available destination from that output for `xcodebuild test`, or use Xcode's selected simulator. Physical-device latency and behavior should be measured on the iPhone used for the demonstration; Mac conversion timings are not iPhone benchmarks.

Verification on 22 October 2026 used Xcode 27.1 (27A9275): the iPhone Release build passed with signing disabled, and the iPhone 16 / iOS 18.6 simulator UI test passed with all eight recordings processed and all four outputs displayed. See [validation evidence](docs/VALIDATION.md) and the captured [prepared session](docs/screenshots/prepared-session.png) and [calculated results](docs/screenshots/model-comparison.png). Installation and timing on a physical iPhone remain to be checked after selecting your signing Team.

## Local artifacts and provenance

Generated model/data assets, `artifacts/`, `build/` and `.venv/` are local outputs excluded by `.gitignore`; they are present in the delivered local project. Keep `artifacts/validation/resource_manifest.json` and preparation/conversion reports with an exported build to identify its inputs and model files.

This is a local research demonstrator, not a clinically validated blood-pressure measurement product. The pinned upstream snapshot has no license file; see the original [upstream NOTICE](../bp-monitor/src/bp_monitor_hybrid/upstream/NOTICE.md) and [attention-fusion NOTICE](../bp-monitor/src/bp_monitor_att_fusion/NOTICE.md). The project does not establish permission to redistribute the models or challenge data, and includes no publication or upload workflow.
