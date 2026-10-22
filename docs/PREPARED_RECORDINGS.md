# Prepared recording JSON, version 1

The iPhone app accepts the model-ready output of the Mac preparation script. Raw `.mat` loading, signal filtering, peak detection, engineered-feature extraction and normalization happen on the Mac. The app loads the resulting arrays and performs every model prediction locally.

## Prepare and import

From `bp-monitor-app`, with its conversion environment activated:

```sh
python tools/prepare_recordings.py \
  --source-root ../bp-monitor \
  --data-root ../bp-monitor/data/bhi-2026-data-challenge-competition-track1 \
  --subjects 0013 0030 \
  --output artifacts/prepared
```

The script accepts four-digit IDs from the official `test/` directory and exports every recording in each selected subject's session. It writes one `subject_XXXX_R.json` per recording and a `manifest.json` describing file checksums, source identities and package versions. The manifest is build metadata, not an importable session.

The programmatic interface is `prepare(data_root, source_root, subject_ids)`, which returns a list of recording objects. No target labels or personal names are copied. Known official IDs receive optional historical model references for validation; the app's decoder does not use those fields.

Import an individual recording JSON through **Open → Import prepared JSON**. For a complete session file, use the already packaged `App/Resources/Data/sample_session.json`, or combine a preparation manifest's records while removing references:

```sh
python - <<'PY'
import json
from pathlib import Path

folder = Path("artifacts/prepared")
manifest = json.loads((folder / "manifest.json").read_text())
records = []
for name in manifest["recordings"]:
    record = json.loads((folder / name).read_text())
    record.pop("reference", None)
    records.append(record)
(folder / "session.json").write_text(
    json.dumps({"schemaVersion": 1, "records": records}, separators=(",", ":")) + "\n"
)
PY
```

Transfer the JSON to the iPhone's Files app and import it. The bundled demonstration uses subjects `0013` and `0030`, four recordings each. `subject_0030_3` has no usable ECG reference and no available beats, exercising the trained missing-input paths.

## Container and identity

A file may contain one recording object or a session envelope:

```json
{
  "schemaVersion": 1,
  "records": []
}
```

The empty `records` array above only illustrates the envelope. A valid session contains **1–100 full recording objects**, unique recording IDs and at most one record per `(subjectID, roundIndex)`. Files must be nonempty and no larger than 64 MiB. The app supports round indices 1–5.

| Recording field | Type / size | Meaning |
|---|---|---|
| `schemaVersion` | Integer `1` | Versioned input contract |
| `id` | Nonempty string, at most 256 characters | Unique recording identity |
| `subjectID` | Nonempty string, at most 256 characters | Groups records for Hybrid postprocessing |
| `roundIndex` | Integer 1–5 | Original within-session recording order |
| `display` | Object | Waveform previews and optional age/sex metadata |
| `gbmFeatures` | 161 numbers or JSON `null` | Ordered raw engineered LightGBM features |
| `tab` | 309 finite numbers | Normalized neural tabular features and missingness flags |
| `accSc` | 7 finite numbers | Normalized acceleration scalar features |
| `ecgOK` | Boolean | Usable R-reference availability |
| `views` | 5 objects | Whole-minute and four quarter-minute context templates |
| `beats` | 19,200 finite numbers | Flattened `[32, 5, 120]` paired-beat tensor |
| `beatValid` | 32 booleans | Actual selected beats versus padding |
| `beatPresent` | Boolean | Whether the beat modality is available |

Each `views` object contains `ppg` with 1,920 values (`[16,120]`), `ecg` with 600 (`[5,120]`) and `acc` with 360 (`[3,120]`). All arrays use **C row-major order**, with sample index varying fastest. For beats, flattened index is `(beat * 5 + channel) * 120 + sample`.

Neural arrays are loaded as float32. LightGBM values retain Double precision; `null` represents a missing feature and follows the stored tree's missing-value routing. Nonfinite neural values are invalid. Do not replace a missing tree feature with zero, change feature order, or normalize LightGBM's inputs with neural statistics.

Channel order is:

- PPG context: IR sites 1–4, red sites 1–4, green sites 1–4, blue sites 1–4.
- Paired context and individual beats: ECG, wavelength-averaged IR, red, green, blue.
- Acceleration context: x, y, z.

`display` contains finite `ppg` and `ecg` arrays with 2–30,000 points, positive `durationSeconds` up to 3,600, and optional `age` / `sex`. The supplied exporter uses the full 60-second recording, reduced to 1,200 block means and centered/scaled for display. These preview arrays never feed inference.

## Preparation contract

The exporter imports the existing repository implementations rather than maintaining another signal-processing algorithm:

1. The official loader reads 16 PPG channels and ECG at 500 Hz and three acceleration axes at 100 Hz. BP targets are disabled.
2. The pinned extractor creates scalar features and five 120-sample template views. Its Butterworth forward/backward filtering, peak detection, PPG-guided ECG fallback, alignment, quality thresholds and sparse-quarter fallback remain in use.
3. The original cache reader performs its CSV numeric parsing, demographic encodings and missing-column handling. Templates pass through the historical float16 storage → float32 load roundtrip.
4. Context normalization uses saved full-training statistics. Nine demographic/session columns and 150 signal columns are standardized, clipped to ±5, missing values filled with zero, then 150 missingness flags are appended. Acceleration scalars use the saved seven-column transform. Waveform normalization matches the frozen neural pipeline.
5. The individual-beat extractor retains joint R-aligned ECG/PPG windows. It keeps IR-correlated beats above 0.7, or all in-bounds beats if fewer than three pass, and applies its float16 cache quantization.
6. The **entire effective beat pool** is normalized before selection: ECG by whole-pool maximum absolute magnitude; each PPG wavelength by whole-pool mean and population standard deviation, with epsilon `1e-6`. Statistics use float64 and output uses float32. If more than 32 beats remain, deterministic rank-bin midpoints select 32; shorter pools receive zero padding.

The full-minute beat pool is shared across all five context views. `beatValid` masks padding and `beatPresent` controls modality availability. Present beats require at least one valid beat and a usable ECG reference (`ecgOK: true`); inconsistent imports are rejected. Unavailable ECG also masks the paired ECG/R-PPG and acceleration context branches. Missing modalities use trained replacement embeddings; they are not inferred from the waveform preview or from an all-zero signal.

The exporter records source checksums, the raw subject-file checksums, the saved preprocessing identity and its numerical environment in the preparation manifest. It performs no training and never refits statistics on imported records.

## Outputs and regression references

The runtime returns pairs in **`[SBP, DBP]` order**, in mmHg. Every neural seed averages five standardized context views in float32 and applies `12 * prediction + bpCenter`. Basic FusionNet retains float32 seed averaging; Mean Beats promotes the seed mean to float64. Five LightGBM models are averaged separately for each target.

For record `r` of subject `s`, the requested standard Hybrid is:

```text
rawBlend[r] = 0.3 * LightGBM[r] + 0.7 * MeanBeats[r]
Hybrid[r] = mean(rawBlend for subject s)
            + offset[roundIndex[r]]
            - mean(offset for imported rounds of subject s)
```

The persisted neural share is `0.7000000000000001`. Grouping uses `subjectID`, not file position. Partial sessions are recentered over only their imported rounds; one-record sessions receive no net offset. A complete session is needed to match the archived full-session result. The separate high-BP stretch in the research outputs is outside this app's displayed Hybrid.

Optional preparation-only `reference` keys are `lightGBM`, `fusionNet`, `meanBeats`, `hybrid` and `hybridRaw`, each holding a historical `[SBP,DBP]` prediction. `hybrid` is the standard session-corrected output; `hybridRaw` precedes correction. They are not measured blood pressures. `package_resources.py` retains them only in `artifacts/validation/reference_session.json` and removes them from the shipping sample. The production decoder does not expose reference fields to the inference engine.

Use a **0.005 mmHg absolute tolerance** for converted neural/native pipeline regression checks. This measures numerical portability, not prediction accuracy or clinical validity. The full-refit checkpoints used all 900 labelled training subjects; the bundled official test subjects have no BP labels available locally.
