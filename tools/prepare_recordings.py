#!/usr/bin/env python3
"""Prepare portable, target-free iPhone demo recordings using the frozen Python pipeline.

The phone imports this JSON and performs actual model inference. Raw MAT extraction
runs here, including the historical float16 cache roundtrip. No BP model is fitted.
All exports retain opaque test IDs; names, contact fields and BP labels are excluded.

Example:
    python tools/prepare_recordings.py --source-root /path/to/bp-monitor \
      --data-root /path/to/bhi-2026-data-challenge-competition-track1 \
      --subjects 0013 0030 --output artifacts/prepared

Programmatic API: prepare(data_root, source_root, subject_ids) returns recording
objects. `reference` contains historical saved predictions when their IDs exist;
its hybrid entry includes session averaging and round correction; hybridRaw
is the raw 0.3 GBM + 0.7 Mean Beats blend. These are regression
references, not clinical truth or newly recomputed predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import re
import sys
import tempfile

# Loading existing repository modules must not modify their source directories.
sys.dont_write_bytecode = True

ORIGINAL_RUN = Path("results/track1/hybrid_upstream/20261012_holdout_v1/final")
MEAN_BEAT_RUN = Path("results/track1/att_fusion/m6_mean_beat_final_v2/20261016_full_refit_r2/final")


def _read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _saved_references(source_root, pd):
    """Return named original-pipeline outputs; labels are never accessed."""
    original = pd.read_csv(source_root / ORIGINAL_RUN / "test_predictions.csv", dtype={"ID": str}, float_precision="round_trip")
    mean = pd.read_csv(source_root / MEAN_BEAT_RUN / "test_predictions.csv", dtype={"ID": str}, float_precision="round_trip")
    result = {}
    for frame, pairs in ((original, {"lightGBM": "gbm_raw", "fusionNet": "fusion_raw"}),
                         (mean, {"meanBeats": "fusion_raw", "hybrid": "blend_round_corrected", "hybridRaw": "blend_raw"})):
        if not frame.ID.is_unique:
            raise ValueError("Historical reference IDs must be unique")
        for _, row in frame.iterrows():
            record = result.setdefault(row.ID, {})
            for model, prefix in pairs.items():
                record[model] = [float(row[prefix + "_SBP"]), float(row[prefix + "_DBP"])]
    return result


def _display_wave(values, np, points=1200):
    """Display-only 20 Hz block averages across the full minute, not model input."""
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(raw) != 30000 or not np.isfinite(raw).all():
        raise ValueError("Display signal must contain 30,000 finite source samples")
    reduced = raw.reshape(points, -1).mean(axis=1)
    # Center and scale only the visual trace, independently from inference arrays.
    centered = reduced - np.median(reduced)
    divisor = max(float(np.percentile(np.abs(centered), 99)), 1e-9)
    return (centered / divisor).astype(np.float32).tolist()


def _validate_record(record, np):
    expected = {"gbmFeatures": 161, "tab": 309, "accSc": 7, "beats": 32 * 5 * 120}
    for key, count in expected.items():
        values = record[key]
        if len(values) != count:
            raise ValueError(f"{record['id']}: {key} has wrong flattened length")
        numeric = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        if np.isinf(numeric).any() or (key != "gbmFeatures" and not np.isfinite(numeric).all()):
            raise ValueError(f"{record['id']}: invalid {key} values")
    if len(record["views"]) != 5 or len(record["beatValid"]) != 32:
        raise ValueError("Exactly five context views and 32 beat slots are required")
    for view in record["views"]:
        for key, channels in (("ppg", 16), ("ecg", 5), ("acc", 3)):
            if len(view[key]) != channels * 120 or not np.isfinite(view[key]).all():
                raise ValueError(f"Invalid {key} view")
    valid = np.asarray(record["beatValid"], dtype=bool)
    if bool(valid.any()) != record["beatPresent"]:
        raise ValueError("Beat availability disagrees with valid slots")
    if record["beatPresent"] and not record["ecgOK"]:
        raise ValueError("Available paired beats require a usable ECG reference")
    if np.asarray(record["beats"]).reshape(32, 5, 120)[~valid].any():
        raise ValueError("Padding slots must be zero")
    json.dumps(record, allow_nan=False)


def prepare(data_root, source_root, subject_ids):
    """Return prepared records for complete opaque test-subject sessions.

    Uses the official loader and pinned extraction/model-preparation functions.
    Fitted full-training statistics are loaded from the final Mean Beats bundle.
    Input subjects must be numeric test IDs (for example '0013'); train labels
    and personal identifiers are never copied into the exported schema.
    """
    source_root, data_root = Path(source_root).resolve(), Path(data_root).resolve()
    if str(source_root / "src") not in sys.path:
        sys.path.insert(0, str(source_root / "src"))
    import numpy as np
    import pandas as pd
    from bp_monitor_hybrid.runtime import load_upstream
    from bp_monitor_att_fusion.beat_extraction import extract_recording
    from bp_monitor_att_fusion.beat_data import normalize_beat_pool, selected_beat_indices

    subjects = [str(value) for value in subject_ids]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Provide unique test subject IDs")
    if any(re.fullmatch(r"[0-9]{4}", value) is None for value in subjects):
        raise ValueError("Subject IDs must be four digits from the test folder")
    upstream = load_upstream(data_root)
    preprocessing = _read(source_root / MEAN_BEAT_RUN / "preprocessing.json")
    model_manifest = _read(source_root / ORIGINAL_RUN / "models/manifest.json")
    columns = model_manifest["columns"]["gbm"]
    if columns != list(upstream.common.DEMO + upstream.common.SIGNAL) or len(columns) != 161:
        raise ValueError("Saved LightGBM feature schema differs from the frozen extractor")
    if preprocessing["columns"]["tab"] != list(upstream.common.DEMO + upstream.fusion.TAB_SIGNAL):
        raise ValueError("Saved neural feature schema differs")
    stats = {key: tuple(pd.Series(value[name], index=value["columns"], dtype=float)
                       for name in ("mean", "std_plus_epsilon"))
             for key, value in preprocessing["stats"].items()}
    references = _saved_references(source_root, pd)
    records = []
    for subject in subjects:
        path = data_root / "test" / f"{subject}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Test subject does not exist: {path}")
        segments = upstream.loader.load_subject_segments(path, channels=upstream.extract.CHANNELS, with_target=False)
        if not segments:
            raise ValueError(f"No recordings in test subject {subject}")
        for round_index, segment in enumerate(segments, start=1):
            extracted = extract_recording(upstream.extract, segment["signal_map"])
            if not extracted.features or set(extracted.templates) != {"ppg", "ecg", "acc"}:
                raise ValueError(f"Raw extraction failed for {segment['file_id']}: {extracted.failure_reason}")
            feature_row = dict(extracted.features)
            feature_row.update(
                ID=segment["file_id"], pid=subject, round=round_index,
                age=segment["age"], sex=str(segment["sex"]).lower(),
                height=segment["height"], weight=segment["weight"],
                med_his=str(segment["medicationHis"]).lower(),
                med_today=str(segment["medicationToday"]).lower(),
                med_type=str(segment["medicationType"]).lower(),
                sbp=np.nan, dbp=np.nan,
            )
            # Execute the original cache reader, including its CSV numeric
            # parser, demographic encodings, missing columns, and half->float
            # waveform loading. This avoids a second preprocessing definition.
            scratch_root = Path(__file__).resolve().parents[1] / "artifacts"
            scratch_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".prepare_", dir=scratch_root) as scratch:
                scratch = Path(scratch)
                pd.DataFrame([feature_row]).to_csv(scratch / "features.csv", index=False)
                np.savez(scratch / "templates.npz", **{
                    name: value.astype(np.float16)[None] for name, value in extracted.templates.items()})
                previous_cache = upstream.common.CACHE
                try:
                    upstream.common.CACHE = scratch
                    frame, templates = upstream.common.load_data(templates=True)
                finally:
                    upstream.common.CACHE = previous_cache
            arrays, _ = upstream.fusion.make_arrays(frame, templates, stats=stats)
            # Keep the exact full effective pool and cache quantization before K32.
            pool, _ = normalize_beat_pool(extracted.waveforms[extracted.effective_mask])
            selected = selected_beat_indices(len(pool))
            beats = np.zeros((32, 5, 120), dtype=np.float32)
            valid = np.zeros(32, dtype=bool)
            beats[:len(selected)] = pool[selected]
            valid[:len(selected)] = True
            raw_features = frame[columns].to_numpy(dtype=np.float64)[0]
            if np.isinf(raw_features).any():
                raise ValueError(f"Nonfinite engineered feature in {segment['file_id']}")
            record = {
                "schemaVersion": 1,
                "id": segment["file_id"], "subjectID": subject, "roundIndex": round_index,
                "display": {"age": float(segment["age"]), "sex": str(segment["sex"]).lower(),
                            "durationSeconds": 60,
                            "ppg": _display_wave(segment["signal_map"][upstream.loader.PPG_CHANNELS[0]], np),
                            "ecg": _display_wave(segment["signal_map"]["ECG"], np)},
                "gbmFeatures": [None if np.isnan(v) else float(v) for v in raw_features],
                "tab": arrays["tab"][0].numpy().tolist(),
                "accSc": arrays["acc_sc"][0].numpy().tolist(),
                "ecgOK": bool(arrays["ecg_ok"][0, 0]),
                "views": [{name: arrays[name][0, view].numpy().reshape(-1).tolist()
                           for name in ("ppg", "ecg", "acc")} for view in range(5)],
                "beats": beats.reshape(-1).tolist(), "beatValid": valid.tolist(),
                "beatPresent": bool(len(pool)),
            }
            if segment["file_id"] in references:
                record["reference"] = references[segment["file_id"]]
            _validate_record(record, np)
            records.append(record)
            print(f"Prepared {record['id']}: ECG={record['ecgOK']}, "
                  f"effective beats={len(pool)}, selected={int(valid.sum())}", file=sys.stderr, flush=True)
    return records


def write_prepared(records, output, *, source_root, data_root):
    output, source_root, data_root = map(Path, (output, source_root, data_root))
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for record in records:
        path = output / (record["id"] + ".json")
        path.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        names.append(path.name)
    subjects = list(dict.fromkeys(record["subjectID"] for record in records))
    sources = ["src/bp_monitor_hybrid/upstream/t1_common.py", "src/bp_monitor_hybrid/upstream/01_extract.py", "src/bp_monitor_hybrid/upstream/03_train_fusion.py",
               "src/bp_monitor_att_fusion/beat_extraction.py", "src/bp_monitor_att_fusion/beat_data.py",
               str(MEAN_BEAT_RUN / "preprocessing.json")]
    manifest = {
        "schemaVersion": 1, "recordings": names, "subjects": subjects,
        "preparation": {
            "kind": "official_test_recording_python_preparation", "modelTrainingPerformed": False,
            "targetLabelsIncluded": False, "personalNamesIncluded": False,
            "flattenOrder": "C_row_major", "contextViews": 5, "maximumBeats": 32,
            "referenceOrigin": "historical_saved_test_predictions_not_ground_truth",
            "hybridReferenceStage": "blend_round_corrected_after_full_subject_session_averaging",
            "displayWaveforms": "full_minute_20Hz_block_means_centered_and_scaled_for_display_only",
            "exporterSHA256": _hash(Path(__file__)),
            "loaderSHA256": _hash(data_root / "Data_Loader.py"),
            "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "torch", "pytorch-lightning")}},
            "sourceSHA256": {name: _hash(source_root / name) for name in sources},
            "rawSubjectSHA256": {subject: _hash(data_root / "test" / (subject + ".mat")) for subject in subjects},
            "files": {name: _hash(output / name) for name in names},
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["0013", "0030"])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/prepared")
    args = parser.parse_args()
    records = prepare(args.data_root, args.source_root, args.subjects)
    manifest = write_prepared(records, args.output, source_root=args.source_root, data_root=args.data_root)
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records), "subjects": manifest["subjects"]}))


if __name__ == "__main__":
    main()
