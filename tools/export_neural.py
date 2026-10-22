#!/usr/bin/env python3
"""Export the frozen five-seed neural ensembles and prove FP32 inference parity.

This reads research source/checkpoints without modifying them. It writes portable
Core ML packages, per-view reference outputs, hashes, and a numerical QA report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn


class PortableEncoder(nn.Module):
    """Replace torch.diff with identical fixed slices; reuse every learned weight."""
    def __init__(self, encoder):
        super().__init__()
        self.conv = encoder.conv
        self.out = encoder.out
        self.paired = hasattr(encoder, 'local_out')
        if self.paired:
            self.local_out = encoder.local_out

    def forward(self, x, extra=None):
        derivative = torch.cat([x[..., :1]-x[..., :1], x[..., 1:]-x[..., :-1]], dim=-1)*4.0
        h = self.conv(torch.cat([x, derivative],dim=1))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)],dim=1)
        if extra is not None:
            pooled = torch.cat([pooled,extra],dim=1)
        global_ = self.out(pooled)
        if self.paired:
            return global_, self.local_out(h.transpose(1,2))
        return global_


def portable_copy(model):
    result = copy.deepcopy(model)
    for name, encoder in list(result.enc.items()):
        if hasattr(encoder,'conv'):
            result.enc[name] = PortableEncoder(encoder)
    if hasattr(result,'beat_encoder'):
        result.beat_encoder = PortableEncoder(result.beat_encoder)
    return result


class FusionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = portable_copy(model)

    def forward(self, tab, ppg, ecg, acc, acc_sc, ecg_ok):
        return self.model(dict(tab=tab, ppg=ppg, ecg=ecg, acc=acc,
                               acc_sc=acc_sc, ecg_ok=ecg_ok))


class MeanBeatsWrapper(nn.Module):
    """Fixed-size equivalent of the selected mean_beat_embeddings model only.

    Sanitize padding BEFORE the beat CNN, mask AFTER it, and use the learned
    missing embedding for an empty pool. No dynamic indexing or Python branch
    depends on data. Context token construction follows matched_global_v1.
    """
    def __init__(self, model):
        super().__init__()
        self.model = portable_copy(model)

    def forward(self, tab, ppg, ecg, acc, acc_sc, present, beats, valid, beat_present):
        model = self.model
        streams = (tab, ppg, ecg[:, :1, :], ecg[:, 1:, :], acc)
        names = ('tab', 'ppg_morph', 'ecg_r', 'ppg_r', 'acc')
        tokens = [model.cls]
        globals_ = []
        for i, name in enumerate(names):
            if name in ('ecg_r', 'ppg_r'):
                global_, local = model.enc[name](streams[i])
                token = (local + model.position).mean(dim=1, keepdim=True)
            elif name == 'acc':
                global_ = model.enc[name](streams[i], acc_sc)
                token = global_[:, None, :]
            else:
                global_ = model.enc[name](streams[i])
                token = global_[:, None, :]
            available = present[:, i:i+1] > 0.5
            globals_.append(torch.where(available, global_, model.missing[i]))
            typed = token + model.type_emb[i]
            missing = (model.missing[i] + model.type_emb[i])[None, None, :]
            tokens.append(torch.where(available[:, :, None], typed, missing))
        active = (valid > 0.5) & (beat_present[:, None] > 0.5)
        safe_beats = torch.where(active[:, :, None, None], beats, torch.zeros_like(beats))
        encoded = model.beat_encoder(safe_beats.reshape(32, 5, 120)).reshape(1, 32, 64)
        counts = active.sum(dim=1, keepdim=True).to(torch.float32)
        pooled = torch.where(active[:, :, None], encoded, torch.zeros_like(encoded)).sum(dim=1)
        pooled = pooled / counts.clamp(min=1.0)
        summary = torch.where(counts > 0, pooled, model.beat_missing[None, :])
        tokens.append((summary + model.beat_type)[:, None, :])
        sequence = torch.cat(tokens, dim=1)
        # All seven tokens are valid for matched_global_v1. Missing modalities
        # have learned replacement tokens, so there is no attention padding.
        updated = model.mixer(sequence)
        return model.head(torch.cat([updated[:, 0], torch.stack(globals_, dim=1).flatten(1)], dim=1))


SHAPES = {'tab': (1,309), 'ppg': (1,16,120), 'ecg': (1,5,120),
          'acc': (1,3,120), 'acc_sc': (1,7), 'ecg_ok': (1,1),
          'present': (1,5), 'beats': (1,32,5,120), 'valid': (1,32), 'beat_present': (1,)}
BASIC_NAMES = ('tab','ppg','ecg','acc','acc_sc','ecg_ok')
MEAN_NAMES = ('tab','ppg','ecg','acc','acc_sc','present','beats','valid','beat_present')


def inputs_for(record, view, family):
    values = dict(tab=record['tab'], ppg=view['ppg'], ecg=view['ecg'], acc=view['acc'],
                  acc_sc=record['accSc'], ecg_ok=[record['ecgOK']],
                  present=[1,1,record['ecgOK'],record['ecgOK'],record['ecgOK']],
                  beats=record['beats'], valid=record['beatValid'], beat_present=[record['beatPresent']])
    names = BASIC_NAMES if family == 'fusion' else MEAN_NAMES
    return {name: np.asarray(values[name], dtype=np.float32).reshape(SHAPES[name]) for name in names}


def original_prediction(model, family, data):
    tensor = {k: torch.from_numpy(v.copy()) for k,v in data.items()}
    if family == 'fusion':
        return model(tensor)
    return model(dict(tab=tensor['tab'], ppg_morph=tensor['ppg'], ecg_r=tensor['ecg'][:,:1],
                      ppg_r=tensor['ecg'][:,1:], acc=tensor['acc'], acc_sc=tensor['acc_sc'],
                      present=tensor['present'] > .5, beat_waveforms=tensor['beats'],
                      beat_valid_mask=tensor['valid'] > .5, beat_present=tensor['beat_present'] > .5,
                      beat_timestamps=torch.zeros((1,32), dtype=torch.float64)))


def synthetic_cases(family):
    generator = np.random.default_rng(20261022)
    names = BASIC_NAMES if family == 'fusion' else MEAN_NAMES
    result = []
    for case, count, ecg in [('all_available',32,1), ('partial_beats',7,1),
                              ('empty_beats',0,1), ('missing_ecg',0,0), ('one_beat',1,1)]:
        data = {name: generator.normal(0,.7,SHAPES[name]).astype(np.float32) for name in names}
        if family == 'fusion':
            data['ecg_ok'][:] = ecg
        else:
            data['present'][:] = [1,1,ecg,ecg,ecg]
            data['valid'][:] = 0
            data['valid'][:,:count] = 1
            data['beat_present'][:] = int(count > 0)
            # Verify invalid storage cannot contaminate predictions.
            data['beats'][:,count:] = np.nan
        result.append((case, data))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source_root, output, prepared, seeds):
    import coremltools as ct
    sys.path.insert(0, str(source_root/'src'))
    from bp_monitor_hybrid.runtime import load_upstream
    from bp_monitor_att_fusion.beat_models import BeatFusionNet
    upstream = load_upstream(source_root/'data/bhi-2026-data-challenge-competition-track1')
    original_run = source_root/'results/track1/hybrid_upstream/20261012_holdout_v1/final/models'
    mean_run = source_root/'results/track1/att_fusion/m6_mean_beat_final_v2/20261016_full_refit_r2'
    config = json.loads((mean_run/'final/preprocessing.json').read_text())
    center = np.asarray(config['bp_center'],np.float32)
    calibration = json.loads((mean_run/'final/calibration.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    (output/'inference_config.json').write_text(json.dumps(dict(
        bpCenter=center.tolist(),bpScale=12.0,neuralWeight=0.7000000000000001,
        roundOffsets=calibration['round_offsets']),indent=2)+'\n')
    records=[]
    if prepared:
        manifest = json.loads((prepared/'manifest.json').read_text())
        paths = [prepared/name for name in manifest['recordings']]
        for path in paths:
            item=json.loads(path.read_text())
            if 'id' in item and 'views' in item:
                records.append(item)
    report=dict(schemaVersion=1,torch=torch.__version__,coremltools=ct.__version__,
                precision='float32',computeUnits='CPU_ONLY',deploymentTarget='iOS17',
                toleranceMMHg=0.005,records=len(records),models=[],status='running')
    predictions={r['id']: {} for r in records}
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(42)
    for family in ('fusion','meanbeats'):
        names = BASIC_NAMES if family=='fusion' else MEAN_NAMES
        for seed in seeds:
            if family=='fusion':
                checkpoint=original_run/f'fusion_seed{seed}.pt'
                model=upstream.fusion.FusionNet(309,7)
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
                wrapper=FusionWrapper(model)
            else:
                checkpoint=mean_run/f'fits/seed{seed}/final.pt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert saved['variant']=='mean_beat_embeddings'
                model=BeatFusionNet('mean_beat_embeddings',seed=seed)
                model.load_state_dict(saved['state_dict'],strict=True)
                wrapper=MeanBeatsWrapper(model)
            model.eval(); wrapper.eval()
            cases=synthetic_cases(family)
            for record in records:
                for v,view in enumerate(record['views']):
                    cases.append((f"{record['id']}/view{v}", inputs_for(record,view,family)))
            wrapper_error=0.0
            reference={}
            with torch.no_grad():
                for name,data in cases:
                    expected=original_prediction(model,family,data).numpy()
                    actual=wrapper(*(torch.from_numpy(data[k]) for k in names)).numpy()
                    assert np.isfinite(actual).all(), name
                    err=float(np.max(np.abs(actual-expected))*12)
                    wrapper_error=max(wrapper_error,err)
                    if err>0.005:
                        raise AssertionError(f'Wrapper parity failed {family} {seed} {name}: {err}mmHg')
                    reference[name]=expected
                example=cases[0][1]
                traced=torch.jit.trace(wrapper,tuple(torch.from_numpy(example[k]) for k in names),check_trace=True)
            print(f'Converting {family} seed{seed}; wrapper error {wrapper_error:.8f} mmHg',flush=True)
            converted=ct.convert(traced,convert_to='mlprogram',minimum_deployment_target=ct.target.iOS17,
                inputs=[ct.TensorType(name=k,shape=SHAPES[k],dtype=np.float32) for k in names],
                outputs=[ct.TensorType(name='prediction',dtype=np.float32)],
                compute_precision=ct.precision.FLOAT32,compute_units=ct.ComputeUnit.CPU_ONLY)
            converted.short_description=f'Frozen BP {family} seed {seed}, standardized SBP DBP, FP32'
            converted.user_defined_metadata['checkpoint_sha256']=sha(checkpoint)
            converted.user_defined_metadata['preprocessing']='prepared-recording-v1'
            package=output/f'{family}_seed{seed}.mlpackage'
            converted.save(str(package))
            # Reload serialized package rather than only testing in-memory graph.
            deployed=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
            max_error=0.0
            errors=[]
            for name,data in cases:
                actual=np.asarray(deployed.predict(data)['prediction'])
                expected=reference[name]
                assert actual.shape==(1,2) and np.isfinite(actual).all(), name
                error=float(np.max(np.abs(actual-expected))*12)
                max_error=max(max_error,error)
                errors.append(dict(case=name,maxAbsMMHg=error))
                if error>0.005:
                    raise AssertionError(f'Core ML parity failed {family} seed{seed} {name}: {error}mmHg')
            for record in records:
                views=np.stack([reference[f"{record['id']}/view{v}"][0] for v in range(5)])
                mmhg=views.mean(axis=0)*np.float32(12)+center
                predictions[record['id']].setdefault(family,[]).append(mmhg.tolist())
            report['models'].append(dict(family=family,seed=seed,checkpointSHA256=sha(checkpoint),
                parameters=sum(p.numel() for p in model.parameters()),cases=len(cases),
                wrapperMaxAbsMMHg=wrapper_error,coreMLMaxAbsMMHg=max_error,caseErrors=errors))
            (output/'neural_parity.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'PASS {family} seed{seed}: {len(cases)} cases, max {max_error:.8f} mmHg',flush=True)
    for record_id,families in predictions.items():
        for family,values in list(families.items()):
            dtype = np.float32 if family == 'fusion' else np.float64
            families[family]=np.asarray(values,dtype).mean(axis=0).tolist()
    report['status']='passed'
    report['maxAbsMMHg']=max(m['coreMLMaxAbsMMHg'] for m in report['models'])
    (output/'neural_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'python_neural_predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='models'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(1000,1005)))
    args=parser.parse_args()
    export(args.source_root.resolve(),args.output.resolve(),args.prepared,args.seeds)
