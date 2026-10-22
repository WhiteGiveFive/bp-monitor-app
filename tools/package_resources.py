#!/usr/bin/env python3
"""Package only parity-checked models and input-only recordings for the app."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def package(root):
    source = root/'artifacts/models'
    report = json.loads((source/'neural_parity.json').read_text())
    required = {(family,seed) for family in ('fusion','meanbeats') for seed in range(1000,1005)}
    actual = {(m['family'],m['seed']) for m in report['models']}
    if report['status']!='passed' or actual!=required:
        raise ValueError('All ten neural models must pass conversion parity before packaging')
    tree_report = json.loads((root/'artifacts/validation/tree_parity_report.json').read_text())
    if tree_report.get('status') not in ('passed','pass'):
        raise ValueError('Native tree parity must pass before packaging')
    models=root/'App/Resources/Models'
    data=root/'App/Resources/Data'
    models.mkdir(parents=True,exist_ok=True)
    data.mkdir(parents=True,exist_ok=True)
    for family,seed in sorted(required):
        subprocess.run(['xcrun','coremlcompiler','compile',str(source/f'{family}_seed{seed}.mlpackage'),
                        str(models),'--platform','ios','--deployment-target','17.0'],check=True)
    for filename in ('trees.json','inference_config.json'):
        shutil.copy2(source/filename,models/filename)
    prepared=root/'artifacts/prepared'
    manifest=json.loads((prepared/'manifest.json').read_text())
    records=[]
    for name in manifest['recordings']:
        path=prepared/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['preparation']['files'][name]:
            raise ValueError(f'Prepared recording checksum mismatch: {name}')
        records.append(json.loads(path.read_text()))
    validation=root/'artifacts/validation'
    (validation/'reference_session.json').write_text(json.dumps(dict(schemaVersion=1,records=records),separators=(',',':'))+'\n')
    # The shipping sample contains inputs only. Saved outputs cannot populate UI.
    inputs=[{k:v for k,v in record.items() if k!='reference'} for record in records]
    (data/'sample_session.json').write_text(json.dumps(dict(schemaVersion=1,records=inputs),separators=(',',':'))+'\n')
    hashes={str(p.relative_to(root/'App/Resources')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'App/Resources').rglob('*')) if p.is_file()}
    receipt=dict(schemaVersion=1,status='packaged',records=len(records),
                 subjects=manifest['subjects'],modelCount=10,treeBoosters=10,
                 sampleContainsPredictions=False,resourceSHA256=hashes)
    (validation/'resource_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'Packaged 10 neural models, 10 tree boosters, and {len(records)} input-only records.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    package(parser.parse_args().project_root.resolve())
