"""Snapshot ended collection cases, then run the reusable conservative filter."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from prepare_rule_teacher_three_source import ART, save, sha256
from sft.rule_teacher_anchor_dedup import main as filter_main, POLICY, SELECTIVE_POLICY


def snapshot_collections(bindings, output):
    """Reuse one snapshot contract for worker roots and formal shard partitions."""
    output = Path(output).resolve()
    if not output.is_relative_to(ART) or (output/'case-manifest.json').exists():
        raise ValueError('fresh authorized output required')
    cases, seen = [], set()
    for binding in bindings:
        cohort, value = binding.split('=', 1)
        root = Path(value).resolve()
        results = sorted(root.glob('worker*/collections/*/shard_*/case_*/result.json'))
        results += sorted(root.glob('case_*/result.json'))
        for result in results:
            folder = result.parent.resolve()
            if folder in seen:
                raise ValueError('duplicate snapshot case:'+str(folder))
            seen.add(folder)
            if not (folder/'case.json').is_file():
                raise ValueError('missing ended case metadata:'+str(folder))
            frozen = sha256(result)
            executions = []
            for execution in sorted((folder/'options').glob('*/execution.json')):
                raw = execution.read_bytes()
                if json.loads(raw).get('accepted_for_training') is True:
                    executions.append(dict(relative_path=str(execution.relative_to(folder)),
                        sha256=hashlib.sha256(raw).hexdigest()))
            if sha256(result) != frozen:
                raise ValueError('case changed while snapshotting:'+str(folder))
            cases.append(dict(path=str(folder), source=cohort, executions=executions,
                sha256={'case.json':sha256(folder/'case.json'), 'result.json':frozen}))
            save(output/'snapshot-progress.json', dict(cases=len(cases), updated_utc=time.time()))
    if not cases:
        raise ValueError('no ended collection cases to filter')
    manifest = output/'case-manifest.json'
    save(manifest,dict(snapshot_utc=time.time(),cases=cases))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', action='append', required=True,
                        help='cohort=/absolute/source/root with worker*/collections/*/shard_* layout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=[POLICY, SELECTIVE_POLICY], default=POLICY)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    manifest = snapshot_collections(args.collection, output)
    cases = json.loads(manifest.read_text())['cases']
    print(json.dumps(dict(snapshot_cases=len(cases), anchors=sum(len(x['executions']) for x in cases))),flush=True)
    filter_main(['--case-manifest',str(manifest),'--output',str(output),'--policy',args.policy])


if __name__ == '__main__':
    main()
