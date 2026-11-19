"""Observe runtime versions without making source edits abort an evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path


def runtime_difference(before, after):
    old, new = before.get('files', {}), after.get('files', {})
    files = {path: {'before': old.get(path), 'after': new.get(path)}
             for path in sorted(old.keys() | new.keys()) if old.get(path) != new.get(path)}
    metadata = {key: {'before': before.get(key), 'after': after.get(key)}
                for key in sorted(before.keys() | after.keys())
                if key != 'files' and before.get(key) != after.get(key)}
    return dict(files=files, metadata=metadata)


def record_runtime_change(root, before, after, *, phase):
    if before == after:
        return
    event = dict(timestamp=datetime.now(timezone.utc).isoformat(), phase=phase,
                 policy='log_only', **runtime_difference(before, after))
    with (Path(root)/'version-drift.jsonl').open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    print('VLN_RUNTIME_CHANGED ' + json.dumps(event, ensure_ascii=False), flush=True)


def observe_runtime(root, previous, current, *, phase):
    record_runtime_change(root, previous, current, phase=phase)
    # Keep the initial ladder-runtime.json intact; store the latest observation separately.
    path = Path(root)/'runtime-latest.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)
    return current
