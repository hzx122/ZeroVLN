"""Verified resume state; incomplete attempts never count as completed archives."""
import fcntl
import json
from pathlib import Path
import shutil
import time
import uuid

from workspace import freeze_template, sha256, verified_result
from version_audit import record_runtime_change


def episode_directory(run, row):
    return Path(run) / 'episodes' / row['dataset'] / row['split'] / row['role'] / row['language'] / row['logical_id']


def allocate_attempt(run, row):
    directory = episode_directory(run, row)
    directory.mkdir(parents=True, exist_ok=True)
    # Reservations survive failed archiving and container restarts.
    index = 1
    while True:
        name = f'attempt-{index:03d}'
        try:
            with (directory / ('.' + name + '.reserved')).open('x'):
                pass
        except FileExistsError:
            index += 1
            continue
        if (directory / name).exists():
            index += 1
            continue
        return directory / name


def prepare_run(run, rows, configuration):
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = run / 'manifest.jsonl'
        encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        if manifest.exists():
            saved = [json.loads(line) for line in manifest.read_text().splitlines()]
            if saved != rows:
                raise ValueError('resume manifest differs from frozen manifest')
        else:
            manifest.write_text(encoded)
        if (run / 'template').exists():
            template = json.loads((run / 'template/manifest.json').read_text())
            if sha256((run / 'template/AGENTS.md').read_bytes()) != template['sha256']:
                raise ValueError('frozen template is corrupt')
        else:
            template = freeze_template(run)
        configuration.update(template=template)
        path = run / 'run.json'
        if path.exists():
            saved = json.loads(path.read_text())
            for key in ('model_url', 'controller_model_url', 'render_url', 'render_mode', 'timeout', 'agent_profile'):
                if saved.get(key) != configuration.get(key):
                    raise ValueError('resume runtime configuration differs: ' + key)
        else:
            path.write_text(json.dumps(configuration, default=str, ensure_ascii=False, indent=2) + '\n')
        with (run / 'restarts.jsonl').open('a') as out:
            out.write(json.dumps(dict(timestamp=time.time(), git_commit=configuration['git_commit'])) + '\n')
        return lock, template
    except BaseException:
        lock.close()
        raise


def load_results(run, rows, normal_only=False):
    found = {}
    for row in rows:
        for attempt in sorted(episode_directory(run, row).glob('attempt-*'), reverse=True):
            try:
                result = verified_result(attempt)
                if any(result.get(key) != value for key, value in row.items()):
                    continue
                if normal_only and (result.get('state') != 'finished' or result.get('reason') != 'stop'):
                    continue
                result['archive'] = str(attempt)
                found[row['logical_id']] = result
                break
            except (OSError, ValueError, KeyError) as exc:
                print('VLN_ARCHIVE_REJECTED ' + json.dumps(dict(path=str(attempt), error=str(exc))), flush=True)
    return found


def import_results(source, run, rows):
    if source is None:
        return
    source = Path(source).resolve()
    if source == run:
        raise ValueError('resume_from must refer to a previous run')
    old = json.loads((source / 'run.json').read_text())
    new = json.loads((run / 'run.json').read_text())
    for key in ('model_url', 'render_url', 'render_mode', 'timeout'):
        if old.get(key) != new.get(key):
            raise ValueError('previous run has a different setting: ' + key)
    record_runtime_change(run, {'template': old['template']}, {'template': new['template']},
                          phase='resume_from_template')
    existing = load_results(run, rows)
    for logical_id, result in load_results(source, rows, normal_only=True).items():
        if logical_id in existing:
            continue
        destination = allocate_attempt(run, result)
        temporary = destination.with_name('.' + destination.name + '.import-' + uuid.uuid4().hex)
        shutil.copytree(result['archive'], temporary, symlinks=True)
        verified_result(temporary)
        temporary.rename(destination)
        with (run / 'imports.jsonl').open('a') as out:
            out.write(json.dumps(dict(logical_id=logical_id, source=result['archive'], destination=str(destination))) + '\n')
