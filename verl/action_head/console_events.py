"""Optional append-only observation of evaluator output for the existing console."""
import json
import os
import threading

_LOCK = threading.Lock()


def model_event(record):
    return dict(t='model', turn=record['turn'], text=record.get('assistant', ''),
                reasoning=record.get('reasoning', ''), dt=float(record.get('dt_s', 0)), actions=[])


def call_events(call):
    return [dict(t='tool', name=call['name'], args=call.get('arguments', {})),
            dict(t='obs', text=json.dumps(call.get('result', call), ensure_ascii=False))]


def emit(event):
    path = os.environ.get('ACTION_HEAD_CONSOLE_EVENTS')
    if not path:
        return
    # One serialized append per event; readers retain incomplete trailing lines.
    with _LOCK:
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()


class Calls(list):
    def append(self, call):
        super().append(call)
        if os.environ.get('ACTION_HEAD_CONSOLE_EVENTS'):
            for event in call_events(call):
                emit(event)


def replay(row):
    for turn in row.get('turns', []):
        yield model_event(turn)
        for call in turn.get('calls', []):
            yield from call_events(call)
    yield dict(t='done', text=json.dumps({k: row.get(k) for k in (
        'scene', 'episode_id', 'success', 'spl', 'agent_stopped', 'calls_used', 'why', 'infra_error')},
        ensure_ascii=False))


class Tail:
    def __init__(self):
        self.path = None
        self.offset = 0

    def read(self, path):
        if path != self.path:
            self.path, self.offset = path, 0
        from dlc_control_io_retry import retry_estale
        offset = self.offset
        def read_complete():
            if not path.exists():
                return [], offset
            events, next_offset = [], offset
            with path.open(encoding='utf-8') as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith('\n'):
                        break
                    events.append(json.loads(line))
                    next_offset = stream.tell()
            return events, next_offset
        events, self.offset = retry_estale(read_complete, str(path))
        return events
