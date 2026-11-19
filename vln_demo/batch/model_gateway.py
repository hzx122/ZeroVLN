"""Transparent per-run model admission with additive increase/multiplicative decrease."""
from collections import deque
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import select
import socket
import threading
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


class AIMD:
    def __init__(self, maximum=64, initial=32, clock=time.monotonic):
        self.maximum = maximum
        self.limit = min(initial, maximum)
        self.active = 0
        self.queue = deque()
        self.condition = threading.Condition()
        self.clock = clock
        self.last_decrease = float('-inf')
        self.previous_preemptions = None
        self.pressure_samples = 0
        self.reason = 'initial'
        self.completed = 0

    def decrease(self, reason):
        with self.condition:
            if self.clock() - self.last_decrease >= 10:
                self.limit = max(1, self.limit // 2)
                self.last_decrease = self.clock()
                self.reason = reason
                self.condition.notify_all()

    def sample(self, preemptions, kv_usage, waiting):
        with self.condition:
            preempted = self.previous_preemptions is not None and preemptions > self.previous_preemptions
            self.previous_preemptions = preemptions
            self.pressure_samples = self.pressure_samples + 1 if kv_usage >= .90 or waiting > 0 else 0
            if preempted or self.pressure_samples >= 2:
                self.decrease('preemption' if preempted else 'server_pressure')
                self.pressure_samples = 0
            elif (kv_usage < .80 and waiting == 0 and self.clock() - self.last_decrease >= 20
                  and (self.queue or self.active >= self.limit)):
                self.limit = min(self.maximum, self.limit + 1)
                self.reason = 'additive_increase'
                self.condition.notify_all()

    def acquire(self, cancelled):
        ticket = object()
        with self.condition:
            self.queue.append(ticket)
            try:
                while self.queue[0] is not ticket or self.active >= self.limit:
                    if cancelled():
                        return False
                    self.condition.wait(.25)
                if cancelled():
                    return False
                self.active += 1
                return True
            finally:
                self.queue.remove(ticket)
                self.condition.notify_all()

    def release(self):
        with self.condition:
            self.active -= 1
            self.completed += 1
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(limit=self.limit, active=self.active, queued=len(self.queue),
                        maximum=self.maximum, completed=self.completed, reason=self.reason)


def metrics_values(text):
    prefix = 'sglang:' if any(line.startswith('sglang:') for line in text.splitlines()) else 'vllm:'
    def value(name, aggregate=sum):
        values = [float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
                  if line.startswith(prefix + name + '{') or line.startswith(prefix + name + ' ')]
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError('missing/nonfinite model metric: ' + name)
        return aggregate(values)
    if prefix == 'sglang:':
        return value('num_retracted_requests_total'), value('token_usage', max), value('num_queue_reqs')
    return value('num_preemptions_total'), value('kv_cache_usage_perc'), value('num_requests_waiting')


@contextmanager
def model_gateway(upstream, maximum, run):
    endpoint = urlsplit(upstream)
    if endpoint.scheme != 'http':
        raise ValueError('resident model binding must be private HTTP')
    admission = AIMD(maximum)
    closing = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def gone(self):
            if closing.is_set():
                return True
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b''
            except OSError:
                return True

        def do_GET(self):
            if self.path == '/health':
                body = json.dumps(admission.snapshot()).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/v1/models':
                self.forward(False)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == '/tokenize':
                # Budget probes use the very same brain as generation. They
                # allocate no decode/KV slot and must retain image UUIDs.
                self.forward(False)
                return
            if self.path not in ('/v1/chat/completions', '/v1/completions', '/v1/responses'):
                self.send_error(404)
                return
            self.forward(True)

        def forward(self, gated):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            acquired = False
            connection = None
            sent = False
            try:
                if gated:
                    acquired = admission.acquire(self.gone)
                    if not acquired:
                        return
                connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3600)
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in ('host', 'connection', 'transfer-encoding', 'accept-encoding')}
                headers['Accept-Encoding'] = 'identity'
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                if response.status in (429, 502, 503, 504):
                    admission.decrease('http_' + str(response.status))
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in ('connection', 'transfer-encoding'):
                        self.send_header(key, value)
                self.send_header('Connection', 'close')
                self.end_headers()
                sent = True
                while data := response.read1(65536):
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # The episode was stopped; close upstream, never replay it.
            except (OSError, http.client.HTTPException) as exc:
                admission.decrease(type(exc).__name__)
                if not sent and not self.gone():
                    self.send_error(502, 'Resident model connection failed')
            finally:
                if connection:
                    connection.close()
                if acquired:
                    admission.release()
                self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()

    def monitor():
        metrics_url = f'{endpoint.scheme}://{endpoint.netloc}/metrics'
        while not closing.is_set():
            record = dict(timestamp=time.time())
            try:
                with build_opener(ProxyHandler({})).open(metrics_url, timeout=3) as response:
                    preemptions, kv, waiting = metrics_values(response.read().decode())
                admission.sample(preemptions, kv, waiting)
                record.update(preemptions=preemptions, kv_usage=kv, server_waiting=waiting)
            except (OSError, ValueError) as exc:
                # Missing metrics freeze increases rather than guessing capacity.
                record['metrics_error'] = type(exc).__name__
            record.update(admission.snapshot())
            try:
                with (Path(run) / 'model-admission.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
            except OSError as exc:
                print('VLN_ADMISSION_LOG_ERROR ' + str(exc), flush=True)
            print('VLN_MODEL_ADMISSION ' + json.dumps(record), flush=True)
            closing.wait(10)

    monitoring = threading.Thread(target=monitor, daemon=True)
    monitoring.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        closing.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
        monitoring.join(timeout=5)
