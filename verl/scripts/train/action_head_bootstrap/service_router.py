"""CPU HTTP routing: scene-affine renderer or least-inflight Qwen replica."""
import argparse
import threading
import requests
from flask import Flask, request, Response
from case_scheduler import renderer_index

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--upstreams', required=True)
p.add_argument('--mode', choices=['scene', 'least-inflight'], required=True)
a = p.parse_args()
urls = a.upstreams.split(',')
counts = [0] * len(urls)
next_index = 0
lock = threading.Lock()
app = Flask(__name__)

@app.route('/<path:path>', methods=['GET', 'POST'])
def route(path):
    global next_index
    body = request.get_json(silent=True) or {}
    with lock:
        if a.mode == 'scene':
            scene = str(body.get('scene_id') or request.args.get('scene_id') or '')
            index = renderer_index(scene, len(urls))
        else:
            # Pure ``min`` always selects replica 0 when all services are
            # idle, which heavily skews bursty short requests. Preserve
            # least-inflight routing, but round-robin among equal minima.
            minimum = min(counts)
            candidates = {i for i, value in enumerate(counts)
                          if value == minimum}
            index = next(
                i % len(urls)
                for i in range(next_index, next_index + len(urls))
                if i % len(urls) in candidates)
            next_index = (index + 1) % len(urls)
        counts[index] += 1
    try:
        upstream = requests.request(request.method, urls[index] + '/' + path,
            params=request.args, data=request.get_data(),
            headers={'Content-Type': request.content_type or 'application/json'}, timeout=1200)
        return Response(upstream.content, status=upstream.status_code,
                        content_type=upstream.headers.get('Content-Type'))
    finally:
        with lock:
            counts[index] -= 1

app.run(host='127.0.0.1', port=a.port, threaded=True)
