"""Bounded retries for stateless RGB-D reads; never replay navigation actions."""
import http.client
import json
import random
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener


def render_bytes(request, timeout=600):
    deadline=time.monotonic()+timeout
    attempt=0
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError('ObjectNav render transport deadline exceeded')
        try:
            with build_opener(ProxyHandler({})).open(request,timeout=remaining) as response:
                return response.read()
        except HTTPError as exc:
            exc.close()
            if exc.code not in (429,502,503,504):raise
            reason='HTTP '+str(exc.code)
        except (ConnectionError, URLError, http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
            reason=type(exc).__name__
        attempt+=1
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('ObjectNav render transport deadline exceeded')
        delay=min(remaining,min(10,2**min(attempt-1,4))+random.random())
        print(json.dumps({'event':'stateless_render_retry','attempt':attempt,'reason':reason,'delay_s':round(delay,2)}),flush=True)
        time.sleep(delay)
