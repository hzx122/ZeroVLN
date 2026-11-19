"""Native LightNav login/reset probe, optionally followed by image inference."""
import argparse
import base64
import json
import math
import os
from pathlib import Path


def probe(host, image=None):
    # This helper runs in its own process; private traffic never uses a proxy.
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    import websocket
    ws = websocket.create_connection(f"ws://{host}:8050", timeout=180 if image else 3)
    try:
        def request(action, data):
            ws.send(json.dumps({"action": action, "data": data}))
            reply = json.loads(ws.recv())
            if reply.get("action") != action or reply.get("data", {}).get("rc") != 0:
                raise RuntimeError(f"LightNav {action} failed: {reply}")
            return reply["data"]
        request("login", {"clientId": "dlc-health"})
        request("reset", {})
        result = {"ok": True, "protocol": "LightNav login/reset"}
        if image:
            reply = request("next", {"seq": 1, "image": base64.b64encode(Path(image).read_bytes()).decode(),
                                     "instruction": "Walk forward through the room."})
            rows = reply.get("actions", {}).get("actions", [])
            if len(rows) != 10 or any(len(row) != 3 or any(not math.isfinite(x) for x in row) for row in rows):
                raise RuntimeError("LightNav did not return ten finite 3D waypoints")
            result["prediction"] = reply
        return result
    finally:
        ws.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--image")
    args = parser.parse_args()
    print(json.dumps(probe(args.host, args.image)))
