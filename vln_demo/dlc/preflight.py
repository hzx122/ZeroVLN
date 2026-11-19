"""Check mounted dependencies and the exact runtime without starting services."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from supervise import standalone_sim_env


def main():
    repo = Path(__file__).resolve().parents[2]
    demo = repo / "vln_demo"
    run = Path(os.environ["VLN_RUN_DIR"])
    records = []

    def require(label, ok, detail):
        records.append({"check": label, "passed": bool(ok), "detail": detail})
        print(f"[preflight] {'ok' if ok else 'FAIL'} {label}", flush=True)

    def command(label, argv, env=None, timeout=180):
        log = run / "logs" / f"preflight-{label}.log"
        try:
            with log.open("w") as out:
                result = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT,
                                        env=env, timeout=timeout)
            require(label, result.returncode == 0, {"exit_code": result.returncode, "log": str(log)})
        except (OSError, subprocess.TimeoutExpired) as exc:
            require(label, False, {"error": str(exc), "log": str(log)})

    gv = Path(os.environ.get("GV", "/opt/navigation-runtime/venv"))
    rs = Path(os.environ.get("RS", "/workspace/agentic_sim/repo/ABot-Navigation/render_server"))
    runtime = Path("/opt/external/user-d5121be6/model/flagos-qwen38-public/runtime")
    workspace = Path(os.environ["DEMO_TERMINAL_CWD"])
    required = [
        gv / "bin/python", rs / "render_sim_dynamic.py",
        Path("/workspace/agentic_sim/3drs_gs_root"),
        Path("/workspace/agentic_sim/abotn_gs_root"),
        Path("/data/zerovln/SAGE-3D/sage_gs_root"),
        Path("/opt/external/user-9656ce61/ppu_sim/3drs_gs_root"),
        Path("/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/mall"),
        Path("/opt/external/user-e7a01947/Model/Qwen3.6-35B-A3B/config.json"),
        Path("/opt/external/user-d5121be6/model/Qwen3.8-Flash-Next/config.json"),
        runtime / "scripts/run_official.py", runtime / "config/rootfs-integrity.json",
        Path(os.environ["VLN_SLEEP_SCRIPT"]), Path(os.environ["VLN_PI_CLI"]),
        workspace / ".pi/settings.json", workspace / ".pi/extensions/vln-console.ts",
        workspace / ".pi/extensions/project-models.ts", workspace / ".pi/skills/vln-visual-reference",
        Path("/opt/external/user-d5121be6/vllm_log_filters.py"),
    ]
    agent_profile = os.environ.get('VLN_AGENT_MODEL_PROFILE', 'next')
    if agent_profile != 'next':
        profile = json.loads((demo/'vllm/sglang_profiles.json').read_text())[agent_profile]
        required = [p for p in required if not p.is_relative_to(runtime)
                    and 'Qwen3.8-Flash-Next' not in str(p)]
        required.extend([Path(profile['model_path'])/'config.json',demo/'vllm'/profile['launcher']])
    if "lepu" in os.environ.get("VLN_REQUIRED_BACKENDS", "").split(","):
        lepu = Path(os.environ.get("LEPU_ASSETS_DIR",
            "/opt/external/user-9656ce61/grpo/locnav_grpo/exp0822/assets_arr5/lepu"))
        required.extend(lepu / name for name in
                        ("walk.json", "floor_meta.json", "lepu_walk_union_g45.npy"))
        required.append(Path(os.environ.get("MALL_ROOT",
            "/opt/external/user-9656ce61/ppu_sim/3drs_gs_root")) /
            "jW00XG5o3g/gs/block0/point_cloud.ply")
    if os.environ.get("VLN_EXTRA_SERVICES") == "1":
        required.extend([repo / "third_party/LightNav-0/src/lightnav/serving/ws_server.py",
                         Path("/opt/external/user-d5121be6/LightNav-0/model/config.json"),
                         Path("/opt/external/user-d5121be6/model/Qwen3.8-27B/config.json")])
    require("mounted-paths", all(p.exists() for p in required),
            {"missing": [str(p) for p in required if not p.exists()]})
    require("workspace-config-link", (workspace / ".pi").resolve() == demo / "config/pi/SpatialOnboarding",
            str((workspace / ".pi").resolve()))
    required_commands = ("curl", "ss", "setsid", "bash", "flock", "nvidia-smi", "mount", "unshare")
    require("system-commands", all(shutil.which(c) for c in required_commands),
            {c: shutil.which(c) for c in required_commands})
    command("gpu-layout", [os.environ["VLN_SLEEP_PYTHON"], "-c", '''
import csv, io, json, subprocess, torch
rows=list(csv.reader(io.StringIO(subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True))))
assert [int(r[0]) for r in rows] == list(range(16)), rows
assert torch.cuda.device_count() == 16, torch.cuda.device_count()
devices=[]
for i, row in enumerate(rows):
    prop=torch.cuda.get_device_properties(i)
    uid=str(getattr(prop,"uuid", "") or "")
    if uid:
        assert uid.removeprefix("GPU-").lower() == row[1].strip().removeprefix("GPU-").lower(), (i, uid, row)
    devices.append({"index":i,"smi_uuid":row[1].strip(),"pci_bus_id":row[2].strip(),"torch_uuid":uid,"torch_name":prop.name})
print(json.dumps({"torch":torch.__version__,"devices":devices}, indent=2))
'''])
    render_env = os.environ.copy()
    render_env["PYTHONPATH"] = ":".join(map(str, [demo / "render_libs", rs / "submodules/diff-plane-rasterization", rs / "submodules/simple-knn", rs]))
    command("render-imports", [str(gv / "bin/python"), "-c",
            "import torch, diff_plane_rasterization, simple_knn._C, flask; print(torch.__version__)"], render_env)
    console_env = standalone_sim_env(os.environ)
    console_env["LD_LIBRARY_PATH"] = str(gv / "lib") + ":" + console_env.get("LD_LIBRARY_PATH", "")
    command("console-imports", [str(gv / "bin/python"), "-c",
            "import numpy, scipy, PIL, habitat_sim, habitat; print('console imports OK')"], console_env)
    nav = Path(os.environ.get("RYNNBRAIN_NAV_DIR",
        "/opt/navigation-runtime/VLMEvalKit/3rdparty/RynnBrain/navigation"))
    command("console-standalone", [str(gv / "bin/python"), "-c", '''
import argparse, runpy, sys
module = runpy.run_path(sys.argv[1])
args = argparse.Namespace(local_rank=0, rank=0, world_size=1, gpu=0, device="cpu")
module["init_distributed_mode"](args)
assert args.distributed is False, vars(args)
print("CPU simulator standalone initialization OK")
''', str(nav / "src/utils/dist.py")], console_env)
    # Exercise the real service import graph and launcher environment: importing
    # vllm alone missed a datasets/pyarrow failure in the old generic CLI.
    command("35b-entrypoint", ["bash", str(demo / "vllm/vllm-vln-35B.sh"), "--check"], timeout=240)
    # The CLI imports its dependency graph from CPFS; a cold metadata cache can
    # take several minutes even though no model or agent session is started.
    command("pi-version", [os.environ["VLN_NODE"], os.environ["VLN_PI_CLI"], "--version"], timeout=600)
    for tool in ("fd", "rg"):
        command("pi-" + tool, [str(Path(os.environ["PI_CODING_AGENT_DIR"]) / "bin" / tool), "--version"])
    if agent_profile == 'next':
        command("next-namespace", ["/usr/bin/python3", "-I", str(runtime / "scripts/run_official.py"), "/bin/true"], timeout=240)
    else:
        command("sglang-import", ["/usr/local/bin/python3", "-c", "import sglang; print(sglang.__version__)"], timeout=240)
    if os.environ.get("VLN_EXTRA_SERVICES") == "1":
        command("objectnav-entrypoint", ["bash", str(demo / "src/objectnav/start.sh"), "--check"], timeout=300)
        for component in ("lightnav", "mujoco"):
            command(component + "-entrypoint", ["bash", str(demo / "dlc/start_extra.sh"), component, "--check"], timeout=300)
        command("lightnav-client", ["/usr/local/bin/python3", "-c", "import websocket; print('websocket-client OK')"])
    # Hash startup code without logging project model configuration or credentials.
    source_files = [Path(__file__), Path(__file__).with_name("entry.sh"),
                    Path(__file__).with_name("defaults.sh"),
                    Path(__file__).with_name("prepare_pi_tools.sh"),
                    Path(__file__).with_name("attempt.sh"),
                    Path(__file__).with_name("supervise.py"), demo / "start_all.sh",
                    demo / "vllm/vllm38next", demo / "vllm/launch_next.py",
                    demo / "vllm/next_prefix_cache.py", demo / "vllm/vllm-vln-35B.sh",
                    demo / "src/run_console.sh", demo / "src/vln_model.py",
                    demo / "src/habitat_topdown.py",
                    demo / "src/sage/env.py",
                    demo / "src/abotn/env.py",
                    demo / "src/hsim/env.py",
                    demo / "src/rs3d/env.py",
                    demo / "src/lightnav_mujoco/env.py",
                    demo / "src/vln_gym/env.py",
                    demo / "src/vln_gym/server.py", repo / "verl/action_head/render_client.py", repo / "verl/action_head/agent_runtime.py",
                    demo / "src/terminal_workspace.py",
                    demo / "src/web/js/store.js", demo / "src/web/js/panels/log.js",
                    demo / "src/web/js/panels/instr.js", demo / "src/web/js/panels/terminal.js",
                    demo / "src/remote_habitat/server.py", demo / "src/remote_habitat/client.py",
                    demo / "src/remote_habitat/start.sh",
                    demo / "src/remote_habitat/policy.py", demo / "src/remote_habitat/worker.py",
                    demo / "src/vln_demo.yaml", demo / "src/agent_tools.py", demo / "src/visual_navigation_tools.py",
                    demo / "src/vlnce_training.py", demo / "src/rxr.py",
                    demo / "src/lepu/assets.py", demo / "src/lepu/env.py",
                    demo / "src/lepu/floorplan.png", demo / "src/mall/assets.py",
                    demo / "src/mall/env.py", demo / "src/visual_navigation/planning.py",
                    demo / "src/webui.py", demo / "src/prompts.py",
                    demo / "dlc/start_extra.sh", demo / "dlc/lightnav_probe.py",
                    demo / "vllm/sglang-qwen3.8-27B-tp1.sh", demo / "config/pi/project-models.ts"]
    receipt = {"at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
               "image": os.environ.get("DOCKER_IMAGE_URL"), "passed": all(r["passed"] for r in records),
               "checks": records, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.exists()}}
    (run / "preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
