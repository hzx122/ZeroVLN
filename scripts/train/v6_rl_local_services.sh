#!/usr/bin/env bash
# Isolated local v6 RL environments. Only the renderer can see physical GPU 15.
# start/run own their children; probe resets these gyms and is for preflight only.
set -euo pipefail

REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python3}
: "${EXPR_NAME:?Set a unique EXPR_NAME for this local RL experiment}"
[[ "$EXPR_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid EXPR_NAME' >&2; exit 2; }
ARTIFACTS=${RL_ARTIFACT_ROOT:-/data/closed_loop_3dgs_rl}
export SERVICES_OUTPUT_DIR=${SERVICES_OUTPUT_DIR:-$ARTIFACTS/${EXPR_NAME}-services}
export TRAIN_GYMS=${TRAIN_GYMS:-8} VALIDATION_GYMS=${VALIDATION_GYMS:-1}
export RENDER_PORT=${RENDER_PORT:-7048} TRAIN_PORT_BASE=${TRAIN_PORT_BASE:-8950}
export VALIDATION_PORT_BASE=${VALIDATION_PORT_BASE:-8970}
export SERVICE_START_TIMEOUT=${SERVICE_START_TIMEOUT:-1200}
export SAGE_RENDER_DEVICE=${SAGE_RENDER_DEVICE:-cuda}
export SAGE_CPU_RENDER_PARALLELISM=${SAGE_CPU_RENDER_PARALLELISM:-1}
export RENDERER_CACHE_SCENES=${RENDERER_CACHE_SCENES:-32}
export TRAIN_SPLIT=train
export CPU_RENDER_PORT=${CPU_RENDER_PORT:-18480}
export CPU_RENDER_PARALLELISM=${CPU_RENDER_PARALLELISM:-4}
export CPU_RENDER_CACHE_SCENES=${CPU_RENDER_CACHE_SCENES:-32}
export CPU_RENDER_SHARED_CACHE=${CPU_RENDER_SHARED_CACHE:-0}
export CPU_RENDER_CACHE_MAX_GIB=${CPU_RENDER_CACHE_MAX_GIB:-128}
export CPU_RENDER_SCENE_LOAD_RESERVE_GIB=${CPU_RENDER_SCENE_LOAD_RESERVE_GIB:-16}
: "${THREE_SOURCE_CASE_MANIFEST:?Set the three-source task pool}"
export THREE_SOURCE_CASE_MANIFEST
export VALIDATION_SPLIT=official_gvln_action_head_p0_v3test100
export PYTHONDONTWRITEBYTECODE=1

# All process control verifies /proc start times before signaling a saved PID.
control() {
  "$PYTHON" - "$@" <<'PY'
import base64, hashlib, json, os, signal, socket, sys, time, urllib.request
from pathlib import Path

action = sys.argv[1]
root = Path(os.environ['SERVICES_OUTPUT_DIR']).resolve()
artifacts = Path(os.environ.get('RL_ARTIFACT_ROOT', '/data/closed_loop_3dgs_rl')).resolve()
if not root.is_relative_to(artifacts) or root == artifacts:
    raise ValueError('SERVICES_OUTPUT_DIR must be an experiment directory on the data disk')
def atomic_json(path, data):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)

import importlib.util, contextlib
helper_path = Path(os.environ['REPO'])/'scripts/train/dlc_control_io_retry.py'
helper_spec = importlib.util.spec_from_file_location('_service_control_io_retry', helper_path)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
helper.install(sys.modules[__name__], root, stats=True)
control_context = contextlib.ExitStack()
control_context.enter_context(helper.bounded(time.monotonic()+int(os.environ['SERVICE_START_TIMEOUT'])))

train_count, val_count = int(os.environ['TRAIN_GYMS']), int(os.environ['VALIDATION_GYMS'])
if train_count not in (8, 16, 32, 64) or not 0 <= val_count <= 16:
    raise ValueError('TRAIN_GYMS must be 8, 16, 32 or 64; VALIDATION_GYMS must be in [0,16]')
train = [int(os.environ['TRAIN_PORT_BASE']) + i for i in range(train_count)]
validation = [int(os.environ['VALIDATION_PORT_BASE']) + i for i in range(val_count)]
render_port = int(os.environ['RENDER_PORT'])
cpu_render_port = int(os.environ["CPU_RENDER_PORT"])
ports = [render_port, cpu_render_port, *train, *validation]
if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
    raise ValueError('service ports must be distinct and in [1024,65535]')
if int(os.environ['SERVICE_START_TIMEOUT']) <= 0 or int(os.environ['RENDERER_CACHE_SCENES']) <= 0:
    raise ValueError('startup timeout and renderer cache size must be positive')
native_cache, native_parallelism = int(os.environ['CPU_RENDER_CACHE_SCENES']), int(os.environ['CPU_RENDER_PARALLELISM'])
if not 1 <= native_cache <= 512 or not 1 <= native_parallelism <= min(native_cache, 16):
    raise ValueError('invalid native renderer cache or parallelism')
native_shared = os.environ['CPU_RENDER_SHARED_CACHE']
native_max_gib = int(os.environ['CPU_RENDER_CACHE_MAX_GIB'])
native_reserve_gib = int(os.environ['CPU_RENDER_SCENE_LOAD_RESERVE_GIB'])
if native_shared not in ('0', '1') or not 1 <= native_reserve_gib <= native_max_gib <= 512:
    raise ValueError('invalid native renderer shared cache memory budget')
sage_parallelism = int(os.environ['SAGE_CPU_RENDER_PARALLELISM'])
if not 1 <= sage_parallelism <= 16 or (os.environ['SAGE_RENDER_DEVICE'] != 'cpu' and sage_parallelism != 1):
    raise ValueError('SAGE CPU render parallelism must be 1..16; GPU mode requires 1')
plan = {
    'schema': 'v6_rl_local_services_v1', 'output_dir': str(root),
    'renderer_gpu': None if os.environ['SAGE_RENDER_DEVICE'] == 'cpu' else 15,
    'renderer_device': os.environ['SAGE_RENDER_DEVICE'], 'renderer_url': f'http://127.0.0.1:{render_port}',
    'sage_cpu_render_parallelism': sage_parallelism,
    'train_split': os.environ['TRAIN_SPLIT'], 'validation_split': os.environ['VALIDATION_SPLIT'],
    'train_urls': [f'http://127.0.0.1:{p}' for p in train],
    'validation_urls': [f'http://127.0.0.1:{p}' for p in validation],
    'gym_max_steps': 0, 'gpu_occupier': False,
    'cpu_renderer_url': f'http://127.0.0.1:{cpu_render_port}',
    'cpu_renderer_parallelism': native_parallelism,
    'cpu_renderer_cache_scenes': native_cache,
    'cpu_renderer_shared_cache': native_shared == '1',
    'cpu_renderer_cache_max_gib': native_max_gib,
    'cpu_renderer_scene_load_reserve_gib': native_reserve_gib,
    'simulation_request_timeout_s': None,
    'case_manifest': os.environ['THREE_SOURCE_CASE_MANIFEST'],
}
pidfile = root / 'owned_pids.jsonl'
def proc(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'start_ticks': fields[19], 'state': fields[0], 'pgid': int(fields[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None
def owned():
    records = [json.loads(line) for line in pidfile.read_text().splitlines()] if pidfile.exists() else []
    for record in records:
        current = proc(record['pid'])
        record['alive'] = bool(current and current['start_ticks'] == record['start_ticks']
                               and current['state'] != 'Z')
    return records
def fetch(url, data=None, timeout=5):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)
def specs():
    # Dataset admission comes from the frozen task pool, not a lazy Gym's
    # currently materialized episode catalog. Actual resets are probed below.
    source = Path(plan['case_manifest'])
    pool = [json.loads(line) for line in source.read_text().splitlines() if line]
    if {row['backend'] for row in pool} != {'sage', 'r2r', 'rxr'}:
        raise RuntimeError('expected all three frozen task sources')
    groups = {'train': [], 'validation': []}
    for label in groups:
        for url in plan[f'{label}_urls']:
            health, spec = fetch(url + '/health'), fetch(url + '/spec', timeout=30)
            if (health.get('env_loaded') is not True or spec.get('splits') != ['train']
                    or set(spec.get('backends', [])) != {'sage', 'r2r', 'rxr'}
                    or spec.get('max_steps') != 0):
                raise RuntimeError(f'{url}: incompatible three-source environment')
            groups[label].append(dict(url=url, split='train', spec=spec,
                planned_case_count=len(pool), source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    return groups

if action == 'plan':
    print(json.dumps(plan, indent=2))
elif action == 'check-free':
    if any(record['alive'] for record in owned()):
        raise RuntimeError('this experiment already owns live service processes; inspect status first')
    for port in ports:
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'refusing to replace an existing service on port {port}')
elif action == 'record':
    pid = int(sys.argv[3])
    current = proc(pid)
    if current is None:
        raise RuntimeError(f'{sys.argv[2]} process {pid} exited before PID registration')
    record = {'kind': sys.argv[2], 'pid': pid, 'pgid': current['pgid'],
              'start_ticks': current['start_ticks']}
    with pidfile.open('a') as handle:
        handle.write(json.dumps(record) + '\n')
elif action in ('stop', 'stop-children'):
    records = owned()
    if action == 'stop':
        supervisors = [record for record in records if record['kind'] == 'supervisor' and record['alive']]
        for record in supervisors:
            os.kill(record['pid'], signal.SIGTERM)
        if supervisors:
            for _ in range(30):
                if not any(record['alive'] for record in owned()):
                    break
                time.sleep(.2)
    records = [record for record in owned() if record['kind'] != 'supervisor' and record['alive']]
    for record in records:
        current = proc(record['pid'])
        if current and current['pgid'] == record['pid']:
            os.killpg(record['pid'], signal.SIGTERM)
    for _ in range(30):
        if not any(record['alive'] and record['kind'] != 'supervisor' for record in owned()):
            break
        time.sleep(.2)
    for record in owned():
        if record['kind'] != 'supervisor' and record['alive']:
            current = proc(record['pid'])
            if current and current['pgid'] == record['pid']:
                os.killpg(record['pid'], signal.SIGKILL)
    print(json.dumps({'stopped_owned_services': str(root)}))
elif action in ('status', 'verify', 'probe'):
    result = {**plan, 'processes': owned()}
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                plan['renderer_url'] + '/ping', timeout=5) as response:
            if b'pong' not in response.read():
                raise RuntimeError('renderer did not return pong')
        cpu_health = fetch(plan['cpu_renderer_url'] + '/health')
        if cpu_health.get('renderer') != 'habitat-mesh-software-egl':
            raise RuntimeError('R2R/RxR renderer is not CPU software EGL')
        result['cpu_renderer'] = cpu_health
        groups = specs()
        result['datasets'] = {label: [{k: v for k, v in item.items() if k != 'spec'} for item in items]
                              for label, items in groups.items()}
        result['ready'] = all(record['alive'] for record in result['processes']) and bool(result['processes'])
        if action == 'verify':
            atomic_json(root / 'services.json', result)
            for label in groups:
                for index, item in enumerate(groups[label]):
                    atomic_json(root / f'{label}_spec_{index:02d}.json', item['spec'])
            (root / 'train_urls.txt').write_text(','.join(plan['train_urls']) + '\n')
            (root / 'validation_urls.txt').write_text(','.join(plan['validation_urls']) + '\n')
        elif action == 'probe':
            probe_root = root / 'probes' / str(time.time_ns())
            probe_root.mkdir(parents=True)
            result['probes'] = []
            # Explicit command only: never reset a live trainer's gyms from status.
            for label, items in groups.items():
                for index, item in enumerate(items):
                    task = next(json.loads(line) for line in Path(plan['case_manifest']).read_text().splitlines()
                                if json.loads(line)['backend'] == 'sage')
                    scene, episode = task['scene'], task['episode_id']
                    started = time.monotonic()
                    response = fetch(item['url'] + '/reset', dict(scene=scene, episode_id=episode,
                        instruction=task['instruction'], backend='sage'), timeout=600)
                    rgb = response.get('obs', {}).pop('rgb_b64', None)
                    if not rgb:
                        raise RuntimeError(f'{item["url"]}: reset returned no RGB image')
                    image = base64.b64decode(rgb, validate=True)
                    if not image.startswith(b'\xff\xd8') or len(image) < 100:
                        raise RuntimeError(f'{item["url"]}: reset RGB is not a JPEG image')
                    stem = f'{label}_{index:02d}'
                    (probe_root / f'{stem}.jpg').write_bytes(image)
                    atomic_json(probe_root / f'{stem}.json', response)
                    result['probes'].append({'url': item['url'], 'scene': scene, 'episode_id': episode,
                                             'seconds': time.monotonic() - started,
                                             'image_sha256': hashlib.sha256(image).hexdigest(),
                                             'image_path': str(probe_root / f'{stem}.jpg')})
            atomic_json(probe_root / 'summary.json', result)
    except Exception as exc:
        result.update(ready=False, error=f'{type(exc).__name__}: {exc}')
        if action == 'probe' and 'probe_root' in locals():
            atomic_json(probe_root / 'failure.json', result)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['ready'] else 1)
else:
    raise ValueError(action)
PY
}

MODE=${1:-status}
case "$MODE" in
  plan|status|stop) control "$MODE"; exit ;;
  probe)
    control probe
    PYTHONPATH="$REPO/verl:$REPO/vln_demo/src:${PYTHONPATH:-}" "$PYTHON" -B \
      "$REPO/scripts/eval/probe_three_source_environments.py" \
      --gym-url "http://127.0.0.1:$TRAIN_PORT_BASE" \
      --renderer-url "http://127.0.0.1:$CPU_RENDER_PORT" \
      --pool "$THREE_SOURCE_CASE_MANIFEST" --include-sage \
      --output "$SERVICES_OUTPUT_DIR/native_probe_$(date +%s%N)"
    exit ;;
  start|run) ;;
  *) echo "usage: EXPR_NAME=... $0 {plan|start|run|status|probe|stop}" >&2; exit 2 ;;
esac
control check-free
mkdir -p "$SERVICES_OUTPUT_DIR/logs"
if [[ "$MODE" == start ]]; then
  rm -f "$SERVICES_OUTPUT_DIR/ready"
  export REPO PYTHON EXPR_NAME
  setsid bash "$REPO/scripts/train/v6_rl_local_services.sh" run \
    > "$SERVICES_OUTPUT_DIR/logs/supervisor.log" 2>&1 < /dev/null &
  supervisor=$!
  deadline=$((SECONDS + SERVICE_START_TIMEOUT))
  until [[ -f "$SERVICES_OUTPUT_DIR/ready" ]]; do
    kill -0 "$supervisor" 2>/dev/null || {
      tail -n 80 "$SERVICES_OUTPUT_DIR/logs/supervisor.log" >&2; exit 2;
    }
    (( SECONDS < deadline )) || { echo 'service startup timed out; inspect supervisor.log' >&2; control stop; exit 2; }
    sleep 2
  done
  control status
  exit
fi

exec 9> "$SERVICES_OUTPUT_DIR/supervisor.lock"
flock -n 9 || { echo 'another service supervisor owns this experiment' >&2; exit 2; }
# Recheck after the lock; a parallel start may have raced the preliminary check.
control check-free
rm -f "$SERVICES_OUTPUT_DIR/ready"
: > "$SERVICES_OUTPUT_DIR/owned_pids.jsonl"
control record supervisor "$$"
RUNTIME_KEY=$(printf '%s' "$SERVICES_OUTPUT_DIR" | sha256sum | cut -c1-12)
LOCAL_RUNTIME_ROOT=${V6_SERVICES_RUNTIME_ROOT:-/tmp/v6rlsvc/$RUNTIME_KEY}
# Relative Gym YAML/data paths must resolve inside this container's checkout.
GYM_WORK_ROOT="$REPO/.local_runtime/rl-gym/$RUNTIME_KEY"
export TMPDIR=$LOCAL_RUNTIME_ROOT/tmp XDG_CACHE_HOME=$LOCAL_RUNTIME_ROOT/cache
export HF_HOME=$XDG_CACHE_HOME/hf TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export PYTORCH_KERNEL_CACHE_PATH=$XDG_CACHE_HOME/kernels TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/extensions
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
export RTC_CACHE_PATH=$XDG_CACHE_HOME/rtc
export PPU_LIB_RTC_CACHE_PATH=$RTC_CACHE_PATH
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" \
  "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$RTC_CACHE_PATH"
cleanup() {
  code=$?
  trap - EXIT INT TERM
  rm -f "$SERVICES_OUTPUT_DIR/ready"
  if control stop-children; then
    wait 2>/dev/null || true
    rm -rf -- "$GYM_WORK_ROOT"
  else
    wait 2>/dev/null || true
    echo "retained Gym workspace after failed child cleanup: $GYM_WORK_ROOT" >&2
  fi
  if [[ -d "$LOCAL_RUNTIME_ROOT/log-spool" ]] &&
      [[ -n $(find "$LOCAL_RUNTIME_ROOT/log-spool" -name '*.log' -print -quit) ]]; then
    echo "retained unarchived service logs: $LOCAL_RUNTIME_ROOT/log-spool" >&2
  else
    rm -rf -- "$LOCAL_RUNTIME_ROOT"
  fi
  printf 'service supervisor exit=%s at %s\n' "$code" "$(date -Is)"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

GV=${RYNNBRAIN_VENV:?Set RYNNBRAIN_VENV to your installed simulation environment}
RS=${RENDER_SERVER_ROOT:-$REPO/third_party/abot_renderer}
export LD_LIBRARY_PATH=$GV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO/vln_demo/render_libs:$RS/submodules/diff-plane-rasterization:$RS/submodules/simple-knn:$RS:$REPO/verl:$REPO/vln_demo/src:${PYTHONPATH:-}
export SAGE_RENDERER_EXTENSION_ROOT="$LOCAL_RUNTIME_ROOT/renderer-extensions"
if [[ "$SAGE_RENDER_DEVICE" != cpu ]]; then
"$PYTHON" -B "$REPO/scripts/train/prepare_renderer_extensions.py" \
  --seed "${RENDERER_EXTENSION_SEED:?GPU renderer requires ABI-matched extension seed}" \
  --local-root "$SAGE_RENDERER_EXTENSION_ROOT" \
  > "$SERVICES_OUTPUT_DIR/logs/renderer-extension-verification.json"
export PYTHONPATH="$PYTHONPATH:$SAGE_RENDERER_EXTENSION_ROOT"
fi
export SAGE_COLLISION=usd_rooms SAGE_FURNITURE_COLLISION=none
export SAGE_ROBOT_RADIUS=0.07 SAGE_CAMERA_HEIGHT=1.2 SAGE_HFOV=115
export SAGE_EPISODES_PER_SCENE=0 SAGE_EPS_PER_SCENE=0 DEMO_MAX_STEPS=0
export SAGE_SPLIT=train SAGE_LAZY_EPISODES=1 SAGE_EPISODE_KEEP=''
unset SAGE_BENCH_PROTOCOL_ROOT VLN_OBSERVATION_PROFILE
export SAGE_USD_COLLISION_MASK_ROOT=${SAGE_USD_COLLISION_MASK_ROOT:?Set the SAGE collision-mask directory}
export SAGE_USD_STATIC_BOUNDARY_MASK_ROOT=$SAGE_USD_COLLISION_MASK_ROOT
export SAGE_RENDER_URL=http://127.0.0.1:$RENDER_PORT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Service stdout is written locally; a separate archiver persists it without
# putting shared-storage latency on reset/render response threads.
service_launch() {
  archive=$1
  shift
  exec setsid "$PYTHON" -B -u "$REPO/scripts/train/service_log_spool.py" \
    --archive "$SERVICES_OUTPUT_DIR/logs/$archive" \
    --local-root "$LOCAL_RUNTIME_ROOT/log-spool" -- "$@"
}

if [[ "$SAGE_RENDER_DEVICE" == cpu ]]; then
  : "${SAGE_CPU_RENDER_CACHE:?Set a container-local CPU renderer compilation cache}"
  renderer_command=(env CUDA_VISIBLE_DEVICES='' "$GV/bin/python" -B -u
    "$RS/render_sim_dynamic.py" --device cpu --cpu-render-parallelism "$SAGE_CPU_RENDER_PARALLELISM")
else
  renderer_command=(env CUDA_VISIBLE_DEVICES=15 "$GV/bin/python"
    "$REPO/verl/scripts/train/action_head_bootstrap/single_gpu_renderer.py"
    "$RS/render_sim_dynamic.py")
fi
service_launch renderer.log "${renderer_command[@]}" \
  --scenes_root "${SAGE_SCENES_ROOT:?Set the SAGE Gaussian scene directory}" \
  --port "$RENDER_PORT" --gpus 15 --sh_degree 3 --max_scenes_per_gpu "$RENDERER_CACHE_SCENES" \
  > "$SERVICES_OUTPUT_DIR/logs/renderer-archive.log" 2>&1 < /dev/null &
renderer=$!
control record renderer "$renderer"
deadline=$((SECONDS + SERVICE_START_TIMEOUT))
until curl --noproxy '*' -fsS -m 3 "$SAGE_RENDER_URL/ping" | grep -q pong; do
  kill -0 "$renderer" 2>/dev/null || { echo 'renderer exited; inspect renderer.log' >&2; exit 2; }
  (( SECONDS < deadline )) || { echo 'renderer startup timeout' >&2; exit 2; }
  sleep 2
done

# Use the system Mesa installation or an explicitly supplied Mesa bundle.
if [[ -n "${MESA_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="$MESA_ROOT/lib:${LD_LIBRARY_PATH:-}"
  export LIBGL_DRIVERS_PATH="$MESA_ROOT/dri"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_ROOT/50_mesa.json"
fi
export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe HABITAT_SIM_GPU_DEVICE_ID=-1 LP_NUM_THREADS=4
export VLN_HABITAT_RENDER_URL=http://127.0.0.1:$CPU_RENDER_PORT
export VLN_RENDER_MODE=batch VLN_RENDER_TIMEOUT=0
: "${RYNNBRAIN_NAV_DIR:?Set the container-local navigation Git checkout}"
export RYNNBRAIN_NAV_DIR
export RYNNVLN_LANGUAGE=en RYNNVLN_TURN_ANGLE=30 RYNNVLN_HFOV=120 RYNNVLN_ALLOW_SLIDING=1
native_cache_options=()
if [[ "$CPU_RENDER_SHARED_CACHE" == 1 ]]; then
  native_cache_options+=(--shared-scene-cache)
fi
service_launch cpu-renderer.log env CUDA_VISIBLE_DEVICES='' "$GV/bin/python" -B -u -m remote_habitat.server \
  --scenes-root "$RYNNBRAIN_NAV_DIR/data/scene_datasets/mp3d" \
  --port "$CPU_RENDER_PORT" --scene-cache-size "$CPU_RENDER_CACHE_SCENES" --render-parallelism "$CPU_RENDER_PARALLELISM" \
  "${native_cache_options[@]}" --scene-cache-max-gib "$CPU_RENDER_CACHE_MAX_GIB" \
  --scene-load-reserve-gib "$CPU_RENDER_SCENE_LOAD_RESERVE_GIB" \
  > "$SERVICES_OUTPUT_DIR/logs/cpu-renderer-archive.log" 2>&1 < /dev/null &
control record cpu-renderer "$!"

read -r BOOT_SCENE BOOT_EPISODE < <("$PYTHON" -B -c \
  'import json,os; from pathlib import Path; r=next(json.loads(x) for x in Path(os.environ["THREE_SOURCE_CASE_MANIFEST"]).read_text().splitlines() if json.loads(x)["backend"]=="sage"); print(r["scene"],r["episode_id"])')
for ((index=0; index<TRAIN_GYMS; index++)); do
  port=$((TRAIN_PORT_BASE + index))
  actor_work="$GYM_WORK_ROOT/$port/work"
  mkdir -p "$actor_work"
  cp "$REPO/vln_demo/src/vln_demo.yaml" "$actor_work/vln_demo.yaml"
  ln -s "$RYNNBRAIN_NAV_DIR/data" "$actor_work/data"
  (
    cd "$actor_work"
    service_launch "train_${port}.log" env CUDA_VISIBLE_DEVICES='' "$GV/bin/python" -B -u \
      -m vln_gym.three_source_action_head --backend sage --split train --max_steps 0 \
      --scene "$BOOT_SCENE" --episode_id "$BOOT_EPISODE" --port "$port" \
      --output_path "$SERVICES_OUTPUT_DIR/gym_outputs/$port"
  ) > "$SERVICES_OUTPUT_DIR/logs/train_${port}-archive.log" 2>&1 < /dev/null &
  control record "train_${port}" "$!"
done

deadline=$((SECONDS + SERVICE_START_TIMEOUT))
until control verify > "$SERVICES_OUTPUT_DIR/logs/readiness.json"; do
  (( SECONDS < deadline )) || { cat "$SERVICES_OUTPUT_DIR/logs/readiness.json" >&2; exit 2; }
  sleep 3
done
touch "$SERVICES_OUTPUT_DIR/ready"
echo "local services ready; explicit preflight: EXPR_NAME=$EXPR_NAME $0 probe"
# One child dying ends this isolated group and makes the failure visible.
set +e
wait -n
child_status=$?
set -e
echo "a service child exited with status $child_status" >&2
exit 1
