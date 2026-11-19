"""Build a resident workspace command using a pinned container-local Git checkout."""
import shlex

from prepare_dlc_workspace_recovery import CACHES


GIT_ASKPASS_SOURCE = r'''#!/usr/bin/env python3
from pathlib import Path
import re,sys,urllib.parse

# Read the user-authorized account file at runtime. Never put its contents in
# the generated launch command, Git remote/config, receipts or training logs.
host = 'git.example.invalid'
prompt = sys.argv[1] if len(sys.argv) == 2 else ''
targets = re.findall(r"https://[^\s'\"]+", prompt)
if len(targets) != 1 or urllib.parse.urlsplit(targets[0]).hostname != host:
    raise SystemExit('Git credential request has an unexpected host')
if prompt.startswith('Username'):
    field = 'username'
elif prompt.startswith('Password'):
    field = 'password'
else:
    raise SystemExit('Git credential request has an unexpected field')
source = Path('/workspace/git_config.txt')
urls = [urllib.parse.urlsplit(value) for value in
        re.findall(r"https://[^\s'\"]+", source.read_text())]
accounts = {(urllib.parse.unquote(url.username), urllib.parse.unquote(url.password))
            for url in urls if url.hostname == host and url.username and url.password}
if len(accounts) != 1:
    raise SystemExit('Git account file must contain one unambiguous account')
username, password = accounts.pop()
print(username if field == 'username' else password)
'''


MEGATRON_PREPARE = r'''NODE=$(printf %02d "$RANK")
ENVROOT=$(dirname "$LOCAL_REPO")/environment
PREP="$OUTPUT_DIR/preparation/node_$NODE"
NODE_SPEC="$OUTPUT_DIR/launch/git_task_node_$NODE.json"
OCCUPIER="$LOCAL_REPO/verl/scripts/train/action_head_sft_data/gpu_occupier.py"
# CPU-only environment preparation; hold the GPUs meanwhile and release
# exactly this holder before the business process starts.
python3 -B "$OCCUPIER" --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --require-gpu-count 16 \
  --ready-file "$RUNTIME/prepare_occupier_ready.json" &
PREP_HOLDER=$!
python3 -B "$LOCAL_REPO/scripts/train/prepare_megatron_history.py" --root "$ENVROOT" --output "$PREP"
prep_code=$?
if [ "$prep_code" != 0 ]; then
  printf '%s\n' "MEGATRON_PREPARE_FAILED code=$prep_code: business not started; holding GPUs"
  while true; do sleep 20; done
fi
kill -TERM "$PREP_HOLDER"; wait "$PREP_HOLDER"
for i in $(seq 1 60); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 2
done
python3 -B - "$SPEC" "$NODE_SPEC" "$PREP/ready.json" <<'PY_SPEC'
import sys
from pathlib import Path
import align_three_source_observations as io
spec=io.load(Path(sys.argv[1]))
spec.update(megatron_environment_ready=sys.argv[3],standby_after_success=True,standby_after_failure=True)
io.write_json(Path(sys.argv[2]),spec)
PY_SPEC
SPEC="$NODE_SPEC"
'''


def rl_entry(repository, commit, output, spec, nodes, *, megatron=False, exports=()):
    """Reuse the Git bootstrap, then immediately run RL once in each Pod.

    ``megatron`` prepares the container-local Megatron environment (holding
    the GPUs during that CPU stage) and binds it into a per-node task spec.
    ``exports`` are literal ``export``/``unset`` lines for the business.
    """
    if nodes not in (4, 8):
        raise ValueError('formal RL requires four or eight complete nodes')
    body = entry(repository, commit, output)
    prefix, boundary, _ = body.partition('sleep 99999999 &\n')
    if not boundary:
        raise ValueError('resident bootstrap boundary changed')
    prefix = prefix.replace('set -uo pipefail\n',
        'set -uo pipefail\nexport LIBRARY_PATH=${LIBRARY_PATH:-}\n'
        'source /usr/local/PPU_SDK/envsetup.sh cuda\n'
        'export RL_DLC_RANK="${RANK:-${PET_NODE_RANK:-${NODE_RANK:-}}}"\n', 1)
    return prefix + f'''export RANK="${{RL_DLC_RANK:?DLC node rank is required}}" WORLD_SIZE={nodes}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if ! python3 -B -c 'import pyarrow,datasets; assert int(pyarrow.__version__.split(".")[0])>=21; assert hasattr(pyarrow,"json_")'; then
  python3 -m pip install --no-deps --no-cache-dir pyarrow==24.0.0 --index-url http://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com
fi
SPEC={shlex.quote(str(spec))}
export ACTION_HEAD_RESIDENT_ENTRY=1
{''.join(line + chr(10) for line in exports)}{MEGATRON_PREPARE if megatron else ''}python3 -B "$LOCAL_REPO/scripts/train/run_git_task.py" --spec "$SPEC" &
TASK_PID=$!
wait "$TASK_PID"
task_code=$?
printf '%s\\n' "GIT_TASK_SUPERVISOR_EXIT pid=$TASK_PID code=$task_code; business will not restart"
# A verified resident handoff terminates the old supervisor with SIGTERM.
# Keep the platform shell and its independent stdout heartbeat alive while
# the new attempt takes ownership. Do not race its GPU release with a holder.
if [ "$task_code" != 143 ]; then
  sleep 99999999 &
  FALLBACK_SLEEP_PID=$!
  python3 -B "$LOCAL_REPO/scripts/train/run_dlc_debug_reservation.py" run \\
    --output "$OUTPUT_DIR/nodes/$HOSTNAME/postrun-reservation" --sleep-pid "$FALLBACK_SLEEP_PID"
fi
while true; do sleep 20; done
'''


def companion_entry(repository, commit, output, spec, nodes):
    """Use the same stdout bootstrap and shared evaluation entry on every Pod."""
    body = entry(repository, commit, output)
    prefix, boundary, standby = body.partition('sleep 99999999 &\n')
    if not boundary or nodes < 1:
        raise ValueError('resident bootstrap boundary or node count changed')
    prefix = prefix.replace('set -uo pipefail\n',
        'set -uo pipefail\nexport EVAL_NODE_RANK="${RANK:-${PET_NODE_RANK:-${NODE_RANK:-}}}"\n'
        'export LIBRARY_PATH=${LIBRARY_PATH:-}\nsource /usr/local/PPU_SDK/envsetup.sh cuda\n', 1)
    return prefix + f'''unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 -B "$LOCAL_REPO/scripts/eval/unified_closedloop_eval.py" resident-watch \\
  --spec {shlex.quote(str(spec))} --rank "${{EVAL_NODE_RANK:?DLC node rank is required}}"
printf 'companion bootstrap exited; inspect original failure; no automatic business restart\\n'
''' + boundary + standby


def entry(repository, commit, output):
    from run_git_task import clean_repository
    import re
    clean_repository(repository)
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('a pushed commit is required')
    local = '/root/rl-code/' + output.name
    return f'''set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OUTPUT_DIR={shlex.quote(str(output))}
export RUNTIME=/tmp/{output.name}/${{HOSTNAME}}
export LOCAL_REPO={shlex.quote(local + '/repo')}
export CODE_COMMIT={shlex.quote(commit)}
export CODE_REPOSITORY={shlex.quote(repository)}
mkdir -p "$RUNTIME" "$(dirname "$LOCAL_REPO")"
# Explicitly authorized existing account credentials, read only by Git.
cat > "$RUNTIME/git-askpass.py" <<'PY_AUTH'
{GIT_ASKPASS_SOURCE.rstrip()}
PY_AUTH
chmod 700 "$RUNTIME/git-askpass.py"
export GIT_ASKPASS="$RUNTIME/git-askpass.py" GIT_TERMINAL_PROMPT=0
for DLC_CACHE_KEY in {CACHES}; do
  export "$DLC_CACHE_KEY=$RUNTIME/cache/${{DLC_CACHE_KEY,,}}"
  mkdir -p "${{!DLC_CACHE_KEY}}"
done
# Bootstrap heartbeat is independent of Git and shared filesystem operations.
python3 -u - <<'PY_HEARTBEAT' &
import json,os,time
from pathlib import Path
pid=os.getppid()
def identity():
    try:return Path(f'/proc/{{pid}}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
ticks=identity()
while ticks and identity()==ticks:
    print(json.dumps(dict(event='workspace_entry_alive',time=time.time(),hostname=os.uname().nodename,
        controller_pid=pid,controller_start_ticks=ticks,stage='see_workspace_status',
        business_progress_reported=False)),flush=True)
    time.sleep(20)
PY_HEARTBEAT
HEARTBEAT_PID=$!
trap 'kill -TERM "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
git_ok=1
fresh=0
if [ ! -d "$LOCAL_REPO/.git" ]; then
  fresh=1
  git clone --quiet --no-checkout -- "$CODE_REPOSITORY" "$LOCAL_REPO" || git_ok=0
fi
if [ "$git_ok" = 1 ]; then
  test "$(git -C "$LOCAL_REPO" remote get-url origin)" = "$CODE_REPOSITORY" || git_ok=0
  if [ "$fresh" = 0 ]; then
    test -z "$(git -C "$LOCAL_REPO" status --porcelain --untracked-files=no)" || git_ok=0
  fi
fi
if [ "$git_ok" = 1 ]; then
  git -C "$LOCAL_REPO" fetch --quiet origin "$CODE_COMMIT" &&
  git -C "$LOCAL_REPO" checkout --quiet --detach "$CODE_COMMIT" &&
  test "$(git -C "$LOCAL_REPO" rev-parse HEAD)" = "$CODE_COMMIT" || git_ok=0
fi
if [ "$git_ok" != 1 ]; then
  printf '%s\\n' 'GIT_CHECKOUT_FAILED: business not started; no source fallback'
  while true; do sleep 20; done
fi
cd "$LOCAL_REPO"
export PYTHONPATH="$LOCAL_REPO/scripts/train:$LOCAL_REPO/scripts/eval:$LOCAL_REPO/verl:$LOCAL_REPO/vln_demo/src"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE NODE_RANK
python3 -B - <<'PY_VERIFY'
import os,json,sys,time
from pathlib import Path
from rule_teacher_training_handoff import save
import run_dlc_debug_reservation as controller
import dlc_control_io_retry as io
root=Path(os.environ['LOCAL_REPO']).resolve()
for module in (controller,io):
    assert Path(module.__file__).resolve().is_relative_to(root),module.__file__
save(Path(os.environ['OUTPUT_DIR'])/'nodes'/os.uname().nodename/'checkout.json',
    dict(repository=os.environ['CODE_REPOSITORY'],commit=os.environ['CODE_COMMIT'],
         checkout=str(root),cwd=os.getcwd(),modules=[controller.__file__,io.__file__],
         pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
PY_VERIFY
if [ "$?" != 0 ]; then
  printf '%s\\n' 'MODULE_SOURCE_CHECK_FAILED: business not started'
  while true; do sleep 20; done
fi
sleep 99999999 &
SLEEP_PID=$!
while true; do
  python3 -B "$LOCAL_REPO/scripts/train/run_dlc_debug_reservation.py" run \
    --output "$OUTPUT_DIR/nodes/$HOSTNAME/node-debug" --sleep-pid "$SLEEP_PID"
  code=$?
  printf 'resident_controller_exit=%s; business is not automatically restarted\\n' "$code"
  sleep 20
done
'''
