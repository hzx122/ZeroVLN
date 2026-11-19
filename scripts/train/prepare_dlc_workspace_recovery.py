"""Freeze the existing workspace controller with control-I/O recovery.

Copies its established dependency snapshot, and reuses the independent stdout
heartbeat. Preparation never submits or starts GPU processes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from dlc_control_io_retry import retry_estale

REPO = Path(__file__).resolve().parents[2]
ART = Path('/data/zerovln/artifacts')
CACHES = ('TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME TRITON_CACHE_DIR '
          'TRITON_HOME TRITON_DUMP_DIR TRITON_OVERRIDE_DIR TORCHINDUCTOR_CACHE_DIR '
          'CUDA_CACHE_PATH PYTORCH_KERNEL_CACHE_PATH TORCH_EXTENSIONS_DIR MPLCONFIGDIR '
          'NUMBA_CACHE_DIR DG_CACHE_DIR DG_JIT_CACHE_DIR RTC_CACHE_PATH PPU_LIB_RTC_CACHE_PATH '
          'HGRTC_CACHE_PATH FLASHINFER_WORKSPACE_BASE FLASHINFER_CUBIN_DIR VLLM_CACHE_ROOT '
          'VLLM_ASSETS_CACHE VLLM_CONFIG_ROOT VLLM_RPC_BASE_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE '
          'HF_MODULES_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR')


def sha(path):
    return hashlib.sha256(retry_estale(Path(path).read_bytes, str(path))).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare(template, output, source, source_snapshot, workspace='pai2_vla_xpu',
            *, name=None, description=None, training_entry=None):
    sys.path.insert(0, str(REPO.parent/'dlc_cli'))
    from workspace_config import resolve_workspace
    if workspace not in ('pai2_vla_xpu', 'pai2_mfm_xpu'):
        raise ValueError('an explicitly authorized workspace is required')
    profile = resolve_workspace(workspace)
    request = json.loads(template.read_text())
    nodes = int(request['podCount'])
    if nodes not in (1, 4) or (source != 'rl' and nodes != 4):
        raise ValueError('resident entry supports four-node training or one-node companion')
    if source not in ('sage', 'r2r', 'rxr', 'rl') or not output.is_relative_to(ART) or output.exists():
        raise ValueError('requires a fresh authorized workspace for sage/r2r/rxr')
    output.mkdir(parents=True)
    platform = output/'platform'; platform.mkdir()
    frozen = output/'source'
    shutil.copytree(source_snapshot, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for dependency_name in ('run_dlc_debug_reservation.py', 'dlc_control_io_retry.py', 'dlc_workspace_stdout_heartbeat.py'):
        shutil.copy2(REPO/'scripts/train'/dependency_name, frozen/'scripts/train'/dependency_name)
    manifest = {str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    save(platform/'source-manifest.json', manifest)
    heartbeat = (frozen/'scripts/train/dlc_workspace_stdout_heartbeat.py').read_text()
    training_body = ''
    rank_capture = ''
    if training_entry is not None:
        training_entry = Path(training_entry).resolve()
        if source != 'rl' or not training_entry.is_relative_to(ART) or not training_entry.is_file():
            raise ValueError('RL resident entry requires an existing frozen data-disk training script')
        rank_capture = 'export RL_DLC_RANK="${RANK:-${PET_NODE_RANK:-${NODE_RANK:-}}}"\n'
        # Run exactly once. The shell's original stdout and independent
        # heartbeat survive the business process exit, including nonzero exit.
        training_body = f'''\nset +e
RANK="$RL_DLC_RANK" WORLD_SIZE={nodes} bash {shlex.quote(str(training_entry))}
RL_EXIT_CODE=$?
set -e
export RL_EXIT_CODE
python3 -B - <<'PY_RL_EXIT'
import json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['RUNTIME'])/'control-source/scripts/train'))
from dlc_control_io_retry import retry_atomic_json
sys.path.insert(0,str(Path(os.environ['RUNTIME'])/'control-source/scripts/eval'))
from v6_rl_checkpoint_companion import atomic_json
code=int(os.environ['RL_EXIT_CODE'])
record=dict(event='rl_business_exit',exit_code=code,time=time.time(),
    hostname=os.uname().nodename,stage='rl_failed_reserving' if code else 'rl_completed_reserving',
    automatic_rl_retry=False,meaning='business process finished; resident entry stays alive')
print(json.dumps(record),flush=True)
target=Path(os.environ['OUTPUT_DIR'])/'nodes'/os.uname().nodename/'rl-exit.json'
target.parent.mkdir(parents=True,exist_ok=True)
retry_atomic_json(lambda: atomic_json(target,record),target)
PY_RL_EXIT
'''
    # Heartbeat source is inline and runs before any shared-file I/O. Its bound
    # PID is the actual workspace shell; unavailable disk status stays unknown.
    body = f'''set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
{rank_capture}\
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE NODE_RANK
export OUTPUT_DIR={shlex.quote(str(output))}
if [ "${{DLC_DEBUG_LAUNCH_SMOKE:-0}}" = "1" ]; then
  python3 -B {shlex.quote(str(frozen/'scripts/train/run_dlc_debug_reservation.py'))} --help >/dev/null
  python3 -B {shlex.quote(str(frozen/'scripts/train/dlc_workspace_stdout_heartbeat.py'))} --help >/dev/null
  printf '%s\\n' WORKSPACE_RECOVERY_ENTRY_OK
  exit 0
fi
export RUNTIME=/tmp/{output.name}/${{HOSTNAME}}
if [ "${{DLC_DEBUG_LAUNCH_HEARTBEAT_SMOKE:-0}}" = "1" ]; then
  export RUNTIME="${{DLC_ENTRY_TEST_RUNTIME:?heartbeat smoke requires its owned test directory}}"
fi
for DLC_CACHE_KEY in {CACHES}; do
  export "$DLC_CACHE_KEY=$RUNTIME/cache/${{DLC_CACHE_KEY,,}}"
  mkdir -p "${{!DLC_CACHE_KEY}}"
done
DEBUG_OUTPUT="$OUTPUT_DIR/nodes/${{HOSTNAME}}/node-debug"
python3 -B - --debug-output "$DEBUG_OUTPUT" --output "$RUNTIME/stdout-heartbeat" <<'PY_WORKSPACE_HEARTBEAT' &
import os as workspace_os, sys as workspace_sys
from pathlib import Path as WorkspacePath
workspace_parent = workspace_os.getppid()
workspace_ticks = WorkspacePath('/proc/' + str(workspace_parent) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
workspace_sys.argv.extend(['--controller-pid', str(workspace_parent), '--controller-start-ticks', workspace_ticks])
{heartbeat}
PY_WORKSPACE_HEARTBEAT
HEARTBEAT_PID=$!
if [ "${{DLC_DEBUG_LAUNCH_HEARTBEAT_SMOKE:-0}}" = "1" ]; then
  sleep 22
  kill -TERM "$HEARTBEAT_PID"
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  printf '%s\\n' WORKSPACE_HEARTBEAT_ENTRY_OK
  exit 0
fi
export FROZEN_CONTROL_SOURCE={shlex.quote(str(frozen))}
export CONTROL_MANIFEST={shlex.quote(str(platform/'source-manifest.json'))}
python3 -B - <<'PY_CONTROL_COPY'
import hashlib,json,os,pathlib,shutil,time
source=pathlib.Path(os.environ['FROZEN_CONTROL_SOURCE'])
target=pathlib.Path(os.environ['RUNTIME'])/'control-source'
while True:
    try:
        manifest=json.loads(pathlib.Path(os.environ['CONTROL_MANIFEST']).read_text())
        target.mkdir(parents=True,exist_ok=True)
        for relative,expected in manifest.items():
            payload=(source/relative).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==expected,relative
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
        print(json.dumps(dict(event='workspace_source_verified',files=len(manifest))),flush=True)
        break
    except OSError as error:
        if error.errno not in (2,116):raise
        print(json.dumps(dict(event='workspace_startup_io_retry',error=repr(error),progress_unavailable=True)),flush=True)
        time.sleep(2)
PY_CONTROL_COPY
export PYTHONPATH="$RUNTIME/control-source:$RUNTIME/control-source/verl:$RUNTIME/control-source/vln_demo/src"
cd "$RUNTIME/control-source"
{training_body}\
sleep 99999999 &
SLEEP_PID=$!
/usr/local/bin/python3.12 -B "$RUNTIME/control-source/scripts/train/run_dlc_debug_reservation.py" run --output "$DEBUG_OUTPUT" --sleep-pid "$SLEEP_PID" &
DEBUG_PID=$!
cleanup() {{
  kill -TERM "$DEBUG_PID" "$SLEEP_PID" "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$DEBUG_PID" 2>/dev/null || true
  wait "$SLEEP_PID" 2>/dev/null || true
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait "$DEBUG_PID"
'''
    (platform/'workspace-entry.sh').write_text(body)
    subprocess.run(['bash','-n',str(platform/'workspace-entry.sh')],check=True)
    request=json.loads(template.read_text())
    if name is None:
        version = output.parent.parent.name.split('-')[2]
        if not (version.startswith('r') and version[1:].isdigit()):
            raise ValueError('workspace must be under a versioned rule-teacher run')
        name = f'rule_teacher_{source}_recovery_4node_20261009_{version}'
    if not name or name.startswith('anonymous') or any(c.isspace() for c in name):
        raise ValueError('workspace name must be nonempty without whitespace or anonymous prefix')
    request.update(name=name,
        userCommand='bash -c '+shlex.quote(body), workDir=str(frozen),
        configPath=str(platform/'workspace-config.json'), parsedProductDir=str(output),
        desc=description or f'{source.upper()}正式训练数据构造故障恢复。四节点每节点16卡，修复工作节点控制文件读写的瞬时错误恢复，保留已完成回合并从独立新attempt接续；沿用v6同深度全指令批量采样与原质量口径，数据和审计日志写独立数据盘目录。')
    request.update(quotaId=profile.resource_id, spaceId=profile.workspace_id,
        dataSources=[dict(dataSourceId=item.data_source_id, dataSourceName='', mountPath=item.mount_path)
                     for item in profile.data_sources])
    assert request['priority']==8 and request['quotaId']==profile.resource_id
    assert request['platType']==2 and request['module']=='vla' and request['podCount']==nodes
    save(platform/'workspace-config.json',dict(source=source,output_root=str(output),
        source_root=str(frozen),source_manifest=sha(platform/'source-manifest.json'),
        original_template=str(template),workspace=workspace,
        controller_io_recovery=True, heartbeat_before_shared_io=True,
        training_entry=str(training_entry) if training_entry else None,
        training_entry_sha256=sha(training_entry) if training_entry else None,
        business_runs_before_reservation=training_entry is not None,
        automatic_business_retry=False))
    save(platform/'create-draft.json',request)
    return dict(output=str(output),files=len(manifest),request=str(platform/'create-draft.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=('sage','r2r','rxr','rl'),required=True)
    p.add_argument('--source-snapshot',type=Path,required=True)
    p.add_argument('--workspace',choices=('pai2_vla_xpu','pai2_mfm_xpu'),default='pai2_vla_xpu')
    p.add_argument('--name')
    p.add_argument('--description')
    p.add_argument('--training-entry', type=Path)
    args=p.parse_args()
    print(json.dumps(prepare(args.template,args.output,args.source,args.source_snapshot,args.workspace,
        name=args.name, description=args.description, training_entry=args.training_entry)))
