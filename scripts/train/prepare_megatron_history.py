"""Prepare a CPU-only, container-local Megatron environment without changing vendor torch/TE."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_git_task import checkout
from rule_teacher_training_handoff import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.is_relative_to('/data/storage') or root.is_relative_to('/data/storage'):
        raise ValueError('environment must be container local')
    if not args.output.resolve().is_relative_to('/data/zerovln'):
        raise ValueError('preparation logs must be on the artifact disk')
    root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    runtime = root/'cache'
    for key in ['TMPDIR','XDG_CACHE_HOME','HF_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                'CUDA_CACHE_PATH','PYTORCH_KERNEL_CACHE_PATH','TORCH_EXTENSIONS_DIR','PIP_CACHE_DIR']:
        path = runtime/key.lower(); path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    def run(command):
        subprocess.run(command, check=True, env=env, timeout=600)
    venv = root/'venv'
    if not venv.exists():
        run([sys.executable,'-m','venv','--system-site-packages',str(venv)])
    python = str(venv/'bin/python')
    base = Path('/workspace')
    art = Path('/data/zerovln/artifacts/v8-cp4-debug-20261015-r1')
    deps = {}
    for name, source, commit in [
        ('mcore',str(base/'envs/git-mirrors/Megatron-LM-v8-cp4.git'),'4b4acac9a1d28ea6829c8d4f566d75698a21249d'),
        ('bridge',str(base/'envs/git-mirrors/Megatron-Bridge-v8-cp4.git'),'c0e164ed2aedac4ad1c877780e2564a19d5d54ec'),
        ('verl','https://git.example.invalid/local-user-03/verl.git','b213f118f8df230bc97f68cdc56043623c6babdb')]:
        deps[name] = checkout(source,commit,root/name)
    wheels = []
    seed = art/'dependency-seeds/upstream35a-cp312'
    for manifest in [seed/'manifest.json', seed/'manifest-transitive.json',
            art.parent/'v8-group-partial-20261019-r1/dependencies/manifest.json']:
        for item in load(manifest):
            path = Path(item['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('wheel checksum mismatch: '+str(path))
            wheels.append(str(path))
    for pattern in ['transformers-5.12.1-*.whl','megatron_energon-7.4.1-*.whl','setuptools-79.0.1-*.whl']:
        found = list((art/'dependencies/wheels').glob(pattern))
        if len(found) != 1:
            raise ValueError('missing fixed wheel: '+pattern)
        wheels.append(str(found[0]))
    native = load(art/'platform/causal_conv_native_install.json')
    if hashlib.sha256(Path(native['wheel']).read_bytes()).hexdigest() != native['sha256']:
        raise ValueError('native causal conv checksum mismatch')
    wheels.append(native['wheel'])
    run([python,'-m','pip','install','--no-deps','--no-index',*wheels])
    for name in ['mcore','bridge']:
        run([python,'-m','pip','install','--no-deps','--no-build-isolation','--no-index','-e',deps[name]['path']])
    main_repo = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = os.pathsep.join(map(str,[main_repo/'scripts/train', main_repo/'verl',
        main_repo/'vln_demo/src',root/'verl',root/'mcore',root/'bridge/src']))
    command = [python,'-B','-c',
        'import json, pathlib; from action_head_rl.megatron_trajectory_text import dependency_report, make_engine_class; '
        'r=dependency_report(); assert r["ready"],r; make_engine_class(); '
        'import torch, transformer_engine, transformers, verl, megatron.core, megatron.bridge, transfer_queue; '
        'assert transfer_queue.__version__ == "0.1.10"; '
        'print(json.dumps(dict(ready=True, api=r, modules={m.__name__:m.__file__ for m in '
        '(torch,transformer_engine,transformers,verl,megatron.core,megatron.bridge,transfer_queue)})))']
    result = subprocess.run(command,env=env,check=True,capture_output=True,text=True,timeout=180)
    (args.output/'import.log').write_text(result.stdout+result.stderr)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    report.update(dependencies=deps,python=python,hostname=os.uname().nodename,
        wheel_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in wheels},
        main_commit=subprocess.check_output(['git','-C',str(main_repo),'rev-parse','HEAD'],text=True).strip(),
        gpu_work=False)
    save(args.output/'ready.json',report)
    print(json.dumps(dict(ready=True,hostname=os.uname().nodename,python=python)),flush=True)


if __name__ == '__main__':
    main()
