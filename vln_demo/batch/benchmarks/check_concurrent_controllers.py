"""Real private R2R/RxR controllers: concurrent motion, frame and STOP isolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner
from workspace import TEMPLATE


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--render-url', required=True)
    parser.add_argument('--controller-model-url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from service_env import service_env
    os.environ.update(service_env(os.environ, args.render_url, args.controller_model_url))
    args.output.mkdir(parents=True, exist_ok=False)
    local = Path(tempfile.mkdtemp(prefix='vln-concurrency-'))
    r2r = runner.manifest_rows('r2r','train','guide','*',0)
    first = r2r[0]
    same = next(r for r in r2r[1:] if r['scene']==first['scene'] and r['instruction']!=first['instruction'])
    different = next(r for r in runner.manifest_rows('rxr','train','guide','*',0) if r['scene']!=first['scene'])
    processes = []
    def request(worker, path, data=None, token=None):
        headers = {'X-VLN-Batch-Token': token or worker['token'], 'x-vln-control-source':'pi-agent'}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        req = Request(worker['url']+path, headers=headers, data=json.dumps(data).encode() if data is not None else None)
        with build_opener(ProxyHandler({})).open(req,timeout=15) as response:
            body = response.read()
            return json.loads(body) if response.headers.get_content_type()=='application/json' else body.decode()
    try:
        for index,row in enumerate((first,same,different)):
            output = args.output / str(index)
            output.mkdir()
            cache = local / str(index)
            cache.mkdir()
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                port = listener.getsockname()[1]
            token = uuid.uuid4().hex
            env = dict(os.environ, VLN_BATCH_TOKEN=token, VLN_BATCH_OUTPUT=str(output),
                VLN_BATCH_MODEL_URL='http://127.0.0.1:19999/v1', VLN_BATCH_CONTROLLER_MODEL_URL=args.controller_model_url, VLN_HABITAT_RENDER_URL=args.render_url,
                VLN_RENDER_MODE='batch', DEMO_DEFAULT_MODEL='qwen35b', DEMO_DEFAULT_DRIVER='react', DEMO_DEFAULT_MODE='think', DEMO_MAX_TOOL_CALLS='10',
                DEMO_BACKEND='rxr' if row['dataset']=='rxr' else 'habitat', RXR_ROLES='guide',
                RXR_LANGUAGES='*', RXR_EPISODES_PER_SCENE='0', DEMO_SYSPROMPT_FILE=str(cache/'unused'),
                TMPDIR=str(cache),XDG_CACHE_HOME=str(cache/'cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1')
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','LOCAL_WORLD_SIZE'):
                env.pop(key,None)
            command = ['bash', str(runner.REPO/'vln_demo/src/run_console.sh'),str(port),'--split','train',
                       '--scene',('rxr-' if row['dataset']=='rxr' else '')+row['scene'],
                       '--episode_id',row['episode_id'],'--out',str(output)]
            with (output/'console.log').open('w') as log:
                process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(dict(process=process,row=row,token=token,url=f'http://127.0.0.1:{port}',cache=cache))
        deadline = time.monotonic()+600
        for worker in processes:
            while True:
                if worker['process'].poll() is not None:
                    raise RuntimeError('controller exited before ready')
                try:
                    state = request(worker,'/status')
                    frame = request(worker,'/frame-snapshot')
                    if state['status']=='idle' and str(frame['source']['episode_id'])==worker['row']['episode_id']:
                        actual = request(worker, '/batch-config')
                        expected = dict(model='qwen35b', driver='react', mode='think', maxtools=10, temp=None, nact=4)
                        assert all(actual['config'].get(k) == v for k, v in expected.items()), actual
                        assert actual['model_endpoint'][0].rstrip('/').removesuffix('/v1') == args.controller_model_url.rstrip('/').removesuffix('/v1')
                        assert actual['model_endpoint'][1] == 'Qwen3.6-35B'
                        (args.output / str(processes.index(worker)) / 'runtime-config.json').write_text(json.dumps(actual, indent=2))
                        worker['initial'] = frame
                        break
                except (OSError,ValueError):
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('controller readiness')
                time.sleep(1)
        for worker in processes:
            for path,data,token in [('/status',None,'wrong'),('/terminal',{'op':'input','data':'bad'},None)]:
                try:
                    request(worker,path,data,token)
                except HTTPError as exc:
                    assert exc.code==403
                else:
                    raise AssertionError('control isolation bypass')
        first_frame = processes[0]['initial']
        payload = dict(model='Qwen3.6-35B', max_tokens=128, temperature=0,
            chat_template_kwargs={'enable_thinking': False}, messages=[{'role':'user','content':[
                {'type':'text','text':'Describe this scene briefly.'},
                {'type':'image_url','image_url':{'url':first_frame['image_data']}}]}])
        req = Request(args.controller_model_url.rstrip('/')+'/chat/completions',
                      data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with build_opener(ProxyHandler({})).open(req, timeout=600) as response:
            inference = json.load(response)
        assert inference['choices'][0]['message'].get('content'), 'controller returned no image response'
        (args.output/'controller-image.json').write_text(json.dumps(inference, indent=2))
        def move(worker):
            source = worker['initial']['source']
            workspace = worker['cache'] / 'workspace'
            workspace.mkdir()
            (workspace / 'AGENTS.md').write_bytes(TEMPLATE.read_bytes())
            probe = subprocess.run([sys.executable, '-B', str(runner.REPO/'vln_demo/batch/sandbox.py'),
                '--workspace', str(workspace), '--', runner.NODE, str(runner.REPO/'vln_demo/batch/benchmarks/check_tools.mjs')],
                cwd=workspace, env=dict(os.environ, PI_VLN_URL=worker['url'], VLN_BATCH_TOKEN=worker['token']),
                capture_output=True, text=True, timeout=660)
            assert probe.returncode == 0, probe.stderr
            tool_result = json.loads(probe.stdout.strip().splitlines()[-1])
            deadline = time.monotonic()+600
            while True:
                state = request(worker,'/status')
                frame = request(worker,'/frame-snapshot')
                if state['status'] in ('done','error') and frame['source']['seq']>source['seq']:
                    assert frame['source']['scene']==source['scene']
                    assert str(frame['source']['episode_id'])==str(source['episode_id'])
                    assert state.get('motion', {}).get('steps', 0) > 0, 'probe must actually move'
                    return dict(state=state['status'],source=frame['source'],motion=state.get('motion'),result=state.get('result'),tools=tool_result)
                if time.monotonic()>deadline:
                    raise TimeoutError('concurrent point action')
                time.sleep(0.2)
        with ThreadPoolExecutor(3) as pool:
            moved = list(pool.map(move,processes))
        snapshots = [request(worker,'/frame-snapshot') for worker in processes]
        request(processes[0],'/batch-finish',{})
        deadline = time.monotonic()+60
        while True:
            stopped = request(processes[0],'/batch-result')
            if stopped.get('state')=='finished':
                break
            if time.monotonic()>deadline:
                raise TimeoutError('STOP')
            time.sleep(0.2)
        assert stopped['explicit_stop']
        for i in (1,2):
            assert request(processes[i],'/batch-result')['state']=='pending'
            assert request(processes[i],'/frame-snapshot')==snapshots[i]
        result = dict(same_scene_different_instructions=True,different_scenes=True,
                      concurrent_actions=moved,stop_and_frames_isolated=True,wrong_token_and_terminal_blocked=True,
                      stop_result=stopped,episodes=[w['row'] for w in processes])
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
    finally:
        for worker in processes:
            runner.stop(worker['process'])
        import shutil
        shutil.rmtree(local)


if __name__=='__main__':
    main()
