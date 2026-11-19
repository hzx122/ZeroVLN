"""Verify text, image, native tool calls and measured prefix reuse before batch admission."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
from urllib.request import Request, ProxyHandler, build_opener
import zlib

from sglang_server import checked_info


def cached_tokens(response):
    # SGLang returns null details on a cold request, then reports warm hits.
    usage = response.get('usage') or {}
    return (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)


def red_image():
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
    raw = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0))
    raw += chunk(b'IDAT', zlib.compress((b'\0'+b'\xff\0\0'*128)*128))+chunk(b'IEND', b'')
    return 'data:image/png;base64,'+base64.b64encode(raw).decode()


def probe(url, config, status_file, output):
    output.mkdir(parents=True, exist_ok=False)
    status = json.loads(status_file.read_text())
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['updated_at'])).total_seconds()
    if not (status['state'] == 'ready' and status['model_id'] == config['model_id'] and 0 <= age < 120):
        raise ValueError('fresh matching model supervisor required')
    opener = build_opener(ProxyHandler({}))
    def call(name, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(url.rstrip('/')+path, data=data, headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=600) as response:
            result = json.load(response)
        (output/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        return result
    receipt = dict(passed=False, model_id=config['model_id'], hostname=status['hostname'], url=url)
    try:
        checked_info(call('server-info', '/server_info'), config)
        common = dict(model=config['model_id'], temperature=0, max_tokens=32,
                      chat_template_kwargs={'enable_thinking': False})
        prompt = 'Read this reference: '+('navigation observation and action. '*2048)+'\nReply OK.'
        hits = []
        for i in range(3):
            r = call(f'prefix-{i}', '/v1/chat/completions', dict(common, messages=[{'role':'user','content':prompt}]))
            if not r['choices'][0]['message'].get('content'):
                raise ValueError('empty text response')
            hits.append(cached_tokens(r))
        if max(hits[1:]) <= 0:
            raise ValueError('repeated requests did not report prefix-cache hits')
        r = call('vision', '/v1/chat/completions', dict(common, messages=[{'role':'user','content':[
            {'type':'text','text':'What is the image color? Reply with one English color word.'},
            {'type':'image_url','image_url':{'url':red_image()}}]}]))
        if 'red' not in str(r['choices'][0]['message'].get('content', '')).lower():
            raise ValueError('image inference did not identify the red test image')
        tool = {'type':'function','function':{'name':'report_status','description':'Report status',
                'parameters':{'type':'object','properties':{'status':{'type':'string','enum':['ok']}},'required':['status']}}}
        r = call('tool', '/v1/chat/completions', dict(common, max_tokens=128,
            messages=[{'role':'user','content':'Call report_status with status ok.'}],
            tools=[tool], tool_choice={'type':'function','function':{'name':'report_status'}}))
        calls = r['choices'][0]['message'].get('tool_calls', [])
        if not calls or calls[0]['function']['name'] != 'report_status' or json.loads(calls[0]['function']['arguments']) != {'status':'ok'}:
            raise ValueError('native tool-call parsing failed')
        receipt.update(passed=True, cached_tokens=hits, text=True, image=True, tools=True,
                       checked_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--url', required=True, help='http://DLC_IP:PORT without /v1')
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('sglang_profiles.json').read_text())[args.profile]
    print(json.dumps(probe(args.url, config, args.status_file, args.output_dir), indent=2))
