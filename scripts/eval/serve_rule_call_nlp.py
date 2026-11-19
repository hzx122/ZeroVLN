#!/usr/bin/env python3
"""Loopback-only CPU syntax service for the rule teacher's isolated runtime."""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vln_demo/src'))
from rule_call_nlp import CallLanguageParser


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--port',type=int,default=19074)
    args=parser.parse_args();nlp=CallLanguageParser(args.model);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,value,status=200):
            data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':self.respond(dict(ready=True,model=nlp.model))
            else:self.respond(dict(error='unknown endpoint'),404)
        def do_POST(self):
            if self.path!='/parse':self.respond(dict(error='unknown endpoint'),404);return
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=65536:self.respond(dict(error='invalid call size'),400);return
            try:
                call=json.loads(self.rfile.read(length))
                if not isinstance(call,dict):raise ValueError('call must be an object')
                with lock:graph=nlp.parse(call)
                self.respond(graph)
            except (ValueError,TypeError) as exc:self.respond(dict(error=str(exc)),400)
    print(json.dumps(dict(event='nlp_ready',port=args.port,model=nlp.model)),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
