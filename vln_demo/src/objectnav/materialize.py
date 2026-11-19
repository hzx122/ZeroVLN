"""Copy JFS render dependencies to CPFS and verify before publishing a manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--assets',type=Path,required=True);args=ap.parse_args()
    rows=[json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    paths=sorted({r[k] for r in rows for k in ('asset_path','navmesh') if r[k].startswith('/jfs/')})
    args.assets.mkdir(parents=True,exist_ok=True)
    def copy(source):
        source=Path(source);dest=args.assets/hashlib.sha256(str(source).encode()).hexdigest()[:16]/source.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            source_hash=sha(source)
            if sha(dest)!=source_hash:raise ValueError('existing asset mismatch: '+str(dest))
        else:
            part=dest.with_suffix(dest.suffix+'.partial');h=hashlib.sha256();before=source.stat()
            with source.open('rb') as src,part.open('xb') as dst:
                for block in iter(lambda:src.read(8*1024*1024),b''):dst.write(block);h.update(block)
                dst.flush();os.fsync(dst.fileno())
            after=source.stat();source_hash=h.hexdigest()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or sha(part)!=source_hash:raise ValueError('asset copy verification failed: '+str(source))
            part.rename(dest)
        print(json.dumps(dict(source=str(source),destination=str(dest),sha256=source_hash)),flush=True)
        return str(source),dict(path=str(dest),sha256=source_hash)
    with ThreadPoolExecutor(4) as pool:mapping=dict(pool.map(copy,paths))
    for r in rows:
        r['asset_provenance']={k:mapping[r[k]]|{'source':r[k]} for k in ('asset_path','navmesh') if r[k] in mapping}
        for k in ('asset_path','navmesh'):
            if r[k] in mapping:r[k]=mapping[r[k]]['path']
    with args.output.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    args.output.with_suffix('.assets.json').write_text(json.dumps(dict(source_manifest=str(args.manifest),source_sha256=sha(args.manifest),manifest_sha256=sha(args.output),assets=mapping),indent=2)+'\n')
    print(json.dumps(dict(complete=True,episodes=len(rows),copied_assets=len(paths))),flush=True)
if __name__=='__main__':main()
