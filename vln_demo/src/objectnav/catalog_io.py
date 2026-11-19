"""Read versioned ObjectNav catalogs, including deterministic gzip manifests."""
import gzip
import json
from pathlib import Path

def read_catalog(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
