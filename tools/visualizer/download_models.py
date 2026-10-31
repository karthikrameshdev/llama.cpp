#!/usr/bin/env python3
"""Download pinned verification models and verify their exact size and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent

def valid(path, model):
    if not path.is_file() or path.stat().st_size != model['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == model['sha256']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for model in json.loads((ROOT / 'model-manifest.json').read_text()):
        path = args.directory / model['local']
        if valid(path, model):
            print('Verified', path.name, flush=True)
            continue
        partial = path.with_suffix('.gguf.partial')
        print('Downloading', model['repo'], model['file'], flush=True)
        request = urllib.request.Request(model['url'], headers={'User-Agent': 'llama-runtime-observatory/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        if not valid(partial, model):
            raise SystemExit(f'Checksum mismatch: {partial}; the existing model has not been replaced')
        partial.replace(path)
        print('Verified', path.name, flush=True)

if __name__ == '__main__':
    main()
