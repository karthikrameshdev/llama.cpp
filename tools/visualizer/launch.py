#!/usr/bin/env python3
"""Build the optional observer and run the local browser application."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def run(command, cwd=ROOT, **kwargs):
    subprocess.run([str(x) for x in command], cwd=cwd, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--model-dir', action='append', help='Allowed local GGUF directory; repeat to add more')
    parser.add_argument('--model', help='Load this model automatically')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--build-dir', default=str(REPO / 'build'))
    parser.add_argument('--trace-budget-mb', type=int, default=512)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.trace_budget_mb < 1:
        parser.error('Use a valid port and a positive trace budget')
    model_dirs = [str(Path(p).resolve()) for p in args.model_dir or [ROOT / 'models']]
    model = str(Path(args.model).resolve()) if args.model else None
    if model and not any(Path(model).is_relative_to(Path(p)) for p in model_dirs):
        model_dirs.append(str(Path(model).parent))
    python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    requirements = ROOT / 'requirements.lock.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = ROOT / '.venv' / '.visualizer-dependencies'
    if not marker.exists() or marker.read_text() != digest:
        if shutil.which('uv'):
            run(['uv', 'pip', 'install', '--python', python, '-r', requirements])
        else:
            run([python, '-m', 'ensurepip'])
            run([python, '-m', 'pip', 'install', '-r', requirements])
        marker.write_text(digest)
    if not shutil.which('cmake') or not shutil.which('npm'):
        raise SystemExit('CMake and Node.js/npm are required. Install them, then rerun this command.')
    sdk = []
    if sys.platform == 'darwin':
        path = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
        sdk = ['-DCMAKE_OSX_SYSROOT=' + path]
    build = Path(args.build_dir).resolve()
    if not list((build / 'bin').glob('*llama-common*')) or not list((build / 'bin').glob('*mtmd*')):
        run(['cmake', '-S', REPO, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DLLAMA_USE_PREBUILT_UI=OFF', *sdk])
        run(['cmake', '--build', build, '--parallel', '4', '--target', 'llama-common', 'mtmd'])
    run(['cmake', '-S', ROOT, '-B', ROOT / 'build', '-DLLAMA_BUILD_DIR=' + str(build), '-DCMAKE_BUILD_TYPE=Release', *sdk])
    run(['cmake', '--build', ROOT / 'build', '--parallel', '4'])
    ui = ROOT / 'ui'
    lock_hash = hashlib.sha256((ui / 'package-lock.json').read_bytes()).hexdigest()
    npm_marker = ui / 'node_modules/.visualizer-lock'
    if not npm_marker.exists() or npm_marker.read_text() != lock_hash:
        run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=ui)
        npm_marker.write_text(lock_hash)
    run(['npm', 'run', 'build'], cwd=ui)
    run([python, '-c', 'from visualizer.protocol import generate; generate()'])
    command = [str(python), '-m', 'visualizer.app', '--port', str(args.port), '--data-dir', str(Path(args.data_dir).resolve()), '--trace-budget-mb', str(args.trace_budget_mb)]
    for directory in model_dirs:
        command.extend(['--model-dir', directory])
    if model:
        command.extend(['--model', model])
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=os.name != 'nt')
    url = f'http://127.0.0.1:{args.port}'
    try:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError('The local service stopped before becoming ready')
            try:
                with urllib.request.urlopen(url + '/api/state', timeout=.5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.1)
        else:
            raise RuntimeError('The local service did not become ready in 30 seconds')
        print(f'\nRuntime Observatory: {url}\nCtrl+C stops the observer and service. Saved traces remain in {args.data_dir}.\n', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=5)

if __name__ == '__main__':
    main()
