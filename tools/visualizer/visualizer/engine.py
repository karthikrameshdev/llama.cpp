import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import grpc
from .protocol import pb, rpc
from .store import TraceBudgetExceeded

class Engine(rpc.TelemetryServicer):
    def __init__(self, store, observer, host='127.0.0.1', port=0):
        self.store = store
        self.observer = str(Path(observer).resolve())
        self.lock = threading.RLock()
        self.commands = queue.Queue(maxsize=64)
        self.ready = threading.Event()
        self.closing = threading.Event()
        self.token = secrets.token_urlsafe(32)
        self.state = {'connected': False, 'status': 'starting', 'model_id': None, 'active_run': None, 'error': None, 'received_events': 0, 'transport': 'gRPC / protobuf'}
        self.rejected = set()
        self.pending_cancel = set()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4), options=[('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        rpc.add_TelemetryServicer_to_server(self, self.server)
        self.port = self.server.add_insecure_port(f'{host}:{port}')
        if not self.port:
            raise RuntimeError('Cannot bind the gRPC collector')
        self.process = None

    def start(self):
        if not Path(self.observer).is_file():
            raise RuntimeError('Native observer is missing; run the launcher to build it')
        self.server.start()
        environment = dict(os.environ, VISUALIZER_GRPC_TOKEN=self.token)
        self.process = subprocess.Popen([sys.executable, '-m', 'visualizer.bridge', '--target', f'127.0.0.1:{self.port}', '--observer', self.observer], env=environment, cwd=Path(__file__).resolve().parents[1])
        return self

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def send(self, kind, body):
        self.commands.put_nowait(pb.Command(kind=kind, payload_json=json.dumps(body), request_id=secrets.token_hex(12)))

    def load(self, model_id, config):
        with self.lock:
            if not self.state['connected']:
                raise ValueError('Engine is not connected')
            if self.state['status'] in ('loading', 'running'):
                raise ValueError('Wait for the current operation or cancel it before loading a model')
            model = self.store.model(model_id)
            if not model:
                raise KeyError(model_id)
            self.state.update(status='loading', model_id=model_id, active_run=None, error=None)
            self.store.set_config(model_id, config)
            inspection_id = 'inspection:' + model_id + ':' + secrets.token_hex(12)
            self.state['inspection_id'] = inspection_id
            self.send('load', dict(config, model_id=model_id, path=model['path'], inspection_id=inspection_id))

    def run(self, prompt, options):
        with self.lock:
            if self.state['status'] != 'ready' or not self.state['connected']:
                raise ValueError('Load a model and wait until it is ready')
            run = self.store.create_run(self.state['model_id'], prompt, options)
            self.state.update(status='running', active_run=run['id'], error=None)
            self.send('run', dict(options, model_id=self.state['model_id'], run_id=run['id'], prompt=prompt))
            return run

    def cancel(self, run_id):
        with self.lock:
            if self.state['active_run'] != run_id:
                raise ValueError('This run is not currently executing')
            if self.store.run(run_id)['status'] == 'queued':
                self.pending_cancel.add(run_id)
            else:
                self.send('cancel', {'run_id': run_id})

    def _event(self, event):
        run_id = event.get('run_id', '')
        if run_id in self.rejected and event['kind'] not in ('run_finished', 'error', 'bridge_error'):
            return
        try:
            self.store.ingest(event, commit=False)
        except TraceBudgetExceeded:
            self.rejected.add(run_id)
            self.store.fail_run(run_id, 'Trace budget exceeded. Capture is incomplete; reduce output length or increase --trace-budget-mb.')
            self.send('cancel', {'run_id': run_id})
            self.store.ingest(dict(event, kind='trace_error', data={'status': 'failed', 'message': 'Trace budget exceeded; incomplete capture'}), commit=False)
            return
        with self.lock:
            self.state['received_events'] += 1
            kind, data = event['kind'], event.get('data', {})
            if kind == 'hello':
                self.state.update(status='idle', connected=True, error=None)
                self.ready.set()
            elif kind == 'run_started' and run_id in self.pending_cancel:
                self.pending_cancel.discard(run_id)
                self.send('cancel', {'run_id': run_id})
            elif kind == 'model_ready':
                self.state.update(status='ready', model_id=event['model_id'], active_run=None, error=None)
            elif kind == 'run_finished':
                self.pending_cancel.discard(run_id)
                self.state.update(status='ready', active_run=None)
            elif kind == 'error':
                self.state.update(status='error' if run_id.startswith('inspection:') else 'ready', active_run=None, error=data.get('message'))
            elif kind == 'bridge_error':
                if self.state['active_run']:
                    self.store.fail_run(self.state['active_run'], data['message'], 'interrupted')
                self.state.update(status='error', connected=False, active_run=None, error=data['message'])

    def Connect(self, request_iterator, context):
        metadata = dict(context.invocation_metadata())
        if metadata.get('authorization') != 'Bearer ' + self.token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Invalid engine session')
        with self.lock:
            if self.state['connected']:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, 'Only one engine session is supported')
        done = threading.Event()

        def receive():
            try:
                for batch in request_iterator:
                    if len(batch.events) > 128:
                        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Event batch too large')
                    with self.lock, self.store.lock, self.store.db:
                        for item in batch.events:
                            event = json.loads(item.payload_json)
                            if event.get('sequence') != item.sequence or event.get('kind') != item.kind:
                                raise ValueError('Malformed event envelope')
                            self._event(event)
            except Exception as exc:
                if not self.closing.is_set():
                    with self.lock:
                        if self.state['active_run']:
                            self.store.fail_run(self.state['active_run'], f'Engine stream interrupted: {exc}', 'interrupted')
                        self.state.update(connected=False, status='error', active_run=None, error=str(exc))
            finally:
                done.set()

        threading.Thread(target=receive, daemon=True, name='telemetry-ingest').start()
        while context.is_active() and not self.closing.is_set() and not done.is_set():
            try:
                yield self.commands.get(timeout=.2)
            except queue.Empty:
                continue
        if not self.closing.is_set():
            with self.lock:
                if self.state['active_run']:
                    self.store.fail_run(self.state['active_run'], 'Engine disconnected', 'interrupted')
                self.state.update(connected=False, status='error', active_run=None)

    def stop(self):
        self.closing.set()
        self.server.stop(0).wait(timeout=5)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        with self.lock:
            if self.state['active_run']:
                self.store.fail_run(self.state['active_run'], 'Application stopped', 'interrupted')
