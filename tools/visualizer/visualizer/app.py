import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import uuid
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Literal
from .engine import Engine
from .store import Store

ROOT = Path(__file__).resolve().parents[1]

class Load(BaseModel):
    model_id: str
    context_size: int = Field(2048, ge=256, le=32768)
    gpu_layers: int = Field(99, ge=0, le=999)
    threads: int = Field(4, ge=1, le=64)
    capture: Literal['routing', 'stats', 'full'] = 'full'
    mmproj: str = ''

class Run(BaseModel):
    prompt: str = Field('', max_length=131072)
    max_tokens: int = Field(32, ge=1, le=1024)
    capture: Literal['routing', 'stats', 'full'] = 'full'
    chat: bool = True
    token_ids: list[int] | None = Field(None, max_length=32768)
    media_id: str | None = None


def create_app(data_dir=None, model_dirs=None, observer=None, trace_budget_mb=512, start_engine=True, autoload=None):
    data_dir = Path(data_dir or ROOT / 'data').resolve()
    roots = [Path(p).resolve() for p in (model_dirs or [ROOT / 'models'])]
    store = Store(data_dir / 'traces.sqlite3', int(trace_budget_mb * 1024 * 1024))
    engine = Engine(store, observer or ROOT / 'build/llama-visualizer-observer')

    def allowed(path):
        path = Path(path).resolve()
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file() or path.suffix.lower() != '.gguf':
            raise HTTPException(403, 'Model must be an existing GGUF inside a configured --model-dir')
        return path

    def scan():
        count = 0
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob('*.gguf'):
                count += 1
                if count > 5000:
                    raise HTTPException(413, 'Model directory contains more than 5,000 GGUF files')
                try:
                    resolved = allowed(path)
                    if 'mmproj' not in resolved.name.lower():
                        store.register(resolved)
                except (OSError, HTTPException):
                    continue
        return store.models()

    @asynccontextmanager
    async def lifespan(app):
        scan()
        if start_engine:
            engine.start()
            if autoload:
                async def load_when_ready():
                    for _ in range(200):
                        if engine.ready.is_set():
                            model = store.register(allowed(autoload))
                            engine.load(model['id'], Load(model_id=model['id']).model_dump(exclude={'model_id'}))
                            return
                        await asyncio.sleep(.1)
                app.state.autoload_task = asyncio.create_task(load_when_ready())
        yield
        if hasattr(app.state, 'autoload_task'):
            app.state.autoload_task.cancel()
        if start_engine:
            engine.stop()
        store.close()

    app = FastAPI(title='llama.cpp Runtime Visualizer', version='1.0.0', lifespan=lifespan)
    app.state.store, app.state.engine = store, engine
    app.state.shutting_down = False
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])

    @app.middleware('http')
    async def local_browser_boundary(request, call_next):
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if origin and origin not in (f'http://{request.headers.get("host")}', f'https://{request.headers.get("host")}'):
                return JSONResponse({'detail': 'Cross-origin mutations are not allowed'}, status_code=403)
            if request.headers.get('x-visualizer-client') != '1':
                return JSONResponse({'detail': 'Missing X-Visualizer-Client header'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'"
        return response

    @app.get('/api/state')
    def state():
        return dict(engine=engine.snapshot(), cursor=store.latest_cursor(), database=str(store.path), model_dirs=[str(p) for p in roots], schema_version=1)

    @app.post('/api/engine/restart', status_code=202)
    def restart_engine():
        nonlocal engine
        if engine.snapshot()['status'] in ('loading', 'running'):
            raise HTTPException(409, 'Cancel the current operation before restarting')
        engine.stop()
        engine = Engine(store, observer or ROOT / 'build/llama-visualizer-observer').start()
        app.state.engine = engine
        return {'status': 'starting'}

    @app.get('/api/models')
    def models():
        return scan()

    @app.post('/api/models/load', status_code=202)
    def load(body: Load):
        model = store.model(body.model_id)
        if not model:
            raise HTTPException(404, 'Unknown model')
        allowed(model['path'])
        config = body.model_dump(exclude={'model_id'})
        if body.mmproj:
            config['mmproj'] = str(allowed(body.mmproj))
        try:
            engine.load(body.model_id, config)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {'model_id': body.model_id, 'status': 'loading'}

    @app.get('/api/models/{model_id}')
    def model(model_id: str):
        result = store.model(model_id)
        if not result:
            raise HTTPException(404, 'Unknown model')
        return result

    @app.get('/api/models/{model_id}/metadata')
    def metadata(model_id: str, key: str = Query(max_length=512), offset: int = Query(0, ge=0), limit: int = Query(128, ge=1, le=4096), component: Literal['model', 'projector'] = 'model', run_id: str | None = None):
        result = model(model_id)
        if run_id:
            saved = get_run(run_id)
            if saved['model_id'] != model_id:
                raise HTTPException(404, 'This model does not belong to the requested run')
            result = saved.get('model_snapshot') or result
        path = (result.get('config') or {}).get('mmproj') if component == 'projector' else result['path']
        if not path:
            raise HTTPException(404, 'No projector recorded for this model')
        path = allowed(path)
        import sys
        sys.path.insert(0, str(ROOT.parents[1] / 'gguf-py'))
        from gguf import GGUFReader
        reader = GGUFReader(path)
        field = reader.fields.get(key)
        if field is None:
            raise HTTPException(404, 'Unknown metadata key')
        value = field.contents(slice(offset, offset + limit)) if len(field.types) > 1 else field.contents()
        if isinstance(value, str):
            total, value = len(value), value[offset:offset + limit]
        elif isinstance(value, list):
            total = len(field.data)
        else:
            total = 1
        return {'key': key, 'offset': offset, 'total': total, 'value': value}

    @app.get('/api/models/{model_id}/weights')
    def weights(model_id: str, tensor: str = Query(max_length=512), component: Literal['model', 'projector'] = 'model',
                axis: Literal['row', 'column'] = 'row', index: int = Query(0, ge=0), plane: int = Query(0, ge=0),
                offset: int = Query(0, ge=0), limit: int = Query(32, ge=1, le=1024), run_id: str | None = None):
        result = model(model_id)
        if run_id:
            saved = get_run(run_id)
            if saved['model_id'] != model_id:
                raise HTTPException(404, 'This model does not belong to the requested run')
            result = saved.get('model_snapshot') or result
        path = (result.get('config') or {}).get('mmproj') if component == 'projector' else result['path']
        if not path:
            raise HTTPException(404, 'No projector recorded')
        path = allowed(path)
        if component == 'model' and (path.stat().st_size != result['size'] or path.stat().st_mtime_ns != result['mtime_ns']):
            raise HTTPException(409, 'Model file changed since this trace was recorded')
        from .tensors import weight_window
        try:
            return weight_window(path, tensor, axis, index, plane, offset, limit)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, NotImplementedError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/runs', status_code=202)
    def run(body: Run):
        if not body.prompt.strip() and not body.token_ids:
            raise HTTPException(422, 'Provide a prompt or explicit token IDs')
        options = body.model_dump(exclude={'prompt', 'media_id'}, exclude_none=True)
        if body.media_id:
            path = data_dir / 'media' / body.media_id
            if not body.media_id.isalnum() or not path.is_file():
                raise HTTPException(404, 'Unknown media upload')
            options['media_path'] = str(path)
        try:
            return engine.run(body.prompt, options)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/api/media', status_code=201)
    async def upload(request: Request):
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > 20 * 1024 * 1024:
                raise HTTPException(413, 'Media is limited to 20 MiB')
            chunks.append(chunk)
        if not size:
            raise HTTPException(422, 'Empty upload')
        media_dir = data_dir / 'media'; media_dir.mkdir(exist_ok=True)
        media_id = uuid.uuid4().hex
        (media_dir / media_id).write_bytes(b''.join(chunks))
        return {'media_id': media_id, 'bytes': size}

    @app.get('/api/runs')
    def runs(model_id: str | None = None, limit: int = Query(100, ge=1, le=500)):
        return store.runs(model_id, limit)

    @app.get('/api/runs/{run_id}')
    def get_run(run_id: str):
        result = store.run(run_id)
        if not result:
            raise HTTPException(404, 'Unknown run')
        return result

    @app.post('/api/runs/{run_id}/cancel', status_code=202)
    def cancel(run_id: str):
        get_run(run_id)
        try:
            engine.cancel(run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {'status': 'cancelling', 'run_id': run_id}

    @app.get('/api/runs/{run_id}/steps')
    def steps(run_id: str):
        return store.steps(run_id)

    @app.get('/api/runs/{run_id}/graph')
    def graph(run_id: str, step: int | None = Query(None, ge=0)):
        if not run_id.startswith('inspection:'):
            get_run(run_id)
        return store.graph(run_id, step)

    @app.get('/api/runs/{run_id}/routes')
    def routes(run_id: str):
        get_run(run_id)
        return store.route_summary(run_id)

    @app.get('/api/runs/{run_id}/playback')
    def playback(run_id: str, step: int = Query(0, ge=0)):
        get_run(run_id)
        return store.playback(run_id, step)

    @app.get('/api/events')
    def events(after: int = Query(0, ge=0), run_id: str | None = None, limit: int = Query(1000, ge=1, le=2000), tail: bool = False):
        return store.events(after, run_id, limit, tail=tail)

    @app.get('/api/stream')
    async def stream(request: Request, after: int = Query(0, ge=0), run_id: str | None = None):
        try:
            after = max(after, int(request.headers.get('last-event-id', '0')))
        except ValueError:
            raise HTTPException(422, 'Invalid Last-Event-ID')
        async def generate():
            cursor, last_ping = after, time.monotonic()
            while not app.state.shutting_down and not await request.is_disconnected():
                batch = await asyncio.to_thread(store.events, cursor, run_id, 500)
                if batch:
                    for event in batch:
                        cursor = event['cursor']
                        yield f'id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n'
                else:
                    if time.monotonic() - last_ping > 10:
                        yield ': keepalive\n\n'; last_ping = time.monotonic()
                    await asyncio.sleep(.08)
        return StreamingResponse(generate(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/runs/{run_id}/export')
    def export(run_id: str, format: Literal['json', 'jsonl'] = 'json'):
        run = get_run(run_id)
        def lines():
            if format == 'json':
                yield json.dumps({'schema_version': 1, 'run': run, 'model': run.get('model_snapshot') or store.model(run['model_id'])}, ensure_ascii=False)[:-1] + ',"events":['
            cursor, first = 0, True
            while True:
                batch = store.events(cursor, run_id, 1000)
                if not batch:
                    break
                for event in batch:
                    cursor = event['cursor']
                    if format == 'json':
                        yield ('' if first else ',') + json.dumps(event, ensure_ascii=False)
                    else:
                        yield json.dumps(event, ensure_ascii=False) + '\n'
                    first = False
            if format == 'json':
                yield ']}'
        return StreamingResponse(lines(), media_type='application/json' if format == 'json' else 'application/x-ndjson', headers={'Content-Disposition': f'attachment; filename="trace-{run_id}.{format}"'})

    @app.get('/api/compare')
    def compare(left: str, right: str):
        a, b = get_run(left), get_run(right)
        ac, bc = store.route_summary(left), store.route_summary(right)
        return {'left': a, 'right': b, 'steps': [len(store.steps(left)), len(store.steps(right))],
                'routing_delta': {k: bc.get(k, 0) - ac.get(k, 0) for k in sorted(ac.keys() | bc.keys())}}

    assets = ROOT / 'ui/dist'
    if assets.is_dir():
        app.mount('/', StaticFiles(directory=assets, html=True), name='browser')
    else:
        @app.get('/')
        def no_ui():
            raise HTTPException(503, 'Browser bundle missing. Run the launcher or npm run build in tools/visualizer/ui.')
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--model-dir', action='append')
    parser.add_argument('--observer')
    parser.add_argument('--model')
    parser.add_argument('--trace-budget-mb', type=int, default=512)
    args = parser.parse_args()
    import uvicorn
    app = create_app(args.data_dir, args.model_dir, args.observer, args.trace_budget_mb, autoload=args.model)
    class LocalServer(uvicorn.Server):
        def handle_exit(self, sig, frame):
            app.state.shutting_down = True
            super().handle_exit(sig, frame)
    LocalServer(uvicorn.Config(app, host='127.0.0.1', port=args.port, access_log=False, timeout_graceful_shutdown=3)).run()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
