from pathlib import Path
import hashlib
import json
import sqlite3
import threading
import time
import uuid

TERMINAL = {'completed', 'cancelled', 'failed', 'interrupted'}

class TraceBudgetExceeded(Exception):
    pass

class Store:
    def __init__(self, path, budget_bytes=512 * 1024 * 1024):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.budget = budget_bytes
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS models (
              id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
              size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, details TEXT, config TEXT);
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id), prompt TEXT NOT NULL,
              status TEXT NOT NULL, started REAL NOT NULL, finished REAL, options TEXT NOT NULL,
              output TEXT NOT NULL DEFAULT '', error TEXT, trace_bytes INTEGER NOT NULL DEFAULT 0, model_snapshot TEXT,
              FOREIGN KEY(model_id) REFERENCES models(id));
            CREATE TABLE IF NOT EXISTS events (
              cursor INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, sequence INTEGER NOT NULL,
              run_id TEXT NOT NULL, model_id TEXT NOT NULL, kind TEXT NOT NULL, step INTEGER,
              received REAL NOT NULL, body TEXT NOT NULL, UNIQUE(session, sequence));
            CREATE INDEX IF NOT EXISTS events_run_step ON events(run_id, step, kind);
            CREATE INDEX IF NOT EXISTS runs_model_started ON runs(model_id, started DESC);
        ''')
        if 'model_snapshot' not in {r['name'] for r in self.db.execute('PRAGMA table_info(runs)')}:
            self.db.execute('ALTER TABLE runs ADD COLUMN model_snapshot TEXT')
        self.db.execute("UPDATE runs SET status='interrupted', finished=?, error='Application stopped before the run finished' WHERE status IN ('queued','running')", (time.time(),))
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def register(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        identity = hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]
        with self.lock, self.db:
            old = self.db.execute('SELECT id FROM models WHERE path=?', (str(path),)).fetchone()
            if old and old['id'] != identity:
                # A replaced model keeps its historical identity/path for replay.
                self.db.execute('UPDATE models SET path=path || ? WHERE id=?', ('#replaced-' + old['id'], old['id']))
            self.db.execute('INSERT OR IGNORE INTO models(id,path,name,size,mtime_ns) VALUES (?,?,?,?,?)',
                            (identity, str(path), path.name, stat.st_size, stat.st_mtime_ns))
        return self.model(identity)

    @staticmethod
    def model_row(row):
        if not row:
            return None
        result = dict(row)
        result['details'] = json.loads(result['details']) if result['details'] else None
        result['config'] = json.loads(result['config']) if result['config'] else None
        return result

    def models(self):
        with self.lock:
            return [self.model_row(r) for r in self.db.execute('SELECT * FROM models ORDER BY name')]

    def model(self, model_id):
        with self.lock:
            return self.model_row(self.db.execute('SELECT * FROM models WHERE id=?', (model_id,)).fetchone())

    def set_config(self, model_id, config):
        with self.lock, self.db:
            self.db.execute('UPDATE models SET config=? WHERE id=?', (json.dumps(config), model_id))

    def create_run(self, model_id, prompt, options):
        run_id = uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute('INSERT INTO runs(id,model_id,prompt,status,started,options,model_snapshot) VALUES (?,?,?,?,?,?,?)',
                            (run_id, model_id, prompt, 'queued', time.time(), json.dumps(options), json.dumps(self.model(model_id))))
        return self.run(run_id)

    @staticmethod
    def run_row(row):
        if not row:
            return None
        result = dict(row)
        result['options'] = json.loads(result['options'])
        result['model_snapshot'] = json.loads(result['model_snapshot']) if result.get('model_snapshot') else None
        return result

    def run(self, run_id):
        with self.lock:
            return self.run_row(self.db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone())

    def runs(self, model_id=None, limit=100):
        with self.lock:
            if model_id:
                rows = self.db.execute('SELECT * FROM runs WHERE model_id=? ORDER BY started DESC LIMIT ?', (model_id, limit))
            else:
                rows = self.db.execute('SELECT * FROM runs ORDER BY started DESC LIMIT ?', (limit,))
            return [self.run_row(r) for r in rows]

    def fail_run(self, run_id, message, status='failed'):
        with self.lock, self.db:
            self.db.execute("UPDATE runs SET status=?, error=?, finished=? WHERE id=? AND status NOT IN ('completed','cancelled','failed','interrupted')", (status, message, time.time(), run_id))

    def ingest(self, event, commit=True):
        kind, run_id = event['kind'], event.get('run_id', '')
        data = event.get('data', {})
        body = json.dumps(event, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        length = len(body.encode())
        with self.lock:
            existing = self.db.execute('SELECT cursor FROM events WHERE session=? AND sequence=?', (event['session'], event['sequence'])).fetchone()
            if existing:
                return existing['cursor']
            run = self.db.execute('SELECT trace_bytes,status FROM runs WHERE id=?', (run_id,)).fetchone()
            if run and run['trace_bytes'] + length > self.budget and kind not in ('run_finished', 'error', 'trace_error'):
                raise TraceBudgetExceeded(run_id)
            cursor = self.db.execute('INSERT INTO events(session,sequence,run_id,model_id,kind,step,received,body) VALUES (?,?,?,?,?,?,?,?)',
                (event['session'], event['sequence'], run_id, event.get('model_id', ''), kind, data.get('step'), time.time(), body)).lastrowid
            if run:
                self.db.execute('UPDATE runs SET trace_bytes=trace_bytes+? WHERE id=?', (length, run_id))
            if kind == 'model_loaded':
                self.db.execute('UPDATE models SET details=? WHERE id=?', (json.dumps(data), event['model_id']))
            if kind == 'run_started':
                self.db.execute("UPDATE runs SET status='running' WHERE id=? AND status='queued'", (run_id,))
            if kind == 'run_finished':
                self.db.execute("UPDATE runs SET status=?, output=?, finished=? WHERE id=? AND status NOT IN ('failed','interrupted')",
                                (data['status'], data.get('output', ''), time.time(), run_id))
            if kind in ('error', 'trace_error') and run:
                self.db.execute("UPDATE runs SET status=?, error=?, finished=? WHERE id=? AND status NOT IN ('failed','interrupted')",
                                (data.get('status', 'failed'), data.get('message', 'Engine failure'), time.time(), run_id))
            if commit:
                self.db.commit()
            return cursor

    def events(self, after=0, run_id=None, limit=1000, kinds=None, tail=False):
        query, args = 'SELECT cursor,body FROM events WHERE cursor>?', [after]
        if run_id is not None:
            query += ' AND run_id=?'; args.append(run_id)
        if kinds:
            query += ' AND kind IN (' + ','.join('?' for _ in kinds) + ')'; args.extend(kinds)
        query += (' ORDER BY cursor DESC LIMIT ?' if tail else ' ORDER BY cursor LIMIT ?'); args.append(limit)
        with self.lock:
            result = [dict(json.loads(r['body']), cursor=r['cursor']) for r in self.db.execute(query, args)]
            return list(reversed(result)) if tail else result

    def latest_cursor(self):
        with self.lock:
            return self.db.execute('SELECT COALESCE(MAX(cursor),0) FROM events').fetchone()[0]

    def resolve_trace(self, run_id):
        if run_id.startswith('inspection:') and run_id.count(':') == 1:
            model_id = run_id.split(':', 1)[1]
            with self.lock:
                row = self.db.execute("SELECT run_id FROM events WHERE model_id=? AND kind='load_started' ORDER BY cursor DESC LIMIT 1", (model_id,)).fetchone()
            if row:
                return row['run_id']
        return run_id

    def steps(self, run_id):
        run_id = self.resolve_trace(run_id)
        with self.lock:
            rows = self.db.execute("SELECT body FROM events WHERE run_id=? AND kind='step_end' ORDER BY cursor", (run_id,))
            return [json.loads(r['body'])['data'] for r in rows]

    def graph(self, run_id, step=None):
        run_id = self.resolve_trace(run_id)
        with self.lock:
            if step is None:
                row = self.db.execute("SELECT MAX(step) AS step FROM events WHERE run_id=? AND kind='step_begin'", (run_id,)).fetchone()
                step = row['step']
            rows = self.db.execute("SELECT body FROM events WHERE run_id=? AND step=? AND kind IN ('node','activation','routing','step_begin','step_end') ORDER BY cursor", (run_id, step)).fetchall()
        nodes, activations, routing, summary = {}, {}, [], {}
        for row in rows:
            event = json.loads(row['body']); data = event['data']
            if event['kind'] == 'node':
                nodes[data['node']['id']] = data['node']
            elif event['kind'] == 'activation':
                activations[data['node_id']] = data
            elif event['kind'] == 'routing':
                routing.append(data)
            else:
                summary.update(data)
        return {'schema_version': 1, 'run_id': run_id, 'step': step, 'nodes': list(nodes.values()), 'activations': activations, 'routing': routing, 'summary': summary}

    def playback(self, run_id, step):
        with self.lock:
            rows = self.db.execute("SELECT cursor,body FROM events WHERE run_id=? AND step=? AND kind IN ('activation','routing','token') ORDER BY cursor", (run_id, step)).fetchall()
        return [dict(json.loads(row['body']), cursor=row['cursor']) for row in rows]

    def route_summary(self, run_id):
        seen, counts = set(), {}
        with self.lock:
            rows = self.db.execute("SELECT body FROM events WHERE run_id=? AND kind='routing'", (run_id,)).fetchall()
        for row in rows:
            data = json.loads(row['body'])['data']
            for selection in data['selections']:
                key = (data['step'], data['component'], data['layer'], selection['token_index'], tuple(selection['experts']))
                if key in seen:
                    continue
                seen.add(key)
                for expert in selection['experts']:
                    label = f"{data['component']}:{data['layer']}:{expert}"
                    counts[label] = counts.get(label, 0) + 1
        return counts
