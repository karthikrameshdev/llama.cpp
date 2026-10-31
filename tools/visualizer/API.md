# Local API and telemetry contract

Base URL: `http://127.0.0.1:8787`. Interactive OpenAPI documentation: `/docs`; machine-readable schema: `/openapi.json`. POST requests require `X-Visualizer-Client: 1` and, if supplied, a matching Origin. Commands are serialized through one native engine. Invalid payloads return 422; overlapping or unavailable engine operations return 409; unknown resources return 404; disallowed paths/origins return 403.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/state` | Engine status, active model/run, durable event cursor, database path |
| GET | `/api/models` | Scan configured roots and list models with available metadata |
| GET | `/api/models/{id}` | Model details, configuration and identity |
| POST | `/api/models/load` | `{model_id, context_size:2048, gpu_layers:99, threads:4, capture:"full", mmproj:""}`; returns 202 |
| GET | `/api/models/{id}/metadata?key=...&offset=0&limit=128` | Page full GGUF arrays/strings; `component=projector` reads the projector, `run_id=...` uses its saved configuration |
| GET | `/api/models/{id}/weights?tensor=...&axis=row&index=0&plane=0&offset=0&limit=32` | Read actual dequantized GGUF values; `axis=column` reads outgoing neuron weights; supports `component` and saved `run_id` |
| POST | `/api/runs` | `{prompt, max_tokens:32, capture:"full", chat:true, token_ids:null, media_id:null}`; returns queued run and 202 |
| POST | `/api/media` | Raw image/audio bytes, maximum 20 MiB; returns opaque `media_id` |
| GET | `/api/runs?model_id=...&limit=100` | Saved prompt archive |
| GET | `/api/runs/{id}` | Status, prompt, options, result, error and retained bytes |
| POST | `/api/runs/{id}/cancel` | Cancel the currently active prompt; returns 202 |
| GET | `/api/runs/{id}/steps` | Recorded step summaries |
| GET | `/api/runs/{id}/graph?step=0` | Exact graph, activations, routing and summary at that step; defaults to latest |
| GET | `/api/runs/{id}/routes` | Deduplicated expert selection counts across the run |
| GET | `/api/runs/{id}/playback?step=0` | Actual activation/routing/token observations in original cursor order for slow replay |
| GET | `/api/runs/{id}/export?format=json` | Stream run/model metadata and all events; `jsonl` streams raw events |
| GET | `/api/events?after=0&run_id=...&limit=1000` | Page durable events by global cursor; `tail=true` returns the latest page in ascending order |
| GET | `/api/stream?after=0&run_id=...` | SSE with durable cursor as event ID; honors Last-Event-ID on reconnect |
| GET | `/api/compare?left=...&right=...` | Run metadata, step counts and expert selection count deltas |
| POST | `/api/engine/restart` | Replace an idle/failed observer session; returns 202 |

Capture is `routing`, `stats` or `full`. Context range is 256–32768, output range 1–1024, prompt limit 131072 characters. Model IDs are derived from resolved path, size and nanosecond mtime; replacing a file creates a new identity while preserving old replay data. Each load has a unique `inspection:<model_id>:<load_id>` trace. The graph alias `inspection:<model_id>` resolves to the latest load.

New runs retain an immutable `model_snapshot` with model/projector metadata and configuration. Export and replay use that snapshot even after another context/projector configuration is loaded. Databases created before snapshot support are migrated on open; older records with no snapshot fall back to the model registry.

## gRPC

`llama.visualizer.v1.Telemetry.Connect(stream EventBatch) returns (stream Command)` is a long-lived bidirectional protobuf stream. The collector binds an ephemeral loopback port. The launched bridge receives the session bearer token through its environment and sends `authorization: Bearer ...` metadata. No token is written into the trace.

Each batch contains up to 128 events. The typed envelope carries sequence, model ID, run ID and kind, plus a JSON event body for flexible architecture-specific details. The bridge adds session identity and preserves native sequence separately. SQLite deduplicates `(session, sequence)`. Commands are `load`, `run`, `cancel` and `shutdown`, with request ID and JSON arguments. The bridge forwards load/run JSON to the standalone native observer; cancellation uses a process signal to interrupt an active synchronous evaluation.

Event bodies have `schema_version:1`, `session`, `sequence`, `native_sequence`, `model_id`, `run_id`, `kind`, `time_us` when emitted by the engine, and `data`. Stored events add `cursor`; database receive time is stored separately. Native monotonic microseconds are not UTC wall-clock timestamps.

Events include `hello`, `load_started`, `model_loaded`, `model_ready`, `run_started`, `step_begin`, `node`, `activation`, `routing`, `step_end`, `token`, `run_finished`, `engine_log`, `observation_error`, `error`, `trace_error` and `bridge_error`.

A `node` contains real operation name, tensor name, shape, strides, dtype, bytes, element count, source node IDs, component, layer, device, buffer, flags and weight/input/operation kind. An `activation` records completed callback wall interval and optional bounded statistics. A `routing` event records the `MUL_MAT_ID.src[2]` source and per-token expert integer selections. `step_end.complete` is false on runtime or observation failure. A failed or interrupted run must not be interpreted as a complete trace even if earlier steps completed.

Full-capture `activation.data.neurons` optionally contains `channels`, `slots`, `token_index`, `expert_ids`, flattened `values`, `meaning` and `scope`. Values are ordered by expert slot, then channel. They cover the first token of that step, within documented limits. The browser's semantic catalog IDs are separate from these raw step/epoch IDs. Stable mapping never rewrites the retained raw events.
