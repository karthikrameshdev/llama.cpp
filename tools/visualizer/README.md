# llama.cpp Runtime Observatory

A local browser application with a **stable model capacity graph**, measured runtime highlights, tensor/weight inspection and per-token mixture-of-experts selections. Original runtime graphs remain recorded per prompt and step. This is an optional standalone tool; existing llama.cpp sources and execution algorithms are unchanged.

## Start

From the repository root:

```sh
python3 tools/visualizer/launch.py --model /absolute/path/to/model.gguf
```

The launcher builds the native observer, installs locked Python and JavaScript dependencies, starts the local service, opens **http://127.0.0.1:8787**, and loads the model. Subsequent launches reuse installed dependencies and incremental native builds. First setup needs internet access. The application itself uses bundled assets and runs locally.

Prerequisites: Python 3.10+, CMake 3.20+, a C++17 compiler, and Node.js 20.19+ or 22.12+ with npm. Use a matching Xcode/SDK toolchain on macOS. The tested host uses Python 3.13, Node 26.8.2, Apple M2 Pro and Metal. Linux/Windows and other GPUs have not been validated for this new tool. POSIX cancellation uses SIGUSR1; Windows currently stops the observer process when cancelling and requires restarting it.

For an existing model library:

```sh
python3 tools/visualizer/launch.py --model-dir /absolute/path/to/models
```

Repeat `--model-dir` for additional directories. The directory containing `--model` is also allowed. Optional arguments include `--no-browser`, `--port 8787`, `--data-dir /path/to/traces`, `--build-dir /path/to/llama-build`, and `--trace-budget-mb 512`. Set GPU layers to zero in Engine settings for CPU execution. A preexisting build must provide compatible **shared** llama, llama-common, ggml and mtmd libraries. The launcher builds them when absent.

Pinned, SHA256-verified small verification models are listed in `model-manifest.json`. Download them with:

```sh
python3 tools/visualizer/download_models.py
python3 tools/visualizer/launch.py --model tools/visualizer/models/granite-3.1-1b-moe-q4_k_m.gguf
```

The three files total approximately 1.5 GB. Review the model publishers' licenses and model cards for your intended use. Model weights, builds, dependencies and local traces are ignored by Git.

## Use the browser

1. Select a model. Its GGUF weight tensors, structural modules, full expert capacity and named operations from a one-token runtime inspection preload into a fixed catalog. The capacity banner includes weight/expert counts and FFN hidden channels. Optional image/audio support requires a compatible `mmproj.gguf` path in Engine settings before loading.
2. Enter a prompt, choose a capture level and select **Run & trace**. The graph follows real prefill/decode execution, including multimodal encoder operations when media is supplied.
3. Switch between **Layers** and **Operations**, filter components, or search. Hover for explanations and a real weight preview. Click a node, or press Enter in search, for dimensions, strides, type, bytes, device, dependencies and measured values. The parameter browser pages actual dequantized rows/columns and expert planes. The channel browser pages FFN neurons and layer output features; select any FFN channel to read its incoming/outgoing weight windows. Drag to pan, scroll to zoom, and toggle 3D.
4. **Experts** shows actual integer IDs selected by MoE dispatch. **Model** exposes architecture metadata and searchable weights; large tokenizer arrays are paged. **Log** shows recent raw observations.
5. Select a saved prompt to inspect its recorded activity on the fixed model. **Execution timeline** selects prefill/decode steps. **Slow replay** advances recorded operations at 2, 10, 40 or 100 observations per second; pause or use **Next operation** for a single advance. **Compare runs** reports expert selection count differences. **JSON / JSONL** downloads the complete retained trace for that prompt.

Prompt execution, step changes, replay and opening Inspect update colors/text on existing objects. They preserve geometry, positions, camera, selection, inspector controls and weight pages. Geometry is rebuilt when a model loads or the user changes graph view/filter/search settings. Runtime tensor IDs and shapes are mapped to stable semantic nodes; previously unseen variants highlight an existing component/layer module. Browse a node's runtime operations for exact step-specific records. Raw graphs and actual source dependencies are still stored independently for every execution step.

The catalog uses a real one-token inspection during loading, but the ready model view is idle. Probe observations remain in the logs/database; prompt and replay observations drive the visible activation highlights.

The capacity banner describes the model catalog, all expert slots and FFN channels from real tensor dimensions. A model does not have a single finite maximum runtime node count independent of input length, batching, media size and backend graph choices. Hidden channels are paged in the inspector rather than allocating millions of browser objects. Catalog module/expert entries are structural capacity; a bright measured highlight requires an actual runtime observation or expert dispatch.

Stop cancels at an execution boundary. Native failures surface in the header; **Restart engine** starts a fresh observer without removing saved traces. Ctrl+C in the launcher stops the local service and observer. Saved runs survive application restarts; interrupted runs are explicitly marked.

## What is measured

Graphs are discovered recursively through the existing `cb_eval` callback and tensor source references, rather than inferred from a model name. Node IDs are scoped to a graph epoch and execution step, including multiple graphs inside multimodal prefill. Initial inspection describes a **real one-token evaluation**, not every possible future input shape. Each prompt retains its actual graph variants.

| Capture | Runtime topology | Completed observations | Tensor values | Expert IDs |
| --- | --- | --- | --- | --- |
| Routing + graph | All callback-visible nodes and dependencies | Routing and named output boundaries | None | Actual dispatch IDs |
| Layer statistics + routing | Same | Routing and named layer/output boundaries | Bounded samples at observed boundaries | Actual dispatch IDs |
| Every operation + samples | Same | Every operation exposed to the evaluation callback | Up to 256 elements for statistics; first 16 values retained. Additional full first-token FFN and layer-output channel vectors within the limits below | Actual dispatch IDs |

Statistics explicitly report `sampled_elements`, `total_elements` and `complete`. A dim topology node does not imply a completed measurement. View/input/weight nodes can appear without a completion callback. Backend fusion and supported callback boundaries determine which operations are exposed; this is not a GPU kernel profiler or a dump of every activation for every token. Wall intervals include observer work and synchronization. Capture can inhibit fusion and substantially slow inference, particularly full capture. It preserves model algorithms and was checked for exact greedy CPU token equivalence on the tested models; timing and device scheduling are affected.

Full capture additionally records named FFN nonlinear outputs and `l_out` layer outputs: all channels at the **first token of each execution step**, including each routed expert slot. Bounds are 65,536 channels and 262,144 values per tensor, with a 4 MiB read limit. Prefill traces do not dump every channel for every input-token position. Single-token decode steps retain that token's full captured channel vectors. Each record names its token index, expert IDs and scope. Missing/unreached/older channel data is labeled unavailable. Layer-output features mix several paths; FFN channels have direct incoming/up-gate and outgoing/down weight windows.

Weight windows read the local GGUF file without changing the engine. Quantized values are decoded with `gguf-py`; unsupported decoders or invalid indexes return explicit errors. A weight page is a precise slice, while output statistics are bounded samples. Every row/column can be paged for supported types; the UI never downloads a multi-gigabyte tensor in one response.

Dense transformer layers usually execute for every prompt. The topology can remain identical while values differ. Routed expert IDs and invocation counts can change with the prompt. Repeated gate/up/down observations remain in the raw trace; heatmaps deduplicate those repeated selections by component, layer, token, step and selected expert set.

The observer must be installed when creating the context. It cannot attach retroactively to an arbitrary stock `llama-cli` or `llama-server` process. This application owns its opt-in execution context and uses greedy sampling, one independent prompt per run, a 256-token microbatch, and configurable context/output limits. Generic text encode/decode and libmtmd image/audio paths are implemented. Audio, encoder-only/encoder-decoder families, speculative generation, distributed backends, concurrent sequences, video, and 7B/20B/30B configurations have not been validated. Existing support in llama.cpp is not a claim that every specialized pipeline is covered by this adapter.

## Transport and storage

```text
Native observer → private JSONL stdio → Python bridge
             ↔ bidirectional protobuf gRPC ↔ local collector / SQLite WAL
             ↔ REST commands + resumable SSE ↔ Three.js browser
```

`proto/telemetry.proto` defines the actual duplex gRPC service. Native libraries have no added gRPC dependency. The bridge batches events into a bounded queue and applies backpressure; it does not silently drop observations. The browser redraws at approximately 10 Hz while every retained event remains in SQLite. A fast operation can finish between browser frames; replay preserves its observations.

`data/traces.sqlite3` stores models, configurations, prompts, results, graph nodes, routing, tokens, engine logs and timestamped events. The default 512 MiB **per-prompt** trace budget cancels and marks an over-budget trace failed/incomplete. Model inspections and overall retention are separate; monitor disk space for long sessions. No automatic history deletion is performed. JSONL export provides pure structured logging. You can also run `build/llama-visualizer-observer` directly with one JSON command per stdin line; observations go to stdout and engine messages to stderr.

The HTTP and gRPC listeners bind to loopback. A per-session bearer token protects gRPC; HTTP mutations require same-origin requests and `X-Visualizer-Client: 1`. Model/projector paths must resolve inside configured model roots. Media uploads are limited to 20 MiB. This is a trusted local workstation application, with no remote multiuser authentication. Prompts, uploaded media, file paths and traces are stored locally; exports contain them.

See [API.md](API.md), [PLAN.md](PLAN.md) and [TESTING.md](TESTING.md) for the contract, implementation plan and verification evidence.
