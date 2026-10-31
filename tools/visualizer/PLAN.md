# Runtime model visualizer

## Requirements and boundaries

Build an opt-in local application around the unmodified llama.cpp libraries. Selecting a model loads its real GGUF metadata, tensor inventory and a runtime inspection graph. Prompt execution produces its own graph variants, operation observations, token stream and actual MoE routing decisions. Dense layers normally all execute; different prompts change values and routing, not necessarily topology.

No files under `src/`, `ggml/`, `include/`, or the existing server/UI are changed. An arbitrary uninstrumented process cannot be inspected retroactively. The native adapter attaches `cb_eval` when creating model contexts, including optional libmtmd contexts. These callbacks preserve model operations and outputs, but synchronization and tensor reads affect latency and backend fusion. Capture levels make that cost explicit.

## Architecture

```
Browser / Three.js graph / history / replay / raw log
    | REST commands + resumable SSE events
Local FastAPI service -> SQLite (WAL), JSONL exports
    | protobuf bidirectional gRPC on loopback
Python engine bridge (bounded queue, no silent event drops)
    | framed JSON commands and observations over private stdio
Native C++ adapter -> existing libllama + libmtmd callbacks
```

The stdio boundary keeps gRPC/Python dependencies out of native libraries. All engine-to-service telemetry and service-to-engine commands cross the real gRPC stream. The native adapter can also be used alone for pure JSONL logging.

## Delivery sequence

1. Native observer: metadata/tensors, recursive operation dependencies, runtime graph discovery, measured node observations, bounded tensor samples and per-token expert IDs. Generic encode/decode handling, chat templates, optional multimodal projector, cancellation and failures.
2. Protobuf contract and engine bridge: duplex commands/events, bounded buffering, readiness, process lifecycle and explicit failure reporting.
3. Local service: model registry restricted to configured directories, serialized engine control, durable per-prompt event history, graph snapshots, paging, SSE resume, cancellation, export and restart recovery.
4. Browser UI: dynamically laid out Three.js graph, layers and operation views, search/filter/inspection, prompt execution, token/step timeline, expert heatmaps, saved runs, replay, comparisons and logs. No simulated activations in the live view.
5. One-command launcher: isolated Python dependencies, local frontend bundle, native adapter build, automatic browser opening and optional model auto-load.
6. Verification: backend unit/integration/error tests; synthetic dense/MoE traces; real Qwen hybrid and Granite MoE models; ten distinct prompts and durable reload; cancellation; tracing output equivalence; live browser controls, graph picking, history/replay/export and layout; core-source invariance.

## Data and accuracy

### Stable model view amendment

Preload a catalog from all backbone/projector weight tensors, actual module roles, full expert capacity, FFN channel dimensions and the runtime inspection. Keep its objects/positions fixed throughout prompts and replay. Match actual step/epoch tensors to semantic catalog entries; map unseen variants to existing component/layer modules. Preserve raw runtime graphs separately. Only model/view/filter changes rebuild geometry. Inspector controls remain mounted while their observed text/values update.

Replay original activation/routing/token events at a selected rate with pause and single-observation stepping. Expose actual paged GGUF row/column/plane values using existing quantization decoders. Provide paged FFN neuron and residual feature channels, with complete bounded first-token vectors and explicit missing-value scope. Keep encoder/projector weights and unused expert slots visible before they execute. Do not describe a logical catalog count as a universal maximum for every possible runtime shape.

- Model identity includes resolved file path, size and modification time. Metadata retains actual architecture keys; tensor inventory retains names, shapes, types and bytes.
- Every execution has a unique run ID. Every decode/encode call has a step ID and phase. Runtime nodes contain operation, dimensions, strides, source IDs, buffer/device information and component.
- Node discovery is separate from completed observation. Callback wall intervals are not GPU kernel timings. Bounded statistics report their sampled element count; unobserved operations are never labeled measured.
- Routing is read from integer expert-ID tensors used by actual `MUL_MAT_ID` operations, including per-token columns and real strides. The UI can aggregate invocation counts while retaining the original observations.
- SQLite is authoritative for reconnect/replay. Traces and graph snapshots are retained separately for each run; JSON/JSONL exports include schema versions and provenance.
- Event volume is bounded by a configured trace budget; exceeding it fails the trace explicitly rather than silently showing an incomplete graph as complete.
- Support follows capabilities exposed by the linked runtime. Specialized generation pipelines must be labeled rather than invented. Initial live model tests cover the host's CPU/Metal backends, not every supported device or model family.
