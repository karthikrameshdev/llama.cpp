# Verification report

Verified locally on 2026-10-30 against fork commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`, macOS / Apple M2 Pro, CPU + Accelerate + Metal. The implementation is isolated to `tools/visualizer/`; the existing tracked repository files, including `src`, `ggml`, `include`, `common` and the existing server/UI, have no changes.

## Results

| Suite | Result | Coverage |
| --- | --- | --- |
| Python tests | **32 passed**, no skips on this host | SQLite durability, recovery, event cursors/tail/deduplication, per-step graph isolation, independent load inspections, model replacement, immutable run snapshots, trace budgets, API/path/origin validation, overlapping runs, cancellation, exports, uploads, comparisons, duplex gRPC/authentication, native failure reporting, CPU equivalence, ordered operation playback, exact F32/F16/Q4_0/Q8_0 row/column/expert-plane windows and index bounds |
| JavaScript tests | **11 passed** | Dependency edges, grouping, weights, MoE classification, routing deduplication, deterministic filters, complete preload including projector/expert capacity, immutable catalog across ten prompt shapes, unseen runtime variants, inactive experts, expert parameter links, recycled native-ID isolation |
| Live integration | **91 passed** | Native engine → gRPC → SQLite → REST; real Qwen/hybrid and Granite MoE; ten distinct prompts and routing signatures; dependency integrity; per-step graphs; export; replay; cancellation; overflow/recovery; metadata paging; image encoding; multi-epoch graphs; saved projector settings/logs; all 512 FFN channels for 8 routed slots in all 24 MoE layers; expert identities; quantized incoming/outgoing weight pages |
| Browser workflow | Passed through the actual local application | Model selection/loading, prompt execution, graph rendering, layers/operations, search/Enter inspection, prefill/decode shape changes, 3D, expert heatmaps, model/projector inventory, projector metadata paging, saved logs, history, timeline/replay, comparison, downloaded JSON, capture help, responsive layout at 1440 and 531 pixels |
| Lifecycle | Passed | Service shutdown with an open SSE browser, restart and durable history, automatic SSE reconnect, forced native process crash, visible failure, browser Restart engine, model reload |

CPU token equivalence compares greedy token IDs with **the evaluation callback disabled at context creation** against full tracing on synthetic dense/MoE models and both real Qwen/Granite models. These are exact-output checks for the tested token inputs, not a universal numerical-equivalence proof on every backend.

The real Granite model exposes 24 layers, 32 experts per layer and top-8 routing. A one-token execution produced **1,777 nodes / 2,323 dependency edges / 1,475 completed observations / 72 raw expert-dispatch observations**. Every source dependency resolved. Ten distinct prompts produced **10 distinct routing signatures**; their original per-step graphs and observations remain separately stored.

Qwen3.5-0.8B exercises hybrid attention/recurrent operations. With its compatible vision projector, the tested image prefill produced **5,601 nodes** spanning `multimodal_encoder` and `decoder`. Separate graph epochs were retained, the recorded node count matched the unique node count, and every dependency resolved. The saved prompt preserved its original projector metadata/configuration after the model was reloaded without that projector.

The browser downloaded and parsed a complete 12-token MoE trace containing **39,926 events** (approximately 29.4 MB). Tensor inspection correctly changed from `32 × 52 × 1 × 1` during prefill to `32 × 1 × 1 × 1` during decode. No JavaScript exceptions were observed in the checked application views. Intentional service/native failures were surfaced and recovered as described above.

Exact model revisions, byte sizes and SHA256 values are in `model-manifest.json`. Models used:

- `ggml-org/Qwen3.5-0.8B-GGUF`, Q4_0.
- `bartowski/granite-3.1-1b-a400m-instruct-GGUF`, Q4_K_M.
- `mradermacher/Qwen3.5-0.8B-GGUF`, matching Q8_0 vision projector.

## Stable visualization regression evidence

The model catalog preloads **242 weight tensors, 768 expert slots and 393,216 FFN channels** for the tested Granite model. It contains approximately 2,526 logical entries including structural modules and named inspection operations. This count describes the fixed catalog, not a universal maximum for every input-dependent runtime graph.

Browser tests ran a 10-token prompt with an already selected node and parameter row. The geometry build counter stayed **3 → 3**, the inspector build counter stayed **1 → 1**, and weight index **7** remained selected while 154 highlight updates were applied. A second 4-token prompt confirmed **3 → 3** geometry builds and **1 → 1** inspector builds while retaining layer-output channel **1023** and its channel family. The recorded channel value at step 3/token 57 was `-0.0007603615522384644`.

Slow replay at 2 observations per second, pause and Next operation advanced actual recorded events while both build counters stayed unchanged. The FFN channel browser read neuron **511**, including real Q4_K incoming rows and Q6_K outgoing columns for expert **3**. Layer output features expose all 1,024 captured channels for the tested layer. Immediate channel-index input, expanded runtime-operation browsing, full expert capacity, readiness gating and hover weight previews are included in the final browser pass.

The updated machine-readable integration evidence is `data/stable-live-verification.json`, with `data/stable-live-verification.log` and `data/stable-unit.log`. Earlier verification files describe the first implementation before the stable-view amendment. Browser screenshots are task outputs.

## Reproduce the current build

From the repository root, first launch the application and download the pinned models as described in README.md. In another terminal:

```sh
cd tools/visualizer
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
npm --prefix ui test
.venv/bin/python tests/verify_live.py --output data/live-verification.json
```

If the virtual environment has uv but no pip, use `uv pip install --python .venv/bin/python -r requirements-dev.txt`. The two synthetic native tests use the local `../../build/context-models/llama-dense.gguf` and `llama-moe.gguf` fixtures and skip when absent. Real native tests skip when the pinned local models or observer have not been prepared. Check the skip count when reproducing; the reported 32-test run had none. Live integration requires all pinned models and a running service; it creates real prompts, changes the loaded model and retains its traces.

Machine-readable live evidence on this workstation is saved in `data/live-verification-final.json`; logs are `data/live-verification-final.log` and `data/unit-final.log`. These generated artifacts and the SQLite database are ignored by Git. Browser screenshots are delivered separately in the task output.

The Python suite emits a Starlette/httpx TestClient deprecation warning. Vite emits its advisory that the bundled Three.js vendor chunk is approximately 504 kB (125 kB gzip); the current application bundle is approximately 48 kB (17 kB gzip). These did not affect verification.

## Validation limits

This establishes the tested local workflows, not an assurance that every possible model, prompt, operating system, GPU or failure mode is issue-free. No 7B/20B/30B model was loaded on this host. Audio, dedicated encoder/encoder-decoder models, Windows cancellation, non-Metal GPUs, distributed execution and concurrent sequences remain unvalidated. Large models/full captures may need larger trace budgets and substantial disk space. Sampled statistics and callback wall intervals have the accuracy limits described in README.md.
