import "./style.css";
import { GraphView, category, routeCounts } from "./graph.js";
import { ModelTopology, explanations } from "./topology.js";
const $ = (id) => document.getElementById(id),
    esc = (v) =>
        String(v ?? "").replace(
            /[&<>"']/g,
            (c) =>
                ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;",
                })[c],
        ),
    pretty = (v) => JSON.stringify(v, null, 2),
    num = (v) =>
        new Intl.NumberFormat("en", {
            notation: Math.abs(v) >= 1e6 ? "compact" : "standard",
            maximumFractionDigits: 1,
        }).format(v || 0);
const state = {
    engine: {},
    models: [],
    runs: [],
    model: null,
    runId: null,
    step: -1,
    steps: [],
    nodes: new Map(),
    activations: {},
    routing: [],
    summary: {},
    logs: [],
    grouped: true,
    weights: true,
    filter: "all",
    search: "",
    live: true,
    selected: null,
    dirty: false,
    structureDirty: false,
    fit: false,
    playing: null,
    mediaId: null,
    cursor: 0,
    graphRequest: 0,
    topology: null,
    topologyKey: null,
    graphReady: false,
    preloadVersion: 0,
    activity: {},
    expertGridKey: null,
    replayRecords: [],
    replayPosition: 0,
    replayStepIndex: 0,
    replayPrepared: false,
    replayGeneration: 0,
};
const emptyInspector = $("inspect-node").innerHTML;
function notice(text, error = false) {
    $("notice").textContent = text;
    $("notice").classList.toggle("error", error);
}
async function api(path, body) {
    const r = await fetch(
        path,
        body === undefined
            ? {}
            : {
                  method: "POST",
                  headers: {
                      "Content-Type": "application/json",
                      "X-Visualizer-Client": "1",
                  },
                  body: JSON.stringify(body),
              },
    );
    const d = await r.json();
    if (!r.ok)
        throw Error(typeof d.detail === "string" ? d.detail : pretty(d.detail));
    return d;
}
function inspectTab(name) {
    document
        .querySelectorAll("[data-inspect]")
        .forEach((b) =>
            b.classList.toggle("active", b.dataset.inspect === name),
        );
    for (const n of ["node", "experts", "model", "log"])
        $("inspect-" + n).hidden = n !== name;
}
document
    .querySelectorAll("[data-inspect]")
    .forEach((b) => (b.onclick = () => inspectTab(b.dataset.inspect)));
let graph;
let hoverId = null;
const hoverWeights = new Map();
try {
    graph = new GraphView($("graph"), selectNode, (n, e) => {
        const t = $("graph-tooltip");
        t.hidden = !n;
        if (n) {
            const weights = state.topology?.weightsFor(n) || [];
            const observed = n.members
                .map((x) => state.topology?.observed(x.id, state.activations))
                .find((x) => x?.activation);
            t.innerHTML =
                "<strong>" +
                esc(
                    n.component +
                        " / " +
                        (n.layer < 0 ? "global" : "layer " + n.layer) +
                        " / " +
                        n.name,
                ) +
                "</strong><p>" +
                esc(explanations[n.category] || explanations.other) +
                "</p><div>" +
                n.members.length +
                " catalog entries · " +
                weights.length +
                " parameter tensors</div><div>" +
                esc(
                    observed?.raw
                        ? observed.raw.shape.join(" × ") +
                              " · " +
                              observed.raw.dtype
                        : "Preloaded model capacity",
                ) +
                "</div><div>" +
                esc(
                    weights
                        .slice(0, 3)
                        .map((w) => w.name)
                        .join("\n"),
                ) +
                '</div><pre class="hover-values">' +
                esc(
                    hoverWeights.get(n.id) ||
                        "Hover to read a weight window. Click for channels and all weight pages.",
                ) +
                "</pre>";
            if (hoverId !== n.id) {
                hoverId = n.id;
                if (weights.length && !hoverWeights.has(n.id)) {
                    const id = n.id;
                    readWeights(weights[0], 0, 0, 0, "row", 8)
                        .then((d) => {
                            hoverWeights.set(
                                id,
                                "Weight row 0, first " +
                                    d.count +
                                    ": " +
                                    d.values
                                        .map(
                                            (v) =>
                                                v?.toPrecision(4) ??
                                                "nonfinite",
                                        )
                                        .join(", "),
                            );
                            if (hoverId === id)
                                t.querySelector(".hover-values").textContent =
                                    hoverWeights.get(id);
                        })
                        .catch((error) => hoverWeights.set(id, error.message));
                }
            }
            const r = $("graph").getBoundingClientRect();
            t.style.left =
                Math.max(0, Math.min(e.clientX - r.left + 12, r.width - 240)) +
                "px";
            t.style.top =
                Math.max(0, Math.min(e.clientY - r.top + 12, r.height - 240)) +
                "px";
        } else {
            hoverId = null;
        }
    });
} catch (e) {
    notice("WebGL unavailable: " + e.message, true);
}
function clearActivity() {
    state.graphRequest++;
    state.nodes.clear();
    state.activations = {};
    state.routing = [];
    state.summary = {};
    state.lastNode = null;
    $("activation-label").textContent =
        "Waiting for the next measured operation · model layout stays fixed";
    state.topology?.bind([]);
    state.dirty = true;
}
function resetModelGraph() {
    stopReplay();
    clearActivity();
    state.steps = [];
    state.step = -1;
    $("timeline").max = 0;
    $("timeline").value = 0;
    $("step-list").innerHTML =
        '<p class="empty-small">Run a prompt to inspect prefill and decode steps.</p>';
    state.topology = null;
    state.topologyKey = null;
    state.graphReady = false;
    $("graph").dataset.topologyReady = "false";
    $("model-capacity").textContent = "Preloading model capacity…";
    state.preloadVersion++;
    state.selected = null;
    state.structureDirty = true;
    state.fit = true;
    state.expertGridKey = null;
    hoverWeights.clear();
    $("inspect-node").innerHTML = emptyInspector;
}
async function preloadModel(force = false) {
    const model = state.model;
    if (!model?.details) return;
    const key =
        model.id +
        ":" +
        (model.config?.mmproj || "") +
        ":" +
        (model.details.projector?.tensors?.length || 0);
    if (!force && state.topologyKey === key) return;
    state.graphReady = false;
    $("graph").dataset.topologyReady = "false";
    const version = ++state.preloadVersion;
    const inspection = await api(
        "/api/runs/" + encodeURIComponent("inspection:" + model.id) + "/graph",
    );
    if (version !== state.preloadVersion || state.model !== model) return;
    state.topology = new ModelTopology(model, inspection);
    state.topology.bind(state.nodes.values());
    state.topologyKey = key;
    state.expertGridKey = null;
    state.selected = null;
    $("inspect-node").innerHTML = emptyInspector;
    state.structureDirty = true;
    state.fit = true;
    state.dirty = true;
    const c = state.topology.capacity;
    $("model-capacity").textContent =
        "PRELOADED CAPACITY  ·  " +
        num(c.nodes) +
        " catalog nodes  ·  " +
        num(c.weights) +
        " weight tensors  ·  " +
        num(c.experts) +
        " experts  ·  " +
        num(c.channels) +
        " FFN channels";
}
function render() {
    if (!state.dirty) return;
    state.dirty = false;
    const nodes = [...(state.topology?.nodes.values() || [])];
    state.activity =
        state.topology?.activity(state.activations, state.routing) || {};
    const active = state.topology?.rawToStable.get(state.lastNode);
    if (state.structureDirty) {
        graph?.setData(
            nodes,
            state.activity,
            {
                grouped: state.grouped,
                weights: state.weights,
                filter: state.filter,
                search: state.search,
            },
            state.fit,
        );
        state.structureDirty = false;
        state.fit = false;
        state.graphReady = Boolean(state.topology);
        $("graph").dataset.topologyReady = String(state.graphReady);
        updateEngine(state.engine);
    } else graph?.updateActivity(state.activity, active);
    $("graph-empty").hidden = nodes.length > 0;
    $("graph-count").textContent =
        num(nodes.length) +
        " preloaded nodes · " +
        num(graph?.layout?.nodes.length) +
        " visible";
    $("metric-nodes").textContent = nodes.length ? num(nodes.length) : "—";
    $("metric-edges").textContent = nodes.length
        ? num(nodes.reduce((a, n) => a + n.sources.length, 0))
        : "—";
    $("metric-observed").textContent = num(
        Object.keys(state.activations).length,
    );
    $("metric-time").textContent = state.summary.duration_us
        ? (state.summary.duration_us / 1000).toFixed(1) + " ms"
        : "—";
    const counts = routeCounts(state.routing);
    $("metric-routes").textContent = counts.size
        ? num([...counts.values()].reduce((a, b) => a + b, 0))
        : "—";
    $("phase-badge").textContent =
        state.summary.phase?.toUpperCase() || "MODEL INSPECTION";
    renderExperts(counts);
    renderLog();
    if (state.lastNode) {
        const raw = state.nodes.get(state.lastNode);
        $("activation-label").textContent = raw
            ? "HIGHLIGHT  ·  " +
              raw.component +
              " / L" +
              raw.layer +
              " / " +
              (raw.name || raw.op)
            : "Following observed execution";
    }
    if (state.selected) updateInspector();
}
setInterval(render, 100);
function setGraph(d) {
    state.step = d.step ?? -1;
    state.nodes = new Map(d.nodes.map((n) => [n.id, n]));
    state.topology?.bind(state.nodes.values());
    state.activations = d.activations;
    state.routing = d.routing;
    state.summary = d.summary;
    state.lastNode = Object.keys(d.activations).at(-1);
    state.dirty = true;
    $("step-label").textContent = state.step < 0 ? "—" : state.step;
    $("timeline").value = Math.max(0, state.step);
}
async function loadGraph(step) {
    if (!state.runId) return;
    await preloadModel();
    const id = state.runId,
        request = ++state.graphRequest,
        d = await api(
            "/api/runs/" +
                encodeURIComponent(id) +
                "/graph" +
                (step == null ? "" : "?step=" + step),
        );
    if (state.runId === id && state.graphRequest === request) {
        setGraph(d);
        if (id.startsWith("inspection:")) {
            state.activations = {};
            state.routing = [];
            state.lastNode = null;
            state.summary = { phase: "model ready" };
            state.step = -1;
            $("step-label").textContent = "—";
        }
    }
}
function selectNode(item) {
    state.selected = item;
    inspectTab("node");
    renderSelected(item);
}
function selectRuntime(id) {
    const n = state.topology?.nodes.get(
        state.topology.rawToStable.get(id) || id,
    );
    if (n)
        selectNode({
            id: n.id,
            name: n.name,
            members: [n],
            category: category(n),
            layer: n.layer,
            component: n.component,
        });
}
function selectedObservation(item) {
    if (item.rawName != null) {
        const raw = [...state.nodes.values()].find(
            (n) =>
                n.name === item.rawName &&
                n.op === item.rawOp &&
                n.component === item.component &&
                n.layer === item.layer,
        );
        return { raw, activation: raw && state.activations[raw.id] };
    }
    return state.topology.observed(item.members[0].id, state.activations);
}
function renderSelected(item) {
    const panel = $("inspect-node");
    panel.dataset.inspectorBuilds =
        Number(panel.dataset.inspectorBuilds || 0) + 1;
    if (item.members.length > 1) {
        panel.innerHTML =
            '<div class="eyebrow">' +
            esc(item.component) +
            " / LAYER " +
            item.layer +
            "</div><h3>" +
            esc(item.name) +
            '</h3><p class="inspector-copy">' +
            item.members.length +
            ' actual nodes. Select one to inspect.</p><div class="group-members">' +
            item.members
                .map(
                    (n) =>
                        '<button data-node="' +
                        esc(n.id) +
                        '">' +
                        esc(n.name || n.op) +
                        "<br>" +
                        esc(n.op) +
                        " · " +
                        n.shape.join(" × ") +
                        "</button>",
                )
                .join("") +
            "</div>";
    } else {
        const catalog = item.members[0],
            observed = selectedObservation(item),
            n = observed.raw || catalog,
            a = observed.activation;
        panel.innerHTML =
            '<div class="eyebrow">' +
            esc(n.component) +
            " / " +
            (n.layer < 0 ? "GLOBAL" : "LAYER " + n.layer) +
            "</div><h3>" +
            esc(n.op) +
            '</h3><p class="node-name">' +
            esc(n.name || n.id) +
            "</p>" +
            Object.entries({
                Shape: n.shape.join(" × "),
                Type: n.dtype,
                Elements: num(n.elements),
                Bytes: num(n.bytes),
                Device: n.device,
                Buffer: n.buffer,
                Observation: a ? "Completed callback" : "Topology only",
                Wall: a ? a.wall_us + " µs" : "Not measured",
            })
                .map(
                    ([k, v]) =>
                        '<div class="property"><span>' +
                        esc(k) +
                        '</span><span data-property="' +
                        k +
                        '">' +
                        esc(v) +
                        "</span></div>",
                )
                .join("") +
            '<div class="detail-label">SOURCE DEPENDENCIES</div><div class="node-sources">' +
            n.sources
                .map(
                    (id) =>
                        '<button data-node="' +
                        esc(id) +
                        '">' +
                        esc(state.nodes.get(id)?.name || id) +
                        "</button>",
                )
                .join("") +
            '</div><div class="detail-label">OUTPUT STATISTICS</div><pre id="node-statistics" class="detail-pre">' +
            esc(
                a?.stats
                    ? pretty(a.stats)
                    : "No output values captured. Full capture samples up to 256 values.",
            ) +
            '</pre><div class="detail-label">STRIDES / FLAGS</div><pre class="detail-pre">' +
            esc(pretty({ strides: n.strides, flags: n.flags })) +
            "</pre>" +
            (state.routing.some((r) => r.node_id === n.id)
                ? '<div class="detail-label">EXPERT DISPATCH</div><pre class="detail-pre">' +
                  esc(pretty(state.routing.filter((r) => r.node_id === n.id))) +
                  "</pre>"
                : "");
    }
    panel
        .querySelectorAll("[data-node]")
        .forEach((b) => (b.onclick = () => selectRuntime(b.dataset.node)));
    panel.insertAdjacentHTML(
        "afterbegin",
        '<p class="inspector-copy">' +
            esc(explanations[item.category] || explanations.other) +
            "</p>",
    );
    panel.insertAdjacentHTML(
        "beforeend",
        '<details class="runtime-binding-browser"><summary>Browse runtime operations in this step</summary><input id="runtime-binding-search" placeholder="Filter actual operation names…" aria-label="Filter runtime operations"><div id="runtime-bindings" class="group-members"></div><p class="accuracy-note">Runtime variants share fixed catalog locations. Open an operation for its exact recorded shape and source tensors.</p></details>',
    );
    const browse = () => {
        const query = $("runtime-binding-search").value.toLowerCase();
        const candidates = [...state.nodes.values()].filter(
            (n) =>
                n.component === item.component &&
                (item.layer < 0 || n.layer === item.layer) &&
                (n.name + " " + n.op).toLowerCase().includes(query),
        );
        $("runtime-bindings").innerHTML =
            candidates
                .slice(0, 100)
                .map(
                    (n) =>
                        '<button data-raw="' +
                        esc(n.id) +
                        '">' +
                        esc(n.name || n.id) +
                        "<br>" +
                        esc(n.op) +
                        "</button>",
                )
                .join("") +
            (candidates.length > 100
                ? '<p class="accuracy-note">100 / ' +
                  candidates.length +
                  " shown. Filter to inspect any operation.</p>"
                : "");
        $("runtime-bindings")
            .querySelectorAll("[data-raw]")
            .forEach(
                (button) =>
                    (button.onclick = () => {
                        const raw = state.nodes.get(button.dataset.raw);
                        const catalog =
                            raw &&
                            state.topology.nodes.get(state.topology.match(raw));
                        if (catalog)
                            selectNode({
                                ...catalog,
                                members: [catalog],
                                rawName: raw.name,
                                rawOp: raw.op,
                            });
                    }),
            );
    };
    $("runtime-binding-search").oninput = browse;
    panel.querySelector(".runtime-binding-browser").ontoggle = (event) => {
        if (event.target.open) browse();
    };
    const weights = state.topology.weightsFor(item);
    if (weights.length) {
        panel.insertAdjacentHTML(
            "beforeend",
            '<section class="tensor-browser"><div class="detail-label">PARAMETERS · ACTUAL GGUF VALUES</div><select id="weight-tensor" aria-label="Parameter tensor">' +
                weights
                    .map(
                        (w, i) =>
                            '<option value="' +
                            i +
                            '">' +
                            esc(w.name) +
                            "</option>",
                    )
                    .join("") +
                '</select><div class="tensor-controls"><label>Axis<select id="weight-axis"><option value="row">Row / incoming</option><option value="column">Column / outgoing</option></select></label><label>Index<input id="weight-index" type="number" min="0" value="0"></label><label>Expert / plane<input id="weight-plane" type="number" min="0" value="' +
                (item.members.find((n) => n.kind === "expert")?.expert || 0) +
                '"></label><label>Offset<input id="weight-offset" type="number" min="0" value="0"></label></div><div class="tensor-actions"><button id="weight-read" class="secondary compact">Read weights</button><button id="weight-next" class="secondary compact">Next 32 values</button></div><pre id="weight-window" class="detail-pre">Loading actual weight values…</pre><p class="accuracy-note">Every weight can be read by index and page. Quantized blocks are dequantized using this repository\'s GGUF decoder.</p></section>',
        );
        const read = async () => {
            const target = $("weight-window");
            try {
                const w = weights[Number($("weight-tensor").value)];
                const d = await readWeights(
                    w,
                    Number($("weight-index").value),
                    Number($("weight-plane").value),
                    Number($("weight-offset").value),
                    $("weight-axis").value,
                );
                if (target.isConnected) target.textContent = pretty(d);
            } catch (error) {
                if (target.isConnected) target.textContent = error.message;
            }
        };
        $("weight-read").onclick = read;
        $("weight-next").onclick = () => {
            $("weight-offset").value = Number($("weight-offset").value) + 32;
            read();
        };
        $("weight-tensor").onchange = () => {
            $("weight-index").value = 0;
            $("weight-offset").value = 0;
            read();
        };
        read();
    }
    const ffnNeurons = state.topology.neurons.get(
        item.component + ":" + item.layer,
    );
    let neurons = ffnNeurons;
    if (neurons) {
        panel.insertAdjacentHTML(
            "beforeend",
            '<section class="neuron-browser"><div class="detail-label">HIDDEN CHANNELS OF THIS LAYER</div><p id="neuron-capacity" class="inspector-copy">' +
                num(neurons.width) +
                " channels per expert · " +
                num(neurons.planes) +
                ' plane(s). Channel details are paged so the model graph stays fixed.</p><div class="tensor-controls"><label>Neuron / channel<input id="neuron-index" aria-label="Neuron index" type="number" min="0" max="' +
                (neurons.width - 1) +
                '" value="0"></label><label>Expert<select id="neuron-expert" aria-label="Neuron expert">' +
                Array.from(
                    { length: neurons.planes },
                    (_, i) => '<option value="' + i + '">' + i + "</option>",
                ).join("") +
                '</select></label></div><select id="neuron-family" aria-label="Channel family"><option value="ffn">FFN hidden neurons</option><option value="layer">Layer output features</option></select><div id="neuron-grid" class="neuron-grid"></div><div class="tensor-actions"><button id="neuron-prev" class="secondary compact">Previous channels</button><button id="neuron-next" class="secondary compact">Next channels</button><button id="neuron-read" class="secondary compact">Read neuron weights</button></div><pre id="neuron-activation" class="detail-pre"></pre><pre id="neuron-weights" class="detail-pre">Select any FFN channel to inspect its incoming and outgoing weights.</pre><p class="accuracy-note">A channel is a tensor feature, not a separate GGML operation. Full capture records FFN and layer output channels for the first token of each step, within explicit capture limits.</p></section>',
        );
        const expert = item.members.find((n) => n.kind === "expert")?.expert;
        if (expert != null) $("neuron-expert").value = expert;
        const grid = () => {
            const selected = Math.max(
                0,
                Math.min(neurons.width - 1, Number($("neuron-index").value)),
            );
            $("neuron-index").value = selected;
            const start = Math.floor(selected / 32) * 32;
            $("neuron-grid").innerHTML = Array.from(
                { length: Math.min(32, neurons.width - start) },
                (_, i) =>
                    '<button data-channel="' +
                    (start + i) +
                    '" class="' +
                    (selected === start + i ? "selected" : "") +
                    '">' +
                    (start + i) +
                    "</button>",
            ).join("");
            $("neuron-grid")
                .querySelectorAll("[data-channel]")
                .forEach(
                    (b) =>
                        (b.onclick = () => {
                            $("neuron-index").value = b.dataset.channel;
                            grid();
                        }),
                );
            updateNeuronActivity();
        };
        $("neuron-index").onchange = grid;
        $("neuron-index").oninput = grid;
        $("neuron-family").onchange = () => {
            const width =
                item.component === "multimodal_encoder"
                    ? state.model.details.projector?.metadata?.[
                          "clip.vision.embedding_length"
                      ]
                    : state.model.details.embedding_size;
            neurons =
                $("neuron-family").value === "ffn"
                    ? ffnNeurons
                    : {
                          width: width || 1,
                          planes: 1,
                          incoming: [],
                          outgoing: [],
                      };
            $("neuron-index").max = neurons.width - 1;
            $("neuron-index").value = Math.min(
                Number($("neuron-index").value),
                neurons.width - 1,
            );
            $("neuron-expert").innerHTML = Array.from(
                { length: neurons.planes },
                (_, i) => '<option value="' + i + '">' + i + "</option>",
            ).join("");
            $("neuron-capacity").textContent =
                num(neurons.width) +
                " channels · " +
                num(neurons.planes) +
                " plane(s).";
            grid();
        };
        $("neuron-expert").onchange = updateNeuronActivity;
        $("neuron-prev").onclick = () => {
            $("neuron-index").value = Math.max(
                0,
                Number($("neuron-index").value) - 32,
            );
            grid();
        };
        $("neuron-next").onclick = () => {
            $("neuron-index").value = Math.min(
                neurons.width - 1,
                Number($("neuron-index").value) + 32,
            );
            grid();
        };
        $("neuron-read").onclick = async () => {
            const target = $("neuron-weights"),
                index = Number($("neuron-index").value),
                plane = Number($("neuron-expert").value);
            if ($("neuron-family").value !== "ffn") {
                target.textContent =
                    "Layer output features mix residual, attention and FFN paths. Use the parameter browser for associated tensors; the FFN family provides direct incoming and outgoing neuron weights.";
                return;
            }
            try {
                const windows = await Promise.all([
                    ...neurons.incoming.map((w) =>
                        readWeights(w, index, plane, 0, "row", 16),
                    ),
                    ...neurons.outgoing.map((w) =>
                        readWeights(w, index, plane, 0, "column", 16),
                    ),
                ]);
                if (target.isConnected)
                    target.textContent = pretty({
                        channel: index,
                        expert: plane,
                        weights: windows,
                    });
            } catch (error) {
                if (target.isConnected) target.textContent = error.message;
            }
        };
        grid();
    }
}
async function readWeights(
    weight,
    index = 0,
    plane = 0,
    offset = 0,
    axis = "row",
    limit = 32,
) {
    const query = new URLSearchParams({
        tensor: weight.name,
        component: weight.file,
        index,
        plane,
        offset,
        axis,
        limit,
    });
    if (state.runId && !state.runId.startsWith("inspection:"))
        query.set("run_id", state.runId);
    return api("/api/models/" + state.model.id + "/weights?" + query);
}
function updateInspector() {
    const item = state.selected;
    if (item?.members.length === 1) {
        const { raw, activation: a } = selectedObservation(item);
        const values = {
            Shape: raw?.shape.join(" × "),
            Elements: raw && num(raw.elements),
            Bytes: raw && num(raw.bytes),
            Observation: a
                ? "Observed this step"
                : "Preloaded; not observed this step",
            Wall: a ? a.wall_us + " µs" : "Not measured",
        };
        for (const [key, value] of Object.entries(values)) {
            const element = $("inspect-node").querySelector(
                '[data-property="' + key + '"]',
            );
            if (element && value != null) element.textContent = value;
        }
        if ($("node-statistics"))
            $("node-statistics").textContent = a?.stats
                ? pretty(a.stats)
                : "No values captured for this node at the selected point.";
    }
    updateNeuronActivity();
}
function updateNeuronActivity() {
    if (!$("neuron-activation") || !state.selected) return;
    const item = state.selected,
        expert = Number($("neuron-expert").value),
        channel = Number($("neuron-index").value);
    const entry = Object.entries(state.activations).find(
        ([id, a]) =>
            a.neurons?.meaning ===
                ($("neuron-family")?.value === "layer"
                    ? "layer output feature channels"
                    : "FFN hidden activation after nonlinearity") &&
            state.nodes.get(id)?.layer === item.layer &&
            state.nodes.get(id)?.component === item.component,
    );
    if (!entry) {
        $("neuron-grid")
            .querySelectorAll("[data-channel]")
            .forEach((button) => {
                button.classList.remove("observed");
                button.title = "No channel value captured at this position";
            });
        $("neuron-activation").textContent =
            "Channel values are unavailable here: the operation may not be reached yet, the trace may be older, or its capture level/limit may exclude channels. New full captures include bounded FFN channel vectors.";
        return;
    }
    const neurons = entry[1].neurons;
    const slot = neurons.expert_ids.length
        ? neurons.expert_ids.indexOf(expert)
        : expert === 0
          ? 0
          : -1;
    const value =
        slot >= 0 && channel < neurons.channels
            ? neurons.values[slot * neurons.channels + channel]
            : undefined;
    $("neuron-activation").textContent =
        slot < 0
            ? "Expert " +
              expert +
              " was not selected for captured token " +
              neurons.token_index +
              "."
            : pretty({
                  channel,
                  expert,
                  token_index: neurons.token_index,
                  value: value ?? null,
                  scope: neurons.scope,
                  step: state.step,
              });
    $("neuron-grid")
        .querySelectorAll("[data-channel]")
        .forEach((b) => {
            const v =
                slot >= 0
                    ? neurons.values[
                          slot * neurons.channels + Number(b.dataset.channel)
                      ]
                    : undefined;
            b.classList.toggle("observed", slot >= 0 && v !== undefined);
            b.title =
                v === undefined
                    ? "Not captured / expert not selected"
                    : "Activation " + v;
        });
}
function renderExperts(counts) {
    let total = 0;
    for (const [k, v] of Object.entries(state.model?.details?.metadata || {}))
        if (k.endsWith(".expert_count") && typeof v === "number")
            total = Math.max(total, v);
    const groups = new Map();
    for (const n of state.topology?.nodes.values() || []) {
        if (n.kind !== "expert") continue;
        const label = n.component + " / L" + String(n.layer).padStart(2, "0");
        if (!groups.has(label)) groups.set(label, new Map());
        groups.get(label).set(n.expert, 0);
    }
    for (const [k, count] of counts) {
        const parts = k.split(":"),
            expert = Number(parts.pop()),
            layer = parts.pop(),
            label = parts.join(":") + " / L" + String(layer).padStart(2, "0");
        if (!groups.has(label)) groups.set(label, new Map());
        groups.get(label).set(expert, count);
    }
    if (state.expertGridKey !== state.topologyKey) {
        $("experts").innerHTML = groups.size
            ? [...groups]
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([label, experts]) => {
                      const n = Math.max(
                              total,
                              ...[...experts.keys()].map((n) => n + 1),
                          ),
                          max = Math.max(1, ...experts.values());
                      return (
                          '<div class="expert-row"><div class="expert-row-label">' +
                          esc(label) +
                          '</div><div class="expert-grid">' +
                          Array.from(
                              { length: Math.min(n, 1024) },
                              (_, i) =>
                                  '<span class="expert-cell ' +
                                  (experts.has(i) ? "used" : "") +
                                  '" data-expert="' +
                                  esc(
                                      label.split(" / L")[0] +
                                          ":" +
                                          Number(label.split(" / L")[1]) +
                                          ":" +
                                          i,
                                  ) +
                                  '" title="Expert ' +
                                  i +
                                  ": " +
                                  (experts.get(i) || 0) +
                                  ' selections" style="--strength:' +
                                  (20 + (30 * (experts.get(i) || 0)) / max) +
                                  '%">' +
                                  i +
                                  "</span>",
                          ).join("") +
                          "</div></div>"
                      );
                  })
                  .join("")
            : '<p class="empty-small">' +
              (total
                  ? "This model has " +
                    total +
                    " experts per layer. Run a prompt to observe routing."
                  : "No routed expert dispatch observed. Dense models may have no routed experts.") +
              "</p>";
        state.expertGridKey = state.topologyKey;
    }
    $("experts")
        .querySelectorAll("[data-expert]")
        .forEach((cell) => {
            const count = counts.get(cell.dataset.expert) || 0;
            if (cell.dataset.count === String(count)) return;
            cell.dataset.count = count;
            cell.classList.toggle("used", count > 0);
            cell.title =
                "Expert " +
                cell.textContent +
                ": " +
                count +
                " actual selections";
            cell.style.setProperty("--strength", count ? "50%" : "20%");
        });
}
function renderModel() {
    const m = state.model,
        d = m?.details;
    $("model-arch").textContent = d?.metadata?.["general.architecture"] || "";
    if (!d) {
        $("model-mini").innerHTML =
            '<span class="muted">' +
            esc(m?.name || "Select a local GGUF") +
            "</span>";
        $("model-details").innerHTML =
            '<p class="empty-small">Waiting for runtime metadata.</p>';
        return;
    }
    const renderKey =
        m.id +
        ":" +
        JSON.stringify([
            m.config,
            d.context_size,
            d.multimodal,
            d.tensors.length,
            d.projector?.tensors.length,
        ]);
    if (state.modelRenderKey === renderKey) return;
    state.modelRenderKey = renderKey;
    $("model-mini").innerHTML =
        esc(d.metadata?.["general.name"] || m.name) +
        '<div class="model-tags"><span class="tag">' +
        num(d.parameters) +
        ' params</span><span class="tag">' +
        d.layers +
        ' layers</span><span class="tag">' +
        esc(d.metadata?.["general.architecture"]) +
        "</span>" +
        (d.hybrid ? '<span class="tag">HYBRID</span>' : "") +
        "</div>";
    const metadataEntries = [
        ...Object.entries(d.metadata).map(([k, v]) => [k, v, "model"]),
        ...Object.entries(d.projector?.metadata || {}).map(([k, v]) => [
            k,
            v,
            "projector",
        ]),
    ];
    const inventory = [
        ...d.tensors.map((t) => ({ ...t, component: "model" })),
        ...(d.projector?.tensors || []).map((t) => ({
            ...t,
            component: "projector",
        })),
    ];
    $("model-details").innerHTML =
        Object.entries({
            Parameters: num(d.parameters),
            Layers: d.layers,
            Embedding: d.embedding_size,
            Vocabulary: num(d.vocab_size),
            "Context / train":
                num(d.context_size) + " / " + num(d.context_train),
            Encoder: d.has_encoder ? "Yes" : "No",
            Decoder: d.has_decoder ? "Yes" : "No",
            Recurrent: d.recurrent ? "Yes" : "No",
            Hybrid: d.hybrid ? "Yes" : "No",
            Multimodal: d.multimodal ? "Projector loaded" : "No projector",
            Runtime: d.runtime_version,
        })
            .map(
                ([k, v]) =>
                    '<div class="property"><span>' +
                    esc(k) +
                    "</span><span>" +
                    esc(v) +
                    "</span></div>",
            )
            .join("") +
        '<div class="detail-label">GGUF METADATA · ' +
        metadataEntries.length +
        " KEYS</div>" +
        metadataEntries
            .map(
                ([k, v, component]) =>
                    '<details class="metadata-key"><summary>' +
                    esc((component === "projector" ? "projector / " : "") + k) +
                    "</summary><pre>" +
                    esc(pretty(v)) +
                    "</pre>" +
                    (v?.paged
                        ? '<button class="secondary compact" data-meta="' +
                          esc(k) +
                          '" data-component="' +
                          component +
                          '" data-offset="0">Read values</button>'
                        : "") +
                    "</details>",
            )
            .join("") +
        '<div class="detail-label">WEIGHT INVENTORY · ' +
        inventory.length +
        ' TENSORS</div><input id="weight-search" class="weight-search" placeholder="Filter weight names…" aria-label="Filter weight inventory"><div id="weights"></div>';
    const weights = () => {
        const ts = inventory.filter((t) =>
            t.name
                .toLowerCase()
                .includes($("weight-search").value.toLowerCase()),
        );
        $("weights").innerHTML =
            ts
                .slice(0, 150)
                .map(
                    (t) =>
                        '<div class="weight-row">' +
                        esc(t.name) +
                        "<small>" +
                        esc(t.component + " · " + t.dtype) +
                        " · " +
                        t.shape.join(" × ") +
                        " · " +
                        num(t.bytes) +
                        " B</small></div>",
                )
                .join("") +
            (ts.length > 150
                ? '<p class="empty-small">150 / ' +
                  ts.length +
                  " shown. Narrow the filter to inspect any tensor.</p>"
                : "");
    };
    weights();
    $("weight-search").oninput = weights;
    $("model-details")
        .querySelectorAll("[data-meta]")
        .forEach(
            (b) =>
                (b.onclick = async () => {
                    try {
                        const d = await api(
                            "/api/models/" +
                                m.id +
                                "/metadata?key=" +
                                encodeURIComponent(b.dataset.meta) +
                                "&offset=" +
                                b.dataset.offset +
                                "&component=" +
                                b.dataset.component +
                                (state.runId &&
                                !state.runId.startsWith("inspection:")
                                    ? "&run_id=" +
                                      encodeURIComponent(state.runId)
                                    : ""),
                        );
                        b.previousElementSibling.textContent = pretty(d);
                        b.dataset.offset = d.offset + 128;
                        b.textContent =
                            d.offset + 128 < d.total
                                ? "Next page"
                                : "Read complete";
                        b.disabled = d.offset + 128 >= d.total;
                    } catch (e) {
                        notice(e.message, true);
                    }
                }),
        );
}
async function refreshHistory() {
    state.runs = await api("/api/runs?limit=100");
    $("history").innerHTML = state.runs.length
        ? state.runs
              .map(
                  (r) =>
                      '<button class="run-item ' +
                      (r.id === state.runId ? "selected" : "") +
                      '" data-run="' +
                      r.id +
                      '"><span class="run-prompt">' +
                      esc((r.prompt || "Token input").slice(0, 200)) +
                      '</span><span class="run-sub"><i class="status-dot ' +
                      esc(r.status) +
                      '"></i>' +
                      esc(r.status) +
                      " · " +
                      new Date(r.started * 1000).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                      }) +
                      "</span></button>",
              )
              .join("")
        : '<p class="empty-small">Your executions will be saved here.</p>';
    $("history")
        .querySelectorAll("[data-run]")
        .forEach((b) => (b.onclick = () => viewRun(b.dataset.run)));
    for (const id of ["compare-left", "compare-right"]) {
        const v = $(id).value;
        $(id).innerHTML = state.runs
            .map(
                (r) =>
                    '<option value="' +
                    r.id +
                    '">' +
                    esc(r.prompt.slice(0, 65)) +
                    "</option>",
            )
            .join("");
        if (state.runs.some((r) => r.id === v)) $(id).value = v;
    }
    if (state.runs.length > 1 && !$("compare-right").dataset.init) {
        $("compare-right").selectedIndex = 1;
        $("compare-right").dataset.init = "1";
    }
}
function exports(id) {
    for (const format of ["json", "jsonl"]) {
        $("export-" + format).href =
            "/api/runs/" + id + "/export?format=" + format;
        $("export-" + format).hidden = false;
    }
}
async function viewRun(id) {
    try {
        stopReplay();
        state.live = false;
        state.runId = id;
        clearActivity();
        const r = await api("/api/runs/" + id);
        state.model =
            r.model_snapshot ||
            state.models.find((m) => m.id === r.model_id) ||
            (await api("/api/models/" + r.model_id));
        renderModel();
        $("response").textContent = r.output || r.error || "No generated text.";
        $("response").classList.remove("empty-response");
        $("response-meta").textContent = r.status;
        exports(id);
        await refreshSteps();
        await loadGraph();
        state.logs = await api(
            "/api/events?run_id=" +
                encodeURIComponent(id) +
                "&limit=300&tail=true",
        );
        state.dirty = true;
        await refreshHistory();
        notice(
            "Saved prompt · " + r.status + " · exact recorded runtime graph",
        );
    } catch (e) {
        notice(e.message, true);
    }
}
async function refreshSteps() {
    if (!state.runId) return;
    const id = state.runId,
        steps = await api("/api/runs/" + encodeURIComponent(id) + "/steps");
    if (state.runId !== id) return;
    state.steps = steps;
    $("timeline").max = Math.max(0, ...steps.map((s) => s.step));
    $("timeline").value = Math.max(0, state.step);
    $("step-list").innerHTML =
        steps
            .map(
                (s) =>
                    '<button class="step-chip ' +
                    (s.step === state.step ? "selected" : "") +
                    '" data-step="' +
                    s.step +
                    '"><strong>' +
                    s.step +
                    " / " +
                    esc(s.phase) +
                    "</strong>" +
                    num(s.node_count) +
                    " nodes<br>" +
                    (s.duration_us / 1000).toFixed(1) +
                    " ms</button>",
            )
            .join("") || '<p class="empty-small">No completed steps yet.</p>';
    $("step-list")
        .querySelectorAll("[data-step]")
        .forEach(
            (b) =>
                (b.onclick = () => {
                    stopReplay();
                    state.live = false;
                    state.replayStepIndex = Math.max(
                        0,
                        state.steps.findIndex(
                            (s) => s.step === Number(b.dataset.step),
                        ),
                    );
                    loadGraph(Number(b.dataset.step));
                }),
        );
}
function renderLog() {
    const q = $("log-filter").value.toLowerCase();
    $("raw-log").textContent = state.logs
        .filter((e) => !state.runId || e.run_id === state.runId)
        .filter((e) => !q || pretty(e).toLowerCase().includes(q))
        .slice(-70)
        .map(
            (e) =>
                "#" +
                e.cursor +
                " " +
                e.kind +
                " " +
                (e.data?.node?.name ||
                    e.data?.node_id ||
                    e.data?.text ||
                    e.data?.message ||
                    JSON.stringify(e.data)?.slice(0, 180) ||
                    ""),
        )
        .join("\n");
    $("event-count").textContent =
        num(state.engine.received_events) + " events";
}
function updateEngine(e) {
    state.engine = e;
    $("restart-engine").hidden = e.status !== "error";
    $("run").disabled =
        e.status !== "ready" ||
        !state.graphReady ||
        state.model?.id !== e.model_id;
    $("load-model").disabled =
        !e.connected || ["loading", "running"].includes(e.status);
    $("cancel").hidden = !e.active_run;
    $("run").hidden = Boolean(e.active_run);
    $("connection").className = "connection " + (e.connected ? "online" : "");
    $("connection").lastChild.textContent = e.connected
        ? "gRPC · " + e.status
        : "Engine · " + e.status;
    if (e.error) notice(e.error, true);
}
$("restart-engine").onclick = async () => {
    try {
        await api("/api/engine/restart", {});
        notice("Restarting the local observer. Select a model once connected.");
        await poll();
    } catch (e) {
        notice(e.message, true);
    }
};
async function poll() {
    try {
        updateEngine((await api("/api/state")).engine);
        if (state.engine.status === "idle" && state.model && !state.runId)
            await loadSelected();
    } catch (e) {
        $("connection").className = "connection";
        $("connection").lastChild.textContent = "Service disconnected";
    }
}
function observation(e) {
    state.cursor = Math.max(state.cursor, e.cursor);
    if (state.live || e.run_id === state.runId) state.logs.push(e);
    if (state.logs.length > 300) state.logs.splice(0, 100);
    const d = e.data;
    if (e.kind === "load_started" && state.live) {
        state.model = state.models.find((m) => m.id === e.model_id);
        $("model-select").value = e.model_id;
        state.runId = e.run_id;
        resetModelGraph();
        for (const format of ["json", "jsonl"])
            $("export-" + format).hidden = true;
        $("response-meta").textContent = "Model inspection";
        $("response").textContent = "Run a prompt to record its execution.";
        $("response").classList.add("empty-response");
        renderModel();
    }
    if (e.kind === "run_started" && state.live && state.runId !== e.run_id) {
        state.runId = e.run_id;
        clearActivity();
        $("response").textContent = "";
        $("response").classList.remove("empty-response");
        $("response-meta").textContent = "Preparing prompt…";
        exports(e.run_id);
    }
    if (e.kind === "model_loaded") {
        const m = state.models.find((m) => m.id === e.model_id);
        if (m) m.details = d;
        if (state.model?.id === e.model_id) {
            state.model = m;
            renderModel();
            preloadModel().catch((error) => notice(error.message, true));
        }
    }
    if (e.kind === "model_ready") {
        poll();
        if (state.live && state.model?.id === e.model_id) {
            state.runId = e.run_id;
            preloadModel(true)
                .then(() => loadGraph())
                .catch((error) => notice(error.message, true));
            notice(
                "Model capacity preloaded · run a prompt to highlight execution.",
            );
        }
    }
    if (["error", "trace_error"].includes(e.kind)) {
        notice(d.message, true);
        poll();
        refreshHistory();
    }
    if (e.kind === "run_finished") {
        poll();
        refreshHistory();
        if (state.runId === e.run_id) {
            $("response-meta").textContent =
                d.status + " · " + d.generated_tokens + " tokens";
            $("response").textContent = d.output || "No generated text.";
            refreshSteps();
            if (state.live) loadGraph();
            notice(
                "Trace " +
                    d.status +
                    " · saved with " +
                    d.steps +
                    " runtime steps.",
            );
        }
    }
    if (e.run_id !== state.runId || !state.live) {
        state.dirty = true;
        return;
    }
    if (e.run_id.startsWith("inspection:")) {
        state.dirty = true;
        return;
    }
    if (e.kind === "step_begin") {
        clearActivity();
        state.step = d.step;
        state.summary = d;
        $("step-label").textContent = d.step;
    }
    if (e.kind === "node") {
        state.nodes.set(d.node.id, d.node);
        state.topology?.bindNode(d.node);
    }
    if (e.kind === "activation") {
        state.activations[d.node_id] = d;
        state.lastNode = d.node_id;
    }
    if (e.kind === "routing") state.routing.push(d);
    if (e.kind === "step_end") {
        state.summary = { ...state.summary, ...d };
        refreshSteps();
    }
    if (e.kind === "token") {
        $("response").textContent += d.text;
        $("response-meta").textContent =
            "Generating · " + (d.index + 1) + " tokens";
    }
    state.dirty = true;
}
async function loadSelected() {
    const id = $("model-select").value;
    if (!id) return;
    try {
        stopReplay();
        state.model = state.models.find((m) => m.id === id);
        state.runId = "inspection:" + id;
        state.live = true;
        clearActivity();
        renderModel();
        notice("Loading model and recording runtime inspection…");
        await api("/api/models/load", {
            model_id: id,
            context_size: Number($("context-size").value),
            gpu_layers: Number($("gpu-layers").value),
            capture: $("capture").value,
            mmproj: $("mmproj").value.trim(),
        });
        await poll();
    } catch (e) {
        notice(e.message, true);
    }
}
$("load-model").onclick = loadSelected;
$("model-select").onchange = loadSelected;
$("prompt-form").onsubmit = async (e) => {
    e.preventDefault();
    if ($("run").disabled) {
        notice(
            state.model?.id !== state.engine.model_id
                ? "Load the model shown in this trace before starting a new prompt."
                : "Waiting for the model capacity graph to preload.",
        );
        return;
    }
    try {
        stopReplay();
        state.live = true;
        clearActivity();
        $("response").textContent = "";
        $("response").classList.remove("empty-response");
        $("response-meta").textContent = "Preparing prompt…";
        const run = await api("/api/runs", {
            prompt: $("prompt").value,
            max_tokens: Number($("max-tokens").value),
            capture: $("capture").value,
            media_id: state.mediaId,
        });
        state.runId = run.id;
        state.step = -1;
        state.steps = [];
        state.model = state.models.find((m) => m.id === run.model_id);
        if (run.model_snapshot) state.model = run.model_snapshot;
        renderModel();
        await preloadModel();
        exports(run.id);
        notice("Recording live runtime observations over gRPC…");
        await poll();
        await refreshHistory();
    } catch (e) {
        notice(e.message, true);
    }
};
$("prompt").onkeydown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter")
        $("prompt-form").requestSubmit();
};
$("cancel").onclick = async () => {
    try {
        await api("/api/runs/" + state.engine.active_run + "/cancel", {});
        notice("Cancelling at the next execution boundary…");
    } catch (e) {
        notice(e.message, true);
    }
};
$("media").onchange = async () => {
    const file = $("media").files[0];
    if (!file) {
        state.mediaId = null;
        return;
    }
    try {
        const r = await fetch("/api/media", {
                method: "POST",
                headers: {
                    "X-Visualizer-Client": "1",
                    "Content-Type": "application/octet-stream",
                },
                body: file,
            }),
            d = await r.json();
        if (!r.ok) throw Error(d.detail);
        state.mediaId = d.media_id;
        notice(
            "Attached " + file.name + ". A compatible mmproj must be loaded.",
        );
    } catch (e) {
        notice(e.message, true);
    }
};
for (const [id, value] of [
    ["view-layers", true],
    ["view-ops", false],
])
    $(id).onclick = () => {
        state.grouped = value;
        $("view-layers").classList.toggle("selected", value);
        $("view-ops").classList.toggle("selected", !value);
        state.structureDirty = true;
        state.fit = true;
        state.dirty = true;
    };
$("show-weights").onchange = (e) => {
    state.weights = e.target.checked;
    state.structureDirty = true;
    state.dirty = true;
};
$("category").onchange = (e) => {
    state.filter = e.target.value;
    state.structureDirty = true;
    state.fit = true;
    state.dirty = true;
};
$("node-search").onkeydown = (e) => {
    if (e.key === "Enter") {
        const query = e.target.value.toLowerCase();
        const found = [...(state.topology?.nodes.values() || [])].find((n) =>
            (
                n.name +
                " " +
                n.op +
                " " +
                n.component +
                " layer:" +
                n.layer +
                " " +
                n.dtype
            )
                .toLowerCase()
                .includes(query),
        );
        if (found) selectRuntime(found.id);
    }
    if (e.key === "Escape") {
        e.target.value = "";
        state.search = "";
        state.structureDirty = true;
        state.dirty = true;
    }
};
$("node-search").oninput = (e) => {
    state.search = e.target.value;
    state.structureDirty = true;
    state.fit = true;
    state.dirty = true;
};
$("fit").onclick = () => graph?.fit();
$("toggle-3d").onclick = () => {
    const v = $("toggle-3d").getAttribute("aria-pressed") !== "true";
    $("toggle-3d").setAttribute("aria-pressed", v);
    graph?.toggle3d(v);
};
for (const name of ["prompt", "timeline", "compare"])
    $("tab-" + name).onclick = () => {
        for (const n of ["prompt", "timeline", "compare"]) {
            $("tab-" + n).classList.toggle("active", n === name);
            $("panel-" + n).hidden = n !== name;
        }
    };
function stopReplay(reset = true) {
    if (state.playing) {
        clearTimeout(state.playing);
        state.playing = null;
    }
    state.replayGeneration++;
    if (reset) {
        state.replayPrepared = false;
        state.replayRecords = [];
        state.replayPosition = 0;
        state.replayStepIndex = 0;
        $("replay-position").textContent =
            "Replay highlights recorded operations in order. Model geometry and camera remain fixed.";
    }
    $("replay").textContent = "▶ Slow replay";
}
$("timeline").oninput = (e) => {
    stopReplay();
    state.live = false;
    state.replayStepIndex = Math.max(
        0,
        state.steps.findIndex((s) => s.step === Number(e.target.value)),
    );
    loadGraph(Number(e.target.value));
};
async function prepareReplayStep(generation) {
    if (state.replayStepIndex >= state.steps.length) return false;
    const step = state.steps[state.replayStepIndex].step,
        id = state.runId;
    const [snapshot, records] = await Promise.all([
        api("/api/runs/" + id + "/graph?step=" + step),
        api("/api/runs/" + id + "/playback?step=" + step),
    ]);
    if (generation !== state.replayGeneration || id !== state.runId)
        return false;
    setGraph(snapshot);
    state.activations = {};
    state.routing = [];
    state.lastNode = null;
    state.replayRecords = records;
    state.replayPosition = 0;
    state.replayPrepared = true;
    state.dirty = true;
    return true;
}
async function nextReplayOperation(generation) {
    if (!state.replayPrepared && !(await prepareReplayStep(generation)))
        return false;
    if (state.replayPosition >= state.replayRecords.length) {
        state.replayStepIndex++;
        state.replayPrepared = false;
        if (!(await prepareReplayStep(generation))) return false;
    }
    if (generation !== state.replayGeneration) return false;
    const event = state.replayRecords[state.replayPosition++];
    if (!event) return false;
    const d = event.data;
    if (event.kind === "activation") {
        state.activations[d.node_id] = d;
        state.lastNode = d.node_id;
    }
    if (event.kind === "routing") state.routing.push(d);
    $("replay-position").textContent =
        "Step " +
        state.step +
        " · observation " +
        state.replayPosition +
        " / " +
        state.replayRecords.length +
        " · " +
        event.kind +
        " · " +
        (state.nodes.get(d.node_id)?.name || d.text || "") +
        " · geometry stays fixed";
    state.dirty = true;
    return true;
}
$("replay").onclick = async () => {
    if (state.playing) {
        stopReplay(false);
        return;
    }
    if (!state.steps.length) {
        notice("Select a saved prompt to replay.");
        return;
    }
    state.live = false;
    const generation = ++state.replayGeneration;
    $("replay").textContent = "Ⅱ Pause";
    const tick = async () => {
        try {
            if (!(await nextReplayOperation(generation))) {
                if (generation === state.replayGeneration) stopReplay();
                return;
            }
            if (generation === state.replayGeneration)
                state.playing = setTimeout(
                    tick,
                    Number($("replay-speed").value),
                );
        } catch (e) {
            notice(e.message, true);
            stopReplay();
        }
    };
    state.playing = setTimeout(tick, 0);
};
$("replay-next").onclick = async () => {
    stopReplay(false);
    state.live = false;
    try {
        if (!(await nextReplayOperation(state.replayGeneration))) stopReplay();
    } catch (error) {
        notice(error.message, true);
    }
};
$("go-live").onclick = async () => {
    stopReplay();
    state.live = true;
    if (state.engine.active_run && state.runId !== state.engine.active_run) {
        state.runId = state.engine.active_run;
        clearActivity();
    }
    await loadGraph();
    notice(
        state.engine.active_run
            ? "Following live execution."
            : "Showing the latest recorded step; no prompt is running.",
    );
};
$("compare").onclick = async () => {
    try {
        const d = await api(
                "/api/compare?left=" +
                    $("compare-left").value +
                    "&right=" +
                    $("compare-right").value,
            ),
            changed = Object.entries(d.routing_delta).filter(([, v]) => v);
        $("comparison").textContent =
            "Execution steps: " +
            d.steps.join(" → ") +
            "\nDifferent expert selection counts: " +
            changed.length +
            "\n\n" +
            (changed
                .slice(0, 100)
                .map(([k, v]) => k + "   " + (v > 0 ? "+" : "") + v)
                .join("\n") ||
                "No routing count differences. Dense-model values may still differ; inspect the saved tensor samples.");
    } catch (e) {
        notice(e.message, true);
    }
};
$("refresh-history").onclick = refreshHistory;
$("log-filter").oninput = () => {
    state.dirty = true;
};
$("help").onclick = () => $("help-dialog").showModal();
$("close-help").onclick = () => $("help-dialog").close();
async function init() {
    try {
        const initial = await api("/api/state");
        state.cursor = initial.cursor;
        updateEngine(initial.engine);
        state.models = await api("/api/models");
        $("model-select").innerHTML = state.models.length
            ? state.models
                  .map(
                      (m) =>
                          '<option value="' +
                          m.id +
                          '">' +
                          esc(m.name) +
                          "</option>",
                  )
                  .join("")
            : '<option value="">No GGUF models found</option>';
        if (initial.engine.model_id)
            $("model-select").value = initial.engine.model_id;
        state.model = state.models.find(
            (m) => m.id === $("model-select").value,
        );
        renderModel();
        await refreshHistory();
        const source = new EventSource("/api/stream?after=" + state.cursor);
        source.onmessage = (e) => {
            try {
                observation(JSON.parse(e.data));
            } catch (error) {
                notice("Could not apply observation: " + error.message, true);
            }
        };
        source.onerror = () =>
            notice(
                "Event stream reconnecting; recorded events will resume from the last cursor.",
            );
        if (initial.engine.status === "ready" && state.model) {
            state.runId = "inspection:" + state.model.id;
            await loadGraph();
            notice("Model capacity ready · waiting for prompt activations.");
        } else if (initial.engine.status === "idle" && state.model)
            await loadSelected();
        else
            notice(
                state.models.length
                    ? "Waiting for the execution engine…"
                    : "Add GGUF files to a configured model directory.",
            );
        setInterval(poll, 1000);
    } catch (e) {
        notice(e.message, true);
    }
}
init();
