import { category } from "./graph.js";

const layerOf = (name) =>
    Number(name.match(/(?:blk|layers|block)\.(\d+)\./)?.[1] ?? -1);
const product = (shape) => shape.reduce((a, b) => a * b, 1);
const moduleId = (component, layer, stage) =>
    `module:${component}:${layer}:${stage}`;
const semanticId = (n) => `op:${n.component}:${n.layer}:${n.op}:${n.name}`;

export const explanations = {
    embedding:
        "Maps token, position or media inputs into feature vectors using learned parameters.",
    normalization:
        "Normalizes feature magnitudes before or after another transformation.",
    attention:
        "Combines query, key and value features so each token can use relevant context.",
    routing:
        "Computes routing scores and selects expert indices for the actual token inputs.",
    expert: "Applies the selected expert's learned feed-forward projections and nonlinear activation.",
    feedforward:
        "Transforms each token's features through learned projections and hidden channels.",
    recurrence:
        "Updates recurrent or state-space features and their persistent context state.",
    output: "Projects final features into model outputs, such as vocabulary logits.",
    other: "A runtime tensor, data transformation or model component shown with its actual metadata.",
};

export class ModelTopology {
    constructor(model, inspection = { nodes: [] }) {
        this.model = model;
        this.nodes = new Map();
        this.bindings = new Map();
        this.rawToStable = new Map();
        this.neurons = new Map();
        this.inventory = [];
        const d = model.details || {};
        const backbone =
            d.has_encoder && !d.has_decoder ? "encoder" : "decoder";
        for (const [file, tensors] of [
            ["model", d.tensors || []],
            ["projector", d.projector?.tensors || []],
        ]) {
            for (const t of tensors) {
                const component =
                    file === "projector"
                        ? "multimodal_encoder"
                        : /(?:^|\.)enc(?:oder)?\./.test(t.name)
                          ? "encoder"
                          : backbone;
                const n = {
                    ...t,
                    id: `weight:${component}:${t.name}`,
                    component,
                    file,
                    layer: layerOf(t.name),
                    op: "NONE",
                    kind: "weight",
                    sources: [],
                    strides: [],
                    flags: 0,
                    elements: product(t.shape),
                    device: "GGUF",
                    buffer: "model file",
                };
                this.inventory.push(n);
                this.nodes.set(n.id, n);
            }
        }
        const components = new Set(this.inventory.map((n) => n.component));
        for (const component of components) {
            this.addModule(component, -1, "other", []);
        }
        for (const n of this.inventory) {
            this.addModule(n.component, n.layer, category(n), [n.id]);
            this.addModule(n.component, n.layer, "other", []);
        }
        for (const n of inspection.nodes) {
            this.addModule(n.component, n.layer, category(n), []);
            if (n.kind === "weight") continue;
            const id = semanticId(n);
            if (!this.nodes.has(id))
                this.nodes.set(id, {
                    ...n,
                    id,
                    sources: [],
                    example_shape: n.shape,
                });
        }
        const seedMap = new Map(
            inspection.nodes.map((n) => [n.id, this.match(n)]),
        );
        for (const n of inspection.nodes) {
            const target = this.nodes.get(seedMap.get(n.id));
            if (!target || target.kind === "module") continue;
            for (const source of n.sources || []) {
                const id = seedMap.get(source);
                if (id && id !== target.id && !target.sources.includes(id))
                    target.sources.push(id);
            }
        }
        for (const up of this.inventory.filter((n) =>
            /ffn_up(?:_exps)?\.weight$/.test(n.name),
        )) {
            const key = `${up.component}:${up.layer}`;
            const width = Number(up.shape[1] || 1),
                planes = product(up.shape.slice(2));
            const incoming = this.inventory.filter(
                (n) =>
                    n.component === up.component &&
                    n.layer === up.layer &&
                    /ffn_(?:up|gate)(?:_exps)?\.weight$/.test(n.name),
            );
            const outgoing = this.inventory.filter(
                (n) =>
                    n.component === up.component &&
                    n.layer === up.layer &&
                    /ffn_down(?:_exps)?\.weight$/.test(n.name),
            );
            this.neurons.set(key, { width, planes, incoming, outgoing });
            if (planes > 1) {
                const router = moduleId(up.component, up.layer, "routing");
                for (let expert = 0; expert < planes; expert++) {
                    const id = `expert:${up.component}:${up.layer}:${expert}`;
                    this.nodes.set(id, {
                        id,
                        name: `Expert ${expert}`,
                        op: "EXPERT",
                        kind: "expert",
                        category: "expert",
                        component: up.component,
                        layer: up.layer,
                        expert,
                        shape: [width],
                        dtype: up.dtype,
                        bytes: 0,
                        elements: width,
                        sources: [...incoming, ...outgoing]
                            .map((n) => n.id)
                            .concat(this.nodes.has(router) ? [router] : []),
                        strides: [],
                        device: "model capacity",
                        buffer: "GGUF expert slices",
                    });
                }
            }
        }
        this.capacity = {
            nodes: this.nodes.size,
            weights: this.inventory.length,
            experts: [...this.nodes.values()].filter((n) => n.kind === "expert")
                .length,
            channels: [...this.neurons.values()].reduce(
                (sum, n) => sum + n.width * n.planes,
                0,
            ),
            parameters: d.parameters || 0,
        };
    }
    addModule(component, layer, stage, weights) {
        const id = moduleId(component, layer, stage);
        if (!this.nodes.has(id))
            this.nodes.set(id, {
                id,
                name: stage,
                category: stage,
                op: "MODULE",
                kind: "module",
                component,
                layer,
                shape: [],
                dtype: "module",
                bytes: 0,
                elements: 0,
                sources: [],
                strides: [],
                device: "model capacity",
                buffer: "architecture catalog",
            });
        for (const weight of weights)
            if (!this.nodes.get(id).sources.includes(weight))
                this.nodes.get(id).sources.push(weight);
    }
    match(n) {
        if (n.kind === "weight") {
            const id = `weight:${n.component}:${n.name}`;
            if (this.nodes.has(id)) return id;
        }
        if (this.nodes.has(semanticId(n))) return semanticId(n);
        for (const id of [
            moduleId(n.component, n.layer, category(n)),
            moduleId(n.component, n.layer, "other"),
            moduleId(n.component, -1, category(n)),
            moduleId(n.component, -1, "other"),
        ]) {
            if (this.nodes.has(id)) return id;
        }
        return null;
    }
    bind(nodes) {
        this.bindings.clear();
        this.rawToStable.clear();
        for (const n of nodes) this.bindNode(n);
    }
    bindNode(n) {
        const id = this.match(n);
        if (!id) return;
        this.rawToStable.set(n.id, id);
        if (!this.bindings.has(id)) this.bindings.set(id, new Map());
        this.bindings.get(id).set(n.id, n);
    }
    activity(activations, routing) {
        const result = {};
        for (const [rawId, activation] of Object.entries(activations)) {
            const id = this.rawToStable.get(rawId);
            if (!id) continue;
            result[id] = activation;
            const raw = this.bindings.get(id)?.get(rawId);
            const module =
                raw && moduleId(raw.component, raw.layer, category(raw));
            if (this.nodes.has(module)) result[module] = activation;
            for (const source of raw?.sources || []) {
                const stable = this.rawToStable.get(source);
                if (this.nodes.get(stable)?.kind === "weight")
                    result[stable] = { measured: true, used_parameter: true };
            }
        }
        for (const r of routing)
            for (const selection of r.selections)
                for (const expert of selection.experts) {
                    const id = `expert:${r.component}:${r.layer}:${expert}`;
                    if (this.nodes.has(id))
                        result[id] = {
                            measured: true,
                            expert,
                            token_index: selection.token_index,
                        };
                }
        return result;
    }
    observed(id, activations) {
        const candidates = [...(this.bindings.get(id)?.values() || [])];
        const raw =
            candidates.find((n) => activations[n.id]?.neurons) ||
            candidates.find((n) => activations[n.id]) ||
            candidates[0];
        return { raw, activation: raw && activations[raw.id] };
    }
    weightsFor(item) {
        const ids = new Set(),
            seen = new Set();
        const visit = (n, depth = 0) => {
            if (!n || seen.has(n.id) || depth > 8 || seen.size > 1000) return;
            seen.add(n.id);
            if (n.kind === "weight") {
                ids.add(n.id);
                return;
            }
            for (const source of n.sources || []) {
                const candidate = this.nodes.get(source);
                if (
                    candidate &&
                    (candidate.layer === item.layer || candidate.layer < 0)
                )
                    visit(candidate, depth + 1);
            }
        };
        item.members.forEach((n) => visit(n));
        if (!ids.size)
            for (const n of this.inventory)
                if (
                    n.component === item.component &&
                    n.layer === item.layer &&
                    (category(n) === item.category || item.name === "other")
                )
                    ids.add(n.id);
        return [...ids].map((id) => this.nodes.get(id));
    }
}
