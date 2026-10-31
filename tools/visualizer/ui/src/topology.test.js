import test from "node:test";
import assert from "node:assert/strict";
import { ModelTopology } from "./topology.js";
const tensor = (name, shape) => ({
    name,
    shape,
    dtype: "f32",
    bytes: shape.reduce((a, b) => a * b, 1) * 4,
});
const model = {
    id: "test",
    details: {
        has_decoder: true,
        layers: 1,
        parameters: 1000,
        tensors: [
            tensor("blk.0.ffn_up_exps.weight", [4, 8, 3, 1]),
            tensor("blk.0.ffn_gate_exps.weight", [4, 8, 3, 1]),
            tensor("blk.0.ffn_down_exps.weight", [8, 4, 3, 1]),
            tensor("blk.0.ffn_gate_inp.weight", [4, 3, 1, 1]),
        ],
        projector: { tensors: [tensor("blk.0.ffn_up.weight", [4, 12, 1, 1])] },
    },
};
const operation = (
    id = "e0n0",
    name = "ffn_moe_swiglu-0",
    shape = [8, 2, 1, 1],
) => ({
    id,
    name,
    shape,
    op: "GLU",
    kind: "operation",
    sources: [],
    component: "decoder",
    layer: 0,
    dtype: "f32",
});
test("preloads all weights, encoder capacity and every expert before a prompt", () => {
    const topology = new ModelTopology(model, { nodes: [operation()] });
    assert.equal(topology.capacity.weights, 5);
    assert.equal(topology.capacity.experts, 3);
    assert.equal(topology.capacity.channels, 36);
    assert(
        [...topology.nodes.values()].some(
            (n) => n.component === "multimodal_encoder",
        ),
    );
    assert.equal(topology.neurons.get("decoder:0").width, 8);
});
test("prompt shapes and native ID reuse do not mutate model geometry", () => {
    const topology = new ModelTopology(model, { nodes: [operation()] });
    const before = JSON.stringify([...topology.nodes]);
    for (let i = 0; i < 10; i++) {
        const raw = operation("e" + i + "n77", "ffn_moe_swiglu-0", [
            8,
            2,
            i + 1,
            1,
        ]);
        topology.bind([raw]);
        const activity = topology.activity(
            { [raw.id]: { measured: true, wall_us: i } },
            [],
        );
        assert(activity[topology.match(raw)]);
        assert.equal(
            topology.observed(topology.match(raw), { [raw.id]: { wall_us: i } })
                .raw.shape[2],
            i + 1,
        );
    }
    assert.equal(JSON.stringify([...topology.nodes]), before);
});
test("unseen runtime variants highlight an existing module without adding nodes", () => {
    const topology = new ModelTopology(model);
    const count = topology.nodes.size;
    const raw = operation("new", "ffn_moe_weighted-0");
    topology.bind([raw]);
    const mapped = topology.match(raw);
    assert.equal(topology.nodes.get(mapped).kind, "module");
    assert(topology.activity({ new: { measured: true } }, [])[mapped]);
    assert.equal(topology.nodes.size, count);
});
test("only actually selected experts highlight within preloaded expert capacity", () => {
    const topology = new ModelTopology(model);
    const activity = topology.activity({}, [
        {
            component: "decoder",
            layer: 0,
            selections: [{ token_index: 2, experts: [1] }],
        },
    ]);
    assert(activity["expert:decoder:0:1"]);
    assert(!activity["expert:decoder:0:0"]);
    assert(topology.nodes.has("expert:decoder:0:0"));
});
test("expert parameter detail retains all incoming and outgoing tensor references", () => {
    const topology = new ModelTopology(model);
    const expert = topology.nodes.get("expert:decoder:0:2");
    const weights = topology.weightsFor({ ...expert, members: [expert] });
    assert(weights.some((n) => n.name.endsWith("ffn_up_exps.weight")));
    assert(weights.some((n) => n.name.endsWith("ffn_down_exps.weight")));
});
test("recycled native IDs never highlight or inspect the wrong semantic node", () => {
    const original = operation();
    const topology = new ModelTopology(model, { nodes: [original] });
    const stable = topology.match(original);
    const collision = operation(original.id, "ffn_moe_weighted-0");
    topology.bind([collision]);
    assert.equal(
        topology.observed(stable, { [collision.id]: { measured: true } }).raw,
        undefined,
    );
    assert(
        !topology.activity({ [collision.id]: { measured: true } }, [])[stable],
    );
    const next = operation("e1n99", "ffn_moe_swiglu-0", [8, 2, 52, 1]);
    topology.bind([next]);
    assert.deepEqual(
        topology.observed(stable, { [next.id]: { measured: true } }).raw.shape,
        [8, 2, 52, 1],
    );
});
