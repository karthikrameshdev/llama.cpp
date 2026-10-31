import test from "node:test";
import assert from "node:assert/strict";
import { category, layoutGraph, routeCounts } from "./graph.js";
const n = (
    id,
    name,
    op = "MUL_MAT",
    sources = [],
    kind = "operation",
    layer = 0,
) => ({
    id,
    name,
    op,
    sources,
    kind,
    layer,
    component: "decoder",
    dtype: "f32",
    shape: [4, 1, 1, 1],
});
test("weights and original dependencies are retained", () => {
    const nodes = [
        n("w", "blk.0.ffn_up.weight", "NONE", [], "weight"),
        n("a", "attn_out-0"),
        n("b", "ffn_up-0", "MUL_MAT", ["w", "a"]),
    ];
    assert.equal(
        layoutGraph(nodes, { grouped: false, weights: true }).edges.length,
        2,
    );
    assert.equal(layoutGraph(nodes, { grouped: false }).edges.length, 1);
});
test("groups aggregate actual edges rather than inventing sequential links", () => {
    const result = layoutGraph([
        n("a", "attn_q-0"),
        n("b", "attn_k-0"),
        n("c", "ffn_up-0", "MUL_MAT", ["a", "b"]),
    ]);
    assert.equal(result.nodes.length, 2);
    assert.equal(result.edges.length, 1);
    assert.equal(result.nodes[0].members.length, 2);
});
test("MoE is recognized from the real operation", () => {
    assert.equal(category(n("a", "unnamed", "MUL_MAT_ID")), "expert");
    assert.equal(category(n("b", "ffn_moe_topk-3", "ARGSORT")), "routing");
    assert.equal(category(n("c", "ssm_state-0")), "recurrence");
});
test("routing deduplicates gate/up/down but retains token and step counts", () => {
    const r = {
        step: 0,
        component: "decoder",
        layer: 1,
        selections: [
            { token_index: 0, experts: [2, 4] },
            { token_index: 1, experts: [3, 4] },
        ],
    };
    const counts = routeCounts([r, r, { ...r, step: 1 }]);
    assert.equal(counts.get("decoder:1:4"), 4);
    assert.equal(counts.get("decoder:1:2"), 2);
});
test("search and filtering are deterministic", () => {
    const nodes = [n("a", "attn_out-0"), n("b", "ffn_up-0")];
    assert.deepEqual(
        layoutGraph(nodes, { filter: "attention" }),
        layoutGraph(nodes, { filter: "attention" }),
    );
    assert.equal(layoutGraph(nodes, { search: "layer:9" }).nodes.length, 0);
    assert.equal(nodes.length, 2);
});
