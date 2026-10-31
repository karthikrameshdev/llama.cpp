import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
export const palette = {
    input: "#78966a",
    embedding: "#91bbfa",
    normalization: "#9bba8c",
    attention: "#58d5b0",
    routing: "#d5a3ff",
    expert: "#f6bc72",
    feedforward: "#d7ac73",
    recurrence: "#f28d9c",
    output: "#c8f36c",
    other: "#779a9c",
};
export function category(n) {
    if (n.category) return n.category;
    const s = n.name.toLowerCase();
    if (/result|output|lm_head/.test(s)) return "output";
    if (/topk|argsort|moe_logits|moe_probs|moe_weights|router|gate_inp/.test(s))
        return "routing";
    if (n.op === "MUL_MAT_ID" || /moe|expert|exps/.test(s)) return "expert";
    if (/norm/.test(s)) return "normalization";
    if (/ssm|recurrent|state|gated_delta|wkv|shortconv/.test(s))
        return "recurrence";
    if (
        /attn|qcur|kcur|vcur|rope|kq_|k_cache|v_cache/.test(s) ||
        /FLASH_ATTN/.test(n.op)
    )
        return "attention";
    if (/ffn|feed_forward|mlp/.test(s)) return "feedforward";
    if (/embd|embed|tok_|position/.test(s)) return "embedding";
    return n.kind === "input" ? "input" : "other";
}
export function routeCounts(routes) {
    const seen = new Set(),
        counts = new Map();
    for (const r of routes)
        for (const s of r.selections) {
            const key = [
                r.step,
                r.component,
                r.layer,
                s.token_index,
                s.experts.join(","),
            ].join(":");
            if (seen.has(key)) continue;
            seen.add(key);
            for (const e of s.experts) {
                const label = [r.component, r.layer, e].join(":");
                counts.set(label, (counts.get(label) || 0) + 1);
            }
        }
    return counts;
}
export function layoutGraph(
    nodes,
    { grouped = true, weights = false, filter = "all", search = "" } = {},
) {
    const visible = nodes.filter(
        (n) =>
            (weights || n.kind !== "weight") &&
            (filter === "all" || category(n) === filter) &&
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
                .includes(search.toLowerCase()),
    );
    const groups = new Map(),
        index = new Map(),
        lanes = new Map(),
        components = [...new Set(nodes.map((n) => n.component))].sort();
    for (const n of visible) {
        const stage = category(n),
            key = grouped ? [n.component, n.layer, stage].join(":") : n.id;
        if (!groups.has(key))
            groups.set(key, {
                id: key,
                name: grouped ? stage : n.name,
                category: stage,
                component: n.component,
                layer: n.layer,
                members: [],
                sources: [],
            });
        groups.get(key).members.push(n);
        index.set(n.id, key);
    }
    for (const n of groups.values()) {
        const lane = [n.component, n.layer, n.category].join(":"),
            ordinal = lanes.get(lane) || 0;
        lanes.set(lane, ordinal + 1);
        n.position = [
            Object.keys(palette).indexOf(n.category) * 85 +
                (grouped ? 0 : (ordinal % 5) * 8 - 16),
            -Math.max(-1, n.layer) * 29 +
                (grouped ? 0 : Math.floor(ordinal / 5) * 5),
            components.indexOf(n.component) * 70,
        ];
    }
    const edges = [],
        seen = new Set();
    for (const n of visible)
        for (const s of n.sources) {
            const a = index.get(s),
                b = index.get(n.id),
                key = a + ">" + b;
            if (!a || !b || a === b || seen.has(key)) continue;
            seen.add(key);
            edges.push([a, b]);
            groups.get(b).sources.push(a);
        }
    return {
        nodes: [...groups.values()],
        edges,
        originalCount: nodes.length,
        visibleCount: visible.length,
    };
}
export class GraphView {
    constructor(container, onSelect, onHover) {
        this.container = container;
        this.onSelect = onSelect;
        this.onHover = onHover;
        this.scene = new THREE.Scene();
        this.content = new THREE.Group();
        this.scene.add(this.content);
        this.camera = new THREE.OrthographicCamera(
            -500,
            500,
            320,
            -320,
            0.1,
            10000,
        );
        this.camera.position.set(400, -300, 1800);
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
        });
        this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        this.renderer.domElement.setAttribute(
            "aria-label",
            "Runtime graph. Drag to pan, scroll to zoom, click to inspect.",
        );
        this.renderer.domElement.setAttribute("role", "img");
        container.prepend(this.renderer.domElement);
        this.controls = new OrbitControls(
            this.camera,
            this.renderer.domElement,
        );
        this.controls.enableRotate = false;
        this.controls.enableDamping = true;
        this.controls.screenSpacePanning = true;
        this.controls.mouseButtons = {
            LEFT: THREE.MOUSE.PAN,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.ROTATE,
        };
        const grid = new THREE.GridHelper(3000, 100, 0x24351e, 0x1d2a18);
        grid.rotation.x = Math.PI / 2;
        grid.position.set(400, -350, -40);
        grid.material.transparent = true;
        grid.material.opacity = 0.5;
        this.scene.add(grid);
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();
        this.renderer.domElement.addEventListener("pointerdown", (e) => {
            this.down = [e.clientX, e.clientY];
        });
        this.renderer.domElement.addEventListener("pointerup", (e) => {
            if (
                this.down &&
                Math.hypot(e.clientX - this.down[0], e.clientY - this.down[1]) <
                    5
            ) {
                const n = this.pick(e);
                if (n) onSelect(n);
            }
        });
        this.renderer.domElement.addEventListener("pointermove", (e) =>
            onHover(this.pick(e), e),
        );
        this.renderer.domElement.addEventListener("pointerleave", () =>
            onHover(null),
        );
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(container);
        this.resize();
        const draw = () => {
            this.frame = requestAnimationFrame(draw);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        draw();
    }
    resize() {
        const w = this.container.clientWidth,
            h = this.container.clientHeight;
        this.renderer.setSize(w, h);
        this.camera.left = (-w / h) * 320;
        this.camera.right = (w / h) * 320;
        this.camera.updateProjectionMatrix();
    }
    pick(e) {
        if (!this.mesh) return null;
        const r = this.renderer.domElement.getBoundingClientRect();
        this.pointer.set(
            ((e.clientX - r.left) / r.width) * 2 - 1,
            (-(e.clientY - r.top) / r.height) * 2 + 1,
        );
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const hits = this.raycaster.intersectObject(this.mesh);
        return hits.length ? this.layout.nodes[hits[0].instanceId] : null;
    }
    setData(nodes, activations, options = {}, fit = false) {
        this.buildCount = (this.buildCount || 0) + 1;
        this.container.dataset.geometryBuilds = this.buildCount;
        for (const c of [...this.content.children]) {
            this.content.remove(c);
            c.geometry?.dispose();
            c.material?.map?.dispose();
            c.material?.dispose();
        }
        this.layout = layoutGraph(nodes, options);
        this.nodeLabels = new Map();
        if (!this.layout.nodes.length) {
            this.mesh = null;
            return;
        }
        this.mesh = new THREE.InstancedMesh(
            new THREE.SphereGeometry(options.grouped ? 4.1 : 1.9, 10, 8),
            new THREE.MeshBasicMaterial(),
            this.layout.nodes.length,
        );
        this.content.add(this.mesh);
        this.edgeGeometry = null;
        const index = new Map(this.layout.nodes.map((n) => [n.id, n])),
            points = [];
        for (const [a, b] of this.layout.edges)
            points.push(...index.get(a).position, ...index.get(b).position);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(points, 3),
        );
        this.edgeGeometry = geometry;
        geometry.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(
                new Float32Array(points.length),
                3,
            ),
        );
        this.content.add(
            new THREE.LineSegments(
                geometry,
                new THREE.LineBasicMaterial({
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.6,
                }),
            ),
        );
        if (options.grouped) {
            const names = {
                embedding: "EMBED",
                normalization: "NORM",
                attention: "ATTN",
                routing: "ROUTE",
                expert: "EXPERT",
                feedforward: "FFN",
                recurrence: "STATE",
                output: "OUT",
                other: "OPS",
                input: "INPUT",
            };
            for (const n of this.layout.nodes) {
                const label = document.createElement("canvas");
                label.width = 256;
                label.height = 48;
                const context = label.getContext("2d");
                context.font = "bold 24px monospace";
                context.fillStyle = "#ffffff";
                context.fillText(names[n.category] || n.category, 0, 32);
                const sprite = new THREE.Sprite(
                    new THREE.SpriteMaterial({
                        map: new THREE.CanvasTexture(label),
                        transparent: true,
                        color: palette[n.category],
                    }),
                );
                sprite.scale.set(150, 28, 1);
                sprite.position.set(
                    n.position[0] + 80,
                    n.position[1] - 6,
                    n.position[2] + 1,
                );
                this.nodeLabels.set(n.id, sprite);
                this.content.add(sprite);
            }
            const layers = new Map();
            for (const n of this.layout.nodes)
                layers.set(n.component + ":" + n.layer, n);
            for (const n of layers.values()) {
                const canvas = document.createElement("canvas");
                canvas.width = 240;
                canvas.height = 48;
                const c = canvas.getContext("2d");
                c.font = "24px monospace";
                c.fillStyle = "#809774";
                c.fillText(
                    n.layer < 0
                        ? n.component.toUpperCase()
                        : n.component.slice(0, 3).toUpperCase() +
                              " / " +
                              String(n.layer).padStart(2, "0"),
                    0,
                    32,
                );
                const sprite = new THREE.Sprite(
                    new THREE.SpriteMaterial({
                        map: new THREE.CanvasTexture(canvas),
                        transparent: true,
                    }),
                );
                sprite.scale.set(115, 28, 1);
                sprite.position.set(-76, n.position[1], n.position[2]);
                this.content.add(sprite);
            }
        }
        this.updateActivity(activations);
        if (fit || !this.fitted) {
            this.fit();
            this.fitted = true;
        }
    }
    updateActivity(activations, active) {
        if (!this.mesh) return;
        const color = new THREE.Color(),
            matrix = new THREE.Matrix4();
        this.layout.nodes.forEach((n, i) => {
            const measured = n.members.some((x) => activations[x.id]),
                current = n.members.some((x) => x.id === active);
            color
                .set(current ? "#ecffc2" : palette[n.category])
                .multiplyScalar(current ? 1.15 : measured ? 0.98 : 0.4);
            this.mesh.setColorAt(i, color);
            this.nodeLabels?.get(n.id)?.material.color.copy(color);
            matrix.makeTranslation(...n.position);
            if (current) matrix.scale(new THREE.Vector3(1.5, 1.5, 1.5));
            this.mesh.setMatrixAt(i, matrix);
        });
        this.mesh.instanceColor.needsUpdate = true;
        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.edgeGeometry) {
            const attribute = this.edgeGeometry.getAttribute("color");
            const observed = new Set(
                this.layout.nodes
                    .filter((n) => n.members.some((m) => activations[m.id]))
                    .map((n) => n.id),
            );
            this.layout.edges.forEach(([a, b], i) => {
                const intensity =
                    observed.has(a) && observed.has(b) ? 0.8 : 0.12;
                for (const vertex of [i * 2, i * 2 + 1])
                    attribute.setXYZ(
                        vertex,
                        0.47 * intensity,
                        0.64 * intensity,
                        0.31 * intensity,
                    );
            });
            attribute.needsUpdate = true;
        }
        this.container.dataset.highlightUpdates =
            Number(this.container.dataset.highlightUpdates || 0) + 1;
    }
    fit() {
        if (!this.layout?.nodes.length) return;
        const box = new THREE.Box3();
        for (const n of this.layout.nodes)
            box.expandByPoint(new THREE.Vector3(...n.position));
        box.expandByScalar(50);
        const center = box.getCenter(new THREE.Vector3()),
            size = box.getSize(new THREE.Vector3());
        this.camera.zoom =
            Math.min(
                (this.camera.right - this.camera.left) / Math.max(size.x, 100),
                640 / Math.max(size.y, 100),
            ) * 0.86;
        this.camera.position.set(center.x, center.y, center.z + 1800);
        this.controls.target.copy(center);
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }
    toggle3d(value) {
        this.controls.enableRotate = value;
        const t = this.controls.target;
        this.camera.position.set(
            t.x + (value ? 430 : 0),
            t.y + (value ? 100 : 0),
            t.z + 1800,
        );
        this.controls.update();
    }
}
