import { defineConfig } from "vite";

export default defineConfig({
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    three: ["three", "three/addons/controls/OrbitControls.js"],
                },
            },
        },
    },
});
