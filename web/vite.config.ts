import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI app runs on :8000. Proxying /api keeps the browser
// same-origin in dev, so no CORS round-trips and no hardcoded host
// in the frontend code. `host: true` exposes the dev server on the
// LAN for the local/LAN demo posture.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
