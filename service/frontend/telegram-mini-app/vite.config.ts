import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendProxyTarget = env.VITE_BACKEND_PROXY_TARGET || "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: "0.0.0.0",
      allowedHosts: [
        "localhost",
        "127.0.0.1",
        "myspace-headdress-playoff.ngrok-free.dev",
        ".ngrok-free.dev",
        ".ngrok-free.app"
      ],
      proxy: {
        "/api": {
          target: backendProxyTarget,
          changeOrigin: true
        }
      }
    }
  };
});
