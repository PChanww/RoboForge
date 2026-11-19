import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const local = loadEnv(mode, process.cwd(), 'VLAB_')
  const target = process.env.VLAB_API_ORIGIN || local.VLAB_API_ORIGIN || 'http://127.0.0.1:8767'
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': { target, changeOrigin: true, ws: true },
        '/assets': { target, changeOrigin: true },
      },
    },
  }
})
