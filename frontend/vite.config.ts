import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.VLAB_API_ORIGIN || 'http://127.0.0.1:8767', changeOrigin: true, ws: true },
      '/assets': { target: process.env.VLAB_API_ORIGIN || 'http://127.0.0.1:8767', changeOrigin: true },
    },
  },
})
