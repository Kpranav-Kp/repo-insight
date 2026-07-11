import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  envDir: "..",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path, 
      },
    },
  },
})
