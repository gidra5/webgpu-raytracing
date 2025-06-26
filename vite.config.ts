import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import path from 'path';

import UnocssPlugin from '@unocss/vite';

export default defineConfig({
  base: '/webgpu-raytracing/',
  plugins: [solidPlugin(), UnocssPlugin()],
  server: {
    port: 3000,
  },
  build: {
    target: 'esnext',
  },
  resolve: {
    alias: {
      '@assets': path.resolve(__dirname, './assets'),
    },
  },
});
