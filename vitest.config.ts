import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config';

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      fileParallelism: false,
      environment: 'jsdom',
      globals: true,
      browser: {
        enabled: true,
        instances: [{ browser: 'chrome' }],
      },
    },
  })
);
