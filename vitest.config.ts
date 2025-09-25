import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    browser: {
      enabled: true,
      name: 'chrome',
    },
  },
});
