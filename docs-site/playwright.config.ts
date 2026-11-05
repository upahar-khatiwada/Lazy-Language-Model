import { defineConfig } from "@playwright/test";

// Tests the static export in out/, so run `bun run build` first.
export default defineConfig({
  testDir: "tests/e2e",
  use: { baseURL: "http://127.0.0.1:3000" },
  webServer: {
    command: "bun run start",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: true,
  },
});
