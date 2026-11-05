import { defineConfig } from "@playwright/test";

// Tests the static export in out/, so run `bun run build` first.
export default defineConfig({
  testDir: "tests/e2e",
  // `bun run start` (python http.server) has a small connection backlog and refuses connections
  // when several browsers load a page at once. That shows up as random test failures.
  workers: 1,
  use: { baseURL: "http://127.0.0.1:3000" },
  webServer: {
    command: "bun run start",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: true,
  },
});
