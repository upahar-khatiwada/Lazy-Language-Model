import { expect, test } from "@playwright/test";

test("the exported home page renders its title", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "LLM Developer Docs" })).toBeVisible();
});
