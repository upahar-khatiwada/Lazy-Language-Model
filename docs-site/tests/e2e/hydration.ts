import { expect, type Locator } from "@playwright/test";

// A prerendered button is visible before React attaches its click handler.
export async function clickWhenHydrated(target: Locator): Promise<void> {
  await expect
    .poll(() =>
      target.evaluate((element) =>
        Object.keys(element).some((key) => key.startsWith("__reactProps")),
      ),
    )
    .toBe(true);
  await target.click();
}
