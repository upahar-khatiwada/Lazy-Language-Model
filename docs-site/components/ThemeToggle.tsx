"use client";

import { Button } from "./Button";

// Writes the override that the inline script in app/layout.tsx restores on the next page load.
function toggleTheme() {
  const root = document.documentElement;
  const current =
    root.dataset.theme ??
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try {
    window.localStorage.setItem("theme", next);
  } catch {
    // Storage can be blocked. The theme still applies to this page.
  }
}

export function ThemeToggle() {
  return (
    <Button onClick={toggleTheme} aria-label="Toggle theme">
      Theme
    </Button>
  );
}
