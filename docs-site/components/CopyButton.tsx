"use client";

import { useState } from "react";
import { Button } from "./Button";

export function CopyButton({ text }: { text: string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    window.setTimeout(() => setStatus("idle"), 2000);
  }

  return (
    <Button onClick={copy} aria-live="polite">
      {status === "idle" ? "Copy" : status === "copied" ? "Copied" : "Copy failed"}
    </Button>
  );
}
