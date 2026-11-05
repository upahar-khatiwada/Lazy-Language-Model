import type { ReactNode } from "react";

// One step of the page authoring protocol. A real h3, written as JSX so that it stays out of the
// on-this-page rail: nine rail entries per concept would hide the concepts themselves.
export function ConceptStep({ children }: { children: ReactNode }) {
  return <h3 data-concept-step="">{children}</h3>;
}
