import type { ComponentProps } from "react";
import styles from "./DataTable.module.css";

// A wide table scrolls inside this wrapper, so it never widens the page.
export function DataTable({
  label = "Table",
  ...props
}: ComponentProps<"table"> & { label?: string }) {
  return (
    <div className={styles.wrap} role="group" aria-label={label} tabIndex={0}>
      <table {...props} />
    </div>
  );
}
