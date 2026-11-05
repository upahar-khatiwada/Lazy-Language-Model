import styles from "./SearchField.module.css";

// Visual element only. Search behaviour is built in a later task.
export function SearchField({ label = "Search documentation" }: { label?: string }) {
  return (
    <label className={styles.field}>
      <span className={styles.label}>{label}</span>
      <input className={styles.input} type="search" placeholder="Search" />
    </label>
  );
}
