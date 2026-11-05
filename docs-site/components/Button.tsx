import type { ComponentProps } from "react";
import styles from "./Button.module.css";

type Variant = "primary" | "secondary";

export function buttonClass(variant: Variant = "secondary"): string {
  return `${styles.button} ${styles[variant]}`;
}

export function Button({
  variant = "secondary",
  type = "button",
  ...props
}: ComponentProps<"button"> & { variant?: Variant }) {
  return <button type={type} className={buttonClass(variant)} {...props} />;
}
