import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import { describe, expect, test } from "vitest";

const css = readFileSync(join(process.cwd(), "styles", "tokens.css"), "utf-8");

function block(pattern: RegExp): Record<string, string> {
  const body = pattern.exec(css)?.[1];
  if (!body) throw new Error(`No block matches ${pattern}`);
  return Object.fromEntries(
    [...body.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((match) => [match[1], match[2]!.trim()]),
  );
}

const light = block(/:root,\s*\[data-theme="light"\]\s*\{([^}]*)\}/);
const darkMedia = block(/:root:not\(\[data-theme="light"\]\)\s*\{([^}]*)\}/);
const dark = block(/\n\[data-theme="dark"\]\s*\{([^}]*)\}/);

function luminance(hex: string): number {
  const channels = [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16) / 255);
  const [r, g, b] = channels.map((c) =>
    c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (high + 0.05) / (low + 0.05);
}

const surfaces = ["--bg", "--surface", "--code-bg"];
const TEXT_PAIRS: [string, string][] = [
  ...surfaces.flatMap((bg) => [
    ["--text", bg] as [string, string],
    ["--muted", bg] as [string, string],
    ["--accent", bg] as [string, string],
  ]),
  ["--text", "--highlight"],
  ["--text", "--hover"],
  ["--muted", "--hover"],
  ["--accent", "--active-bg"],
  ["--muted", "--active-bg"],
  ["--on-accent", "--accent"],
  ["--bg", "--text"],
  ["--syntax-keyword", "--code-bg"],
  ["--syntax-string", "--code-bg"],
  ["--syntax-number", "--code-bg"],
  ["--syntax-comment", "--code-bg"],
  ["--syntax-function", "--code-bg"],
  // Highlighted code lines. --syntax-comment is replaced by --text there, in CodeBlock.module.css.
  ["--syntax-keyword", "--highlight"],
  ["--syntax-string", "--highlight"],
  ["--syntax-number", "--highlight"],
  ["--syntax-function", "--highlight"],
  ["--badge-amber-text", "--badge-amber-bg"],
  ["--badge-green-text", "--badge-green-bg"],
  ["--badge-violet-text", "--badge-violet-bg"],
];

describe.each([
  ["light", light],
  ["dark", dark],
])("%s theme contrast", (_name, tokens) => {
  test.each(TEXT_PAIRS)("%s on %s meets AA", (foreground, background) => {
    expect(contrast(tokens[foreground]!, tokens[background]!)).toBeGreaterThanOrEqual(4.5);
  });

  test("the input border and focus ring reach 3:1 against page surfaces", () => {
    for (const bg of ["--bg", "--surface"]) {
      expect(contrast(tokens["--muted"]!, tokens[bg]!)).toBeGreaterThanOrEqual(3);
      expect(contrast(tokens["--accent"]!, tokens[bg]!)).toBeGreaterThanOrEqual(3);
    }
  });
});

test("the two copies of the dark theme are identical", () => {
  expect(darkMedia).toEqual(dark);
});

test("light and dark define the same tokens", () => {
  expect(Object.keys(dark).sort()).toEqual(Object.keys(light).sort());
});

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return [".css", ".ts", ".tsx"].includes(extname(path)) ? [path] : [];
  });
}

const NAMED_COLOUR =
  /(?:^|[\s;{])(?:color|background(?:-color)?|border(?:-\w+)*|outline(?:-color)?|fill|stroke|box-shadow)\s*:[^;]*\b(?:white|black|gray|grey|red|green|blue|yellow|orange|purple|pink|teal|amber|violet)\b/m;
const FUNCTION_COLOUR = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix)\(/;
const HEX_COLOUR = /#[0-9a-fA-F]{3,8}\b/;

test("colour literals appear only in styles/tokens.css", () => {
  const files = ["app", "components", "styles"]
    .flatMap((dir) => sourceFiles(join(process.cwd(), dir)))
    .filter((path) => !path.endsWith("tokens.css"));
  expect(files.length).toBeGreaterThan(10);

  const offenders = files.filter((path) => {
    // Token names such as var(--badge-green-bg) contain colour words but are not literals.
    const text = readFileSync(path, "utf-8").replace(/var\(--[\w-]+\)/g, "var()");
    return HEX_COLOUR.test(text) || FUNCTION_COLOUR.test(text) || NAMED_COLOUR.test(text);
  });
  expect(offenders).toEqual([]);
});
