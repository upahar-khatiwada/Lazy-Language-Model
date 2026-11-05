import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const config = [
  {
    ignores: [
      "node_modules/",
      ".next/",
      "out/",
      "next-env.d.ts",
      "test-results/",
      "playwright-report/",
    ],
  },
  ...nextVitals,
  ...nextTypescript,
];

export default config;
