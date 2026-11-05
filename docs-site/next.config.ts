import type { NextConfig } from "next";

// Set NEXT_PUBLIC_BASE_PATH (for example "/Lazy-Language-Model-LLM-") for GitHub Pages.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
};

export default nextConfig;
