import { allEquationSlugs, equationDetail } from "@/lib/equation-links";

// One small file per equation, read when a popover opens. Equations are rendered here, at build
// time, so no KaTeX JavaScript reaches the browser and pages do not carry every equation.

export const dynamic = "force-static";
export const dynamicParams = false;

export function generateStaticParams() {
  return allEquationSlugs().map((slug) => ({ file: `${slug}.json` }));
}

export async function GET(_request: Request, { params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  const detail = equationDetail(file.replace(/\.json$/, ""));
  if (!detail) return new Response("Unknown equation", { status: 404 });
  return Response.json(detail);
}
