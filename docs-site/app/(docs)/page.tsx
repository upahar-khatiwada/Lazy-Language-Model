import Link from "next/link";
import { DocMain } from "@/components/DocMain";
import { NAV } from "@/data/nav";
import { listContentSlugs } from "@/lib/content";
import { flattenNav, resolveNav } from "@/lib/nav";

export default function HomePage() {
  const first = flattenNav(resolveNav(NAV, listContentSlugs())).find((page) => page.enabled);
  return (
    <DocMain>
      <h1>LLM Developer Docs</h1>
      {first ? (
        <p>
          <Link href={first.href}>{first.title}</Link>
        </p>
      ) : null}
    </DocMain>
  );
}
