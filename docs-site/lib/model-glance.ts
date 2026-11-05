import type { GlanceGroup } from "@/data/home";
import type { RepoConfig, RepoMeta } from "./repo";

export type GlanceValueRow = {
  label: string;
  name: string;
  value: string;
  permalink: string;
  note?: string;
  // Nothing in the repository reads this constant, so changing it has no effect.
  unread: boolean;
};

// `head_dim` has the value text "emb_dim // num_heads", so derived constants print their value.
export function configValueText(constant: RepoConfig["constants"][number]): string {
  return constant.derived ? String(constant.value) : constant.valueText;
}

export function buildGlance(groups: GlanceGroup[], config: RepoConfig, repo: RepoMeta) {
  return groups.map((group) => ({
    title: group.title,
    rows: group.rows.map((row): GlanceValueRow => {
      const constant = config.constants.find((candidate) => candidate.name === row.config);
      if (!constant) throw new Error(`config.json has no constant named ${row.config}`);
      return {
        label: row.label,
        name: constant.name,
        value: configValueText(constant),
        permalink: `${repo.githubBaseUrl}/blob/${repo.sha}/${config.path}#L${constant.line}`,
        note: row.note,
        unread: constant.readBy.length === 0,
      };
    }),
  }));
}
