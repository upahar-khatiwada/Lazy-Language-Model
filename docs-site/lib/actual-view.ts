// Pure logic for Actual Model Output: no file access, so a client component can use it.

export type ActualAttentionLayer = {
  layer: number;
  heads: { head: number; A: number[][] }[];
};

export type ActualAttentionData = {
  window: {
    split: string;
    index: number;
    ids: number[];
    tokens: string[];
    targets: number[];
    text: string;
  };
  T: number;
  H: number;
  L: number;
  dh: number;
  largestRowSumGap: number;
  layers: ActualAttentionLayer[];
};

const ROW_SUM_TOLERANCE = 1e-9;

// Every promise attention_forward makes about A, as a list of what is broken.
export function attentionProblems({ T, H, L, layers, window }: ActualAttentionData): string[] {
  const problems: string[] = [];
  if (window.tokens.length !== T) problems.push(`${window.tokens.length} tokens, T is ${T}`);
  if (layers.length !== L) problems.push(`${layers.length} layers, L is ${L}`);
  for (const { layer, heads } of layers) {
    if (heads.length !== H) problems.push(`layer ${layer}: ${heads.length} heads, H is ${H}`);
    for (const { head, A } of heads) {
      const where = `layer ${layer}, head ${head}`;
      if (A.length !== T) problems.push(`${where}: ${A.length} rows, T is ${T}`);
      A.forEach((row, i) => {
        if (row.length !== T) problems.push(`${where}, row ${i}: ${row.length} columns`);
        const sum = row.reduce((total, value) => total + value, 0);
        if (Math.abs(sum - 1) > ROW_SUM_TOLERANCE) {
          problems.push(`${where}, row ${i}: sums to ${sum}`);
        }
        row.forEach((value, j) => {
          if (j > i && value !== 0) problems.push(`${where}: masked cell ${i},${j} is ${value}`);
        });
      });
    }
  }
  return problems;
}

// A cell is masked when its column comes after its row: attention_forward's np.triu(..., 1).
export function causalMask(T: number): boolean[][] {
  return Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => j > i));
}

// Real tokens repeat and can be blank, and a label must be unique and fit a matrix header.
export function tokenLabels(tokens: string[]): string[] {
  return tokens.map((token, i) => `${i} ${visibleToken(token).slice(0, 4)}`);
}

export function visibleToken(token: string): string {
  return token.replaceAll(" ", "␠").replaceAll("\n", "\\n").replaceAll("\t", "\\t");
}

export function shortHash(sha256: string): string {
  return sha256.slice(0, 12);
}
