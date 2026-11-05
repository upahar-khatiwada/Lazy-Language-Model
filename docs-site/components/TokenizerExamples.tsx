import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import {
  applyMerges,
  buildTokenizer,
  countWords,
  decode,
  encode,
  mergeTokens,
} from "@/lib/math/bpe";
import type { Pair } from "@/lib/math/bpe";
import { DataTable } from "./DataTable";
import { showToken, Tokens } from "./Tokens";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases and results come from lib/math/bpe.ts, so each table has a
// Python result to compare with in tests/math.golden.test.ts. All of it is hand-picked text.

const NOT_THE_MODEL = "A tiny text chosen by hand, not the corpus. These are not the real model's.";

export function WordCountExample({ concept }: { concept: string }) {
  const texts = ["count_words.stepper", "count_words.classes"].map(
    (id) => getGoldenCase("bpe", id).inputs.text as string,
  );
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Pre-tokens and their counts"
        kind="illustrative"
        caption="Tiny texts chosen by hand, not the corpus. ␠ is a space and \n is a newline."
      >
        <DataTable label="Pre-token counts" data-testid="word-counts">
          <thead>
            <tr>
              <th scope="col">Text</th>
              <th scope="col">Pre-tokens with their counts</th>
            </tr>
          </thead>
          <tbody>
            {texts.map((text) => (
              <tr key={text}>
                <td>
                  <code>{JSON.stringify(text)}</code>
                </td>
                <td>
                  <Tokens
                    tokens={countWords(text).map(([word, n]) => `${showToken(word)} × ${n}`)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function MergeExample({ concept }: { concept: string }) {
  const cases = ["merge_tokens.simple", "merge_tokens.overlap_left_to_right"].map(
    (id) => getGoldenCase("bpe", id).inputs as { tokens: string[]; pair: Pair; merged: string },
  );
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure title="merge_tokens" kind="illustrative" caption="Hand-picked tokens.">
        <DataTable label="merge_tokens inputs and results" data-testid="merge-example">
          <thead>
            <tr>
              <th scope="col">Tokens</th>
              <th scope="col">Pair</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {cases.map(({ tokens, pair, merged }) => (
              <tr key={tokens.join()}>
                <td>
                  <Tokens tokens={tokens} />
                </td>
                <td>
                  <Tokens tokens={pair} />
                </td>
                <td>
                  <Tokens tokens={mergeTokens(tokens, pair, merged)} />
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function VocabularyExample({ concept }: { concept: string }) {
  const text = getGoldenCase("bpe", "build_tokenizer.stepper").inputs.text as string;
  const { itos } = buildTokenizer(text);
  const characters = new Set(text);
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure title={`The vocabulary of "${text}"`} kind="illustrative" caption={NOT_THE_MODEL}>
        <DataTable label="Vocabulary" data-testid="vocabulary">
          <thead>
            <tr>
              <th scope="col">Id</th>
              <th scope="col">Token</th>
              <th scope="col">Comes from</th>
            </tr>
          </thead>
          <tbody>
            {itos.map((token, id) => (
              <tr key={token}>
                <td>{id}</td>
                <td>
                  <code>{showToken(token)}</code>
                </td>
                <td>
                  {id === 0
                    ? "The unknown token, always first"
                    : characters.has(token)
                      ? "Character, by first appearance"
                      : "Merged token, by merge order"}
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function EncodeExample({ concept }: { concept: string }) {
  const cases = [
    "encode.known_word",
    "encode.unseen_punctuation",
    "encode.unseen_inside_word",
    "encode.each_unseen_character",
    "encode.unk_text_unseen",
  ].map((id) => {
    const { train_text, text } = getGoldenCase("bpe", id).inputs as Record<string, string>;
    return { id, trainText: train_text!, text: text! };
  });
  const { trainText } = cases[0]!;
  const tok = buildTokenizer(trainText);
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title={`Encoding and decoding with the tokenizer of "${trainText}"`}
        kind="illustrative"
        caption={NOT_THE_MODEL}
      >
        <DataTable label="Encode and decode" data-testid="encode-example">
          <thead>
            <tr>
              <th scope="col">Text</th>
              <th scope="col">Tokens</th>
              <th scope="col">Ids</th>
              <th scope="col">Decoded</th>
            </tr>
          </thead>
          <tbody>
            {cases.map(({ id, text }) => {
              const ids = encode(tok, text);
              return (
                <tr key={id} data-case={id}>
                  <td>
                    <code>{text}</code>
                  </td>
                  <td>
                    <Tokens tokens={applyMerges(text, tok.merges)} />
                  </td>
                  <td>
                    <code data-testid="encode-ids">{JSON.stringify(ids)}</code>
                  </td>
                  <td>
                    <code data-testid="encode-decoded">{decode(tok, ids)}</code>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}
