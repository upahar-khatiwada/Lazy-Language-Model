import { blockBackward, blockForward, matrixOf, vectorOf } from "./block";
import type { BlockCache, FlatParams, Gradients } from "./block";
import { embedBackward, embedForward } from "./embeddings";
import {
  crossEntropy,
  crossEntropyBackward,
  layerNormBackward,
  layerNormForward,
  linearBackward,
  linearForward,
  type LayerNormCache,
} from "./ops";
import type { Matrix } from "./tensor";

// The fields of transformer.ModelConfig that these two functions read.
export type ModelSizes = {
  context_len: number;
  num_heads: number;
  num_layers: number;
  ln_eps: number;
};

export type ModelCache = { blocks: BlockCache[]; lnf: LayerNormCache; Xf: Matrix };

// Mirrors transformer.forward. stages holds X after the embeddings and after each block.
export function forward(params: FlatParams, ids: number[], cfg: ModelSizes) {
  if (ids.length > cfg.context_len) throw new Error("sequence longer than the position table");

  let X = embedForward(matrixOf(params, "tok_emb"), matrixOf(params, "pos_emb"), ids).X0;
  const stages = [X];
  const blocks: BlockCache[] = [];
  for (let l = 0; l < cfg.num_layers; l++) {
    const block = blockForward(X, params, l, cfg.num_heads, cfg.ln_eps);
    X = block.Y;
    stages.push(X);
    blocks.push(block.cache);
  }

  const final = layerNormForward(
    X,
    vectorOf(params, "lnf_g"),
    vectorOf(params, "lnf_b"),
    cfg.ln_eps,
  );
  const Xf = final.Y;
  const logits = linearForward(Xf, matrixOf(params, "head_w"), vectorOf(params, "head_b"));
  return { logits, stages, cache: { blocks, lnf: final.cache, Xf } satisfies ModelCache };
}

// Mirrors transformer.loss_and_grads, in the same order. dX holds each value that variable takes:
// after the final LayerNorm, then after each block, last block first. calls names each function it
// calls, so that a test can compare the order with a real run.
export function lossAndGrads(
  params: FlatParams,
  ids: number[],
  targets: number[],
  cfg: ModelSizes,
) {
  const calls: string[] = [];
  const logged = <T>(name: string, run: () => T): T => {
    calls.push(name);
    return run();
  };

  const { logits, cache } = logged("forward", () => forward(params, ids, cfg));
  const { loss, probs } = logged("cross_entropy_forward", () => crossEntropy(logits, targets));
  const dZ = logged("cross_entropy_backward", () => crossEntropyBackward(probs, targets));
  const head = logged("linear_backward", () =>
    linearBackward(cache.Xf, matrixOf(params, "head_w"), dZ),
  );
  const final = logged("layer_norm_backward", () =>
    layerNormBackward(head.dX, cache.lnf, vectorOf(params, "lnf_g")),
  );

  const grads: Gradients = {
    head_w: head.dW,
    head_b: head.db,
    lnf_g: final.dgamma,
    lnf_b: final.dbeta,
  };
  const dX = [final.dX];
  for (let l = cfg.num_layers - 1; l >= 0; l--) {
    const block = logged(`block_backward(${l})`, () =>
      blockBackward(dX.at(-1)!, cache.blocks[l]!, params, l, cfg.num_heads),
    );
    Object.assign(grads, block.grads);
    dX.push(block.dX);
  }
  const tables = logged("embed_backward", () =>
    embedBackward(
      dX.at(-1)!,
      ids,
      matrixOf(params, "tok_emb").length,
      matrixOf(params, "pos_emb").length,
    ),
  );
  grads.tok_emb = tables.tok;
  grads.pos_emb = tables.pos;

  return { loss, probs, dZ, dXf: head.dX, dX, grads, calls };
}
