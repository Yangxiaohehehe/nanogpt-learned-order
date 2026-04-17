# out-wikitext103-seq256-random-b1-permute-block

This directory stores block-attention analysis for the checkpoint
`/home/devbox/project/AOGPT-test-order/nanogpt_learned_order/out/base/permute/seq256/block1/out-wikitext103-seq256-random-b1-permute-block/ckpt.pt` evaluated with `mode=Random`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each top-level group:

- `reveal/`: reveal-order block coordinates in the current permuted input frame
- `current_original/`: reordered back to the original block order of the current permuted input frame
- `true_original/`: reordered further back to the true unpermuted data block order
