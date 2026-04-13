# block_attention_b32_permute_block_random

This directory stores block-attention analysis for the permuted-data checkpoint
`out-wikitext103-random-b32-permute-block-base/ckpt.pt` evaluated with `mode=Random`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each group:

- `reveal/`: reveal-order block coordinates in the current permuted input frame
- `current_original/`: reordered back to the original block order of the current permuted input frame
- `true_original/`: reordered further back to the true unpermuted data block order
