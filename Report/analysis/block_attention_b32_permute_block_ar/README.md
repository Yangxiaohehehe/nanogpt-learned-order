# block_attention_b32_permute_block_ar

This directory stores block-attention analysis for the permuted-data checkpoint
`out-wikitext103-random-b32-permute-block-base/ckpt.pt` evaluated with `mode=AR`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each group:

- `current_frame/`: canonical current-frame view kept from the `reveal` export
- `true_original/`: block matrix mapped back to the true unpermuted data block order

`original` in the current permuted input frame is omitted here because under `mode=AR` it is
effectively redundant with the canonical current-frame view for quick inspection.
