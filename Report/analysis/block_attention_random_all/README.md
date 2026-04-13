# block_attention_random_all

This directory stores block-attention analysis for the `out-wikitext103-random-b64/ckpt.pt`
checkpoint evaluated with `mode=Random`.

Subdirectories:

- `with_none/`: predictor-aligned block attention, including `[None]`
- `without_none/`: real-token-only block attention, excluding `[None]`
- `diff/`: `with_none - without_none`

Each subdirectory keeps both frames:

- `*_reveal.*`: reveal-order block matrix
- `*_original.*`: block matrix reordered back to original block order
