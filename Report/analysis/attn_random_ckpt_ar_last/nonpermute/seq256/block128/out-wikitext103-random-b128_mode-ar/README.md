# out-wikitext103-random-b128_mode-ar

This directory stores block-attention analysis for the checkpoint
`out/base/nonpermute/seq256/block128/out-wikitext103-random-b128-attn/ckpt.pt` evaluated with `mode=AR`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each top-level group:

- files are exported directly at that level because current-frame reveal order is the canonical view
