# out-wikitext103-seq256-random-b64_mode-ar

This directory stores block-attention analysis for the checkpoint
`out/base/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64/ckpt.pt` evaluated with `mode=AR`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each top-level group:

- files are exported directly at that level because current-frame reveal order is the canonical view
