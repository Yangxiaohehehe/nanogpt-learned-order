# block_attention_ar_all

This directory stores block-attention analysis for the `out-wikitext103-random-b64/ckpt.pt`
checkpoint evaluated with `mode=AR`.

Subdirectories:

- `with_none/`: predictor-aligned block attention, including `[None]`
- `without_none/`: real-token-only block attention, excluding `[None]`
- `diff/`: `with_none - without_none`

Because `mode=AR` uses the identity block order, `reveal` and `original` are effectively the
same view here. To reduce clutter, only one canonical copy is kept in each subdirectory:

- `block_attention_<type>.npy`
- `block_attention_<type>.png`
- `block_attention_<type>_metadata.json`
