# out-wikitext103-seq256-random-b32

This directory stores block-attention analysis for the checkpoint
`/home/devbox/project/AOGPT-test-order/nanogpt_learned_order/out/base/nonpermute/seq256/block32/out-wikitext103-seq256-random-b32/ckpt.pt` evaluated with `mode=Random`.

Top-level groups:

- `with_none/`: predictor-aligned block attention including `[None]`
- `without_none/`: real-token-only block attention excluding `[None]`
- `diff/`: `with_none - without_none`

Inside each top-level group:

- `random/`: reveal-order view from a single Random rollout average
- `current_l2r/`: the same Random-rollout matrix reordered back to current l2r order

Important note:

- this does not run a second l2r forward pass
- `current_l2r/` is only a coordinate remapping of the Random attention result
