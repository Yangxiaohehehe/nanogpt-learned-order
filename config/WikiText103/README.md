# WikiText103 Config Layout

This directory is organized by block granularity and stage:

- `block16/`: 16 blocks total, `block_order_block_len = 16`
- `block32/`: 32 blocks total, `block_order_block_len = 8`
- `block64/`: 64 blocks total, `block_order_block_len = 4`
- `legacy/`: older exploratory configs that do not fit the main block-order pipeline

Inside each block directory:

- `standard/`: stage-1 backbone training configs
- `policy/`: order-head / policy-backbone / joint configs
- `eval/`: evaluation-only configs such as local swap search

Compatibility note:

The top-level files such as `random.py`, `random_b32.py`, and
`prefix_policy_order_head.py` are kept as thin wrappers so existing commands
continue to work.

Recommended direct entry points:

- 16 blocks:
  - `config/WikiText103/block16/standard/random.py`
  - `config/WikiText103/block16/standard/ar.py`
- 32 blocks:
  - `config/WikiText103/block32/standard/random.py`
  - `config/WikiText103/block32/standard/ar.py`
- 64 blocks:
  - `config/WikiText103/block64/standard/random.py`
  - `config/WikiText103/block64/standard/ar.py`
