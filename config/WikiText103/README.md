# WikiText103 Config Layout

The active layout is now organized first by sequence length, then by whether
the input data is permuted, and finally by the number of reveal blocks:

- `seq256/non_permute/block16/`
- `seq256/non_permute/block32/`
- `seq256/non_permute/block64/`
- `seq256/non_permute/block128/`
- `seq256/permute/block16/`
- `seq256/permute/block32/`
- ...
- `seq512/non_permute/block16/`
- `seq512/permute/block128/`

Inside each block directory, the standard entry points are:

- `random.py`
- `ar.py`
- `segment_curriculum.py`

These are all standalone config files, so each one can be customized
independently for block-specific or run-specific settings such as W&B names.

Legacy paths such as `block16/standard/`, `block32/standard/`, and
`block_permute/` are kept for backward compatibility with existing commands
and historical reports.
