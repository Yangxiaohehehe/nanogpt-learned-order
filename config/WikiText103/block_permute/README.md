This directory stores config presets for block-level permutation experiments.

These configs assume:

- `permute_data = True`
- `permute_mode = 'block'`

The goal is to keep each block internally contiguous while applying a fixed
permutation over block identities / positions across the whole sequence.

Suggested usage:

- start with `block32_random_permute.py`
- then run `block32_segment_curriculum_permute.py`

If needed, additional block16 / block64 presets can be added here with the
same naming pattern.
