# Report Index

This file is a lightweight map of the current report folders.

## Main folders

### `different_AR_RANDOM`
Purpose:
- compare AR and Random trajectories under a fixed checkpoint

Contents:
- per-step loss
- per-step entropy
- target/max probability curves
- summary json

Use when:
- reasoning about why AR differs from Random at the trajectory level

### `signal_benchmark_generated`
Purpose:
- benchmark candidate order signals under `adjacent` search

Use when:
- comparing signal quality with the weakest local-search move set

### `signal_benchmark_insert_front`
Purpose:
- benchmark candidate order signals under `insert_front` moves

Use when:
- checking whether explicitly front-loading moves help l2r recovery

### `signal_benchmark_insert_anywhere`
Purpose:
- benchmark candidate order signals under `insert_anywhere` moves

Use when:
- checking whether larger action spaces reveal stronger useful signals

### `ar_likeness_benchmark_random_only_b200_mean`
Purpose:
- evaluate AR-likeness score components under a fixed Random checkpoint
- compare `mode=AR` vs `mode=Random`

Important note:
- this is evaluation-only
- these scores are not training targets

Use when:
- validating whether trajectory-level statistics can distinguish AR-like behavior under the same backbone

### `swap`
Purpose:
- export local swap evaluation csv

## Reading priority

If you are actively working on signal design, the most relevant folders are:

1. `different_AR_RANDOM`
2. `signal_benchmark_insert_anywhere`
3. `ar_likeness_benchmark_random_only_b200_mean`

If you are actively working on training pipelines, the most relevant inputs are not here, but in:
- `config/WikiText103/`
- `out-wikitext103-random/`
- `out-wikitext103-ar/`
