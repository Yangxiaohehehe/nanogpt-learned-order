# seq256

This directory stores Hugging Face GPT-style AR attention exports.

- `model_name`: `openai-community/gpt2`
- `dataset`: `wikitext103`
- `split`: `val`
- `predictor_len`: `256`
- `layer_reduce`: `last`
- `head_reduce`: `mean`

Important comparison note:

- Hugging Face GPT models do not contain an explicit AO-GPT-style `[None]` predictor token
- `with_none/` is exported as a predictor-aligned proxy using `attn[:-1, :-1]`
- `without_none/` is exported as a real-token-only proxy using `attn[1:, 1:]`
- there is no exact `diff = with_none - without_none` analogue here

Available setup labels:

- `block1`: num_blocks=256, block_len=1
- `block16`: num_blocks=16, block_len=16
- `block32`: num_blocks=32, block_len=8
- `block64`: num_blocks=64, block_len=4
- `block128`: num_blocks=128, block_len=2
