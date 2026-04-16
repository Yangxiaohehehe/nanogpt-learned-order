from dataclasses import dataclass

from AOGPT_block import AOGPT as BlockAOGPT
from AOGPT_token import AOGPT as TokenAOGPT


@dataclass
class AOGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    block_order_block_len: int = 16
    order_impl: str = "block"
    force_manual_attention: bool = False


def AOGPT(config):
    order_impl = str(getattr(config, "order_impl", "block"))
    if order_impl == "token":
        return TokenAOGPT(config)
    if order_impl == "block":
        return BlockAOGPT(config)
    raise ValueError(f"Unsupported order_impl={order_impl!r}. Expected 'block' or 'token'.")
