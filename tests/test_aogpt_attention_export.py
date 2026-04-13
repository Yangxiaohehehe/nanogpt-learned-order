import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AOGPT import AOGPT, AOGPTConfig


def test_return_attentions_with_manual_backend():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
            block_order_block_len=2,
            force_manual_attention=True,
        )
    )
    inputs = torch.randint(0, 32, (1, 8))

    outputs = model(inputs, mode="AR", return_attentions=True)

    logits, loss, attentions = outputs
    assert logits.shape == (1, 9, 32)
    assert loss.ndim == 0
    assert len(attentions) == 2
    for layer_att in attentions:
        assert layer_att.shape == (1, 2, 9, 9)
        assert torch.isfinite(layer_att).all()
