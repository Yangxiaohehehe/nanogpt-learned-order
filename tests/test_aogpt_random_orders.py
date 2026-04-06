import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AOGPT import AOGPT, AOGPTConfig


def test_sample_random_orders_returns_tensor_with_expected_shape():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=True,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))

    orders = model.sample_random_orders(inputs)

    assert orders.shape == inputs.shape
    assert orders.device == inputs.device
    expected = torch.arange(inputs.shape[1])
    for row in orders:
        assert torch.equal(torch.sort(row).values, expected)


def test_learned_mode_returns_valid_permutations_and_order_stats():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=True,
            learned_order=True,
            learned_order_noise_scale=0.0,
            learned_order_warmup_steps=0,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))

    logits, loss, order_info = model(
        inputs,
        mode="Learned",
        learned_step=10,
        return_order_info=True,
    )

    assert logits.shape == (2, 9, 32)
    assert loss.ndim == 0
    assert order_info["orders"].shape == inputs.shape
    expected = torch.arange(inputs.shape[1], device=inputs.device)
    for row in order_info["orders"]:
        assert torch.equal(torch.sort(row).values, expected)
    assert order_info["priority_weights"].shape == inputs.shape
    assert order_info["warmup_active"] is False
    assert order_info["loss_unweighted"] >= 0.0
    assert order_info["loss_weighted"] >= 0.0


def test_learned_mode_marks_warmup_random_fallback():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=True,
            learned_order=True,
            learned_order_warmup_steps=50,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))

    _, _, order_info = model(
        inputs,
        mode="Learned",
        learned_step=0,
        return_order_info=True,
    )

    assert order_info["warmup_active"] is True
    assert order_info["orders"].shape == inputs.shape


def test_learned_mode_without_weighting_matches_unweighted_loss():
    model = AOGPT(
        AOGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=True,
            learned_order=True,
            learned_order_warmup_steps=0,
            learned_order_loss_weighting=False,
        )
    )
    inputs = torch.randint(0, 32, (2, 8))

    _, _, order_info = model(
        inputs,
        mode="Learned",
        learned_step=10,
        return_order_info=True,
    )

    assert order_info["loss_weighted"] == order_info["loss_unweighted"]
