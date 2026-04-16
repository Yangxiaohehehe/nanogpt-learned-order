import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
    token_losses_to_block_losses,
)
from path_layout import default_eval_out_dir


@dataclass
class EvalConfig:
    ckpt_path: Path
    out_dir: Optional[Path]
    data_dir: Optional[Path] = None
    dataset: Optional[str] = None
    split: str = "val"
    batch_size: int = 8
    block_size: Optional[int] = None
    num_batches: int = 100
    num_random_seeds: int = 5
    swap_pairs: List[Tuple[int, int]] = field(default_factory=list)
    seed: int = 12345
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"


def load_checkpoint(ckpt_path: Path, device: str) -> Dict:
    return torch.load(ckpt_path, map_location=device)


def load_model(ckpt_path: Path, device: str):
    checkpoint = load_checkpoint(ckpt_path, device)
    model_config = checkpoint["model_args"]
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("model_type", "aogpt") != "aogpt":
        raise ValueError("eval_ckpt_modes.py currently supports only AO-GPT checkpoints.")

    model = AOGPT(AOGPTConfig(**model_config))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def resolve_data_dir(config: EvalConfig, checkpoint: Dict) -> Path:
    if config.data_dir is not None:
        return config.data_dir

    dataset = config.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not determine dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / dataset


def resolve_block_size(config: EvalConfig, checkpoint: Dict) -> int:
    if config.block_size is not None:
        return config.block_size
    return int(checkpoint["model_args"]["block_size"])


def load_split_tokens(data_dir: Path, split: str) -> np.memmap:
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def load_token_permutation_from_checkpoint(checkpoint: Dict, block_size: int):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation")
    block_len = int(config.get("block_order_block_len", 1))
    num_blocks = block_size // block_len
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)
    inverse_token_perm = block_permutation_to_token_permutation(inverse_block_perm, block_len=block_len)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
        "inverse_token_perm": inverse_token_perm,
    }


def make_batch_sampler(tokens: np.memmap, batch_size: int, block_size: int, seed: int, token_perm: Optional[torch.Tensor] = None):
    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than block_size.")
    rng = np.random.default_rng(seed)

    def sample(device: Optional[str] = None) -> torch.Tensor:
        starts = rng.integers(0, max_start, size=batch_size)
        batch = torch.stack(
            [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
        )
        if token_perm is not None:
            batch = batch[:, token_perm]
        if device is not None:
            batch = batch.to(device)
            return batch
        return batch

    return sample


def build_block_orders(
    batch_size: int,
    num_blocks: int,
    device: torch.device,
    mode: str,
    generator: torch.Generator,
    swap_pair: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if mode == "AR":
        return torch.arange(num_blocks, device=device).unsqueeze(0).expand(batch_size, -1)
    if mode == "Swap":
        if swap_pair is None:
            raise ValueError("Swap mode requires a swap_pair.")
        pos_a, pos_b = swap_pair
        if not (1 <= pos_a <= num_blocks and 1 <= pos_b <= num_blocks):
            raise ValueError(f"Swap positions must be within [1, {num_blocks}], got {swap_pair}.")
        orders = torch.arange(num_blocks, device=device).unsqueeze(0).expand(batch_size, -1).clone()
        idx_a = pos_a - 1
        idx_b = pos_b - 1
        orders[:, [idx_a, idx_b]] = orders[:, [idx_b, idx_a]]
        return orders
    if mode == "Random":
        return torch.stack(
            [torch.randperm(num_blocks, generator=generator, device=device) for _ in range(batch_size)]
        )
    raise ValueError(f"Unsupported mode: {mode}")


def build_token_orders_from_block_orders(block_orders: torch.Tensor, block_len: int) -> torch.Tensor:
    return expand_block_orders_to_token_orders(block_orders.long(), block_len=block_len)


@torch.no_grad()
def compute_per_step_curve(
    model: AOGPT,
    idx: torch.Tensor,
    mode: str,
    generator: torch.Generator,
    autocast_context,
    swap_pair: Optional[Tuple[int, int]] = None,
    orders: Optional[torch.Tensor] = None,
    block_orders: Optional[torch.Tensor] = None,
) -> np.ndarray:
    if orders is None:
        if block_orders is None:
            block_orders = build_block_orders(
                idx.size(0),
                model.num_blocks,
                idx.device,
                mode,
                generator,
                swap_pair=swap_pair,
            )
        orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
    with autocast_context:
        logits, _ = model(idx, mode=None, orders=orders)
    targets = model.shuffle(idx, orders)
    token_losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).view(targets.size(0), targets.size(1))
    block_losses = token_losses_to_block_losses(token_losses, block_len=model.block_order_block_len)
    return block_losses.mean(dim=0).float().cpu().numpy()


@torch.no_grad()
def compute_per_step_stats(
    model: AOGPT,
    idx: torch.Tensor,
    mode: str,
    generator: torch.Generator,
    autocast_context,
    swap_pair: Optional[Tuple[int, int]] = None,
    orders: Optional[torch.Tensor] = None,
    block_orders: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    if orders is None:
        if block_orders is None:
            block_orders = build_block_orders(
                idx.size(0),
                model.num_blocks,
                idx.device,
                mode,
                generator,
                swap_pair=swap_pair,
            )
        orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
    with autocast_context:
        logits, _ = model(idx, mode=None, orders=orders)

    targets = model.shuffle(idx, orders)
    step_logits = logits[:, :-1, :]
    log_probs = F.log_softmax(step_logits, dim=-1)
    probs = log_probs.exp()

    token_losses = F.cross_entropy(
        step_logits.reshape(-1, step_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).view(targets.size(0), targets.size(1))
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_probs = target_log_probs.exp()
    max_probs = probs.max(dim=-1).values
    entropy = -(probs * log_probs).sum(dim=-1)

    def pool_metric(values: torch.Tensor) -> np.ndarray:
        pooled = values.float().view(values.size(0), model.num_blocks, model.block_order_block_len).mean(dim=-1)
        return pooled.mean(dim=0).cpu().numpy()

    return {
        "loss": pool_metric(token_losses),
        "target_prob": pool_metric(target_probs),
        "max_prob": pool_metric(max_probs),
        "entropy": pool_metric(entropy),
    }


def get_autocast_context(config: EvalConfig):
    if "cuda" not in config.device or config.dtype == "float32":
        return nullcontext()
    ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config.dtype]
    return torch.amp.autocast(device_type="cuda", dtype=ptdtype)


def aggregate_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(curves)
    return stacked.mean(axis=0), stacked.std(axis=0)


def aggregate_metric_curves(metric_runs: List[Dict[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    metric_names = metric_runs[0].keys()
    return {
        metric_name: aggregate_curves([run[metric_name] for run in metric_runs])
        for metric_name in metric_names
    }


def count_inversions(values: List[int]) -> int:
    def sort_count(items):
        if len(items) <= 1:
            return items, 0

        mid = len(items) // 2
        left, left_inv = sort_count(items[:mid])
        right, right_inv = sort_count(items[mid:])

        merged = []
        i = 0
        j = 0
        split_inv = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                split_inv += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, left_inv + right_inv + split_inv

    _, inversions = sort_count(list(values))
    return inversions


def kendall_tau_to_ar(order: np.ndarray) -> float:
    n = len(order)
    if n < 2:
        return 1.0
    inversions = count_inversions(order.tolist())
    return 1.0 - (4.0 * inversions) / (n * (n - 1))


def summarize_kendall_taus(taus: List[float], order_type: str) -> Dict[str, float]:
    tau_array = np.asarray(taus, dtype=np.float64)
    return {
        "order_type": order_type,
        "kendall_tau_mean": float(tau_array.mean()),
        "kendall_tau_std": float(tau_array.std()),
        "kendall_tau_min": float(tau_array.min()),
        "kendall_tau_max": float(tau_array.max()),
        "num_samples": int(len(tau_array)),
    }


def derive_random_seeds(seed: int, count: int) -> List[int]:
    rng = np.random.default_rng(seed)
    seeds = []
    seen = set()
    while len(seeds) < count:
        candidate = int(rng.integers(0, np.iinfo(np.int32).max))
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return seeds


def normalize_swap_pairs(swap_pairs: List[Tuple[int, int]], num_blocks: int) -> List[Tuple[int, int]]:
    normalized = []
    seen = set()
    for pair in swap_pairs:
        if len(pair) != 2:
            raise ValueError(f"Each swap pair must contain exactly 2 positions, got {pair}.")
        pos_a, pos_b = int(pair[0]), int(pair[1])
        if pos_a == pos_b:
            raise ValueError(f"Swap positions must be different, got {pair}.")
        if not (1 <= pos_a <= num_blocks and 1 <= pos_b <= num_blocks):
            raise ValueError(f"Swap positions must be within [1, {num_blocks}], got {pair}.")
        key = (pos_a, pos_b)
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def swap_label(swap_pair: Tuple[int, int]) -> str:
    return f"swap_{swap_pair[0]}_{swap_pair[1]}"


def save_csv(
    out_dir: Path,
    ar_mean,
    ar_std,
    swap_curves,
    random_seed_curves,
    random_mean,
    random_std,
    original_ar_mean=None,
    original_ar_std=None,
):
    csv_path = out_dir / "per_step_loss.csv"
    swap_columns = list(swap_curves.keys())
    seed_columns = [f"random_seed_{seed}" for seed in random_seed_curves]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["block_position", "ar_mean", "ar_std"]
        if original_ar_mean is not None:
            header.extend(["original_position_ar_mean", "original_position_ar_std"])
        header.extend([*swap_columns, *seed_columns, "random_mean", "random_std"])
        writer.writerow(header)
        for position in range(len(ar_mean)):
            row = [position, float(ar_mean[position]), float(ar_std[position])]
            if original_ar_mean is not None:
                row.extend([float(original_ar_mean[position]), float(original_ar_std[position])])
            row.extend(
                [
                    *[float(swap_curves[label][position]) for label in swap_columns],
                    *[float(random_seed_curves[seed][position]) for seed in random_seed_curves],
                    float(random_mean[position]),
                    float(random_std[position]),
                ]
            )
            writer.writerow(row)
    return csv_path


def save_metric_csv(
    out_dir: Path,
    metric_name: str,
    ar_mean,
    ar_std,
    swap_curves,
    random_seed_curves,
    random_mean,
    random_std,
    original_ar_mean=None,
    original_ar_std=None,
):
    csv_path = out_dir / f"per_step_{metric_name}.csv"
    swap_columns = list(swap_curves.keys())
    seed_columns = [f"random_seed_{seed}" for seed in random_seed_curves]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["block_position", "ar_mean", "ar_std"]
        if original_ar_mean is not None:
            header.extend(["original_position_ar_mean", "original_position_ar_std"])
        header.extend([*swap_columns, *seed_columns, "random_mean", "random_std"])
        writer.writerow(header)
        for position in range(len(ar_mean)):
            row = [position, float(ar_mean[position]), float(ar_std[position])]
            if original_ar_mean is not None:
                row.extend([float(original_ar_mean[position]), float(original_ar_std[position])])
            row.extend(
                [
                    *[float(swap_curves[label][position]) for label in swap_columns],
                    *[float(random_seed_curves[seed][position]) for seed in random_seed_curves],
                    float(random_mean[position]),
                    float(random_std[position]),
                ]
            )
            writer.writerow(row)
    return csv_path


def plot_curves(
    out_dir: Path,
    ar_mean,
    ar_std,
    swap_curves,
    random_seed_curves,
    random_mean,
    random_std,
    title: str,
    original_ar_mean=None,
    original_ar_std=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ar_mean))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, ar_mean, label="Current Position L2R / AR", linewidth=2)
    if original_ar_mean is not None:
        ax.plot(x, original_ar_mean, label="Original Position L2R", linewidth=2, linestyle=":")
    for label, curve in swap_curves.items():
        ax.plot(x, curve, linewidth=1.75, alpha=0.9, linestyle="--", label=label)
    for seed, curve in random_seed_curves.items():
        ax.plot(x, curve, linewidth=1.5, alpha=0.8, label=f"Random (seed={seed})")
    ax.set_xlabel("Block Reveal Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_step_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, ar_mean, label="Current Position L2R / AR", linewidth=2)
    ax.fill_between(x, ar_mean - ar_std, ar_mean + ar_std, alpha=0.2)
    if original_ar_mean is not None:
        ax.plot(x, original_ar_mean, label="Original Position L2R", linewidth=2, linestyle=":")
        ax.fill_between(x, original_ar_mean - original_ar_std, original_ar_mean + original_ar_std, alpha=0.15)
    for label, curve in swap_curves.items():
        ax.plot(x, curve, linewidth=1.5, linestyle="--", label=label)
    ax.plot(x, random_mean, label="Random mean", linewidth=2)
    ax.fill_between(x, random_mean - random_std, random_mean + random_std, alpha=0.2)
    ax.set_xlabel("Block Reveal Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{title} (AR batch std, Random seed std)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_step_loss_with_band.png", dpi=180)
    plt.close(fig)


def plot_metric_curves(
    out_dir: Path,
    metric_name: str,
    metric_label: str,
    ar_mean,
    ar_std,
    swap_curves,
    random_seed_curves,
    random_mean,
    random_std,
    title: str,
    original_ar_mean=None,
    original_ar_std=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ar_mean))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, ar_mean, label="Current Position L2R / AR", linewidth=2)
    if original_ar_mean is not None:
        ax.plot(x, original_ar_mean, label="Original Position L2R", linewidth=2, linestyle=":")
    for label, curve in swap_curves.items():
        ax.plot(x, curve, linewidth=1.75, alpha=0.9, linestyle="--", label=label)
    for seed, curve in random_seed_curves.items():
        ax.plot(x, curve, linewidth=1.5, alpha=0.8, label=f"Random (seed={seed})")
    ax.set_xlabel("Block Reveal Step")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"per_step_{metric_name}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, ar_mean, label="Current Position L2R / AR", linewidth=2)
    ax.fill_between(x, ar_mean - ar_std, ar_mean + ar_std, alpha=0.2)
    if original_ar_mean is not None:
        ax.plot(x, original_ar_mean, label="Original Position L2R", linewidth=2, linestyle=":")
        ax.fill_between(x, original_ar_mean - original_ar_std, original_ar_mean + original_ar_std, alpha=0.15)
    for label, curve in swap_curves.items():
        ax.plot(x, curve, linewidth=1.5, linestyle="--", label=label)
    ax.plot(x, random_mean, label="Random mean", linewidth=2)
    ax.fill_between(x, random_mean - random_std, random_mean + random_std, alpha=0.2)
    ax.set_xlabel("Block Reveal Step")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{title} (AR batch std, Random seed std)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"per_step_{metric_name}_with_band.png", dpi=180)
    plt.close(fig)


def save_summary(out_dir: Path, config: EvalConfig, checkpoint: Dict, random_seeds, ar_mean, random_mean, original_ar_mean=None):
    modes = ["AR"]
    if config.swap_pairs:
        modes.append("Swap")
    modes.append("Random")
    summary = {
        "ckpt_path": str(config.ckpt_path),
        "out_dir": str(config.out_dir),
        "dataset": config.dataset or checkpoint.get("config", {}).get("dataset"),
        "split": config.split,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "num_batches": config.num_batches,
        "num_random_seeds": config.num_random_seeds,
        "swap_pairs": [list(pair) for pair in config.swap_pairs],
        "seed": config.seed,
        "random_seeds": random_seeds,
        "device": config.device,
        "dtype": config.dtype,
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "modes": modes,
        "ar_curve_mean": float(np.mean(ar_mean)),
        "original_position_ar_curve_mean": None if original_ar_mean is None else float(np.mean(original_ar_mean)),
        "random_curve_mean": float(np.mean(random_mean)),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def save_order_distance_csv(out_dir: Path, order_distance_to_ar: Dict[str, Dict[str, float]]):
    csv_path = out_dir / "order_distance_to_ar.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_name",
                "order_type",
                "kendall_tau_mean",
                "kendall_tau_std",
                "kendall_tau_min",
                "kendall_tau_max",
                "num_samples",
            ]
        )
        for order_name, stats in order_distance_to_ar.items():
            writer.writerow(
                [
                    order_name,
                    stats["order_type"],
                    stats["kendall_tau_mean"],
                    stats["kendall_tau_std"],
                    stats["kendall_tau_min"],
                    stats["kendall_tau_max"],
                    stats["num_samples"],
                ]
            )
    return csv_path


def run_evaluation(config: EvalConfig):
    config.ckpt_path = Path(config.ckpt_path)
    if config.out_dir is None:
        config.out_dir = default_eval_out_dir(REPO_ROOT, config.ckpt_path, "eval_ckpt_modes")
    else:
        config.out_dir = Path(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(config.ckpt_path, config.device)
    config.block_size = resolve_block_size(config, checkpoint)
    config.swap_pairs = normalize_swap_pairs(config.swap_pairs, model.num_blocks)
    data_dir = resolve_data_dir(config, checkpoint)
    tokens = load_split_tokens(data_dir, config.split)
    permutation_state = load_token_permutation_from_checkpoint(checkpoint, config.block_size)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    inverse_block_perm = None if permutation_state is None else permutation_state["inverse_block_perm"].to(config.device)
    sample_batch = make_batch_sampler(tokens, config.batch_size, config.block_size, config.seed, token_perm=token_perm)
    autocast_context = get_autocast_context(config)
    batches = [sample_batch() for _ in range(config.num_batches)]
    total_samples = config.num_batches * config.batch_size

    ar_metric_runs = []
    ar_generator = torch.Generator(device="cuda" if "cuda" in config.device else "cpu")
    ar_generator.manual_seed(config.seed)
    for batch in batches:
        batch_device = batch.to(config.device)
        block_orders = build_block_orders(
            batch_device.size(0),
            model.num_blocks,
            batch_device.device,
            "AR",
            ar_generator,
        )
        orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
        ar_metric_runs.append(
            compute_per_step_stats(
                model,
                batch_device,
                "AR",
                ar_generator,
                autocast_context,
                orders=orders,
                block_orders=block_orders,
            )
        )

    ar_metrics = aggregate_metric_curves(ar_metric_runs)
    ar_mean, ar_std = ar_metrics["loss"]
    order_distance_to_ar = {
        "AR": summarize_kendall_taus([1.0] * total_samples, "AR"),
    }
    original_ar_metrics = None
    if inverse_block_perm is not None:
        original_ar_metric_runs = []
        original_tau = kendall_tau_to_ar(inverse_block_perm.detach().cpu().numpy())
        for batch in batches:
            batch_device = batch.to(config.device)
            block_orders = inverse_block_perm.unsqueeze(0).expand(batch_device.size(0), -1)
            orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
            original_ar_metric_runs.append(
                compute_per_step_stats(
                    model,
                    batch_device,
                    "AR",
                    ar_generator,
                    autocast_context,
                    orders=orders,
                    block_orders=block_orders,
                )
            )
        original_ar_metrics = aggregate_metric_curves(original_ar_metric_runs)
        order_distance_to_ar["original_position_AR"] = summarize_kendall_taus(
            [original_tau] * total_samples,
            "OriginalPositionAR",
        )
    swap_curves = {}
    swap_metric_curves = {"target_prob": {}, "max_prob": {}, "entropy": {}}
    for pair in config.swap_pairs:
        pair_metric_runs = []
        swap_generator = torch.Generator(device="cuda" if "cuda" in config.device else "cpu")
        swap_generator.manual_seed(config.seed)
        for batch in batches:
            batch_device = batch.to(config.device)
            block_orders = build_block_orders(
                batch_device.size(0),
                model.num_blocks,
                batch_device.device,
                "Swap",
                swap_generator,
                swap_pair=pair,
            )
            orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
            pair_metric_runs.append(
                compute_per_step_stats(
                    model,
                    batch_device,
                    "Swap",
                    swap_generator,
                    autocast_context,
                    swap_pair=pair,
                    orders=orders,
                    block_orders=block_orders,
                )
            )
        aggregated_pair_metrics = aggregate_metric_curves(pair_metric_runs)
        swap_curves[swap_label(pair)] = aggregated_pair_metrics["loss"][0]
        swap_metric_curves["target_prob"][swap_label(pair)] = aggregated_pair_metrics["target_prob"][0]
        swap_metric_curves["max_prob"][swap_label(pair)] = aggregated_pair_metrics["max_prob"][0]
        swap_metric_curves["entropy"][swap_label(pair)] = aggregated_pair_metrics["entropy"][0]
        swap_order = build_block_orders(
            batches[0].size(0),
            model.num_blocks,
            batches[0].to(config.device).device,
            "Swap",
            swap_generator,
            swap_pair=pair,
        )[0].detach().cpu().numpy()
        swap_tau = kendall_tau_to_ar(swap_order)
        order_distance_to_ar[swap_label(pair)] = summarize_kendall_taus([swap_tau] * total_samples, "Swap")

    random_seeds = derive_random_seeds(config.seed, config.num_random_seeds)
    random_seed_curves = {}
    random_metric_curves = {"target_prob": {}, "max_prob": {}, "entropy": {}}
    random_seed_stats = {}
    for random_seed in random_seeds:
        generator = torch.Generator(device="cuda" if "cuda" in config.device else "cpu")
        generator.manual_seed(random_seed)
        seed_metric_runs = []
        seed_taus = []
        for batch in batches:
            batch_device = batch.to(config.device)
            block_orders = build_block_orders(
                batch_device.size(0),
                model.num_blocks,
                batch_device.device,
                "Random",
                generator,
            )
            orders = build_token_orders_from_block_orders(block_orders, model.block_order_block_len)
            seed_metric_runs.append(
                compute_per_step_stats(
                    model,
                    batch_device,
                    "Random",
                    generator,
                    autocast_context,
                    orders=orders,
                    block_orders=block_orders,
                )
            )
            seed_taus.extend([kendall_tau_to_ar(order) for order in block_orders.detach().cpu().numpy()])
        aggregated_seed_metrics = aggregate_metric_curves(seed_metric_runs)
        random_seed_curves[random_seed] = aggregated_seed_metrics["loss"][0]
        random_metric_curves["target_prob"][random_seed] = aggregated_seed_metrics["target_prob"][0]
        random_metric_curves["max_prob"][random_seed] = aggregated_seed_metrics["max_prob"][0]
        random_metric_curves["entropy"][random_seed] = aggregated_seed_metrics["entropy"][0]
        stat_key = f"random_seed_{random_seed}"
        random_seed_stats[stat_key] = summarize_kendall_taus(seed_taus, "Random")
        order_distance_to_ar[stat_key] = random_seed_stats[stat_key]

    random_mean, random_std = aggregate_curves(list(random_seed_curves.values()))
    target_prob_mean, target_prob_std = aggregate_curves(list(random_metric_curves["target_prob"].values()))
    max_prob_mean, max_prob_std = aggregate_curves(list(random_metric_curves["max_prob"].values()))
    entropy_mean, entropy_std = aggregate_curves(list(random_metric_curves["entropy"].values()))
    order_distance_to_ar["random_seed_stats"] = {
        "order_type": "Random",
        "kendall_tau_mean": float(
            np.mean([stats["kendall_tau_mean"] for stats in random_seed_stats.values()])
        ),
        "kendall_tau_std": float(
            np.std([stats["kendall_tau_mean"] for stats in random_seed_stats.values()])
        ),
        "kendall_tau_min": float(
            np.min([stats["kendall_tau_mean"] for stats in random_seed_stats.values()])
        ),
        "kendall_tau_max": float(
            np.max([stats["kendall_tau_mean"] for stats in random_seed_stats.values()])
        ),
        "num_samples": int(total_samples * len(random_seed_stats)),
    }

    save_csv(
        config.out_dir,
        ar_mean,
        ar_std,
        swap_curves,
        random_seed_curves,
        random_mean,
        random_std,
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["loss"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["loss"][1],
    )
    save_order_distance_csv(config.out_dir, order_distance_to_ar)
    plot_curves(
        config.out_dir,
        ar_mean,
        ar_std,
        swap_curves,
        random_seed_curves,
        random_mean,
        random_std,
        title=f"Per-Step Loss Comparison ({config.num_batches} batches, {config.num_random_seeds} random seeds)",
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["loss"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["loss"][1],
    )
    save_metric_csv(
        config.out_dir,
        "target_prob",
        ar_metrics["target_prob"][0],
        ar_metrics["target_prob"][1],
        swap_metric_curves["target_prob"],
        random_metric_curves["target_prob"],
        target_prob_mean,
        target_prob_std,
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["target_prob"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["target_prob"][1],
    )
    save_metric_csv(
        config.out_dir,
        "max_prob",
        ar_metrics["max_prob"][0],
        ar_metrics["max_prob"][1],
        swap_metric_curves["max_prob"],
        random_metric_curves["max_prob"],
        max_prob_mean,
        max_prob_std,
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["max_prob"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["max_prob"][1],
    )
    save_metric_csv(
        config.out_dir,
        "entropy",
        ar_metrics["entropy"][0],
        ar_metrics["entropy"][1],
        swap_metric_curves["entropy"],
        random_metric_curves["entropy"],
        entropy_mean,
        entropy_std,
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["entropy"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["entropy"][1],
    )
    plot_metric_curves(
        config.out_dir,
        "target_prob",
        "Average True-Token Probability",
        ar_metrics["target_prob"][0],
        ar_metrics["target_prob"][1],
        swap_metric_curves["target_prob"],
        random_metric_curves["target_prob"],
        target_prob_mean,
        target_prob_std,
        title=f"Per-Block True-Token Probability ({config.num_batches} batches, {config.num_random_seeds} random seeds)",
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["target_prob"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["target_prob"][1],
    )
    plot_metric_curves(
        config.out_dir,
        "max_prob",
        "Average Max Probability",
        ar_metrics["max_prob"][0],
        ar_metrics["max_prob"][1],
        swap_metric_curves["max_prob"],
        random_metric_curves["max_prob"],
        max_prob_mean,
        max_prob_std,
        title=f"Per-Block Max Probability ({config.num_batches} batches, {config.num_random_seeds} random seeds)",
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["max_prob"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["max_prob"][1],
    )
    plot_metric_curves(
        config.out_dir,
        "entropy",
        "Average Predictive Entropy",
        ar_metrics["entropy"][0],
        ar_metrics["entropy"][1],
        swap_metric_curves["entropy"],
        random_metric_curves["entropy"],
        entropy_mean,
        entropy_std,
        title=f"Per-Block Predictive Entropy ({config.num_batches} batches, {config.num_random_seeds} random seeds)",
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["entropy"][0],
        original_ar_std=None if original_ar_metrics is None else original_ar_metrics["entropy"][1],
    )
    summary_path = save_summary(
        config.out_dir,
        config,
        checkpoint,
        random_seeds,
        ar_mean,
        random_mean,
        original_ar_mean=None if original_ar_metrics is None else original_ar_metrics["loss"][0],
    )
    summary = json.loads(summary_path.read_text())
    summary["order_distance_to_ar"] = order_distance_to_ar
    summary["target_prob_curve_mean"] = float(np.mean(ar_metrics["target_prob"][0]))
    summary["random_target_prob_curve_mean"] = float(np.mean(target_prob_mean))
    summary["max_prob_curve_mean"] = float(np.mean(ar_metrics["max_prob"][0]))
    summary["random_max_prob_curve_mean"] = float(np.mean(max_prob_mean))
    summary["entropy_curve_mean"] = float(np.mean(ar_metrics["entropy"][0]))
    summary["random_entropy_curve_mean"] = float(np.mean(entropy_mean))
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "ar_mean": ar_mean,
        "ar_std": ar_std,
        "swap_curves": swap_curves,
        "random_seed_curves": random_seed_curves,
        "random_mean": random_mean,
        "random_std": random_std,
        "ar_metrics": ar_metrics,
        "swap_metric_curves": swap_metric_curves,
        "random_metric_curves": random_metric_curves,
        "random_target_prob_mean": target_prob_mean,
        "random_target_prob_std": target_prob_std,
        "random_max_prob_mean": max_prob_mean,
        "random_max_prob_std": max_prob_std,
        "random_entropy_mean": entropy_mean,
        "random_entropy_std": entropy_std,
        "order_distance_to_ar": order_distance_to_ar,
        "original_ar_metrics": original_ar_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AO-GPT checkpoint under AR and Random generation orders.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--num_random_seeds", type=int, default=5)
    parser.add_argument("--swap_pairs", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parsed = vars(parser.parse_args())
    raw_swap_pairs = parsed.pop("swap_pairs") or []
    if len(raw_swap_pairs) % 2 != 0:
        raise ValueError("--swap_pairs requires an even number of integers, e.g. --swap_pairs 1 5 2 6")
    parsed["swap_pairs"] = list(zip(raw_swap_pairs[::2], raw_swap_pairs[1::2]))
    return EvalConfig(**parsed)


def main():
    config = parse_args()
    result = run_evaluation(config)
    print(
        json.dumps(
            {
                "ar_curve_mean": float(np.mean(result["ar_mean"])),
                "random_curve_mean": float(np.mean(result["random_mean"])),
                "out_dir": str(config.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
