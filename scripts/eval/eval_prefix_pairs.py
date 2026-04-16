import argparse
import csv
import pickle
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import compute_prefix_auc, expand_block_orders_to_token_orders, token_losses_to_block_losses
from path_layout import default_eval_out_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ordered two-block prefixes for an AO-GPT checkpoint."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument(
        "--suffix_mode",
        type=str,
        default="ascending",
        choices=["ascending", "random"],
        help="How to complete the remaining blocks after the first two fixed blocks.",
    )
    parser.add_argument(
        "--random_suffix_samples",
        type=int,
        default=1,
        help="Number of random suffix samples per pair when suffix_mode=random.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    return parser.parse_args()


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def load_checkpoint(ckpt_path: Path, device: str):
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_data_dir(args, checkpoint):
    if args.data_dir is not None:
        return args.data_dir
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str):
    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than block_size.")
    starts = rng.integers(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    )
    return batch.to(device)


def build_pair_block_orders(pair, num_blocks, batch_size, device, suffix_mode, generator=None):
    first, second = pair
    remaining = [idx for idx in range(num_blocks) if idx not in pair]
    if suffix_mode == "ascending":
        suffix = torch.tensor(remaining, dtype=torch.long, device=device)
        full_order = torch.tensor([first, second], dtype=torch.long, device=device)
        full_order = torch.cat([full_order, suffix], dim=0)
        return full_order.unsqueeze(0).expand(batch_size, -1)
    if suffix_mode == "random":
        orders = []
        prefix = torch.tensor([first, second], dtype=torch.long, device=device)
        remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=device)
        for _ in range(batch_size):
            perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
            suffix = remaining_tensor[perm]
            orders.append(torch.cat([prefix, suffix], dim=0))
        return torch.stack(orders, dim=0)
    raise ValueError(f"Unsupported suffix_mode: {suffix_mode}")


@torch.no_grad()
def evaluate_pair(
    model,
    tokens,
    pair,
    batch_size,
    num_batches,
    suffix_mode,
    random_suffix_samples,
    seed,
    device,
    autocast_context,
):
    rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    torch_generator.manual_seed(seed)

    prefix2_values = []
    prefix4_values = []
    full_loss_values = []

    for batch_idx in range(num_batches):
        idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device)
        num_suffix_rollouts = random_suffix_samples if suffix_mode == "random" else 1
        for suffix_sample_idx in range(num_suffix_rollouts):
            if suffix_mode == "random":
                local_seed = seed + batch_idx * 9973 + suffix_sample_idx * 101
                torch_generator.manual_seed(local_seed)
            block_orders = build_pair_block_orders(
                pair=pair,
                num_blocks=model.num_blocks,
                batch_size=idx.size(0),
                device=device,
                suffix_mode=suffix_mode,
                generator=torch_generator,
            )
            token_orders = expand_block_orders_to_token_orders(
                block_orders,
                block_len=model.block_order_block_len,
            )
            with autocast_context:
                _, loss, token_losses = model(
                    idx,
                    mode=None,
                    orders=token_orders,
                    return_token_loss=True,
                )
            block_losses = token_losses_to_block_losses(
                token_losses,
                block_len=model.block_order_block_len,
            )
            prefix2_values.append(float(compute_prefix_auc(block_losses, 2).item()))
            prefix4_values.append(float(compute_prefix_auc(block_losses, min(4, model.num_blocks)).item()))
            full_loss_values.append(float(loss.item()))

    return {
        "prefix2_auc": float(np.mean(prefix2_values)),
        "prefix4_auc": float(np.mean(prefix4_values)),
        "full_loss": float(np.mean(full_loss_values)),
    }


def export_csv(rows, out_csv: Path):
    rows = sorted(rows, key=lambda row: (row["prefix2_auc"], row["prefix4_auc"], row["full_loss"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "first_block",
                "second_block",
                "prefix2_auc",
                "prefix4_auc",
                "full_loss",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    int(rank),
                    int(row["first_block"]),
                    int(row["second_block"]),
                    float(row["prefix2_auc"]),
                    float(row["prefix4_auc"]),
                    float(row["full_loss"]),
                ]
            )


def main():
    args = parse_args()
    if args.out_csv is None:
        args.out_csv = default_eval_out_file(
            REPO_ROOT,
            args.ckpt_path,
            "eval_prefix_pairs",
            f"prefix_pairs_{args.suffix_mode}.csv",
        )
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    _ = data_dir / "meta.pkl"
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)

    rows = []
    for first_block in range(model.num_blocks):
        for second_block in range(model.num_blocks):
            if first_block == second_block:
                continue
            metrics = evaluate_pair(
                model=model,
                tokens=tokens,
                pair=(first_block, second_block),
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                suffix_mode=args.suffix_mode,
                random_suffix_samples=args.random_suffix_samples,
                seed=args.seed,
                device=args.device,
                autocast_context=autocast_context,
            )
            rows.append(
                {
                    "first_block": first_block,
                    "second_block": second_block,
                    **metrics,
                }
            )

    export_csv(rows, args.out_csv)
    print(f"saved pair ranking csv to {args.out_csv}")

    rows_sorted = sorted(rows, key=lambda row: (row["prefix2_auc"], row["prefix4_auc"], row["full_loss"]))
    print("top-10 ordered block pairs:")
    for rank, row in enumerate(rows_sorted[:10], start=1):
        print(
            f"  rank {rank}: ({row['first_block']}, {row['second_block']}) "
            f"prefix2_auc={row['prefix2_auc']:.6f}, "
            f"prefix4_auc={row['prefix4_auc']:.6f}, "
            f"full_loss={row['full_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
