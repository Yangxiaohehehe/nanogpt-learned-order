import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from AOGPT import AOGPT, AOGPTConfig
from order_utils import build_ascending_block_orders, evaluate_block_order_quality


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed candidate pool of block orders over many batches and rank them by metrics."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("out/out-wikitext103-random-b32/ckpt.pt"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/metric_candidate_leaderboard_b32"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--late_drop_weight", type=float, default=0.2)
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


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_checkpoint(ckpt_path: Path, device: str):
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
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
    return Path(__file__).resolve().parent / "data" / dataset


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


def summarize_window(window_losses, final_loss, prefix, tv_weight, late_drop_weight):
    first = window_losses[:, 0]
    last = window_losses[:, -1]
    area = window_losses.mean(dim=-1)
    slope = first - last
    variance = window_losses.var(dim=-1, unbiased=False)
    if window_losses.size(1) < 2:
        total_variation = torch.zeros_like(area)
        max_step_jump = torch.zeros_like(area)
    else:
        deltas = window_losses[:, 1:] - window_losses[:, :-1]
        total_variation = deltas.abs().sum(dim=-1)
        max_step_jump = deltas.abs().amax(dim=-1)
    late_drop = torch.clamp(last - final_loss, min=0.0)
    return {
        f"{prefix}_area": -area,
        f"{prefix}_slope": slope,
        f"{prefix}_variance": -variance,
        f"{prefix}_total_variation": -total_variation,
        f"{prefix}_max_step_jump": -max_step_jump,
        f"{prefix}_late_drop": -late_drop,
        f"{prefix}_area_plus_slope": -area + slope,
        f"{prefix}_area_plus_tv": -area - tv_weight * total_variation,
        f"{prefix}_area_plus_late_drop": -area - late_drop_weight * late_drop,
        f"{prefix}_area_plus_slope_plus_tv": -area + slope - tv_weight * total_variation,
    }


def compute_candidate_metrics(block_losses, early_k, tv_weight, late_drop_weight):
    block_losses = block_losses.float()
    early = block_losses[:, :early_k]
    full = block_losses
    final_loss = block_losses[:, -1]
    metrics = {}
    metrics.update(summarize_window(early, final_loss, "early", tv_weight, late_drop_weight))
    metrics.update(summarize_window(full, final_loss, "full", tv_weight, late_drop_weight))
    return metrics


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    rng = np.random.default_rng(args.seed)
    autocast_context = get_autocast_context(args.device, args.dtype)
    order_generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    order_generator.manual_seed(args.seed)

    num_blocks = model.num_blocks
    early_k = max(2, num_blocks // 4)

    random_candidates = torch.stack(
        [torch.randperm(num_blocks, generator=order_generator, device=args.device) for _ in range(int(args.num_candidates))],
        dim=0,
    )
    l2r_candidate = build_ascending_block_orders(1, num_blocks, args.device)
    candidate_orders = torch.cat([random_candidates, l2r_candidate], dim=0)
    candidate_labels = [f"rand_{i:02d}" for i in range(int(args.num_candidates))] + ["l2r"]
    l2r_idx = candidate_orders.size(0) - 1

    metric_sums = None
    total_eval_count = 0

    for _ in range(int(args.num_batches)):
        idx = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        tiled_orders = candidate_orders.unsqueeze(0).expand(idx.size(0), -1, -1)
        flat_orders = tiled_orders.reshape(idx.size(0) * candidate_orders.size(0), num_blocks)
        flat_idx = idx.unsqueeze(1).expand(idx.size(0), candidate_orders.size(0), idx.size(1)).reshape(
            idx.size(0) * candidate_orders.size(0),
            idx.size(1),
        )

        chunk_size = max(1, int(args.candidate_eval_batch_size))
        metric_chunks = []
        for start in range(0, flat_orders.size(0), chunk_size):
            end = min(flat_orders.size(0), start + chunk_size)
            metrics = evaluate_block_order_quality(
                model,
                flat_idx[start:end],
                flat_orders[start:end],
                prefix_k=early_k,
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            metric_chunks.append(
                compute_candidate_metrics(
                    metrics["block_losses"],
                    early_k=early_k,
                    tv_weight=float(args.tv_weight),
                    late_drop_weight=float(args.late_drop_weight),
                )
            )

        batch_metrics = {}
        for key in metric_chunks[0]:
            batch_metrics[key] = torch.cat([chunk[key] for chunk in metric_chunks], dim=0).view(
                idx.size(0),
                candidate_orders.size(0),
            )

        if metric_sums is None:
            metric_sums = {key: torch.zeros(candidate_orders.size(0), dtype=torch.float64) for key in batch_metrics}

        for key, values in batch_metrics.items():
            metric_sums[key] += values.detach().cpu().double().sum(dim=0)
        total_eval_count += idx.size(0)

    metric_means = {key: (values / total_eval_count).tolist() for key, values in metric_sums.items()}
    metric_names = list(metric_means.keys())

    leaderboards = {}
    for source_metric in metric_names:
        source_values = np.asarray(metric_means[source_metric], dtype=np.float64)
        sorted_idx = np.argsort(-source_values)
        l2r_rank = int(np.where(sorted_idx == l2r_idx)[0][0]) + 1
        cutoff = max(int(args.top_k), l2r_rank)
        rows = []
        for rank_pos, candidate_idx in enumerate(sorted_idx[:cutoff], start=1):
            row = {
                "rank": int(rank_pos),
                "candidate_idx": int(candidate_idx),
                "candidate_label": candidate_labels[candidate_idx],
                "is_l2r": bool(candidate_idx == l2r_idx),
                "order_prefix": [int(v) for v in candidate_orders[candidate_idx, :8].detach().cpu().tolist()],
            }
            for metric_name in metric_names:
                row[metric_name] = float(metric_means[metric_name][candidate_idx])
            rows.append(row)
        leaderboards[source_metric] = {
            "l2r_rank": l2r_rank,
            "cutoff_used": cutoff,
            "rows": rows,
        }

    run_meta = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_candidates": args.num_candidates,
        "includes_l2r_candidate": True,
        "top_k": args.top_k,
        "num_blocks": num_blocks,
        "early_k": early_k,
        "tv_weight": args.tv_weight,
        "late_drop_weight": args.late_drop_weight,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "note": (
            "Each leaderboard ranks a fixed candidate pool of random orders plus l2r by the mean metric value "
            "over all evaluated batches. Metrics are provided in both early_* form (first 25% of block reveal steps) "
            "and full_* form (the full block order trajectory). If l2r is outside top_k, rows are extended until "
            "l2r appears."
        ),
    }

    save_json(args.out_dir / "run_meta.json", run_meta)
    save_json(args.out_dir / "leaderboards.json", leaderboards)

    print(f"saved candidate leaderboards to {args.out_dir}")
    for metric_name, payload in leaderboards.items():
        print(f"{metric_name}: l2r_rank={payload['l2r_rank']}, cutoff={payload['cutoff_used']}")


if __name__ == "__main__":
    main()
