import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import evaluate_block_order_quality


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare candidate-order metric rankings on a fixed checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("out/out-wikitext103-random-b32/ckpt.pt"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/leaderboards/metric_rank_benchmark_b32"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--top_frac", type=float, default=0.25)
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


def sample_random_block_orders(batch_size, num_candidates, num_blocks, device, generator):
    return torch.stack(
        [
            torch.stack(
                [torch.randperm(num_blocks, generator=generator, device=device) for _ in range(num_candidates)],
                dim=0,
            )
            for _ in range(batch_size)
        ],
        dim=0,
    )


def compute_candidate_metrics(block_losses, early_k):
    early = block_losses[:, :early_k].float()
    first = early[:, 0]
    last_early = early[:, -1]
    last_all = block_losses[:, -1].float()
    area = early.mean(dim=-1)
    slope = first - last_early
    variance = early.var(dim=-1, unbiased=False)
    if early.size(1) < 2:
        total_variation = torch.zeros_like(area)
    else:
        total_variation = (early[:, 1:] - early[:, :-1]).abs().sum(dim=-1)
    late_drop = torch.clamp(last_early - last_all, min=0.0)
    area_plus_slope = -area + slope
    area_plus_tv = -area - 0.3 * total_variation
    return {
        "area": -area,
        "slope": slope,
        "variance": -variance,
        "total_variation": -total_variation,
        "late_drop": -late_drop,
        "area_plus_slope": area_plus_slope,
        "area_plus_tv": area_plus_tv,
    }


def rank_positions_desc(values):
    sorted_idx = torch.argsort(values, dim=-1, descending=True)
    ranks = torch.empty_like(sorted_idx)
    rank_ids = torch.arange(values.size(-1), device=values.device).unsqueeze(0).expand_as(sorted_idx)
    ranks.scatter_(1, sorted_idx, rank_ids)
    return ranks


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
    top_k = max(1, int(round(args.num_candidates * float(args.top_frac))))

    metric_names = None
    pair_rank_values = {}
    pair_top_overlap = {}
    metric_top_values = {}
    metric_all_values = {}
    for batch_idx in range(int(args.num_batches)):
        idx = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        candidate_orders = sample_random_block_orders(
            idx.size(0),
            int(args.num_candidates),
            num_blocks,
            idx.device,
            order_generator,
        )
        flat_orders = candidate_orders.reshape(idx.size(0) * int(args.num_candidates), num_blocks)
        flat_idx = idx.unsqueeze(1).expand(idx.size(0), int(args.num_candidates), idx.size(1)).reshape(
            idx.size(0) * int(args.num_candidates),
            idx.size(1),
        )

        metric_chunks = []
        chunk_size = max(1, int(args.candidate_eval_batch_size))
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
            metric_chunks.append(compute_candidate_metrics(metrics["block_losses"], early_k=early_k))

        combined_metrics = {}
        for key in metric_chunks[0]:
            combined_metrics[key] = torch.cat([chunk[key] for chunk in metric_chunks], dim=0).view(
                idx.size(0),
                int(args.num_candidates),
            )

        if metric_names is None:
            metric_names = list(combined_metrics.keys())
            for src in metric_names:
                metric_top_values[src] = []
                metric_all_values[src] = []
                for dst in metric_names:
                    pair_rank_values[(src, dst)] = []
                    pair_top_overlap[(src, dst)] = []

        rank_tables = {name: rank_positions_desc(values) for name, values in combined_metrics.items()}

        for sample_idx in range(idx.size(0)):
            per_metric_top = {}
            for name in metric_names:
                values = combined_metrics[name][sample_idx]
                sorted_idx = torch.argsort(values, descending=True)
                top_idx = sorted_idx[:top_k]
                per_metric_top[name] = top_idx
                metric_top_values[name].extend(values[top_idx].detach().cpu().tolist())
                metric_all_values[name].extend(values.detach().cpu().tolist())

            for src in metric_names:
                src_top = per_metric_top[src]
                src_set = set(int(v) for v in src_top.detach().cpu().tolist())
                for dst in metric_names:
                    dst_ranks = rank_tables[dst][sample_idx][src_top].float()
                    pair_rank_values[(src, dst)].append(float(dst_ranks.mean().item()))
                    dst_top_set = set(int(v) for v in per_metric_top[dst].detach().cpu().tolist())
                    overlap = len(src_set & dst_top_set) / float(top_k)
                    pair_top_overlap[(src, dst)].append(overlap)

    rank_summary = []
    overlap_summary = []
    for src in metric_names:
        rank_row = {"source_metric": src}
        overlap_row = {"source_metric": src}
        for dst in metric_names:
            rank_row[f"mean_rank_under_{dst}"] = float(np.mean(pair_rank_values[(src, dst)]))
            overlap_row[f"top_overlap_with_{dst}"] = float(np.mean(pair_top_overlap[(src, dst)]))
        rank_summary.append(rank_row)
        overlap_summary.append(overlap_row)

    metric_value_summary = []
    for name in metric_names:
        metric_value_summary.append(
            {
                "metric": name,
                "mean_all_value": float(np.mean(metric_all_values[name])),
                "mean_top25_value": float(np.mean(metric_top_values[name])),
                "top25_minus_all": float(np.mean(metric_top_values[name]) - np.mean(metric_all_values[name])),
            }
        )

    run_meta = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_candidates": args.num_candidates,
        "top_frac": args.top_frac,
        "top_k": top_k,
        "num_blocks": num_blocks,
        "early_k": early_k,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "note": (
            "Top-25% means the top candidate orders under each metric among randomly sampled candidate orders. "
            "This is a ranking-consistency analysis, not a training signal."
        ),
    }

    save_json(args.out_dir / "run_meta.json", run_meta)
    save_json(args.out_dir / "rank_summary.json", rank_summary)
    save_json(args.out_dir / "overlap_summary.json", overlap_summary)
    save_json(args.out_dir / "metric_value_summary.json", metric_value_summary)

    print(f"saved metric-rank benchmark to {args.out_dir}")
    for row in rank_summary:
        print(row)


if __name__ == "__main__":
    main()
