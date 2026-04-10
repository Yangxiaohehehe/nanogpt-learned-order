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
from order_utils import token_losses_to_block_losses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AR-likeness trajectory scores by comparing AR vs Random rollouts."
    )
    parser.add_argument(
        "--ckpt_paths",
        type=Path,
        nargs="*",
        default=[
            Path("out-wikitext103-random/ckpt.pt"),
        ],
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/analysis/ar_likeness_benchmark"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--num_random_rollouts", type=int, default=8)
    parser.add_argument("--early_k", type=int, default=4)
    parser.add_argument(
        "--save_per_sample_rows",
        action="store_true",
        help="If set, save per-sample row details. Disabled by default to keep outputs compact.",
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


@torch.no_grad()
def compute_block_loss_trajectory(model, idx, mode, block_len, autocast_context):
    with autocast_context:
        _, _, token_losses = model(
            idx,
            mode=mode,
            return_token_loss=True,
        )
    return token_losses_to_block_losses(token_losses, block_len=block_len)


def compute_ar_likeness_components(block_losses, early_k):
    early_k = max(2, min(int(early_k), block_losses.size(1)))
    early = block_losses[:, :early_k]
    late = block_losses[:, early_k:]

    first = early[:, 0]
    last_early = early[:, -1]
    last_all = block_losses[:, -1]

    early_area = early.mean(dim=-1)
    early_slope = first - last_early
    early_variance = early.var(dim=-1, unbiased=False)
    late_drop = torch.clamp(last_early - last_all, min=0.0)
    total_drop = torch.clamp(first - last_all, min=0.0)
    late_drop_ratio = late_drop / total_drop.clamp_min(1e-6)
    if late.size(1) > 0:
        late_mean_gap = late.mean(dim=-1) - early.mean(dim=-1)
    else:
        late_mean_gap = torch.zeros_like(early_area)

    return {
        "early_area": early_area,
        "early_slope": early_slope,
        "early_variance": early_variance,
        "late_drop": late_drop,
        "late_drop_ratio": late_drop_ratio,
        "late_mean_gap": late_mean_gap,
    }


def build_score_specs():
    return [
        ("area_only", {"early_area": -1.0}),
        ("slope_only", {"early_slope": 1.0}),
        ("variance_only", {"early_variance": -1.0}),
        ("late_drop_only", {"late_drop": -1.0}),
        ("area_plus_slope", {"early_area": -1.0, "early_slope": 1.0}),
        ("area_plus_variance", {"early_area": -1.0, "early_variance": -1.0}),
        ("area_plus_late_drop", {"early_area": -1.0, "late_drop": -1.0}),
        (
            "area_slope_variance",
            {"early_area": -1.0, "early_slope": 1.0, "early_variance": -1.0},
        ),
        (
            "area_slope_late_drop",
            {"early_area": -1.0, "early_slope": 1.0, "late_drop": -1.0},
        ),
        (
            "area_slope_variance_late_drop",
            {"early_area": -1.0, "early_slope": 1.0, "early_variance": -1.0, "late_drop": -1.0},
        ),
        (
            "full_with_ratio",
            {
                "early_area": -1.0,
                "early_slope": 1.0,
                "early_variance": -1.0,
                "late_drop_ratio": -1.0,
            },
        ),
    ]


def standardize_components(component_store):
    stats = {}
    normalized = {}
    for name, values in component_store.items():
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1e-6)
        stats[name] = {"mean": float(mean.item()), "std": float(std.item())}
        normalized[name] = (values - mean) / std
    return normalized, stats


def summarize_component(name, ar_values, random_values):
    margin = ar_values - random_values
    return {
        "component": name,
        "ar_mean": float(ar_values.mean().item()),
        "random_mean": float(random_values.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "ar_better_rate": float((margin > 0).float().mean().item()),
    }


def summarize_score(name, ar_scores, random_scores, weights):
    margin = ar_scores - random_scores
    return {
        "score_name": name,
        "weights": weights,
        "evaluation_note": (
            "AR-likeness scores are evaluation-only diagnostics for checking whether AR trajectories score higher "
            "than Random trajectories. They are not training targets or propagated supervision signals."
        ),
        "ar_mean": float(ar_scores.mean().item()),
        "random_mean": float(random_scores.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std(unbiased=False).item()),
        "ar_win_rate": float((margin > 0).float().mean().item()),
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    autocast_context = get_autocast_context(args.device, args.dtype)
    score_specs = build_score_specs()

    overall_ranking = []

    for ckpt_path in args.ckpt_paths:
        checkpoint = load_checkpoint(ckpt_path, args.device)
        data_dir = resolve_data_dir(args, checkpoint)
        tokens = load_tokens(data_dir, args.split)
        model = build_model(checkpoint, args.device)

        ar_components_all = {}
        random_components_all = {}
        per_sample_rows = []
        ar_block_loss_rows = []
        random_block_loss_rows = []

        for batch_idx in range(int(args.num_batches)):
            idx = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
            ar_block_losses = compute_block_loss_trajectory(
                model,
                idx,
                mode="AR",
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            ar_components = compute_ar_likeness_components(ar_block_losses, args.early_k)

            for rollout_idx in range(int(args.num_random_rollouts)):
                random_block_losses = compute_block_loss_trajectory(
                    model,
                    idx,
                    mode="Random",
                    block_len=model.block_order_block_len,
                    autocast_context=autocast_context,
                )
                random_components = compute_ar_likeness_components(random_block_losses, args.early_k)

                for name, values in ar_components.items():
                    ar_components_all.setdefault(name, []).append(values.detach().cpu())
                    random_components_all.setdefault(name, []).append(random_components[name].detach().cpu())

                ar_block_loss_rows.append(ar_block_losses.detach().cpu())
                random_block_loss_rows.append(random_block_losses.detach().cpu())

                if args.save_per_sample_rows:
                    for sample_idx in range(idx.size(0)):
                        row = {
                            "batch_idx": batch_idx,
                            "sample_idx": sample_idx,
                            "random_rollout_idx": rollout_idx,
                            "ar_block_losses": [float(v) for v in ar_block_losses[sample_idx].detach().cpu().tolist()],
                            "random_block_losses": [float(v) for v in random_block_losses[sample_idx].detach().cpu().tolist()],
                        }
                        for name, values in ar_components.items():
                            row[f"ar_{name}"] = float(values[sample_idx].item())
                            row[f"random_{name}"] = float(random_components[name][sample_idx].item())
                        per_sample_rows.append(row)

        ar_components_cat = {name: torch.cat(values, dim=0) for name, values in ar_components_all.items()}
        random_components_cat = {name: torch.cat(values, dim=0) for name, values in random_components_all.items()}

        combined_components = {
            name: torch.cat([ar_components_cat[name], random_components_cat[name]], dim=0)
            for name in ar_components_cat
        }
        normalized_components, component_stats = standardize_components(combined_components)
        num_ar = next(iter(ar_components_cat.values())).size(0)
        normalized_ar = {name: values[:num_ar] for name, values in normalized_components.items()}
        normalized_random = {name: values[num_ar:] for name, values in normalized_components.items()}

        component_summaries = []
        for name in ar_components_cat:
            direction = 1.0 if name == "early_slope" else -1.0
            component_summaries.append(
                summarize_component(
                    name,
                    direction * normalized_ar[name],
                    direction * normalized_random[name],
                )
            )
        component_summaries = sorted(
            component_summaries,
            key=lambda row: (-row["ar_better_rate"], -row["margin_mean"]),
        )

        score_summaries = []
        for score_name, weights in score_specs:
            ar_score = torch.zeros_like(next(iter(normalized_ar.values())))
            random_score = torch.zeros_like(next(iter(normalized_random.values())))
            for component_name, weight in weights.items():
                ar_score = ar_score + float(weight) * normalized_ar[component_name]
                random_score = random_score + float(weight) * normalized_random[component_name]
            score_summaries.append(
                summarize_score(score_name, ar_score, random_score, weights)
            )
        score_summaries = sorted(
            score_summaries,
            key=lambda row: (-row["ar_win_rate"], -row["margin_mean"]),
        )

        ar_block_loss_mean = torch.cat(ar_block_loss_rows, dim=0).mean(dim=0)
        random_block_loss_mean = torch.cat(random_block_loss_rows, dim=0).mean(dim=0)
        mean_curve_summary = {
            "evaluation_note": (
                "These are mean block-loss trajectories over reveal/generate steps. "
                "AR is both reveal-step order and original position because AR is l2r; "
                "Random is reveal-step order only, not original-position order."
            ),
            "ar_block_loss_mean": [float(v) for v in ar_block_loss_mean.tolist()],
            "random_block_loss_mean": [float(v) for v in random_block_loss_mean.tolist()],
            "ar_minus_random_block_loss_mean": [float(v) for v in (ar_block_loss_mean - random_block_loss_mean).tolist()],
            "num_trajectory_samples": int(ar_block_loss_mean.numel() * 0 + torch.cat(ar_block_loss_rows, dim=0).size(0)),
        }

        ckpt_name = ckpt_path.parent.name
        ckpt_out_dir = args.out_dir / ckpt_name
        if args.save_per_sample_rows:
            save_json(ckpt_out_dir / "per_sample_rows.json", per_sample_rows)
        save_json(ckpt_out_dir / "mean_curve_summary.json", mean_curve_summary)
        save_json(ckpt_out_dir / "component_stats.json", component_stats)
        save_json(ckpt_out_dir / "component_ranking.json", component_summaries)
        save_json(ckpt_out_dir / "score_ranking.json", score_summaries)
        save_json(
            ckpt_out_dir / "run_meta.json",
            {
                "ckpt_path": str(ckpt_path),
                "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
                "split": args.split,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "num_random_rollouts": args.num_random_rollouts,
                "early_k": args.early_k,
                "seed": args.seed,
                "device": args.device,
                "dtype": args.dtype,
                "evaluation_policy": {
                    "goal": "Check whether AR trajectories score higher than Random trajectories.",
                    "note": (
                        "These AR-likeness scores are evaluation-only and must not be used directly as "
                        "training targets or propagated supervision signals."
                    ),
                },
            },
        )

        best_score = score_summaries[0]
        overall_ranking.append(
            {
                "checkpoint": ckpt_name,
                "best_score_name": best_score["score_name"],
                "best_ar_win_rate": best_score["ar_win_rate"],
                "best_margin_mean": best_score["margin_mean"],
            }
        )

        print(f"[{ckpt_name}] best AR-likeness score: {best_score['score_name']}")
        print(
            f"  ar_win_rate={best_score['ar_win_rate']:.4f}, "
            f"margin_mean={best_score['margin_mean']:.4f}"
        )
        print("  top-5 score combos:")
        for row in score_summaries[:5]:
            print(
                f"    {row['score_name']}: "
                f"ar_win_rate={row['ar_win_rate']:.4f}, "
                f"margin_mean={row['margin_mean']:.4f}"
            )

    save_json(args.out_dir / "overall_ranking.json", overall_ranking)


if __name__ == "__main__":
    main()
