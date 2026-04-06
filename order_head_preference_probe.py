import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    build_ascending_block_orders,
    evaluate_block_order_quality,
    expand_block_orders_to_token_orders,
    generate_adjacent_swap_candidates,
    pairwise_order_preference_loss,
    prefix_plackett_luce_logprob,
    sample_random_block_orders,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the order head on random-only preference pairs and evaluate with an l2r probe."
    )
    parser.add_argument("--ckpt_path", type=Path, default=Path("out/out-wikitext103-random-b32/ckpt.pt"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/order_head_preference_probe_b32"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_batches", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=200)
    parser.add_argument("--train_num_candidates", type=int, default=32)
    parser.add_argument("--eval_num_candidates", type=int, default=256)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--signal", type=str, default="early_area_plus_tv", choices=[
        "early_area",
        "early_total_variation",
        "early_area_plus_tv",
        "full_area",
        "full_total_variation",
        "full_area_plus_tv",
    ])
    parser.add_argument(
        "--train_pair_mode",
        type=str,
        default="search_improved_adjacent",
        choices=["random_top_bottom", "search_improved_adjacent"],
    )
    parser.add_argument(
        "--tie_break_signal",
        type=str,
        default="full_area_plus_tv",
        choices=[
            "early_area",
            "early_total_variation",
            "early_area_plus_tv",
            "full_area",
            "full_total_variation",
            "full_area_plus_tv",
        ],
    )
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--rank_prefix_k", type=int, default=8)
    parser.add_argument("--signal_early_k", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--preference_margin", type=float, default=0.0)
    parser.add_argument("--log_every_batches", type=int, default=10)
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


def freeze_backbone_except_order_head(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("policy_order_head.")


def forward_policy_hidden_states(model, idx, generator, autocast_context, track_grad):
    random_block_orders = sample_random_block_orders(
        idx.size(0),
        model.num_blocks,
        idx.device,
        generator=generator,
    )
    random_token_orders = expand_block_orders_to_token_orders(
        random_block_orders,
        block_len=model.block_order_block_len,
    )
    grad_ctx = nullcontext() if track_grad else torch.no_grad()
    with grad_ctx:
        with autocast_context:
            outputs = model(
                idx,
                mode=None,
                orders=random_token_orders,
                return_hidden=True,
            )
    _, _, hidden_states = outputs
    return hidden_states


def compute_signal_from_block_losses(block_losses, signal, early_k, tv_weight):
    block_losses = block_losses.float()
    early_k = max(1, min(int(early_k), block_losses.size(1)))
    final_loss = block_losses[:, -1]

    def build_metrics(window_losses):
        area = window_losses.mean(dim=-1)
        if window_losses.size(1) < 2:
            total_variation = torch.zeros_like(area)
        else:
            total_variation = (window_losses[:, 1:] - window_losses[:, :-1]).abs().sum(dim=-1)
        return {
            "area": -area,
            "total_variation": -total_variation,
            "area_plus_tv": -area - float(tv_weight) * total_variation,
        }

    early_metrics = build_metrics(block_losses[:, :early_k])
    full_metrics = build_metrics(block_losses)

    metric_map = {
        "early_area": early_metrics["area"],
        "early_total_variation": early_metrics["total_variation"],
        "early_area_plus_tv": early_metrics["area_plus_tv"],
        "full_area": full_metrics["area"],
        "full_total_variation": full_metrics["total_variation"],
        "full_area_plus_tv": full_metrics["area_plus_tv"],
    }
    return metric_map[signal]


def build_candidate_orders(batch_size, num_blocks, num_candidates, device, generator, include_l2r=False):
    random_orders = torch.stack(
        [sample_random_block_orders(batch_size, num_blocks, device, generator=generator) for _ in range(int(num_candidates))],
        dim=1,
    )
    if not include_l2r:
        return random_orders
    l2r = build_ascending_block_orders(batch_size, num_blocks, device).unsqueeze(1)
    return torch.cat([random_orders, l2r], dim=1)


def evaluate_candidate_pool(
    model,
    idx,
    candidate_orders,
    signal,
    signal_early_k,
    tv_weight,
    candidate_eval_batch_size,
    autocast_context,
):
    batch_size, num_candidates, num_blocks = candidate_orders.shape
    flat_orders = candidate_orders.reshape(batch_size * num_candidates, num_blocks)
    flat_idx = idx.unsqueeze(1).expand(batch_size, num_candidates, idx.size(1)).reshape(
        batch_size * num_candidates,
        idx.size(1),
    )

    signal_chunks = []
    kendall_chunks = []
    chunk_size = max(1, int(candidate_eval_batch_size))
    for start in range(0, flat_orders.size(0), chunk_size):
        end = min(flat_orders.size(0), start + chunk_size)
        metrics = evaluate_block_order_quality(
            model,
            flat_idx[start:end],
            flat_orders[start:end],
            prefix_k=max(2, min(signal_early_k, num_blocks)),
            block_len=model.block_order_block_len,
            autocast_context=autocast_context,
        )
        signal_chunks.append(
            compute_signal_from_block_losses(
                metrics["block_losses"],
                signal=signal,
                early_k=signal_early_k,
                tv_weight=tv_weight,
            )
        )
        kendall_chunks.append(metrics["kendall_per_sample"])

    signal_scores = torch.cat(signal_chunks, dim=0).view(batch_size, num_candidates)
    kendall = torch.cat(kendall_chunks, dim=0).view(batch_size, num_candidates)
    kendall_distance = (1.0 - kendall) / 2.0
    return {
        "signal_scores": signal_scores,
        "kendall": kendall,
        "kendall_distance": kendall_distance,
    }


def build_train_preference_batch(
    model,
    idx,
    num_candidates,
    signal,
    tie_break_signal,
    train_pair_mode,
    signal_early_k,
    tv_weight,
    candidate_eval_batch_size,
    order_generator,
    autocast_context,
):
    if train_pair_mode == "search_improved_adjacent":
        raw_orders = sample_random_block_orders(
            idx.size(0),
            model.num_blocks,
            idx.device,
            generator=order_generator,
        )
        swap_candidates = generate_adjacent_swap_candidates(raw_orders)
        candidate_orders = torch.cat([raw_orders.unsqueeze(1), swap_candidates], dim=1)
        pool_metrics = evaluate_candidate_pool(
            model,
            idx,
            candidate_orders,
            signal=signal,
            signal_early_k=signal_early_k,
            tv_weight=tv_weight,
            candidate_eval_batch_size=candidate_eval_batch_size,
            autocast_context=autocast_context,
        )
        tie_break_metrics = evaluate_candidate_pool(
            model,
            idx,
            candidate_orders,
            signal=tie_break_signal,
            signal_early_k=signal_early_k,
            tv_weight=tv_weight,
            candidate_eval_batch_size=candidate_eval_batch_size,
            autocast_context=autocast_context,
        )
        signal_scores = pool_metrics["signal_scores"]
        tie_break_scores = tie_break_metrics["signal_scores"]
        # Lexicographic max: main signal first, then tie-break.
        combined_scores = signal_scores + 1e-4 * tie_break_scores
        best_idx = combined_scores.argmax(dim=1)
        batch_index = torch.arange(idx.size(0), device=idx.device)
        preferred_orders = candidate_orders[batch_index, best_idx]
        other_orders = raw_orders
        quality_gap = signal_scores[batch_index, best_idx] - signal_scores[:, 0]
        tie_break_gap = tie_break_scores[batch_index, best_idx] - tie_break_scores[:, 0]
        return {
            "preferred_orders": preferred_orders,
            "other_orders": other_orders,
            "quality_gap": quality_gap,
            "tie_break_gap": tie_break_gap,
            "preferred_signal": signal_scores[batch_index, best_idx],
            "other_signal": signal_scores[:, 0],
        }

    candidate_orders = build_candidate_orders(
        idx.size(0),
        model.num_blocks,
        num_candidates,
        idx.device,
        generator=order_generator,
        include_l2r=False,
    )
    pool_metrics = evaluate_candidate_pool(
        model,
        idx,
        candidate_orders,
        signal=signal,
        signal_early_k=signal_early_k,
        tv_weight=tv_weight,
        candidate_eval_batch_size=candidate_eval_batch_size,
        autocast_context=autocast_context,
    )
    signal_scores = pool_metrics["signal_scores"]
    best_idx = signal_scores.argmax(dim=1)
    worst_idx = signal_scores.argmin(dim=1)
    batch_index = torch.arange(idx.size(0), device=idx.device)
    preferred_orders = candidate_orders[batch_index, best_idx]
    other_orders = candidate_orders[batch_index, worst_idx]
    quality_gap = signal_scores[batch_index, best_idx] - signal_scores[batch_index, worst_idx]
    return {
        "preferred_orders": preferred_orders,
        "other_orders": other_orders,
        "quality_gap": quality_gap,
        "tie_break_gap": torch.zeros_like(quality_gap),
        "preferred_signal": signal_scores[batch_index, best_idx],
        "other_signal": signal_scores[batch_index, worst_idx],
    }


def rank_candidates_with_head(scores, candidate_orders, rank_prefix_k):
    batch_size, num_candidates, num_blocks = candidate_orders.shape
    expanded_scores = scores.unsqueeze(1).expand(batch_size, num_candidates, num_blocks).reshape(
        batch_size * num_candidates,
        num_blocks,
    )
    flat_orders = candidate_orders.reshape(batch_size * num_candidates, num_blocks)
    flat_logprob = prefix_plackett_luce_logprob(
        expanded_scores,
        flat_orders,
        prefix_k=rank_prefix_k,
    )
    return flat_logprob.view(batch_size, num_candidates)


def maybe_print_progress(prefix, step_idx, total_steps, metrics):
    print(
        f"[{prefix}] batch {step_idx}/{total_steps} "
        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"gap={metrics['gap']:.4f}"
    )


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    train_tokens = load_tokens(data_dir, args.train_split)
    eval_tokens = load_tokens(data_dir, args.eval_split)

    model = build_model(checkpoint, args.device)
    freeze_backbone_except_order_head(model)
    model.train()

    order_head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        order_head_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    rng_train = np.random.default_rng(args.seed)
    rng_eval = np.random.default_rng(args.seed + 1)
    autocast_context = get_autocast_context(args.device, args.dtype)
    order_generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    order_generator.manual_seed(args.seed)

    train_history = []
    for batch_idx in range(1, int(args.train_batches) + 1):
        idx = sample_batch(train_tokens, args.batch_size, model.config.block_size, rng_train, args.device)
        hidden_states = forward_policy_hidden_states(
            model,
            idx,
            generator=order_generator,
            autocast_context=autocast_context,
            track_grad=True,
        )
        with autocast_context:
            scores = model.score_prefix_policy(idx, hidden_states, detach_inputs=False)
        preference_targets = build_train_preference_batch(
            model,
            idx,
            num_candidates=int(args.train_num_candidates),
            signal=args.signal,
            tie_break_signal=args.tie_break_signal,
            train_pair_mode=args.train_pair_mode,
            signal_early_k=int(args.signal_early_k),
            tv_weight=float(args.tv_weight),
            candidate_eval_batch_size=int(args.candidate_eval_batch_size),
            order_generator=order_generator,
            autocast_context=autocast_context,
        )
        loss, pref_acc, _, _ = pairwise_order_preference_loss(
            scores,
            preference_targets["preferred_orders"],
            preference_targets["other_orders"],
            prefix_k=int(args.rank_prefix_k),
            margin=float(args.preference_margin),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_metrics = {
            "loss": float(loss.detach().item()),
            "accuracy": float(pref_acc.detach().item()),
            "gap": float(preference_targets["quality_gap"].mean().item()),
            "tie_break_gap": float(preference_targets["tie_break_gap"].mean().item()),
            "preferred_signal": float(preference_targets["preferred_signal"].mean().item()),
            "other_signal": float(preference_targets["other_signal"].mean().item()),
        }
        train_history.append(step_metrics)
        if batch_idx == 1 or batch_idx % max(1, int(args.log_every_batches)) == 0:
            maybe_print_progress("train", batch_idx, int(args.train_batches), step_metrics)

    model.eval()
    eval_accum = {
        "mean_l2r_rank": 0.0,
        "mean_l2r_head_score": 0.0,
        "l2r_top1_rate": 0.0,
        "l2r_top5_rate": 0.0,
        "mean_top1_kendall_tau": 0.0,
        "mean_top1_kendall_distance": 0.0,
        "mean_top1_signal": 0.0,
        "mean_l2r_signal": 0.0,
    }
    total_eval_samples = 0
    l2r_candidate_index = int(args.eval_num_candidates)

    with torch.no_grad():
        for batch_idx in range(1, int(args.eval_batches) + 1):
            idx = sample_batch(eval_tokens, args.batch_size, model.config.block_size, rng_eval, args.device)
            hidden_states = forward_policy_hidden_states(
                model,
                idx,
                generator=order_generator,
                autocast_context=autocast_context,
                track_grad=False,
            )
            with autocast_context:
                scores = model.score_prefix_policy(idx, hidden_states, detach_inputs=True)

            candidate_orders = build_candidate_orders(
                idx.size(0),
                model.num_blocks,
                int(args.eval_num_candidates),
                idx.device,
                generator=order_generator,
                include_l2r=True,
            )
            head_scores = rank_candidates_with_head(scores, candidate_orders, int(args.rank_prefix_k))
            sort_idx = torch.argsort(head_scores, dim=1, descending=True)
            l2r_rank = (sort_idx == l2r_candidate_index).nonzero(as_tuple=False)[:, 1] + 1

            pool_metrics = evaluate_candidate_pool(
                model,
                idx,
                candidate_orders,
                signal=args.signal,
                signal_early_k=int(args.signal_early_k),
                tv_weight=float(args.tv_weight),
                candidate_eval_batch_size=int(args.candidate_eval_batch_size),
                autocast_context=autocast_context,
            )

            batch_indices = torch.arange(idx.size(0), device=idx.device)
            top1_idx = sort_idx[:, 0]
            eval_accum["mean_l2r_rank"] += float(l2r_rank.float().sum().item())
            eval_accum["mean_l2r_head_score"] += float(head_scores[:, l2r_candidate_index].sum().item())
            eval_accum["l2r_top1_rate"] += float((l2r_rank == 1).float().sum().item())
            eval_accum["l2r_top5_rate"] += float((l2r_rank <= 5).float().sum().item())
            eval_accum["mean_top1_kendall_tau"] += float(pool_metrics["kendall"][batch_indices, top1_idx].sum().item())
            eval_accum["mean_top1_kendall_distance"] += float(
                pool_metrics["kendall_distance"][batch_indices, top1_idx].sum().item()
            )
            eval_accum["mean_top1_signal"] += float(pool_metrics["signal_scores"][batch_indices, top1_idx].sum().item())
            eval_accum["mean_l2r_signal"] += float(pool_metrics["signal_scores"][:, l2r_candidate_index].sum().item())
            total_eval_samples += idx.size(0)

            if batch_idx == 1 or batch_idx % max(1, int(args.log_every_batches)) == 0:
                print(
                    f"[eval] batch {batch_idx}/{int(args.eval_batches)} "
                    f"mean_l2r_rank_so_far={eval_accum['mean_l2r_rank']/total_eval_samples:.4f} "
                    f"top1_rate_so_far={eval_accum['l2r_top1_rate']/total_eval_samples:.4f} "
                    f"top5_rate_so_far={eval_accum['l2r_top5_rate']/total_eval_samples:.4f}"
                )

    train_summary = {
        "mean_preference_loss": float(np.mean([row["loss"] for row in train_history])),
        "mean_preference_accuracy": float(np.mean([row["accuracy"] for row in train_history])),
        "mean_preference_gap": float(np.mean([row["gap"] for row in train_history])),
        "mean_tie_break_gap": float(np.mean([row["tie_break_gap"] for row in train_history])),
        "mean_preferred_signal": float(np.mean([row["preferred_signal"] for row in train_history])),
        "mean_other_signal": float(np.mean([row["other_signal"] for row in train_history])),
    }
    eval_summary = {key: float(value / total_eval_samples) for key, value in eval_accum.items()}

    run_meta = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "batch_size": args.batch_size,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "train_num_candidates": args.train_num_candidates,
        "eval_num_candidates": args.eval_num_candidates,
        "candidate_eval_batch_size": args.candidate_eval_batch_size,
        "signal": args.signal,
        "tie_break_signal": args.tie_break_signal,
        "train_pair_mode": args.train_pair_mode,
        "signal_early_k": args.signal_early_k,
        "rank_prefix_k": args.rank_prefix_k,
        "tv_weight": args.tv_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "preference_margin": args.preference_margin,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "note": (
            "Training uses random-only candidate pools to construct better-vs-worse preference pairs. "
            "Evaluation adds l2r only as a probe candidate; it is not used as a training target."
        ),
    }

    checkpoint_out = {
        "model": model.state_dict(),
        "model_args": checkpoint["model_args"],
        "source_ckpt_path": str(args.ckpt_path),
        "run_meta": run_meta,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "run_meta.json", run_meta)
    save_json(args.out_dir / "train_summary.json", train_summary)
    save_json(args.out_dir / "eval_summary.json", eval_summary)
    torch.save(checkpoint_out, args.out_dir / "order_head_probe_ckpt.pt")

    print(f"saved outputs to {args.out_dir}")
    print(json.dumps({"train": train_summary, "eval": eval_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
