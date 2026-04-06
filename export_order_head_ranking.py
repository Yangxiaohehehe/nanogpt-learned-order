import argparse
import csv
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from AOGPT import AOGPT, AOGPTConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export block-level order head ranking from an AO-GPT checkpoint."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--score_mode", type=str, default="AR", choices=["AR", "Random"])
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


def load_meta_vocab_size(data_dir: Path):
    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    return meta.get("vocab_size")


@torch.no_grad()
def collect_scores(model, tokens, batch_size, num_batches, score_mode, seed, device, autocast_context):
    rng = np.random.default_rng(seed)
    block_size = model.config.block_size
    score_batches = []
    for _ in range(num_batches):
        idx = sample_batch(tokens, batch_size, block_size, rng, device)
        with autocast_context:
            _, _, hidden_states = model(
                idx,
                mode=score_mode,
                return_hidden=True,
            )
            scores = model.score_prefix_policy(idx, hidden_states, detach_inputs=True)
        score_batches.append(scores.float().cpu())
    return torch.cat(score_batches, dim=0)


def export_csv(out_csv: Path, avg_scores, std_scores):
    sorted_indices = np.argsort(-avg_scores)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "origin_block_index", "avg_score", "std_score"])
        for rank, block_idx in enumerate(sorted_indices, start=1):
            writer.writerow(
                [
                    int(rank),
                    int(block_idx),
                    float(avg_scores[block_idx]),
                    float(std_scores[block_idx]),
                ]
            )


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    _ = load_meta_vocab_size(data_dir)
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)

    all_scores = collect_scores(
        model=model,
        tokens=tokens,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        score_mode=args.score_mode,
        seed=args.seed,
        device=args.device,
        autocast_context=autocast_context,
    )
    avg_scores = all_scores.mean(dim=0).numpy()
    std_scores = all_scores.std(dim=0, unbiased=False).numpy()
    export_csv(args.out_csv, avg_scores, std_scores)

    print(f"saved ranking csv to {args.out_csv}")
    print(f"score_mode={args.score_mode}, split={args.split}, samples={all_scores.size(0)}")
    print("top-5 blocks:")
    sorted_indices = np.argsort(-avg_scores)
    for rank, block_idx in enumerate(sorted_indices[:5], start=1):
        print(
            f"  rank {rank}: block {int(block_idx)}, "
            f"avg_score={float(avg_scores[block_idx]):.6f}, "
            f"std_score={float(std_scores[block_idx]):.6f}"
        )


if __name__ == "__main__":
    main()
