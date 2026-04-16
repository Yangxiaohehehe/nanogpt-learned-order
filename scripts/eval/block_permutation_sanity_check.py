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
from order_utils import (
    block_permutation_to_token_permutation,
    build_ascending_block_orders,
    build_fixed_block_permutation,
    evaluate_block_order_quality,
    invert_permutation,
)
from path_layout import default_eval_out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check block permutation checkpoints by comparing current-frame and recovered-original l2r."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
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
    parser.add_argument("--seed", type=int, default=12345)
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


def load_block_permutation_from_checkpoint(checkpoint, num_blocks, block_len):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None):
    max_start = len(tokens) - block_size
    starts = rng.integers(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    ).to(device)
    if token_perm is not None:
        batch = batch[:, token_perm.to(device)]
    return batch


def summarize_order_metrics(model, tokens, order, args, token_perm, autocast_context):
    rng = np.random.default_rng(args.seed)
    full_loss_sum = 0.0
    prefix_auc_sum = 0.0
    kendall_sum = 0.0
    num_samples = 0

    for _ in range(int(args.num_batches)):
        idx = sample_batch(
            tokens,
            batch_size=int(args.batch_size),
            block_size=model.config.block_size,
            rng=rng,
            device=args.device,
            token_perm=token_perm,
        )
        tiled_orders = order.unsqueeze(0).expand(idx.size(0), -1)
        metrics = evaluate_block_order_quality(
            model,
            idx,
            tiled_orders,
            prefix_k=model.num_blocks,
            block_len=model.block_order_block_len,
            autocast_context=autocast_context,
        )
        full_loss_sum += float(metrics["full_loss_per_sample"].sum().item())
        prefix_auc_sum += float(metrics["prefix_auc_per_sample"].sum().item())
        kendall_sum += float(metrics["kendall_per_sample"].sum().item())
        num_samples += int(idx.size(0))

    return {
        "mean_full_loss": full_loss_sum / max(1, num_samples),
        "mean_prefix_auc": prefix_auc_sum / max(1, num_samples),
        "mean_kendall_tau_to_current_l2r": kendall_sum / max(1, num_samples),
        "num_samples": num_samples,
    }


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = default_eval_out_dir(REPO_ROOT, args.ckpt_path, "block_permutation_sanity_check")
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    perm_state = load_block_permutation_from_checkpoint(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
    )
    if perm_state is None:
        raise ValueError("Checkpoint does not contain block permutation metadata.")

    block_perm = perm_state["block_perm"]
    inverse_block_perm = perm_state["inverse_block_perm"]
    token_perm = perm_state["token_perm"]

    current_l2r = build_ascending_block_orders(1, model.num_blocks, args.device).squeeze(0)
    recovered_original_l2r = inverse_block_perm.to(args.device)

    current_metrics = summarize_order_metrics(
        model,
        tokens,
        current_l2r,
        args,
        token_perm,
        autocast_context,
    )
    recovered_metrics = summarize_order_metrics(
        model,
        tokens,
        recovered_original_l2r,
        args,
        token_perm,
        autocast_context,
    )

    payload = {
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": args.split,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "num_blocks": int(model.num_blocks),
            "block_order_block_len": int(model.block_order_block_len),
        },
        "block_perm": [int(v) for v in block_perm.tolist()],
        "inverse_block_perm": [int(v) for v in inverse_block_perm.tolist()],
        "current_frame_l2r": [int(v) for v in current_l2r.detach().cpu().tolist()],
        "recovered_original_l2r_in_current_frame": [int(v) for v in recovered_original_l2r.detach().cpu().tolist()],
        "metrics": {
            "current_frame_l2r": current_metrics,
            "recovered_original_l2r_in_current_frame": recovered_metrics,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "block_permutation_sanity.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("block_perm:", payload["block_perm"])
    print("inverse_block_perm:", payload["inverse_block_perm"])
    print("current-frame l2r:", payload["current_frame_l2r"])
    print("original-frame recovered l2r:", payload["recovered_original_l2r_in_current_frame"])
    print("current-frame l2r mean_full_loss:", current_metrics["mean_full_loss"])
    print("original-frame recovered l2r mean_full_loss:", recovered_metrics["mean_full_loss"])
    print(f"saved sanity check to {out_path}")


if __name__ == "__main__":
    main()
