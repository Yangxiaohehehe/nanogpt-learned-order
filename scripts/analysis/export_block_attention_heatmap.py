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
from order_utils import build_fixed_block_permutation, block_permutation_to_token_permutation


EXPORT_TYPES = ("with_none", "without_none", "diff")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export AO-GPT block attention heatmaps with/without [None] and their difference."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", type=str, default="Random", choices=["AR", "Random"])
    parser.add_argument(
        "--order_frame",
        type=str,
        default="original",
        choices=["reveal", "original"],
        help=(
            "Retained for backward compatibility. The script now always exports both "
            "reveal and original frames regardless of this value."
        ),
    )
    parser.add_argument("--layer_reduce", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--head_reduce", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--force_manual_attention", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=(
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float32"
        ),
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


def build_model(checkpoint, device: str, force_manual_attention: bool):
    model_args = dict(checkpoint["model_args"])

    if "force_manual_attention" in getattr(AOGPTConfig, "__annotations__", {}):
        model_args["force_manual_attention"] = bool(force_manual_attention)

    model = AOGPT(AOGPTConfig(**model_args))

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    incompatible = model.load_state_dict(state_dict, strict=False)

    if hasattr(model, "set_attention_backend"):
        model.set_attention_backend(force_manual_attention)

    model.to(device)
    model.eval()

    ignored_policy_keys = [
        key for key in incompatible.unexpected_keys if key.startswith("policy_order_head.")
    ]
    other_unexpected_keys = [
        key for key in incompatible.unexpected_keys if not key.startswith("policy_order_head.")
    ]
    allowed_missing_keys = {
        f"transformer.h.{layer_idx}.attn.bias" for layer_idx in range(model.config.n_layer)
    }
    other_missing_keys = [
        key for key in incompatible.missing_keys if key not in allowed_missing_keys
    ]

    if ignored_policy_keys:
        print(f"ignored {len(ignored_policy_keys)} legacy policy_order_head keys from checkpoint")
    if other_unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys that were not recognized: {other_unexpected_keys}"
        )
    if other_missing_keys:
        raise RuntimeError(
            f"Missing checkpoint keys that were not expected analysis-only buffers: {other_missing_keys}"
        )

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
    if max_start < 0:
        raise ValueError("Dataset split is shorter than block_size.")
    starts = rng.integers(0, max_start + 1, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    )
    return batch.to(device), starts.tolist()


def build_orders(model, idx, mode):
    if mode == "AR":
        token_orders = model.set_ascending_orders(idx)
    elif mode == "Random":
        token_orders = model.sample_random_orders(idx)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    block_len = model.block_order_block_len
    block_orders = token_orders.view(token_orders.size(0), model.num_blocks, block_len)[:, :, 0] // block_len
    return token_orders, block_orders


def normalize_perm_1d(values, expected_len: int, name: str):
    perm = torch.as_tensor(values, dtype=torch.long, device="cpu")
    if perm.ndim != 1 or perm.numel() != expected_len:
        raise ValueError(
            f"{name} must be a 1D permutation of length {expected_len}, got shape={tuple(perm.shape)}."
        )
    expected = torch.arange(expected_len, dtype=torch.long, device="cpu")
    if not torch.equal(torch.sort(perm).values, expected):
        raise ValueError(f"{name} is not a valid permutation of [0, {expected_len - 1}].")
    return perm


def resolve_data_permutation(checkpoint, num_blocks: int, block_len: int):
    data_permutation = checkpoint.get("data_permutation")
    if not data_permutation:
        return None

    permute_mode = data_permutation.get("permute_mode")
    if permute_mode != "block":
        raise ValueError(
            f"Unsupported checkpoint data_permutation permute_mode={permute_mode!r}. Only 'block' is supported."
        )

    if "block_perm" in data_permutation:
        block_perm = normalize_perm_1d(
            data_permutation["block_perm"],
            expected_len=num_blocks,
            name="checkpoint.data_permutation.block_perm",
        )
    else:
        permute_seed = data_permutation.get("permute_seed")
        if permute_seed is None:
            raise ValueError(
                "Checkpoint data_permutation is missing both block_perm and permute_seed."
            )
        block_perm = build_fixed_block_permutation(num_blocks, permute_seed).to(dtype=torch.long, device="cpu")

    if "inverse_block_perm" in data_permutation:
        inverse_block_perm = normalize_perm_1d(
            data_permutation["inverse_block_perm"],
            expected_len=num_blocks,
            name="checkpoint.data_permutation.inverse_block_perm",
        )
    else:
        inverse_block_perm = invert_permutation(block_perm)

    if not torch.equal(invert_permutation(block_perm), inverse_block_perm):
        raise ValueError("checkpoint.data_permutation inverse_block_perm is inconsistent with block_perm.")

    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)

    return {
        "permute_mode": permute_mode,
        "permute_seed": data_permutation.get("permute_seed"),
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def maybe_apply_data_permutation(idx, data_permutation):
    if data_permutation is None:
        return idx
    token_perm = data_permutation["token_perm"].to(device=idx.device)
    return idx[:, token_perm]


def get_export_frames(data_permutation):
    if data_permutation is None:
        return ("reveal", "original")
    return ("reveal", "original", "true_original")


def reduce_attention_batch(attn_outputs, layer_reduce, head_reduce):
    layers = [layer_att.detach().float().cpu() for layer_att in attn_outputs]
    if layer_reduce == "last":
        attn = layers[-1]
    else:
        attn = torch.stack(layers, dim=0).mean(dim=0)

    if head_reduce == "first":
        return attn[:, 0]
    return attn.mean(dim=1)


def extract_logits_loss_attentions(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError(
            "Expected model.forward_fn(..., return_attentions=True) to return a tuple/list."
        )
    if len(outputs) < 3:
        raise RuntimeError(
            f"Expected at least 3 outputs from model.forward_fn(..., return_attentions=True), got {len(outputs)}."
        )

    logits = outputs[0]
    loss = outputs[1]
    attentions = None

    for candidate in reversed(outputs[2:]):
        if isinstance(candidate, (list, tuple)) and candidate:
            if all(torch.is_tensor(item) and item.ndim == 4 for item in candidate):
                attentions = candidate
                break

    if attentions is None:
        raise RuntimeError(
            "Could not find attention tensors in model.forward_fn(..., return_attentions=True) outputs."
        )

    return logits, loss, attentions


def invert_permutation(perm_1d):
    inverse = torch.empty_like(perm_1d)
    inverse[perm_1d] = torch.arange(perm_1d.numel(), dtype=perm_1d.dtype, device=perm_1d.device)
    return inverse


def reorder_block_attention_to_original(block_matrix, block_order):
    inverse = invert_permutation(block_order)
    return block_matrix[inverse][:, inverse]


def aggregate_predictor_aligned_attention_to_block(attn_2d, block_len):
    """
    Aggregate token attention into predictor-aligned reveal blocks.

    AO-GPT predicts targets from shift_logits = logits[..., :-1, :], so the
    predictor-aligned attention region is attn[:-1, :-1]. These positions include
    predictor 0 = [None], therefore reveal block 0 contains [None] plus the first
    (block_len - 1) revealed real-token predictor positions.
    """
    shifted_attn = attn_2d[:-1, :-1]
    num_predictor_positions = shifted_attn.size(0)
    if num_predictor_positions % block_len != 0:
        raise ValueError(
            f"Predictor-aligned attention length {num_predictor_positions} "
            f"is not divisible by block_len={block_len}."
        )

    num_blocks = num_predictor_positions // block_len
    return shifted_attn.view(num_blocks, block_len, num_blocks, block_len).mean(dim=(1, 3))


def aggregate_real_token_attention_to_block(attn_2d, block_len):
    """
    Aggregate attention among real-token reveal positions only.

    This removes the [None] row/column first, then groups contiguous revealed real-token
    positions into full blocks of size block_len. Every exported block therefore contains
    exactly block_len real tokens.
    """
    real_token_attn = attn_2d[1:, 1:]
    num_real_positions = real_token_attn.size(0)
    if num_real_positions % block_len != 0:
        raise ValueError(
            f"Real-token attention length {num_real_positions} is not divisible by block_len={block_len}."
        )

    num_blocks = num_real_positions // block_len
    return real_token_attn.view(num_blocks, block_len, num_blocks, block_len).mean(dim=(1, 3))


def build_block_views_for_sample(attn_2d, block_order, block_len, data_permutation=None):
    with_none_reveal = aggregate_predictor_aligned_attention_to_block(attn_2d, block_len=block_len)
    without_none_reveal = aggregate_real_token_attention_to_block(attn_2d, block_len=block_len)

    with_none_original = reorder_block_attention_to_original(with_none_reveal, block_order)
    without_none_original = reorder_block_attention_to_original(without_none_reveal, block_order)

    diff_reveal = with_none_reveal - without_none_reveal
    diff_original = with_none_original - without_none_original

    outputs = {
        "with_none": {
            "reveal": with_none_reveal,
            "original": with_none_original,
        },
        "without_none": {
            "reveal": without_none_reveal,
            "original": without_none_original,
        },
        "diff": {
            "reveal": diff_reveal,
            "original": diff_original,
        },
    }

    if data_permutation is not None:
        data_block_perm = data_permutation["block_perm"]
        for export_type in EXPORT_TYPES:
            outputs[export_type]["true_original"] = reorder_block_attention_to_original(
                outputs[export_type]["original"],
                data_block_perm,
            )

    return outputs


def aggregate_batch_block_attentions(attn_batch, block_orders, block_len, data_permutation=None):
    export_frames = get_export_frames(data_permutation)
    aggregated = {
        export_type: {frame: [] for frame in export_frames} for export_type in EXPORT_TYPES
    }

    for sample_idx in range(attn_batch.size(0)):
        per_sample = build_block_views_for_sample(
            attn_batch[sample_idx],
            block_order=block_orders[sample_idx].detach().cpu(),
            block_len=block_len,
            data_permutation=data_permutation,
        )
        for export_type in EXPORT_TYPES:
            for frame in export_frames:
                aggregated[export_type][frame].append(per_sample[export_type][frame])

    return {
        export_type: {
            frame: torch.stack(aggregated[export_type][frame], dim=0) for frame in export_frames
        }
        for export_type in EXPORT_TYPES
    }


def get_alignment_metadata(export_type: str):
    if export_type == "with_none":
        return {
            "attention_alignment": "predictor_aligned_reveal_blocks_with_none",
            "attention_alignment_detail": (
                "attention is first aligned to shift loss positions with attn[:-1, :-1]. "
                "Blocks are then formed over predictor positions, so reveal block 0 includes "
                "[None] plus the first (block_len - 1) revealed real-token predictor positions. "
                "This is the generation-mechanism view of block attention."
            ),
        }
    if export_type == "without_none":
        return {
            "attention_alignment": "real_token_only_reveal_blocks_without_none",
            "attention_alignment_detail": (
                "attention is restricted to real-token positions only via attn[1:, 1:]. "
                "Blocks are re-formed over contiguous revealed real tokens, and every block "
                "contains exactly block_len real tokens. This is the real block-relationship view "
                "of block attention."
            ),
        }
    if export_type == "diff":
        return {
            "attention_alignment": "difference_with_none_minus_without_none",
            "attention_alignment_detail": (
                "difference is computed as with_none - without_none after each matrix is first "
                "aggregated in reveal blocks and then optionally reordered to original block order. "
                "This isolates the extra effect introduced by the start predictor [None]."
            ),
        }
    raise ValueError(f"Unsupported export_type: {export_type}")


def get_frame_metadata(frame: str, data_permutation):
    if frame == "reveal":
        return {
            "frame_alignment": "reveal_order",
            "frame_alignment_detail": (
                "rows and columns are indexed by reveal-order blocks in the current model input frame."
            ),
        }
    if frame == "original":
        return {
            "frame_alignment": "current_input_original_order",
            "frame_alignment_detail": (
                "reveal-order block matrices are reordered by the sampled block order back to the original "
                "block indices of the current model input frame."
            ),
        }
    if frame == "true_original":
        if data_permutation is None:
            raise ValueError("true_original frame requested without checkpoint data_permutation.")
        return {
            "frame_alignment": "true_data_original_order",
            "frame_alignment_detail": (
                "after reveal-order matrices are reordered to the current input frame original order, they are "
                "reordered again by the checkpoint's fixed data permutation back to the true unpermuted data "
                "block order."
            ),
        }
    raise ValueError(f"Unsupported frame: {frame}")


def save_heatmap_png(matrix, out_path: Path, title: str, *, cmap="viridis", vmin=None, vmax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("key block")
    plt.ylabel("query block")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def export_results(
    args,
    model,
    data_dir,
    total_samples,
    loss_sum,
    logits_shape,
    batch_start_offsets,
    matrix_sums,
    data_permutation,
):
    if total_samples <= 0:
        raise RuntimeError("No attention matrices were aggregated.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_frames = get_export_frames(data_permutation)

    diff_abs_max = max(
        float((matrix_sums["diff"][frame] / total_samples).abs().max().item())
        for frame in export_frames
    )

    for export_type in EXPORT_TYPES:
        alignment = get_alignment_metadata(export_type)
        for frame in export_frames:
            matrix = matrix_sums[export_type][frame] / total_samples

            base_name = f"block_attention_{export_type}_{frame}"
            np.save(args.out_dir / f"{base_name}.npy", matrix.numpy())

            metadata = {
                "ckpt_path": str(args.ckpt_path),
                "data_dir": str(data_dir),
                "split": args.split,
                "mode": args.mode,
                "requested_order_frame_arg": args.order_frame,
                "export_frame": frame,
                "export_type": export_type,
                "layer_reduce": args.layer_reduce,
                "head_reduce": args.head_reduce,
                "force_manual_attention": bool(args.force_manual_attention),
                "seed": int(args.seed),
                "num_batches": int(args.num_batches),
                "batch_size": int(args.batch_size),
                "num_samples_aggregated": int(total_samples),
                "mean_loss": float(loss_sum / total_samples),
                "logits_shape": logits_shape,
                "num_blocks": int(model.num_blocks),
                "block_len": int(model.block_order_block_len),
                "axis_labels": list(range(model.num_blocks)),
                "frame_reorder_logic": (
                    "reveal matrices are aggregated first; original matrices are obtained by "
                    "reordering the block-level reveal matrices with the sample block order; "
                    "for permuted-data checkpoints, true_original matrices are then obtained by "
                    "reordering current-frame original matrices with the checkpoint's fixed data permutation."
                ),
                "batch_start_offsets_preview": batch_start_offsets[: min(32, len(batch_start_offsets))],
            }
            metadata.update(alignment)
            metadata.update(get_frame_metadata(frame, data_permutation))
            if data_permutation is not None:
                metadata["data_permutation"] = {
                    "permute_mode": data_permutation["permute_mode"],
                    "permute_seed": data_permutation["permute_seed"],
                    "block_perm": data_permutation["block_perm"].tolist(),
                    "inverse_block_perm": data_permutation["inverse_block_perm"].tolist(),
                }

            (args.out_dir / f"{base_name}_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            if export_type == "diff":
                png_ok = save_heatmap_png(
                    matrix.numpy(),
                    args.out_dir / f"{base_name}.png",
                    title=f"block attention diff | mode={args.mode} | frame={frame}",
                    cmap="coolwarm",
                    vmin=-diff_abs_max,
                    vmax=diff_abs_max,
                )
            else:
                png_ok = save_heatmap_png(
                    matrix.numpy(),
                    args.out_dir / f"{base_name}.png",
                    title=f"block attention {export_type} | mode={args.mode} | frame={frame}",
                )

            if png_ok:
                print(f"saved heatmap png to {args.out_dir / f'{base_name}.png'}")
            else:
                print(f"matplotlib not installed; skipped png export for {base_name}")


def main():
    args = parse_args()

    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device, force_manual_attention=args.force_manual_attention)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    rng = np.random.default_rng(args.seed)
    autocast_context = get_autocast_context(args.device, args.dtype)
    data_permutation = resolve_data_permutation(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
    )
    export_frames = get_export_frames(data_permutation)

    total_samples = 0
    loss_sum = 0.0
    logits_shape = None
    batch_start_offsets = []
    matrix_sums = {
        export_type: {frame: None for frame in export_frames} for export_type in EXPORT_TYPES
    }

    for batch_idx in range(int(args.num_batches)):
        idx, starts = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        idx = maybe_apply_data_permutation(idx, data_permutation)
        token_orders, block_orders = build_orders(model, idx, args.mode)
        batch_start_offsets.extend(int(v) for v in starts)

        with torch.no_grad():
            with autocast_context:
                outputs = model.forward_fn(
                    idx,
                    token_orders,
                    return_attentions=True,
                )

        logits, loss, attn_outputs = extract_logits_loss_attentions(outputs)
        logits_shape = list(logits.shape)

        attn_batch = reduce_attention_batch(
            attn_outputs,
            layer_reduce=args.layer_reduce,
            head_reduce=args.head_reduce,
        )

        batch_matrices = aggregate_batch_block_attentions(
            attn_batch,
            block_orders,
            block_len=model.block_order_block_len,
            data_permutation=data_permutation,
        )

        for export_type in EXPORT_TYPES:
            for frame in export_frames:
                batch_sum = batch_matrices[export_type][frame].sum(dim=0)
                if matrix_sums[export_type][frame] is None:
                    matrix_sums[export_type][frame] = batch_sum
                else:
                    matrix_sums[export_type][frame] += batch_sum

        batch_samples = int(attn_batch.size(0))
        total_samples += batch_samples
        loss_sum += float(loss.item()) * batch_samples

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx + 1 == int(args.num_batches):
            print(f"processed {batch_idx + 1}/{args.num_batches} batches")

    export_results(
        args=args,
        model=model,
        data_dir=data_dir,
        total_samples=total_samples,
        loss_sum=loss_sum,
        logits_shape=logits_shape,
        batch_start_offsets=batch_start_offsets,
        matrix_sums=matrix_sums,
        data_permutation=data_permutation,
    )
    print(f"saved block attention arrays, pngs, and metadata to {args.out_dir}")


if __name__ == "__main__":
    main()
