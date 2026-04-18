import argparse
import json
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _patch_torchvision_unavailable_when_lzma_missing():
    """
    Work around environments where torchvision is installed but importing it fails
    because the Python build lacks the optional `_lzma` extension.

    Newer `transformers` versions may touch vision helpers while importing text-only
    models such as GPT-2. For our compare-to-GPT script we do not need torchvision at
    all, so we safely make it appear unavailable in this specific failure mode.
    """
    try:
        import lzma  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "_lzma":
            return

    original_find_spec = importlib.util.find_spec

    def wrapped_find_spec(name, package=None):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = wrapped_find_spec


_patch_torchvision_unavailable_when_lzma_missing()

from transformers import AutoModelForCausalLM


DEFAULT_COMPARE_SETUPS = "1,16,32,64,128"
EXPORT_TYPES = ("with_none", "without_none")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Hugging Face GPT-style AR attention heatmaps using the same "
            "predictor-aligned block aggregation logic as the local AO-GPT analysis."
        )
    )
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--predictor_len",
        type=int,
        default=256,
        help=(
            "Number of predictor positions to compare against AO-GPT exports. "
            "The script samples predictor_len + 1 tokens so the shifted AR attention "
            "submatrix is predictor_len x predictor_len."
        ),
    )
    parser.add_argument(
        "--compare_setups",
        type=str,
        default=DEFAULT_COMPARE_SETUPS,
        help=(
            "Comma-separated setup labels aligned with the repository convention. "
            "For seq256 this usually means 1,16,32,64,128."
        ),
    )
    parser.add_argument("--layer_reduce", type=str, default="last", choices=["mean", "last"])
    parser.add_argument("--head_reduce", type=str, default="mean", choices=["mean", "first"])
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
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Attention backend hint passed to from_pretrained when supported.",
    )
    return parser.parse_args()


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def slugify_model_name(model_name: str):
    safe = model_name.replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe.strip("_")


def resolve_out_dir(args):
    if args.out_dir is not None:
        return args.out_dir
    return REPO_ROOT / "Report" / "compare2gpt" / slugify_model_name(args.model_name) / f"seq{args.predictor_len}"


def resolve_data_dir(args):
    if args.data_dir is not None:
        return args.data_dir
    return REPO_ROOT / "data" / args.dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def sample_batch(tokens, batch_size: int, sample_len: int, rng, device: str):
    max_start = len(tokens) - sample_len
    if max_start < 0:
        raise ValueError("Dataset split is shorter than the requested sample_len.")
    starts = rng.integers(0, max_start + 1, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + sample_len].astype(np.int64)) for start in starts]
    )
    return batch.to(device), starts.tolist()


def reduce_attention_batch(attn_outputs, layer_reduce, head_reduce):
    layers = [layer_att.detach().float().cpu() for layer_att in attn_outputs]
    if layer_reduce == "last":
        attn = layers[-1]
    else:
        attn = torch.stack(layers, dim=0).mean(dim=0)

    if head_reduce == "first":
        return attn[:, 0]
    return attn.mean(dim=1)


def parse_compare_setups(compare_setups: str, predictor_len: int):
    labels = []
    for raw in compare_setups.split(","):
        raw = raw.strip()
        if not raw:
            continue
        label = int(raw)
        if label <= 0:
            raise ValueError(f"Invalid compare setup label: {label}")
        if label == 1:
            block_len = 1
            num_blocks = predictor_len
        else:
            if predictor_len % label != 0:
                raise ValueError(
                    f"predictor_len={predictor_len} is not divisible by compare setup label={label}."
                )
            num_blocks = label
            block_len = predictor_len // label
        labels.append(
            {
                "label": label,
                "num_blocks": num_blocks,
                "block_len": block_len,
            }
        )
    if not labels:
        raise ValueError("No valid compare setups were provided.")
    return labels


def aggregate_predictor_attention_to_block(attn_2d, block_len):
    if attn_2d.size(0) != attn_2d.size(1):
        raise ValueError(f"Expected a square attention matrix, got {tuple(attn_2d.shape)}.")
    num_positions = attn_2d.size(0)
    if num_positions % block_len != 0:
        raise ValueError(
            f"Predictor-aligned attention length {num_positions} is not divisible by block_len={block_len}."
        )
    num_blocks = num_positions // block_len
    return attn_2d.view(num_blocks, block_len, num_blocks, block_len).mean(dim=(1, 3))


def aggregate_batch_predictor_attentions(attn_batch, setup_specs):
    aggregated = {spec["label"]: [] for spec in setup_specs}
    for sample_idx in range(attn_batch.size(0)):
        sample_attn = attn_batch[sample_idx]
        for spec in setup_specs:
            aggregated[spec["label"]].append(
                aggregate_predictor_attention_to_block(sample_attn, block_len=spec["block_len"])
            )
    return {
        label: torch.stack(matrices, dim=0)
        for label, matrices in aggregated.items()
    }


def get_export_type_metadata(export_type: str):
    if export_type == "with_none":
        return {
            "attention_alignment": "predictor_aligned_hf_ar_with_none_proxy",
            "attention_alignment_detail": (
                "For standard Hugging Face AR models there is no explicit AO-GPT-style `[None]` "
                "predictor token. This export therefore uses the predictor-aligned submatrix "
                "`attn[:-1, :-1]` as a practical proxy for AO-GPT's `with_none` view."
            ),
            "export_note": (
                "This is a proxy `with_none` view, not a strict reproduction of AO-GPT's explicit "
                "start predictor semantics."
            ),
        }
    if export_type == "without_none":
        return {
            "attention_alignment": "real_token_only_hf_ar_without_none_proxy",
            "attention_alignment_detail": (
                "This export uses the real-token-only submatrix `attn[1:, 1:]`, which removes the "
                "first predictor/key position before aggregation. It is the closest practical proxy "
                "to AO-GPT's `without_none` view under a standard AR model."
            ),
            "export_note": (
                "This is a proxy `without_none` view for standard AR models that do not include an "
                "explicit `[None]` predictor token."
            ),
        }
    raise ValueError(f"Unsupported export_type: {export_type}")


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


def write_root_readme(out_dir: Path, args, setup_specs):
    lines = [
        f"# {out_dir.name}",
        "",
        "This directory stores Hugging Face GPT-style AR attention exports.",
        "",
        f"- `model_name`: `{args.model_name}`",
        f"- `dataset`: `{args.dataset}`",
        f"- `split`: `{args.split}`",
        f"- `predictor_len`: `{args.predictor_len}`",
        f"- `layer_reduce`: `{args.layer_reduce}`",
        f"- `head_reduce`: `{args.head_reduce}`",
        "",
        "Important comparison note:",
        "",
        "- Hugging Face GPT models do not contain an explicit AO-GPT-style `[None]` predictor token",
        "- `with_none/` is exported as a predictor-aligned proxy using `attn[:-1, :-1]`",
        "- `without_none/` is exported as a real-token-only proxy using `attn[1:, 1:]`",
        "- there is no exact `diff = with_none - without_none` analogue here",
        "",
        "Available setup labels:",
        "",
    ]
    for spec in setup_specs:
        lines.append(
            f"- `block{spec['label']}`: num_blocks={spec['num_blocks']}, block_len={spec['block_len']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_results(
    args,
    out_dir: Path,
    data_dir: Path,
    setup_specs,
    total_samples: int,
    loss_sum: float,
    logits_shape,
    batch_start_offsets,
    matrix_sums,
):
    if total_samples <= 0:
        raise RuntimeError("No attention matrices were aggregated.")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_root_readme(out_dir, args, setup_specs)

    for spec in setup_specs:
        label = spec["label"]
        for export_type in EXPORT_TYPES:
            matrix = matrix_sums[export_type][label] / total_samples
            setup_dir = out_dir / f"block{label}" / export_type
            setup_dir.mkdir(parents=True, exist_ok=True)

            base_name = f"block_attention_{export_type}"
            np.save(setup_dir / f"{base_name}.npy", matrix.numpy())

            metadata = {
                "model_name": args.model_name,
                "data_dir": str(data_dir),
                "split": args.split,
                "seed": int(args.seed),
                "num_batches": int(args.num_batches),
                "batch_size": int(args.batch_size),
                "num_samples_aggregated": int(total_samples),
                "mean_loss": float(loss_sum / total_samples),
                "logits_shape": logits_shape,
                "predictor_len": int(args.predictor_len),
                "sample_input_len": int(args.predictor_len + 1),
                "compare_setup_label": int(label),
                "num_blocks": int(spec["num_blocks"]),
                "block_len": int(spec["block_len"]),
                "axis_labels": list(range(spec["num_blocks"])),
                "layer_reduce": args.layer_reduce,
                "head_reduce": args.head_reduce,
                "export_type": export_type,
                "frame_alignment": "true_l2r_order",
                "frame_alignment_detail": (
                    "rows and columns are indexed directly by the standard autoregressive left-to-right "
                    "token order after the selected proxy alignment."
                ),
                "token_id_space_assumption": (
                    "the input bin file is assumed to use GPT-2 BPE token ids compatible with the selected "
                    "Hugging Face model."
                ),
                "batch_start_offsets_preview": batch_start_offsets[: min(32, len(batch_start_offsets))],
            }
            metadata.update(get_export_type_metadata(export_type))
            (setup_dir / f"{base_name}_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            png_ok = save_heatmap_png(
                matrix.numpy(),
                setup_dir / f"{base_name}.png",
                title=(
                    f"HF GPT attention {export_type} | model={args.model_name} | "
                    f"setup=block{label} | predictor_len={args.predictor_len}"
                ),
            )
            if png_ok:
                print(f"saved heatmap png to {setup_dir / f'{base_name}.png'}")
            else:
                print(f"matplotlib not installed; skipped png export for {setup_dir / base_name}")


def load_model(args):
    load_kwargs = {"output_attentions": True}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.to(args.device)
    model.eval()
    return model


def main():
    args = parse_args()
    out_dir = resolve_out_dir(args)
    data_dir = resolve_data_dir(args)
    tokens = load_tokens(data_dir, args.split)
    setup_specs = parse_compare_setups(args.compare_setups, predictor_len=args.predictor_len)
    model = load_model(args)
    rng = np.random.default_rng(args.seed)
    autocast_context = get_autocast_context(args.device, args.dtype)

    total_samples = 0
    loss_sum = 0.0
    logits_shape = None
    batch_start_offsets = []
    matrix_sums = {
        export_type: {spec["label"]: None for spec in setup_specs}
        for export_type in EXPORT_TYPES
    }

    sample_input_len = args.predictor_len + 1

    for batch_idx in range(int(args.num_batches)):
        input_ids, starts = sample_batch(
            tokens,
            batch_size=args.batch_size,
            sample_len=sample_input_len,
            rng=rng,
            device=args.device,
        )
        batch_start_offsets.extend(int(v) for v in starts)

        with torch.no_grad():
            with autocast_context:
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_attentions=True,
                    use_cache=False,
                )

        if outputs.attentions is None:
            raise RuntimeError("The selected Hugging Face model did not return attention tensors.")

        logits_shape = list(outputs.logits.shape)
        attn_batch = reduce_attention_batch(
            outputs.attentions,
            layer_reduce=args.layer_reduce,
            head_reduce=args.head_reduce,
        )
        export_batches = {
            "with_none": attn_batch[:, :-1, :-1],
            "without_none": attn_batch[:, 1:, 1:],
        }

        for export_type, export_batch in export_batches.items():
            batch_matrices = aggregate_batch_predictor_attentions(
                export_batch,
                setup_specs=setup_specs,
            )
            for spec in setup_specs:
                label = spec["label"]
                batch_sum = batch_matrices[label].sum(dim=0)
                if matrix_sums[export_type][label] is None:
                    matrix_sums[export_type][label] = batch_sum
                else:
                    matrix_sums[export_type][label] += batch_sum

        batch_samples = int(attn_batch.size(0))
        total_samples += batch_samples
        loss_sum += float(outputs.loss.item()) * batch_samples

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx + 1 == int(args.num_batches):
            print(f"processed {batch_idx + 1}/{args.num_batches} batches")

    export_results(
        args=args,
        out_dir=out_dir,
        data_dir=data_dir,
        setup_specs=setup_specs,
        total_samples=total_samples,
        loss_sum=loss_sum,
        logits_shape=logits_shape,
        batch_start_offsets=batch_start_offsets,
        matrix_sums=matrix_sums,
    )
    print(f"saved Hugging Face GPT attention arrays, pngs, and metadata to {out_dir}")


if __name__ == "__main__":
    main()
