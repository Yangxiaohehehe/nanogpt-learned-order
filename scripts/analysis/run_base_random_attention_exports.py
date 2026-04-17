"""
Batch-export Random-mode attention heatmaps for random checkpoints under out/base.

Typical usage:
python scripts/analysis/run_base_random_attention_exports.py \
  --batch_size 32 \
  --num_batches 32
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE_ROOT = REPO_ROOT / "out" / "base"
ANALYSIS_ROOT = REPO_ROOT / "Report" / "analysis" / "attn"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "export_block_attention_heatmap.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find random checkpoints under out/base/{permute,nonpermute} and export "
            "Random-mode attention heatmaps into Report/analysis/attn with mirrored subpaths."
        )
    )
    parser.add_argument("--out_base_root", type=Path, default=OUT_BASE_ROOT)
    parser.add_argument("--analysis_root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--layer_reduce", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--head_reduce", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force_manual_attention", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_random_checkpoints(out_base_root: Path):
    checkpoints = []
    for ckpt_path in sorted(out_base_root.glob("**/ckpt.pt")):
        run_dir_name = ckpt_path.parent.name.lower()
        rel_parts = ckpt_path.relative_to(out_base_root).parts
        if "random" not in run_dir_name:
            continue
        if not rel_parts:
            continue
        # Only keep the two requested families under out/base.
        if rel_parts[0] not in {"permute", "nonpermute"}:
            continue
        checkpoints.append(ckpt_path)
    return checkpoints


def build_output_dir(analysis_root: Path, out_base_root: Path, ckpt_path: Path):
    rel_ckpt_dir = ckpt_path.parent.relative_to(out_base_root)
    return analysis_root / rel_ckpt_dir


def build_command(args, ckpt_path: Path, out_dir: Path):
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--ckpt_path",
        str(ckpt_path),
        "--out_dir",
        str(out_dir),
        "--split",
        args.split,
        "--batch_size",
        str(args.batch_size),
        "--num_batches",
        str(args.num_batches),
        "--seed",
        str(args.seed),
        "--mode",
        "Random",
        "--layer_reduce",
        args.layer_reduce,
        "--head_reduce",
        args.head_reduce,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.force_manual_attention:
        cmd.append("--force_manual_attention")
    return cmd


def main():
    args = parse_args()
    checkpoints = discover_random_checkpoints(args.out_base_root)
    if not checkpoints:
        raise SystemExit(f"No random checkpoints found under {args.out_base_root}")

    print(f"found {len(checkpoints)} random checkpoints under {args.out_base_root}")

    failures = []
    for index, ckpt_path in enumerate(checkpoints, start=1):
        out_dir = build_output_dir(args.analysis_root, args.out_base_root, ckpt_path)
        cmd = build_command(args, ckpt_path, out_dir)

        print(f"[{index}/{len(checkpoints)}] exporting {ckpt_path}")
        print("command:", shlex.join(cmd))
        print("output :", out_dir)

        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=REPO_ROOT)
        if completed.returncode != 0:
            failures.append((ckpt_path, completed.returncode))
            print(f"FAILED with code {completed.returncode}: {ckpt_path}")
        else:
            print(f"done: {ckpt_path}")

    if failures:
        print("\nSome exports failed:")
        for ckpt_path, code in failures:
            print(f"- code={code} :: {ckpt_path}")
        raise SystemExit(1)

    print(f"\nAll exports completed successfully into {args.analysis_root}")


if __name__ == "__main__":
    main()
