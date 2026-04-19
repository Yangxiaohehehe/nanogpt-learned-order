"""
Train a config while saving iteration checkpoints, then scan selected steps with
structured pair mining to inspect how high-scoring pairs emerge over training.

Typical usage:
python scripts/runner/train_and_pair_scan.py \
  config/WikiText103/seq256/permute/block1/random.py \
  --steps 500,1000,1500,2000,3000 \
  --scan_out_root Report/analysis/early_pair_scan/permute/seq256/block1
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def parse_csv_list(raw_value, cast_fn):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def run_command(cmd, cwd):
    print("[runner] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_config_file(config_path: Path):
    namespace = {}
    exec(config_path.read_text(encoding="utf-8"), namespace)
    return namespace


def infer_config_label(config_path: Path, config_ns):
    out_dir = str(config_ns.get("out_dir", "")).strip()
    if out_dir:
        return Path(out_dir).name
    return config_path.stem


def build_train_scan_roots(config_path: Path, config_ns, scan_out_root_arg):
    label = infer_config_label(config_path, config_ns)
    train_out_dir = REPO_ROOT / "out" / "analysis_pair_scan" / label
    if scan_out_root_arg:
        scan_out_root = scan_out_root_arg if scan_out_root_arg.is_absolute() else REPO_ROOT / scan_out_root_arg
    else:
        scan_out_root = REPO_ROOT / "Report" / "analysis" / "early_pair_scan" / label
    checkpoint_dir = train_out_dir / "checkpoints"
    return train_out_dir, checkpoint_dir, scan_out_root


def build_train_cmd(args, config_path: Path, train_out_dir: Path, checkpoint_dir: Path, max_step: int):
    cmd = [
        sys.executable,
        "train.py",
        str(config_path),
        f"--out_dir={train_out_dir}",
        f"--max_iters={int(max_step)}",
        "--save_iter_checkpoints=True",
        f"--save_iter_checkpoint_dir={checkpoint_dir}",
        "--save_iter_checkpoint_keep=0",
    ]
    if bool(args.disable_wandb):
        cmd.append("--wandb_log=False")
    if str(args.device).strip():
        cmd.append(f"--device={args.device}")
    if str(args.dtype).strip():
        cmd.append(f"--dtype={args.dtype}")
    if bool(args.no_compile):
        cmd.append("--compile=False")
    for override in args.train_override:
        cmd.append(override)
    return cmd


def build_scan_cmd(args, checkpoint_dir: Path, scan_out_root: Path, steps_csv: str):
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "runner" / "early_signal_scan.py"),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--out_root={scan_out_root}",
        f"--steps={steps_csv}",
        f"--split={args.split}",
        f"--batch_size={int(args.batch_size)}",
        f"--num_batches={int(args.num_batches)}",
        f"--pair_mining_batches={int(args.pair_mining_batches)}",
        f"--pair_eval_batch_size={int(args.pair_eval_batch_size)}",
        f"--candidate_eval_batch_size={int(args.candidate_eval_batch_size)}",
        f"--random_pool_size={int(args.random_pool_size)}",
        f"--structured_pool_size={int(args.structured_pool_size)}",
        f"--top_pair_pool_size={int(args.top_pair_pool_size)}",
        f"--aggregate_top_k_pairs={int(args.aggregate_top_k_pairs)}",
        f"--prefix_len={int(args.prefix_len)}",
        f"--segment_len={int(args.segment_len)}",
        f"--pair_score_k={int(args.pair_score_k)}",
        f"--tv_weight={float(args.tv_weight)}",
        f"--log_every_batches={int(args.log_every_batches)}",
        f"--seed={int(args.seed)}",
    ]
    if str(args.dataset).strip():
        cmd.append(f"--dataset={args.dataset}")
    if args.data_dir is not None:
        cmd.append(f"--data_dir={args.data_dir}")
    if str(args.device).strip():
        cmd.append(f"--device={args.device}")
    if str(args.dtype).strip():
        cmd.append(f"--dtype={args.dtype}")
    if bool(args.skip_existing):
        cmd.append("--skip_existing")
    for override in args.scan_override:
        cmd.append(override)
    return cmd


def build_manifest(
    config_path: Path,
    train_out_dir: Path,
    checkpoint_dir: Path,
    scan_out_root: Path,
    steps,
    args,
):
    return {
        "config_path": str(config_path),
        "train_out_dir": str(train_out_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "scan_out_root": str(scan_out_root),
        "steps": [int(v) for v in steps],
        "train_overrides": list(args.train_override),
        "scan_overrides": list(args.scan_override),
        "benchmark": {
            "split": str(args.split),
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "pair_mining_batches": int(args.pair_mining_batches),
            "pair_eval_batch_size": int(args.pair_eval_batch_size),
            "candidate_eval_batch_size": int(args.candidate_eval_batch_size),
            "random_pool_size": int(args.random_pool_size),
            "structured_pool_size": int(args.structured_pool_size),
            "top_pair_pool_size": int(args.top_pair_pool_size),
            "aggregate_top_k_pairs": int(args.aggregate_top_k_pairs),
            "prefix_len": int(args.prefix_len),
            "segment_len": int(args.segment_len),
            "pair_score_k": int(args.pair_score_k),
            "tv_weight": float(args.tv_weight),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a config while saving iteration checkpoints, then run early pair/segment "
            "scans on selected steps. Useful for seq256/block1 permute or non-permute studies."
        )
    )
    parser.add_argument("config", type=Path, help="Training config, e.g. config/.../random.py")
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Comma-separated training steps to preserve/evaluate, e.g. 500,1000,1500,2000",
    )
    parser.add_argument("--scan_out_root", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--pair_mining_batches", type=int, default=20)
    parser.add_argument("--pair_eval_batch_size", type=int, default=4)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--random_pool_size", type=int, default=64)
    parser.add_argument("--structured_pool_size", type=int, default=64)
    parser.add_argument("--top_pair_pool_size", type=int, default=128)
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--segment_len", type=int, default=6)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--train_override",
        action="append",
        default=[],
        help="Extra override passed through to train.py, e.g. --train_override=--learning_rate=5e-4",
    )
    parser.add_argument(
        "--scan_override",
        action="append",
        default=[
            "--pair_mining_mode=attention_pruned",
            "--attn_top_k=10",
            "--attn_num_batches=32",
            "--attn_batch_size=32",
            "--attn_mode=Random",
            "--attn_symmetrize=mean",
            "--attn_export_type=with_none",
            "--skip_candidate_pool_eval",
        ],
        help="Extra override passed through to early_signal_scan.py / structured_candidate_benchmark.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    steps = parse_csv_list(args.steps, int)
    if not steps:
        raise ValueError("No valid --steps were provided.")
    max_step = max(int(v) for v in steps)
    steps_csv = ",".join(str(int(v)) for v in steps)

    config_ns = load_config_file(config_path)
    train_out_dir, checkpoint_dir, scan_out_root = build_train_scan_roots(
        config_path,
        config_ns,
        args.scan_out_root,
    )
    train_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scan_out_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        config_path=config_path,
        train_out_dir=train_out_dir,
        checkpoint_dir=checkpoint_dir,
        scan_out_root=scan_out_root,
        steps=steps,
        args=args,
    )
    manifest_path = scan_out_root / "runner_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[runner] saved manifest to {manifest_path}")

    if not bool(args.skip_train):
        run_command(
            build_train_cmd(
                args=args,
                config_path=config_path,
                train_out_dir=train_out_dir,
                checkpoint_dir=checkpoint_dir,
                max_step=max_step,
            ),
            cwd=REPO_ROOT,
        )

    run_command(
        build_scan_cmd(
            args=args,
            checkpoint_dir=checkpoint_dir,
            scan_out_root=scan_out_root,
            steps_csv=steps_csv,
        ),
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
