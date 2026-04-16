from pathlib import Path


def checkpoint_rel_dir(repo_root: Path, ckpt_path: Path) -> Path:
    repo_root = repo_root.resolve()
    ckpt_path = ckpt_path.resolve()
    try:
        rel = ckpt_path.relative_to(repo_root)
    except ValueError:
        return ckpt_path.parent.name and Path(ckpt_path.parent.name) or Path("external_checkpoint")
    if rel.name == "ckpt.pt":
        return rel.parent
    return rel.parent / rel.stem


def default_eval_out_dir(repo_root: Path, ckpt_path: Path, task_name: str) -> Path:
    return repo_root / "Report" / "eval" / checkpoint_rel_dir(repo_root, ckpt_path) / task_name


def default_eval_out_file(repo_root: Path, ckpt_path: Path, task_name: str, filename: str) -> Path:
    return default_eval_out_dir(repo_root, ckpt_path, task_name) / filename
