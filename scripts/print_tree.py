#!/usr/bin/env python3
"""
Print a clean project directory tree.

Usage examples (run from repo root):
  python scripts/print_tree.py
  python scripts/print_tree.py --max-depth 3
  python scripts/print_tree.py --sizes --output project_tree.txt
  python scripts/print_tree.py --ignore ".git,node_modules,results/*.npz,data/raw/*"
  python scripts/print_tree.py --root path/to/project --only-dirs
"""
from __future__ import annotations
import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime

DEFAULT_IGNORES = [
    # VCS / tooling
    ".git", ".github", ".svn", ".hg", ".DS_Store",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".vscode", ".idea",
    # Python
    "__pycache__", "*.egg-info", "build", "dist",
    # Envs
    ".venv", "venv", "env", ".conda", ".env",
    # JS
    "node_modules",
    # Notebooks
    ".ipynb_checkpoints",
    # Results / data (tweak these for your repo)
    "results/figs", "results/artifacts", "results/*.npz", "results/*.npy",
    "results/runs.jsonl*", "results/*.pickle", "results/*.pkl",
    "data/raw/*", "data/cache", "data/tmp", "tmp", "logs",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print project directory tree.")
    p.add_argument("--root", type=str, default=".", help="Root directory to scan (default: .)")
    p.add_argument("--max-depth", type=int, default=5, help="Maximum depth to print (default: 5, 0 = only root).")
    p.add_argument("--ignore", type=str, default=",".join(DEFAULT_IGNORES),
                   help="Comma-separated glob patterns to ignore.")
    p.add_argument("--only-dirs", action="store_true", help="List directories only (no files).")
    p.add_argument("--sizes", action="store_true", help="Show file sizes (human-readable).")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow directory symlinks.")
    p.add_argument("--output", type=str, default="", help="Write to file instead of stdout (e.g., project_tree.txt).")
    return p.parse_args()

def human_size(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if num < 1024:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}EB"

def compile_ignores(patterns_csv: str) -> List[str]:
    pats = [p.strip() for p in patterns_csv.split(",") if p.strip()]
    return pats if pats else []

def matches_any(path: Path, patterns: Iterable[str], root: Path) -> bool:
    """
    Return True if 'path' matches any ignore glob.
    Matches both the name and the POSIX relative path.
    """
    name = path.name
    rel = path.relative_to(root).as_posix() if path != root else ""
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or (rel and fnmatch.fnmatch(rel, pat)):
            return True
        # Also allow component-level ignores like ".git" anywhere in the path
        for part in path.parts:
            if fnmatch.fnmatch(part, pat):
                return True
    return False

def list_children(dir_path: Path, ignores: List[str], root: Path, only_dirs: bool) -> Tuple[List[Path], List[Path]]:
    dirs, files = [], []
    try:
        for entry in dir_path.iterdir():
            if matches_any(entry, ignores, root):
                continue
            if entry.is_dir():
                dirs.append(entry)
            elif not only_dirs:
                files.append(entry)
    except PermissionError:
        pass
    # Sort: directories first, then files; case-insensitive
    dirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())
    return dirs, files

def build_tree_lines(
    root: Path,
    max_depth: int,
    ignores: List[str],
    only_dirs: bool,
    sizes: bool,
    follow_symlinks: bool,
) -> Tuple[List[str], int, int]:
    lines: List[str] = []
    n_dirs = 0
    n_files = 0

    def recurse(cur: Path, prefix: str, depth: int):
        nonlocal n_dirs, n_files
        if depth > max_depth:
            return
        dirs, files = list_children(cur, ignores, root, only_dirs)
        entries = dirs + ([] if only_dirs else files)

        for i, entry in enumerate(entries):
            last = (i == len(entries) - 1)
            connector = "└── " if last else "├── "
            suffix = ""
            if sizes and entry.is_file():
                try:
                    suffix = f"  ({human_size(entry.stat().st_size)})"
                except Exception:
                    suffix = ""
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(f"{prefix}{connector}{name}{suffix}")

            if entry.is_dir():
                n_dirs += 1
                if depth < max_depth and (follow_symlinks or not entry.is_symlink()):
                    new_prefix = prefix + ("    " if last else "│   ")
                    recurse(entry, new_prefix, depth + 1)
            else:
                n_files += 1

    # Header line for root
    lines.append(root.resolve().as_posix() + "/")
    recurse(root, prefix="", depth=1)
    return lines, n_dirs, n_files

def main():
    args = parse_args()
    root = Path('../').resolve()
    ignores = compile_ignores(args.ignore)

    lines, n_dirs, n_files = build_tree_lines(
        root=root,
        max_depth=max(0, args.max_depth),
        ignores=ignores,
        only_dirs=args.only_dirs,
        sizes=args.sizes,
        follow_symlinks=args.follow_symlinks,
    )

    summary = f"\n{n_dirs} directories, {n_files} files  ·  depth≤{args.max_depth}  ·  generated {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
    output = "\n".join(lines) + summary

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote tree to {out_path}")
    else:
        print(output)

if __name__ == "__main__":
    main()
