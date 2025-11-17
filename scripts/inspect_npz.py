#!/usr/bin/env python3
"""
Inspect a .npz file: list arrays, show shapes/dtypes/sizes, preview data, and export an array.

Usage (from repo root):
  python scripts/inspect_npz.py results/traces/synth_spikes_n512_s005__hs_em__r1__orig.npz
  python scripts/inspect_npz.py path/to/file.npz --keys
  python scripts/inspect_npz.py path/to/file.npz --array beta --head 10
  python scripts/inspect_npz.py path/to/file.npz --array tau2_path --stats
  python scripts/inspect_npz.py path/to/file.npz --array alpha_hat --export alpha_hat.csv
  python scripts/inspect_npz.py path/to/file.npz --array beta --export beta.npy
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple

def human_size(num: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if num < 1024:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"

def summarize_array(name: str, arr: np.ndarray) -> str:
    shape = "×".join(str(d) for d in arr.shape) if arr.shape else "scalar"
    nbytes = arr.nbytes
    return f"{name}: shape=({shape}), dtype={arr.dtype}, size={arr.size}, nbytes={human_size(nbytes)}"

def print_table(rows: list[Tuple[str, str, str, str]]) -> None:
    # Simple fixed-width columns
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    header = ("name", "shape", "dtype", "nbytes")
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    fmt = "  {:%d}  {:%d}  {:%d}  {:%d}" % tuple(widths)
    print(fmt.format(*header))
    print(fmt.format(*("-"*widths[0], "-"*widths[1], "-"*widths[2], "-"*widths[3])))
    for r in rows:
        print(fmt.format(*r))

def export_array(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".npy":
        np.save(out_path, arr)
    elif suffix == ".npz":
        np.savez_compressed(out_path, arr=arr)
    elif suffix == ".csv":
        if arr.ndim == 1:
            np.savetxt(out_path, arr, delimiter=",")
        elif arr.ndim == 2:
            np.savetxt(out_path, arr, delimiter=",")
        else:
            raise ValueError("CSV export supports only 1D or 2D arrays.")
    elif suffix in (".txt",):
        np.savetxt(out_path, arr.reshape(-1), fmt="%.6g")
    else:
        raise ValueError(f"Unsupported export format: {suffix} (use .npy, .npz, .csv, or .txt)")
    print(f"Exported to {out_path}")

def main():
    p = argparse.ArgumentParser(description="Inspect .npz contents.")
    p.add_argument("npz_path", type=str, help="Path to .npz file")
    p.add_argument("--keys", action="store_true", help="Only list array names")
    p.add_argument("--array", type=str, default="", help="Name of array to preview/export")
    p.add_argument("--head", type=int, default=10, help="Preview first N elements (flattened) if --array is set")
    p.add_argument("--stats", action="store_true", help="Show min/max/mean/std for the selected array")
    p.add_argument("--export", type=str, default="", help="Export selected array to .npy/.npz/.csv/.txt")
    p.add_argument("--allow-pickle", action="store_true", help="Allow loading object arrays (default: off)")
    args = p.parse_args()

    path = Path(args.npz_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with np.load(path, allow_pickle=args.allow_pickle) as data:
        keys = list(data.keys())
        if args.keys and not args.array:
            print(f"{path}  ({len(keys)} arrays)")
            for k in keys:
                print("  -", k)
            return

        # Summary table
        rows = []
        for k in keys:
            arr = data[k]
            rows.append((k, "×".join(str(d) for d in arr.shape) if arr.shape else "scalar",
                         str(arr.dtype), human_size(arr.nbytes)))
        print(f"{path}  ({len(keys)} arrays)\n")
        print_table(rows)
        print()

        if args.array:
            if args.array not in data:
                raise SystemExit(f"Array '{args.array}' not found. Available: {', '.join(keys)}")
            arr = data[args.array]
            print("Selected:", summarize_array(args.array, arr))
            if args.stats and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                print(f"  min={np.min(arr):.6g}, max={np.max(arr):.6g}, mean={np.mean(arr):.6g}, std={np.std(arr):.6g}")
            # Preview
            flat = arr.reshape(-1)
            n = min(args.head, flat.size)
            if n > 0:
                np.set_printoptions(edgeitems=3, threshold=10, linewidth=120)
                print(f"  head[{n}]:", flat[:n])
            # Export
            if args.export:
                export_array(arr, Path(args.export))

if __name__ == "__main__":
    main()
