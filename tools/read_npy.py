#!/usr/bin/env python3
"""Inspect .npy tensors used by the Qwen2.5 FPGA golden trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one .npy file or scan a directory of .npy files."
    )
    parser.add_argument("path", type=Path, help="Path to a .npy file or a directory.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan a directory for .npy files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of files to print when path is a directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=16,
        help="Number of flattened values to print from each array.",
    )
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Only print metadata and statistics, not sample values.",
    )
    return parser.parse_args()


def iter_npy_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix != ".npy":
            raise ValueError(f"Not a .npy file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(path.glob(pattern))


def format_number(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def numeric_stats(array: np.ndarray) -> list[str]:
    if not np.issubdtype(array.dtype, np.number):
        return ["stats: non-numeric array"]

    array32 = array.astype(np.float32, copy=False)
    finite = np.isfinite(array32)
    finite_count = int(finite.sum())
    total = int(array32.size)

    if finite_count == 0:
        return [f"finite: 0 / {total}", "stats: no finite values"]

    finite_values = array32[finite]
    return [
        f"finite: {finite_count} / {total}",
        f"min: {format_number(finite_values.min())}",
        f"max: {format_number(finite_values.max())}",
        f"mean: {format_number(finite_values.mean())}",
        f"std: {format_number(finite_values.std())}",
    ]


def inspect_npy(path: Path, sample_size: int, show_values: bool) -> None:
    array = np.load(path, allow_pickle=False)
    print(f"file: {path}")
    print(f"shape: {array.shape}")
    print(f"ndim: {array.ndim}")
    print(f"dtype: {array.dtype}")
    print(f"elements: {array.size}")

    for line in numeric_stats(array):
        print(line)

    if show_values and array.size > 0:
        sample = array.reshape(-1)[:sample_size]
        values = ", ".join(format_number(v) for v in sample)
        print(f"sample[{len(sample)}]: {values}")

    print()


def main() -> None:
    args = parse_args()
    files = iter_npy_files(args.path, args.recursive)

    if not files:
        print(f"No .npy files found: {args.path}")
        return

    selected = files[: args.limit]
    for file_path in selected:
        inspect_npy(file_path, args.sample_size, not args.no_values)

    if len(files) > len(selected):
        print(f"Skipped {len(files) - len(selected)} files. Increase --limit to print more.")


if __name__ == "__main__":
    main()
# 用法示例：python LLM_FPGA\tools\read_npy.py "LLM_FPGA\qwen25_0p5b_instruct_full_generation_trace\00_prefill_full_prompt\layer_00_transformer_block\attention_rmsnorm_input.npy"