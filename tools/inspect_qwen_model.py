#!/usr/bin/env python3
"""Inspect a local Qwen/Qwen2.5 model directory.

This script reads config.json and safetensors metadata without running inference.
It is meant to answer:
  - What is the model architecture?
  - What are the key Qwen2 dimensions?
  - What weight tensors exist and what are their shapes?
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Qwen/Qwen2.5 config and safetensors weight shapes."
    )
    parser.add_argument("model_dir", type=Path, help="Local model directory.")
    parser.add_argument(
        "--max-weights",
        type=int,
        default=120,
        help="Maximum number of weight tensors to print.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only print weight names containing this substring.",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Only print config and logical module structure.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def product(shape: tuple[int, ...] | list[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def human_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def print_config_summary(config: dict[str, Any]) -> None:
    hidden_size = get(config, "hidden_size")
    num_heads = get(config, "num_attention_heads")
    num_kv_heads = get(config, "num_key_value_heads")
    head_dim = get(config, "head_dim")
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    print("== Config ==")
    rows = [
        ("model_type", get(config, "model_type")),
        ("architectures", get(config, "architectures")),
        ("torch_dtype", get(config, "torch_dtype")),
        ("vocab_size", get(config, "vocab_size")),
        ("hidden_size", hidden_size),
        ("intermediate_size", get(config, "intermediate_size")),
        ("num_hidden_layers", get(config, "num_hidden_layers")),
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("gqa_repeat", (num_heads // num_kv_heads) if num_heads and num_kv_heads else None),
        ("rms_norm_eps", get(config, "rms_norm_eps")),
        ("rope_theta", get(config, "rope_theta")),
        ("max_position_embeddings", get(config, "max_position_embeddings")),
        ("attention_bias", get(config, "attention_bias")),
        ("tie_word_embeddings", get(config, "tie_word_embeddings")),
    ]
    for name, value in rows:
        print(f"{name}: {value}")
    print()


def print_logical_structure(config: dict[str, Any]) -> None:
    vocab = get(config, "vocab_size", "V")
    hidden = get(config, "hidden_size", "H")
    inter = get(config, "intermediate_size", "I")
    layers = get(config, "num_hidden_layers", "N")
    heads = get(config, "num_attention_heads")
    kv_heads = get(config, "num_key_value_heads")
    head_dim = get(config, "head_dim")
    if head_dim is None and isinstance(hidden, int) and isinstance(heads, int):
        head_dim = hidden // heads
    q_out = heads * head_dim if isinstance(heads, int) and isinstance(head_dim, int) else "A*D"
    kv_out = kv_heads * head_dim if isinstance(kv_heads, int) and isinstance(head_dim, int) else "K*D"

    print("== Logical Qwen2 Module Structure ==")
    print("Qwen2ForCausalLM")
    print(f"  model.embed_tokens.weight                  [{vocab}, {hidden}]")
    print(f"  model.layers[0..{layers - 1 if isinstance(layers, int) else 'N-1'}]")
    print(f"    input_layernorm.weight                   [{hidden}]")
    print("    self_attn")
    print(f"      q_proj.weight                          [{q_out}, {hidden}]")
    print(f"      k_proj.weight                          [{kv_out}, {hidden}]")
    print(f"      v_proj.weight                          [{kv_out}, {hidden}]")
    print(f"      o_proj.weight                          [{hidden}, {q_out}]")
    print(f"    post_attention_layernorm.weight          [{hidden}]")
    print("    mlp")
    print(f"      gate_proj.weight                       [{inter}, {hidden}]")
    print(f"      up_proj.weight                         [{inter}, {hidden}]")
    print(f"      down_proj.weight                       [{hidden}, {inter}]")
    print(f"  model.norm.weight                          [{hidden}]")
    print(f"  lm_head.weight                             [{vocab}, {hidden}]")
    print()


def load_weight_index(model_dir: Path) -> dict[str, Any] | None:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if not index_files:
        return None
    return load_json(index_files[0])


def print_weight_index_summary(model_dir: Path) -> None:
    index = load_weight_index(model_dir)
    print("== Weight Files ==")
    if index is None:
        safetensors = sorted(model_dir.glob("*.safetensors"))
        bins = sorted(model_dir.glob("*.bin"))
        print(f"safetensors files: {len(safetensors)}")
        for file_path in safetensors[:20]:
            print(f"  {file_path.name}")
        print(f"bin files: {len(bins)}")
        for file_path in bins[:20]:
            print(f"  {file_path.name}")
        print()
        return

    weight_map = index.get("weight_map", {})
    metadata = index.get("metadata", {})
    files = Counter(weight_map.values())
    total_size = metadata.get("total_size")
    print(f"index file: {sorted(model_dir.glob('*.safetensors.index.json'))[0].name}")
    print(f"weight tensors in index: {len(weight_map)}")
    if total_size is not None:
        print(f"total_size bytes: {total_size} ({total_size / (1024 ** 3):.3f} GiB)")
    print("shards:")
    for file_name, count in sorted(files.items()):
        print(f"  {file_name}: {count} tensors")
    print()


def iter_safetensor_shapes(model_dir: Path) -> list[tuple[str, tuple[int, ...], str, str]]:
    rows: list[tuple[str, tuple[int, ...], str, str]] = []
    for file_path in sorted(model_dir.glob("*.safetensors")):
        with file_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size).decode("utf-8"))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            shape = tuple(int(dim) for dim in value["shape"])
            dtype = value["dtype"]
            rows.append((key, shape, dtype, file_path.name))
    return sorted(rows, key=lambda row: row[0])


def print_weight_shapes(model_dir: Path, max_weights: int, name_filter: str) -> None:
    print("== Weight Tensor Shapes ==")
    rows = iter_safetensor_shapes(model_dir)
    if name_filter:
        rows = [row for row in rows if name_filter in row[0]]

    total_params = sum(product(shape) for _, shape, _, _ in rows)
    print(f"matched tensors: {len(rows)}")
    print(f"matched parameters: {total_params} ({human_count(total_params)})")
    print()

    for index, (name, shape, dtype, file_name) in enumerate(rows[:max_weights], start=1):
        params = product(shape)
        shape_text = ", ".join(str(dim) for dim in shape)
        print(f"{index:04d} {name}")
        print(f"     shape=({shape_text}) dtype={dtype} params={params} file={file_name}")

    if len(rows) > max_weights:
        print(f"... skipped {len(rows) - max_weights} tensors. Increase --max-weights to print more.")
    print()


def print_first_layer_weight_guide(config: dict[str, Any]) -> None:
    print("== First Layer Weight Names For FPGA Module Bring-up ==")
    names = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for name in names:
        print(name)
    print()

    print("FPGA note:")
    print("  Activation .npy gives module input/output samples.")
    print("  Weight tensors above are still required for parameterized modules.")
    print()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    config = load_json(config_path)
    print(f"model_dir: {model_dir}")
    print()
    print_config_summary(config)
    print_logical_structure(config)
    print_weight_index_summary(model_dir)
    print_first_layer_weight_guide(config)

    if not args.no_weights:
        print_weight_shapes(model_dir, args.max_weights, args.filter)


if __name__ == "__main__":
    main()
