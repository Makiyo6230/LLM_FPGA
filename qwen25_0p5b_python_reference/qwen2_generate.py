"""Pure Python/PyTorch Qwen2.5-0.5B greedy inference reference.

This script does not instantiate AutoModelForCausalLM and does not call
model.generate(). It loads model.safetensors directly, then composes the module
functions from qwen2_module_reference.py into a full Qwen2 forward/decode loop.

Tokenizer loading still uses Transformers because tokenization is CPU input
preparation, not an FPGA compute module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from qwen2_module_reference import (
    Qwen2Shape,
    apply_rotary_pos_emb,
    attention_core,
    attention_output_projection,
    greedy_next_token,
    lm_head,
    project_qkv,
    rmsnorm,
    swiglu_mlp,
)


DEFAULT_PROMPT = "你好，请用一句话介绍你自己。"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure Qwen2.5-0.5B reference inference.")
    parser.add_argument("--model-path", default="/home/nvidia/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_config(model_path: Path) -> dict:
    return json.loads((model_path / "config.json").read_text(encoding="utf-8"))


def load_weights(model_path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    weights = load_file(str(model_path / "model.safetensors"), device="cpu")
    return {name: tensor.to(device=device, dtype=dtype) for name, tensor in weights.items()}


def make_shape(config: dict) -> Qwen2Shape:
    return Qwen2Shape(
        hidden_size=int(config["hidden_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])),
        vocab_size=int(config["vocab_size"]),
    )


def rotary_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if q_len == 1:
        return None
    mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    past_len = kv_len - q_len
    query_positions = torch.arange(q_len, device=device).unsqueeze(1) + past_len
    key_positions = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = mask.masked_fill(key_positions <= query_positions, 0)
    return mask.view(1, 1, q_len, kv_len)


def decoder_layer(
    hidden_states: torch.Tensor,
    layer_idx: int,
    weights: dict[str, torch.Tensor],
    shape: Qwen2Shape,
    config: dict,
    position_ids: torch.Tensor,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    prefix = f"model.layers.{layer_idx}"
    eps = float(config["rms_norm_eps"])
    rope_theta = float(config["rope_theta"])

    residual = hidden_states
    hidden_states = rmsnorm(hidden_states, weights[f"{prefix}.input_layernorm.weight"], eps)

    query, key, value = project_qkv(
        hidden_states,
        weights[f"{prefix}.self_attn.q_proj.weight"],
        weights[f"{prefix}.self_attn.k_proj.weight"],
        weights[f"{prefix}.self_attn.v_proj.weight"],
        weights.get(f"{prefix}.self_attn.q_proj.bias"),
        weights.get(f"{prefix}.self_attn.k_proj.bias"),
        weights.get(f"{prefix}.self_attn.v_proj.bias"),
        shape,
    )
    cos, sin = rotary_cos_sin(position_ids, shape.head_dim, rope_theta, hidden_states.dtype, hidden_states.device)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if past_key_value is not None:
        key = torch.cat((past_key_value[0], key), dim=2)
        value = torch.cat((past_key_value[1], value), dim=2)
    new_past_key_value = (key, value)

    mask = causal_mask(query.shape[2], key.shape[2], hidden_states.device, hidden_states.dtype)
    _, _, _, _, context = attention_core(
        query,
        key,
        value,
        mask,
        shape.num_attention_heads,
        shape.num_key_value_heads,
    )
    attn_output = attention_output_projection(
        context,
        weights[f"{prefix}.self_attn.o_proj.weight"],
        weights.get(f"{prefix}.self_attn.o_proj.bias"),
    )
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = rmsnorm(hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
    _, _, _, _, mlp_output = swiglu_mlp(
        hidden_states,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )
    hidden_states = residual + mlp_output
    return hidden_states, new_past_key_value


def qwen2_forward(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    shape: Qwen2Shape,
    config: dict,
    position_ids: torch.Tensor,
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])
    next_past_key_values = []
    num_layers = int(config["num_hidden_layers"])

    for layer_idx in range(num_layers):
        past_key_value = None if past_key_values is None else past_key_values[layer_idx]
        hidden_states, new_past_key_value = decoder_layer(
            hidden_states,
            layer_idx,
            weights,
            shape,
            config,
            position_ids,
            past_key_value,
        )
        next_past_key_values.append(new_past_key_value)

    hidden_states = rmsnorm(hidden_states, weights["model.norm.weight"], float(config["rms_norm_eps"]))
    logits = lm_head(hidden_states, weights["model.embed_tokens.weight"])
    return logits, next_past_key_values


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([chat_text], return_tensors="pt")
    input_ids = model_inputs.input_ids.to(device)

    config = load_config(model_path)
    shape = make_shape(config)

    load_start = time.time()
    weights = load_weights(model_path, device, dtype)
    load_seconds = time.time() - load_start

    generated_ids: list[int] = []
    past_key_values = None
    next_token = None

    generate_start = time.time()
    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
        logits, past_key_values = qwen2_forward(input_ids, weights, shape, config, position_ids, past_key_values)
        next_token = greedy_next_token(logits).view(1, 1)
        generated_ids.append(int(next_token.item()))

        for step_index in range(1, args.max_new_tokens):
            position_ids = torch.tensor([[input_ids.shape[1] + step_index - 1]], device=device, dtype=torch.long)
            logits, past_key_values = qwen2_forward(next_token, weights, shape, config, position_ids, past_key_values)
            next_token = greedy_next_token(logits).view(1, 1)
            generated_ids.append(int(next_token.item()))

    generate_seconds = time.time() - generate_start
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    metadata = {
        "model_path": str(model_path),
        "prompt": args.prompt,
        "messages": messages,
        "chat_text": chat_text,
        "input_token_count": int(input_ids.shape[1]),
        "input_ids": input_ids[0].detach().cpu().tolist(),
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_tokens": generated_tokens,
        "response": response,
        "load_seconds": round(load_seconds, 4),
        "generate_seconds": round(generate_seconds, 4),
        "tokens_per_second": round(len(generated_ids) / generate_seconds, 4) if generate_seconds > 0 else None,
        "dtype": args.dtype,
        "decode_mode": f"pure python qwen2_module_reference greedy max_new_tokens={args.max_new_tokens}",
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
