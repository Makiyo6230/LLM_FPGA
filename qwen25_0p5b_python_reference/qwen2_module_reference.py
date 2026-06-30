"""Module-level Qwen2 reference code for FPGA implementation.

This file is intentionally small and explicit. It mirrors the computation
order used by Qwen2.5 decoder layers, so each function can be compared against
the exported .npy golden tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Qwen2Shape:
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 151936


QWEN25_0P5B_SHAPE = Qwen2Shape()


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2 RMSNorm.

    Golden tensors:
      *_rmsnorm_input.npy -> hidden_states
      *_rmsnorm_output.npy -> return value
    """

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


def linear(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Linear projection used by Q/K/V/O/MLP/lm_head."""

    return F.linear(hidden_states, weight, bias)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims for RoPE."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen2 RoPE to query and key.

    Expected shapes:
      query: [batch, q_heads, seq, head_dim]
      key:   [batch, kv_heads, seq, head_dim]
      cos:   [batch, seq, head_dim]
      sin:   [batch, seq, head_dim]
    """

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    Input shape:  [batch, kv_heads, seq, head_dim]
    Output shape: [batch, kv_heads * n_rep, seq, head_dim]
    """

    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def project_qkv(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    shape: Qwen2Shape = QWEN25_0P5B_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Q/K/V projections and reshape to attention layout."""

    batch_size, seq_len, _ = hidden_states.shape
    query = linear(hidden_states, q_weight, q_bias)
    key = linear(hidden_states, k_weight, k_bias)
    value = linear(hidden_states, v_weight, v_bias)

    query = query.view(batch_size, seq_len, shape.num_attention_heads, shape.head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, shape.num_key_value_heads, shape.head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, shape.num_key_value_heads, shape.head_dim).transpose(1, 2)
    return query, key, value


def attention_core(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen2 eager attention core.

    Returns:
      key_repeated
      value_repeated
      scores_after_mask
      softmax_weights
      weighted_value_sum
    """

    num_key_value_groups = num_attention_heads // num_key_value_heads
    key_repeated = repeat_kv(key_cache, num_key_value_groups)
    value_repeated = repeat_kv(value_cache, num_key_value_groups)

    scores = torch.matmul(query, key_repeated.transpose(2, 3)) / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_repeated.shape[-2]]

    softmax_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weighted_value_sum = torch.matmul(softmax_weights, value_repeated)
    return key_repeated, value_repeated, scores, softmax_weights, weighted_value_sum


def attention_output_projection(
    context: torch.Tensor,
    o_weight: torch.Tensor,
    o_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Merge heads and apply output projection."""

    batch_size, num_heads, seq_len, head_dim = context.shape
    context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, num_heads * head_dim)
    return linear(context, o_weight, o_bias)


def swiglu_mlp(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen2 SwiGLU MLP.

    Returns:
      gate_proj
      up_proj
      silu_gate
      gate_up_mul
      down_proj
    """

    gate_proj = linear(hidden_states, gate_weight, gate_bias)
    up_proj = linear(hidden_states, up_weight, up_bias)
    silu_gate = F.silu(gate_proj)
    gate_up_mul = silu_gate * up_proj
    down_proj = linear(gate_up_mul, down_weight, down_bias)
    return gate_proj, up_proj, silu_gate, gate_up_mul, down_proj


def lm_head(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute vocabulary logits."""

    return linear(hidden_states, weight, bias)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Greedy decode token selection."""

    return torch.argmax(logits[:, -1, :], dim=-1)


def layer_flow_order() -> list[str]:
    """Canonical FPGA module implementation order for one Qwen2 decoder layer."""

    return [
        "transformer_block_input",
        "attention_rmsnorm",
        "qkv_projection",
        "rope",
        "kv_cache_update",
        "gqa_kv_repeat",
        "qk_matmul_scale",
        "causal_mask",
        "softmax",
        "weighted_value_sum",
        "attention_output_projection",
        "attention_residual_add",
        "mlp_rmsnorm",
        "mlp_gate_up_projection",
        "silu",
        "gate_times_up",
        "mlp_down_projection",
        "mlp_residual_add",
        "transformer_block_output",
    ]
