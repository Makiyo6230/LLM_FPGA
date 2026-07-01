# Qwen2.5-0.5B FPGA 模块拆分与验证指南

本文档用于把 `Qwen2.5-0.5B-Instruct` 的 Python 推理代码、模块级 `.npy` golden 数据和 FPGA 实现任务对齐起来。目标不是重新解释 Transformer，而是明确：

- 哪段推理代码对应哪个硬件模块。
- FPGA 每个模块需要实现什么输入、计算和输出。
- 如何用现有 `.npy` 数据逐模块验证。

## 0. 当前资料位置

```text
LLM_FPGA/
  README.md
  Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md
  Qwen2.5-0.5B_FPGA模块拆分与验证指南.md

  qwen25_0p5b_python_reference/
    manual_prefill_decode.py
    run_full_generate_trace.py
    qwen2_module_reference.py
    snapshots/
      modeling_qwen2_fpga_instrumented.py
      fpga_recorder.py

  qwen25_0p5b_instruct_full_generation_trace/
    metadata.json
    name_mapping.json
    00_prefill_full_prompt/
    01_decode_token_01/
    ...
    07_decode_token_07/
    generation_token_vocab_logits/
```

推荐阅读顺序：

```text
1. qwen25_0p5b_python_reference/manual_prefill_decode.py
   理解从 prompt 到 token 的显式 prefill/decode 推理循环。

2. qwen25_0p5b_python_reference/qwen2_module_reference.py
   理解 RMSNorm、Linear、RoPE、Attention、SwiGLU MLP、lm_head 的最小参考实现。

3. qwen25_0p5b_python_reference/snapshots/modeling_qwen2_fpga_instrumented.py
   对照 HuggingFace Qwen2 源码和 record_tensor 插桩点。

4. qwen25_0p5b_instruct_full_generation_trace/
   用每个模块保存下来的 .npy 数据做 FPGA 逐级验证。
```

## 1. 模型参数

本项目当前 demo 使用 `Qwen/Qwen2.5-0.5B-Instruct`。模型结构是 Qwen2 架构，核心特点是 RoPE、RMSNorm、SwiGLU MLP、GQA attention、QKV bias、tie word embeddings。

```text
模型: Qwen2.5-0.5B-Instruct
参数规模: 约 0.49B
层数 N: 24
hidden_size H: 896
intermediate_size I: 4864
vocab_size V: 151936

attention heads A: 14
key/value heads K: 2
GQA repeat G: A / K = 7
head_dim D: 64

RMSNorm eps: 1e-6
trace 模型 dtype: float16
generation logits 保存 dtype: float32
```

常用张量大小：

```text
hidden / token
  = H * 2
  = 896 * 2
  = 1792 bytes

Q / token / layer
  = A * D * 2
  = 14 * 64 * 2
  = 1792 bytes

K / token / layer
  = K * D * 2
  = 2 * 64 * 2
  = 256 bytes

V / token / layer
  = K * D * 2
  = 2 * 64 * 2
  = 256 bytes

KV cache / token / layer
  = K * (D + D) * 2
  = 2 * (64 + 64) * 2
  = 512 bytes

KV cache / token / all layers
  = 24 * 512
  = 12288 bytes
  = 12 KiB
```

## 2. 一次推理的总流程

一次请求分为 `prefill` 和 `decode` 两类 forward。

```text
prompt
  -> tokenizer.apply_chat_template
  -> tokenizer(...) 得到 input_ids / attention_mask
  -> prefill forward(input_ids = 全 prompt, past_key_values = None)
  -> lm_head logits
  -> argmax 得到第 1 个生成 token
  -> decode forward(input_ids = 上一个生成 token, past_key_values = KV cache)
  -> lm_head logits
  -> argmax 得到下一个 token
  -> 重复 decode
```

对应代码主线：

```text
manual_prefill_decode.py
  line 83: tokenizer.apply_chat_template(...)
  line 88: AutoModelForCausalLM.from_pretrained(...)
  line 98: past_key_values = None
  line 102: prefill forward
  line 103: 保存 outputs.past_key_values
  line 104: argmax 得到 next_token
  line 115-119: decode forward，传入 past_key_values
  line 121: 更新 past_key_values
  line 122: argmax 得到 next_token

run_full_generate_trace.py
  line 106: model.generate(...)
  用于跑 HuggingFace generate 路径，保存完整 trace。
```

FPGA runtime 最终要复现的是 `manual_prefill_decode.py` 这条显式路径，而不是依赖 `model.generate()` 的调度封装。

## 3. Golden 数据组织方式

本次 golden case：

```text
prompt: 你好，请用一句话介绍你自己。
input_token_count: 36
max_new_tokens: 8
generated response: 我是Qwen，由阿里云开发
record step count: 8
```

目录含义：

```text
00_prefill_full_prompt/
  prefill 阶段
  输入是完整 prompt，T = 36

01_decode_token_01/
  decode 第 1 步
  输入是上一步生成的 1 个 token，T = 1
  KV cache 长度从 36 增长到 37

02_decode_token_02/
  decode 第 2 步
  输入 T = 1
  KV cache 长度从 37 增长到 38

...

07_decode_token_07/
  decode 第 7 步
  输入 T = 1
  KV cache 长度增长到 43
```

每个 step 下面有 24 个 layer 目录：

```text
layer_00_transformer_block/
layer_01_transformer_block/
...
layer_23_transformer_block/
```

建议 FPGA 验证时先固定：

```text
step = 00_prefill_full_prompt
layer = layer_00_transformer_block
```

把第 0 层的模块打通后，再扩展到 24 层和 decode。

## 3.1 CPU/FPGA 分工与 golden 对应总表

`00_prefill_full_prompt` 的数据可以和本文档模块对应，但不是所有模块都必须在 FPGA 上实现。这里先把分工说清楚：

```text
CPU host 必须负责:
  文本 prompt、chat template、tokenizer、attention_mask/position_ids 准备、调度 prefill/decode。

FPGA 第一阶段建议负责:
  从 embedding 后的 hidden states 开始，完成 24 层 Transformer block、final RMSNorm、必要时 lm_head。

FPGA 完整部署可继续负责:
  token embedding lookup、lm_head + argmax、KV cache 管理。

不建议 FPGA 负责:
  tokenizer 和 chat template。它们是字符串/BPE 处理，放 CPU 更自然。
```

模块和数据对应关系：

| 文档模块                            | 主要实现位置                   | 是否建议 FPGA 实现 | `00_prefill_full_prompt` 可用验证数据                                                                                                                                                         |
| ----------------------------------- | ------------------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模块 0 Tokenizer 与输入准备         | CPU host                       | 不建议             | 输入在顶层`metadata.json` 的 `prompt/messages/chat_text`；输出是 `model_input_token_ids.npy`、`position_ids.npy`                                                                        |
| 模块 1 Token Embedding              | CPU 或 FPGA                    | 可选               | `model_input_token_ids.npy` -> `token_embedding_output.npy`                                                                                                                                 |
| 模块 2 Transformer Block 输入/输出  | FPGA 调度边界                  | 需要               | `layer_XX_transformer_block/transformer_block_input.npy` -> `transformer_block_output.npy`                                                                                                  |
| 模块 3 Attention 前 RMSNorm         | FPGA                           | 需要               | `attention_rmsnorm_input.npy` -> `attention_rmsnorm_output.npy`                                                                                                                             |
| 模块 4 Q/K/V Linear                 | FPGA                           | 需要               | `attention_input_after_rmsnorm.npy` -> `attention_q_projection.npy` / `attention_k_projection.npy` / `attention_v_projection.npy`                                                       |
| 模块 5 Reshape Head 与 RoPE         | FPGA，cos/sin 可由 CPU 预先给  | 需要               | `attention_q_before_rope.npy` / `attention_k_before_rope.npy` + `attention_rope_cos.npy` / `attention_rope_sin.npy` -> `attention_q_after_rope.npy` / `attention_k_after_rope.npy`  |
| 模块 6 KV Cache Update              | FPGA 或 runtime cache manager  | 需要               | `attention_k_after_rope.npy` / `attention_value_states.npy` -> `attention_k_cache_after_update.npy` / `attention_v_cache_after_update.npy`                                              |
| 模块 7 GQA Repeat                   | FPGA 内部地址映射或显式 repeat | 需要               | `attention_k_cache_after_update.npy` / `attention_v_cache_after_update.npy` -> `attention_k_repeated_for_gqa.npy` / `attention_v_repeated_for_gqa.npy`                                  |
| 模块 8 QK/Mask/Softmax              | FPGA                           | 需要               | `attention_q_after_rope.npy` + `attention_k_repeated_for_gqa.npy` -> `attention_qk_scores_before_mask.npy` -> `attention_qk_scores_after_mask.npy` -> `attention_softmax_weights.npy` |
| 模块 9 Attention weighted V         | FPGA                           | 需要               | `attention_softmax_weights.npy` + `attention_v_repeated_for_gqa.npy` -> `attention_weighted_value_sum.npy` / `attention_context_before_output_projection.npy`                           |
| 模块 10 Attention output projection | FPGA                           | 需要               | `attention_output_projection_input.npy` -> `attention_output_projection.npy`                                                                                                                |
| 模块 11 Attention residual add      | FPGA                           | 需要               | `transformer_block_input.npy` + `attention_module_output.npy` -> `attention_residual_add_output.npy`                                                                                      |
| 模块 12 MLP 前 RMSNorm              | FPGA                           | 需要               | `mlp_rmsnorm_input.npy` -> `mlp_rmsnorm_output.npy`                                                                                                                                         |
| 模块 13 SwiGLU MLP                  | FPGA                           | 需要               | `mlp_input_after_rmsnorm.npy` -> `mlp_gate_projection.npy` / `mlp_up_projection.npy` -> `mlp_silu_gate_activation.npy` -> `mlp_silu_gate_times_up.npy` -> `mlp_down_projection.npy` |
| 模块 14 MLP residual add            | FPGA                           | 需要               | `attention_residual_add_output.npy` + `mlp_module_output.npy` -> `transformer_block_output.npy`                                                                                           |
| 模块 15 Final RMSNorm               | FPGA                           | 需要               | `final_rmsnorm_input.npy` -> `final_rmsnorm_output.npy`                                                                                                                                     |
| 模块 16 LM Head 与 argmax           | FPGA 或 CPU 分阶段实现         | 可选到需要         | `lm_head_input.npy` -> `lm_head_vocab_logits.npy`；token 选择可对比 `metadata.json` 的 `generated_ids`                                                                                  |

注意：

```text
1. .npy trace 主要保存 activation 中间量，不保存全部权重。
2. Linear / Embedding / lm_head / RMSNorm 验证时还需要模型权重。
3. Tokenizer 的输入是文本，所以在 metadata.json，不是 .npy。
4. 早期 FPGA demo 可以从 token_embedding_output.npy 开始，先跳过 tokenizer 和 embedding。
5. 如果只验证单模块，可以直接用该模块输入 .npy 作为 FPGA 输入，用对应输出 .npy 做 golden。
```

## 4. Python 源码到 Qwen2 模块路径

HuggingFace 模型调用路径：

```text
Qwen2ForCausalLM.forward
  -> Qwen2Model.forward
      -> token embedding
      -> rotary embedding cos/sin
      -> 24 * Qwen2DecoderLayer.forward
          -> input RMSNorm
          -> Qwen2Attention.forward
              -> q_proj / k_proj / v_proj
              -> RoPE
              -> KV cache update
              -> GQA repeat_kv
              -> QK^T / scale / mask / softmax
              -> attention @ V
              -> o_proj
          -> residual add
          -> post-attention RMSNorm
          -> Qwen2MLP.forward
              -> gate_proj
              -> up_proj
              -> SiLU
              -> multiply
              -> down_proj
          -> residual add
      -> final RMSNorm
  -> lm_head
  -> greedy argmax
```

核心源码位置：

```text
qwen2_module_reference.py
  line 29: rmsnorm
  line 44: linear
  line 58: apply_rotary_pos_emb
  line 81: repeat_kv
  line 95: project_qkv
  line 118: attention_core
  line 149: attention_output_projection
  line 161: swiglu_mlp
  line 188: lm_head
  line 194: greedy_next_token
  line 200: layer_flow_order

snapshots/modeling_qwen2_fpga_instrumented.py
  line 36: Qwen2MLP
  line 136: apply_rotary_pos_emb
  line 161: repeat_kv
  line 173: eager_attention_forward
  line 206: Qwen2Attention
  line 309: Qwen2RMSNorm
  line 329: Qwen2DecoderLayer
  line 403: Qwen2Model
  line 495: Qwen2ForCausalLM
```

## 5. 模块 0：Tokenizer 与输入准备

这部分通常不放到 FPGA 上实现，保留在 CPU host。

实现位置：

```text
CPU 必须实现。
FPGA 不建议实现。
```

```text
Python 代码:
  manual_prefill_decode.py line 83
  tokenizer.apply_chat_template(...)

输入:
  metadata.json 中的 prompt / messages / chat_text

输出:
  input_ids
  attention_mask
  position_ids
```

Golden 数据：

```text
qwen25_0p5b_instruct_full_generation_trace/metadata.json
00_prefill_full_prompt/model_input_token_ids.npy
00_prefill_full_prompt/position_ids.npy
```

FPGA 分工：

```text
CPU:
  负责 tokenizer、chat template、input_ids、attention_mask、position_ids。

FPGA:
  接收 input_ids 或 embedding 后的 hidden states。
```

验证重点：

```text
1. input_ids 必须和 metadata.json 里的 input_ids 完全一致。
2. prefill position_ids 通常是 [0, 1, ..., 35]。
3. decode position_ids 要等于当前 cache 已有长度。
4. 这个模块验证的是 CPU tokenizer 结果，不是 FPGA 计算结果。
```

## 6. 模块 1：Token Embedding

```text
功能:
  把 token id 查表成 hidden 向量。

Python 源码:
  modeling_qwen2_fpga_instrumented.py line 439: record model.input_ids
  modeling_qwen2_fpga_instrumented.py line 440: record model.embedding

输入:
  input_ids: int64, shape = [B, T]
  embed_tokens.weight: shape = [V, H]

输出:
  hidden_states: shape = [B, T, H]
```

Golden 数据：

```text
00_prefill_full_prompt/model_input_token_ids.npy
00_prefill_full_prompt/token_embedding_output.npy

01_decode_token_01/model_input_token_ids.npy
01_decode_token_01/token_embedding_output.npy
```

典型 shape：

```text
prefill:
  input_ids = [1, 36]
  embedding = [1, 36, 896]

decode:
  input_ids = [1, 1]
  embedding = [1, 1, 896]
```

FPGA 实现建议：

```text
1. 如果 embedding 放 FPGA：实现 BRAM/HBM 查表，按 token id 取 896 个 fp16。
2. 如果 embedding 放 CPU：CPU 直接把 token_embedding_output 送到 FPGA。
3. 早期 demo 建议先把 embedding 作为 FPGA 输入，减少权重搬运复杂度。
```

推荐分阶段：

```text
第一阶段:
  CPU 读取/计算 token_embedding_output.npy，把它作为 FPGA 的第一份输入。
  FPGA 从 layer_00_transformer_block/transformer_block_input.npy 开始验证。

第二阶段:
  FPGA 接收 model_input_token_ids.npy，并实现 embedding lookup。
  输出对比 token_embedding_output.npy。
```

验证方法：

```text
FPGA 输出 token_embedding_output_fpga.npy
对比 golden token_embedding_output.npy
要求 shape 完全一致，max_abs_diff 接近 0。
```

## 7. 模块 2：Transformer Block 输入

每层 block 的输入来自上一层输出，第 0 层来自 embedding。

```text
Python 源码:
  modeling_qwen2_fpga_instrumented.py line 352: block.input
  modeling_qwen2_fpga_instrumented.py line 379: block.output

Golden 数据:
  {step}/layer_XX_transformer_block/transformer_block_input.npy
  {step}/layer_XX_transformer_block/transformer_block_output.npy
```

FPGA 实现建议：

```text
1. 每一层输入输出 shape 都是 [B, T, H]。
2. prefill 中 T = prompt token 数。
3. decode 中 T = 1，但 attention 内部会访问历史 KV cache。
4. 可以先实现单层 block，再在 host 侧循环调用 24 次。
```

验证方法：

```text
先固定 layer_00:
  输入 transformer_block_input.npy
  输出 transformer_block_output.npy

单层通过后，再验证:
  layer_00 output == layer_01 input
  layer_01 output == layer_02 input
```

## 8. 模块 3：Attention 前 RMSNorm

```text
功能:
  y = x / sqrt(mean(x^2) + eps) * weight

Python 源码:
  qwen2_module_reference.py line 29: rmsnorm
  modeling_qwen2_fpga_instrumented.py line 309: Qwen2RMSNorm
  modeling_qwen2_fpga_instrumented.py line 354: input_rmsnorm.input
  modeling_qwen2_fpga_instrumented.py line 356: input_rmsnorm.output

输入:
  hidden_states: [B, T, H]
  input_layernorm.weight: [H]

输出:
  attention_input_after_rmsnorm: [B, T, H]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_rmsnorm_input.npy
{step}/layer_XX_transformer_block/attention_rmsnorm_output.npy
{step}/layer_XX_transformer_block/attention_input_after_rmsnorm.npy
```

FPGA 实现建议：

```text
1. 对每个 token 的 896 维 hidden 独立做 reduce sum(x^2)。
2. mean = sum / 896。
3. rsqrt(mean + eps)，再乘 weight。
4. 建议 fp32 累加，输出 fp16 或与 golden 对齐的 dtype。
```

验证方法：

```text
输入 attention_rmsnorm_input.npy
输出对比 attention_rmsnorm_output.npy
同时确认 attention_rmsnorm_output.npy 与 attention_input_after_rmsnorm.npy 一致。
```

## 9. 模块 4：Q/K/V Linear Projection

```text
功能:
  q = x @ q_proj.weight.T + q_proj.bias
  k = x @ k_proj.weight.T + k_proj.bias
  v = x @ v_proj.weight.T + v_proj.bias

Python 源码:
  qwen2_module_reference.py line 44: linear
  qwen2_module_reference.py line 95: project_qkv
  modeling_qwen2_fpga_instrumented.py line 238: attention.q_proj
  modeling_qwen2_fpga_instrumented.py line 240: attention.k_proj
  modeling_qwen2_fpga_instrumented.py line 242: attention.v_proj

输入:
  attention_input_after_rmsnorm: [B, T, H]

输出:
  q projection: [B, T, A * D] = [B, T, 896]
  k projection: [B, T, K * D] = [B, T, 128]
  v projection: [B, T, K * D] = [B, T, 128]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_input_after_rmsnorm.npy
{step}/layer_XX_transformer_block/attention_q_projection.npy
{step}/layer_XX_transformer_block/attention_k_projection.npy
{step}/layer_XX_transformer_block/attention_v_projection.npy
```

FPGA 实现建议：

```text
1. 这是标准 GEMM / GEMV。
2. prefill: T = 36，可以按 [T, H] x [H, out] 做矩阵乘。
3. decode: T = 1，本质是 3 个向量矩阵乘。
4. Q/K/V 可融合成一次大 projection，也可先分成 3 个模块验证。
```

验证方法：

```text
先单独验证 q_proj，再验证 k_proj/v_proj。
fp16 权重 + fp32 accumulate 的结果通常比纯 fp16 accumulate 更稳定。
```

## 10. 模块 5：Reshape Head 与 RoPE

```text
功能:
  q projection reshape 到 [B, A, T, D]
  k projection reshape 到 [B, K, T, D]
  对 q/k 应用 RoPE:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

Python 源码:
  qwen2_module_reference.py line 58: apply_rotary_pos_emb
  modeling_qwen2_fpga_instrumented.py line 136: apply_rotary_pos_emb
  modeling_qwen2_fpga_instrumented.py line 247: attention.q_before_rope
  modeling_qwen2_fpga_instrumented.py line 248: attention.k_before_rope
  modeling_qwen2_fpga_instrumented.py line 252: attention.rope_cos
  modeling_qwen2_fpga_instrumented.py line 253: attention.rope_sin
  modeling_qwen2_fpga_instrumented.py line 255: attention.q_after_rope
  modeling_qwen2_fpga_instrumented.py line 256: attention.k_after_rope

输出:
  q_after_rope: [B, A, T, D]
  k_after_rope: [B, K, T, D]
  v_states: [B, K, T, D]
```

Golden 数据：

```text
{step}/rotary_embedding_cos.npy
{step}/rotary_embedding_sin.npy
{step}/layer_XX_transformer_block/attention_q_before_rope.npy
{step}/layer_XX_transformer_block/attention_k_before_rope.npy
{step}/layer_XX_transformer_block/attention_rope_cos.npy
{step}/layer_XX_transformer_block/attention_rope_sin.npy
{step}/layer_XX_transformer_block/attention_q_after_rope.npy
{step}/layer_XX_transformer_block/attention_k_after_rope.npy
{step}/layer_XX_transformer_block/attention_value_states.npy
```

FPGA 实现建议：

```text
1. RoPE 是逐元素计算，适合流式处理。
2. cos/sin 可由 CPU 预计算后传入，也可由 FPGA 查表。
3. 早期验证建议直接读取 golden cos/sin，先保证旋转公式正确。
4. 注意 q 的 head 数是 14，k/v 的 head 数是 2。
```

推荐分阶段：

```text
第一阶段:
  CPU 或测试脚本直接提供 attention_rope_cos.npy / attention_rope_sin.npy。
  FPGA 只实现 q/k 的旋转公式。

第二阶段:
  FPGA 根据 position_ids 生成或查表 cos/sin。
  再对比 attention_rope_cos.npy / attention_rope_sin.npy。
```

验证方法：

```text
输入 q_before_rope/k_before_rope + rope_cos/rope_sin
输出对比 q_after_rope/k_after_rope。
```

## 11. 模块 6：KV Cache Update

```text
功能:
  prefill: 把当前 36 个 token 的 K/V 写入 cache。
  decode: 把当前 1 个 token 的 K/V append 到历史 cache 后面。

Python 源码:
  manual_prefill_decode.py line 103: past_key_values = outputs.past_key_values
  manual_prefill_decode.py line 118: decode forward 传入 past_key_values
  manual_prefill_decode.py line 121: past_key_values = outputs.past_key_values
  modeling_qwen2_fpga_instrumented.py line 269: past_key_values.update(...)
  modeling_qwen2_fpga_instrumented.py line 270: attention.k_after_cache_update
  modeling_qwen2_fpga_instrumented.py line 271: attention.v_after_cache_update

Cache shape:
  K cache: [B, K, cache_len, D]
  V cache: [B, K, cache_len, D]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_k_after_rope.npy
{step}/layer_XX_transformer_block/attention_v_projection.npy
{step}/layer_XX_transformer_block/attention_k_cache_after_update.npy
{step}/layer_XX_transformer_block/attention_v_cache_after_update.npy
```

典型 shape：

```text
00_prefill_full_prompt/layer_00_transformer_block/attention_k_cache_after_update.npy
  [1, 2, 36, 64]

01_decode_token_01/layer_00_transformer_block/attention_k_cache_after_update.npy
  [1, 2, 37, 64]

07_decode_token_07/layer_00_transformer_block/attention_k_cache_after_update.npy
  [1, 2, 43, 64]
```

FPGA 实现建议：

```text
1. 每层维护独立 K cache 和 V cache。
2. cache 写地址由 layer_id、head_id、position_id 决定。
3. prefill 可连续写 36 个位置。
4. decode 每步只 append 一个位置，但 attention 会读取 [0, cache_len) 的全部历史。
```

验证方法：

```text
prefill:
  k_cache_after_update 应等于当前 step 的 k_after_rope 全量写入。

decode:
  k_cache_after_update 前 cache_len-1 个位置应等于上一 step cache。
  最后一个位置应等于当前 token 的 k_after_rope。
```

## 12. 模块 7：GQA Repeat

```text
功能:
  Q heads = 14
  KV heads = 2
  每个 KV head repeat 7 次，扩展到 14 个 head。

Python 源码:
  qwen2_module_reference.py line 81: repeat_kv
  modeling_qwen2_fpga_instrumented.py line 161: repeat_kv
  modeling_qwen2_fpga_instrumented.py line 186: attention.key_repeated
  modeling_qwen2_fpga_instrumented.py line 187: attention.value_repeated

输入:
  key_states: [B, 2, cache_len, 64]
  value_states: [B, 2, cache_len, 64]

输出:
  key_repeated: [B, 14, cache_len, 64]
  value_repeated: [B, 14, cache_len, 64]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_k_cache_after_update.npy
{step}/layer_XX_transformer_block/attention_v_cache_after_update.npy
{step}/layer_XX_transformer_block/attention_k_repeated_for_gqa.npy
{step}/layer_XX_transformer_block/attention_v_repeated_for_gqa.npy
```

FPGA 实现建议：

```text
1. 物理上不一定真的复制 7 份，可在 attention 读 cache 时做 head 映射。
2. q_head h 对应 kv_head = h // 7。
3. 为了和 golden 对齐，调试阶段可以先显式输出 repeated tensor。
```

验证方法：

```text
检查:
  repeated[:, 0..6, :, :] 都来自 kv head 0
  repeated[:, 7..13, :, :] 都来自 kv head 1
```

## 13. 模块 8：QK MatMul、Scale、Mask、Softmax

```text
功能:
  scores = Q @ K^T / sqrt(D)
  scores += causal/attention mask
  weights = softmax(scores)

Python 源码:
  qwen2_module_reference.py line 118: attention_core
  modeling_qwen2_fpga_instrumented.py line 173: eager_attention_forward
  modeling_qwen2_fpga_instrumented.py line 190: attention.scores_unmasked
  modeling_qwen2_fpga_instrumented.py line 193: attention.scores_masked
  modeling_qwen2_fpga_instrumented.py line 196: attention.softmax

输入:
  q_after_rope: [B, 14, T, 64]
  k_repeated: [B, 14, cache_len, 64]

输出:
  scores_before_mask: [B, 14, T, cache_len]
  scores_after_mask: [B, 14, T, cache_len]
  softmax_weights: [B, 14, T, cache_len]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_q_after_rope.npy
{step}/layer_XX_transformer_block/attention_k_repeated_for_gqa.npy
{step}/layer_XX_transformer_block/attention_qk_scores_before_mask.npy
{step}/layer_XX_transformer_block/attention_qk_scores_after_mask.npy
{step}/layer_XX_transformer_block/attention_softmax_weights.npy
{step}/layer_XX_transformer_block/attention_weights_after_dropout_or_softmax.npy
```

典型 shape：

```text
prefill:
  attention_qk_scores_after_mask = [1, 14, 36, 36]

decode token 01:
  attention_qk_scores_after_mask = [1, 14, 1, 37]

decode token 07:
  attention_qk_scores_after_mask = [1, 14, 1, 43]
```

FPGA 实现建议：

```text
1. QK matmul 是 attention 的核心计算。
2. prefill 是 T x T 三角 mask，decode 是 1 x cache_len。
3. softmax 建议使用 max-subtract 保证数值稳定:
   exp(scores - max(scores)) / sum(exp(scores - max(scores)))
4. mask 后的极小值会影响 softmax，必须和 PyTorch 行为对齐。
```

验证方法：

```text
分三段验证:
  A. qk_scores_before_mask
  B. qk_scores_after_mask
  C. attention_softmax_weights

如果 C 不一致，先看 A 是否一致，再看 mask 和 softmax 近似。
```

## 14. 模块 9：Attention Weighted Value Sum

```text
功能:
  context = softmax_weights @ V

Python 源码:
  modeling_qwen2_fpga_instrumented.py line 199: attention.value_aggregation
  modeling_qwen2_fpga_instrumented.py line 300: attention.context

输入:
  softmax_weights: [B, 14, T, cache_len]
  value_repeated: [B, 14, cache_len, 64]

输出:
  context: [B, 14, T, 64]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_softmax_weights.npy
{step}/layer_XX_transformer_block/attention_v_repeated_for_gqa.npy
{step}/layer_XX_transformer_block/attention_weighted_value_sum.npy
{step}/layer_XX_transformer_block/attention_context_before_output_projection.npy
```

FPGA 实现建议：

```text
1. decode 阶段是每个 head 做一个长度 cache_len 的加权和。
2. prefill 阶段是对每个 query position 做加权和。
3. 输出需要 reshape/transpose 回 [B, T, A * D]，也就是 [B, T, 896]。
```

验证方法：

```text
先对比 attention_weighted_value_sum.npy。
再对比 attention_context_before_output_projection.npy。
```

## 15. 模块 10：Attention Output Projection

```text
功能:
  attention_output = context @ o_proj.weight.T

Python 源码:
  qwen2_module_reference.py line 149: attention_output_projection
  modeling_qwen2_fpga_instrumented.py line 302: attention.o_proj_input
  modeling_qwen2_fpga_instrumented.py line 304: attention.o_proj

输入:
  attention_output_projection_input: [B, T, 896]

输出:
  attention_output_projection: [B, T, 896]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/attention_output_projection_input.npy
{step}/layer_XX_transformer_block/attention_output_projection.npy
{step}/layer_XX_transformer_block/attention_module_output.npy
```

FPGA 实现建议：

```text
1. 标准 Linear。
2. decode 是 GEMV，prefill 是小 batch GEMM。
3. 可和前一个 context reshape 直接串流，减少中间写回。
```

验证方法：

```text
输出应同时对齐:
  attention_output_projection.npy
  attention_module_output.npy
```

## 16. 模块 11：Attention Residual Add

```text
功能:
  hidden = residual + attention_output

Python 源码:
  modeling_qwen2_fpga_instrumented.py line 367: attention.output
  modeling_qwen2_fpga_instrumented.py line 369: attention.residual_output

输入:
  transformer_block_input: [B, T, 896]
  attention_module_output: [B, T, 896]

输出:
  attention_residual_add_output: [B, T, 896]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/transformer_block_input.npy
{step}/layer_XX_transformer_block/attention_module_output.npy
{step}/layer_XX_transformer_block/attention_residual_add_output.npy
```

FPGA 实现建议：

```text
逐元素 add，可流式实现。
```

验证方法：

```text
transformer_block_input + attention_module_output
  == attention_residual_add_output
```

## 17. 模块 12：MLP 前 RMSNorm

```text
功能:
  对 attention residual 后的 hidden 再做 RMSNorm。

Python 源码:
  modeling_qwen2_fpga_instrumented.py line 373: post_attention_rmsnorm.input
  modeling_qwen2_fpga_instrumented.py line 375: post_attention_rmsnorm.output

Golden 数据:
  {step}/layer_XX_transformer_block/mlp_rmsnorm_input.npy
  {step}/layer_XX_transformer_block/mlp_rmsnorm_output.npy
  {step}/layer_XX_transformer_block/mlp_input_after_rmsnorm.npy
```

FPGA 实现建议：

```text
和 Attention 前 RMSNorm 使用同一个硬件模块，只换 weight。
```

验证方法：

```text
输入 mlp_rmsnorm_input.npy
输出对比 mlp_rmsnorm_output.npy
```

## 18. 模块 13：SwiGLU MLP

```text
功能:
  gate = x @ gate_proj.weight.T
  up   = x @ up_proj.weight.T
  act  = silu(gate)
  mid  = act * up
  out  = mid @ down_proj.weight.T

Python 源码:
  qwen2_module_reference.py line 161: swiglu_mlp
  modeling_qwen2_fpga_instrumented.py line 36: Qwen2MLP
  modeling_qwen2_fpga_instrumented.py line 51: mlp.gate_proj
  modeling_qwen2_fpga_instrumented.py line 53: mlp.up_proj
  modeling_qwen2_fpga_instrumented.py line 55: mlp.silu
  modeling_qwen2_fpga_instrumented.py line 57: mlp.gate_up_mul
  modeling_qwen2_fpga_instrumented.py line 59: mlp.down_proj

输入:
  mlp_input_after_rmsnorm: [B, T, 896]

中间:
  gate_proj: [B, T, 4864]
  up_proj: [B, T, 4864]
  silu_gate: [B, T, 4864]
  silu_gate_times_up: [B, T, 4864]

输出:
  mlp_down_projection: [B, T, 896]
  mlp_module_output: [B, T, 896]
```

Golden 数据：

```text
{step}/layer_XX_transformer_block/mlp_input_after_rmsnorm.npy
{step}/layer_XX_transformer_block/mlp_gate_projection.npy
{step}/layer_XX_transformer_block/mlp_up_projection.npy
{step}/layer_XX_transformer_block/mlp_silu_gate_activation.npy
{step}/layer_XX_transformer_block/mlp_silu_gate_times_up.npy
{step}/layer_XX_transformer_block/mlp_down_projection.npy
{step}/layer_XX_transformer_block/mlp_module_output.npy
```

FPGA 实现建议：

```text
1. gate_proj 和 up_proj 都是 896 -> 4864 的 Linear。
2. SiLU(x) = x * sigmoid(x)，需要近似或查表实现。
3. down_proj 是 4864 -> 896 的 Linear。
4. MLP 权重和计算量很大，是 FPGA 设计的核心吞吐模块之一。
```

验证方法：

```text
按顺序逐段对比:
  gate_projection
  up_projection
  silu_gate_activation
  silu_gate_times_up
  down_projection
  mlp_module_output
```

## 19. 模块 14：MLP Residual Add 与 Block Output

```text
功能:
  block_output = attention_residual_add_output + mlp_module_output

Python 源码:
  modeling_qwen2_fpga_instrumented.py line 377: mlp.output
  modeling_qwen2_fpga_instrumented.py line 379: block.output

Golden 数据:
  {step}/layer_XX_transformer_block/attention_residual_add_output.npy
  {step}/layer_XX_transformer_block/mlp_module_output.npy
  {step}/layer_XX_transformer_block/transformer_block_output.npy
```

FPGA 实现建议：

```text
逐元素 add，可流式实现。
```

验证方法：

```text
attention_residual_add_output + mlp_module_output
  == transformer_block_output
```

## 20. 模块 15：Final RMSNorm

24 层 transformer block 结束后，模型会做最终 RMSNorm。

```text
Python 源码:
  modeling_qwen2_fpga_instrumented.py line 485: model.final_rmsnorm_input
  modeling_qwen2_fpga_instrumented.py line 487: model.final_rmsnorm_output

输入:
  layer_23_transformer_block/transformer_block_output.npy

输出:
  final_rmsnorm_output.npy
```

Golden 数据：

```text
{step}/final_rmsnorm_input.npy
{step}/final_rmsnorm_output.npy
{step}/lm_head_input.npy
```

FPGA 实现建议：

```text
复用 RMSNorm 模块，换 final norm weight。
```

验证方法：

```text
final_rmsnorm_output.npy 应与 lm_head_input.npy 对齐。
```

## 21. 模块 16：LM Head 与 Greedy Argmax

```text
功能:
  logits = hidden @ lm_head.weight.T
  next_token = argmax(logits[-1])

Python 源码:
  qwen2_module_reference.py line 188: lm_head
  qwen2_module_reference.py line 194: greedy_next_token
  modeling_qwen2_fpga_instrumented.py line 552: lm_head.input
  modeling_qwen2_fpga_instrumented.py line 556: lm_head.logits
  manual_prefill_decode.py line 104: argmax
  manual_prefill_decode.py line 122: argmax

输入:
  lm_head_input: [B, T, 896]
  lm_head.weight: [151936, 896]

输出:
  lm_head_vocab_logits: [B, T, 151936]
  next_token_id: scalar per batch
```

Golden 数据：

```text
{step}/lm_head_input.npy
{step}/lm_head_vocab_logits.npy

generation_token_vocab_logits/
  prefill_next_token_vocab_logits.npy
  decode_step_0001_next_token_vocab_logits.npy
  ...
  decode_step_0007_next_token_vocab_logits.npy
```

FPGA 实现建议：

```text
1. lm_head 是 896 -> 151936 的大矩阵乘，权重访存压力很高。
2. 早期 demo 可以只输出 logits 后在 CPU 做 argmax。
3. 真正部署时建议 FPGA 做分块矩阵乘和分块 argmax，只返回 token id。
4. 如果 embedding 和 lm_head 权重绑定，需要注意权重布局复用。
```

推荐分阶段：

```text
第一阶段:
  FPGA 不实现 lm_head，只输出 final_rmsnorm_output.npy 对齐。
  CPU/Python 用 lm_head 权重算 logits 和 argmax。

第二阶段:
  FPGA 实现 lm_head 分块矩阵乘，输出 lm_head_vocab_logits.npy。

第三阶段:
  FPGA 只返回 argmax token id。
  用 metadata.json 的 generated_ids 验证 token 是否一致。
```

验证方法：

```text
1. 先对比 lm_head_vocab_logits.npy。
2. 再只检查 argmax token 是否等于 metadata.json 中 generated_ids 对应 token。
3. 当量化后 logits 数值不完全一致时，top-1/top-k 是否一致比逐元素误差更重要。
```

## 22. 推荐 FPGA 实现顺序

第一版不要从 tokenizer 开始，也不要先追求完整端到端。建议先把 CPU/FPGA 边界固定为：

```text
CPU:
  prompt -> tokenizer -> model_input_token_ids.npy / position_ids.npy
  可选: token embedding -> token_embedding_output.npy

FPGA:
  从 layer_00_transformer_block/transformer_block_input.npy 开始
  逐模块输出自己的 *_fpga.npy
  用同名 golden .npy 做比较
```

```text
阶段 A: 单算子验证
  1. RMSNorm
  2. Linear
  3. RoPE
  4. repeat_kv
  5. softmax
  6. SiLU

阶段 B: Attention 子图
  1. q_proj/k_proj/v_proj
  2. RoPE
  3. KV cache update
  4. GQA repeat 或 head 映射读取
  5. QK matmul + mask + softmax
  6. softmax @ V
  7. o_proj
  8. residual add

阶段 C: MLP 子图
  1. post-attention RMSNorm
  2. gate_proj/up_proj
  3. SiLU + multiply
  4. down_proj
  5. residual add

阶段 D: 单层 block
  layer_00 输入 transformer_block_input.npy
  输出 transformer_block_output.npy

阶段 E: 24 层串联
  逐层检查 layer_i output == layer_{i+1} input

阶段 F: 完整 prefill
  从 token_embedding_output.npy 到 prefill logits。

阶段 G: decode with KV cache
  从 01_decode_token_01 到 07_decode_token_07 逐步验证 cache 增长和 token 输出。
```

## 23. `.npy` 验证脚本模板

建议每个 FPGA 模块都输出一个 `.npy`，再用 Python 对比。

```python
from pathlib import Path

import numpy as np





def compare_tensor(golden_path, fpga_path, rtol=1e-2, atol=1e-2):
    golden = np.load(golden_path)
    fpga = np.load(fpga_path)

    assert golden.shape == fpga.shape, (golden.shape, fpga.shape)

    golden32 = golden.astype(np.float32)
    fpga32 = fpga.astype(np.float32)
    diff = fpga32 - golden32

    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    ok = np.allclose(fpga32, golden32, rtol=rtol, atol=atol)

    denom = np.linalg.norm(golden32.ravel()) * np.linalg.norm(fpga32.ravel())
    cosine = float(np.dot(golden32.ravel(), fpga32.ravel()) / denom) if denom != 0 else 1.0

    print("golden:", golden_path)
    print("fpga:  ", fpga_path)
    print("shape: ", golden.shape)
    print("max_abs:", max_abs)
    print("mean_abs:", mean_abs)
    print("cosine: ", cosine)
    print("allclose:", ok)
    return ok





root = Path("qwen25_0p5b_instruct_full_generation_trace")
compare_tensor(
    root / "00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_output.npy",
    Path("fpga_outputs/attention_rmsnorm_output.npy"),
)
```

建议阈值：

```text
未量化 fp16 对齐:
  RMSNorm / Linear / Residual: atol=1e-2, rtol=1e-2
  Softmax / Attention: 先看 max_abs，再看 top token 是否保持一致

量化实现:
  不强求所有中间张量 allclose
  更关注每层 cosine、最终 logits top-k、最终 generated token 是否一致
```

## 24. 端到端验收标准

最小 demo 验收：

```text
输入 prompt:
  你好，请用一句话介绍你自己。

期望生成 token ids:
  [104198, 48, 16948, 3837, 67071, 102661, 99718, 100013]

期望文本:
  我是Qwen，由阿里云开发
```

推荐验收层级：

```text
1. 单模块输出能对齐对应 .npy。
2. layer_00 完整 block 输出能对齐。
3. 24 层 prefill 输出 final_rmsnorm_output/lm_head_vocab_logits 能对齐。
4. prefill argmax token 等于 generated_ids[0]。
5. decode 每一步 cache_len 和 next token 对齐。
6. 8 个 token 全部生成一致。
```

## 25. 常见风险点

```text
1. prefill 和 decode 的 T 不同:
   prefill T=36，decode T=1，但 decode attention 的 cache_len 会增长。

2. GQA 不等于 MHA:
   Q head 是 14，KV head 是 2，必须做 repeat 或 head 映射。

3. RoPE 的维度广播容易错:
   q/k shape 是 [B, heads, T, D]，cos/sin 要按 position 对齐。

4. RMSNorm 累加精度:
   建议 fp32 accumulate，否则误差可能从第一层开始放大。

5. Softmax 近似:
   attention 中最容易出现数值偏差，建议先使用高精度实现对齐，再做近似优化。

6. LM head 权重巨大:
   151936 vocab 会带来明显带宽压力，端到端 demo 可以先 host argmax，后续再硬化。

7. 文件名与模块名:
   当前 trace 已经把 0001/0002 这类编号改成语义化名称，优先使用这些语义化文件名。
```

## 26. 一句话总结

下一步是从 `RMSNorm -> Linear -> RoPE -> Attention -> MLP -> 单层 Block -> 24 层 Prefill -> Decode KV Cache` 逐级实现 FPGA 模块；每一级都直接读取本目录下的 `.npy` 作为输入和 golden 输出，保证模块级误差可定位、可复现、可回归。