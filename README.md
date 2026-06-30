# LLM_FPGA

Qwen2.5-0.5B-Instruct FPGA inference reference project.

This repository collects three things needed for module-by-module FPGA
implementation and verification:

1. A full PyTorch golden trace for Qwen2.5-0.5B-Instruct generation.
2. Python reference code for prompt-to-token inference and module-level math.
3. A FlightLLM FPGA LLM demo project as an engineering reference.

The current target is not a production server. The goal is to build a
correctness-first workflow: export PyTorch intermediate tensors, implement the
same modules in FPGA/C++/HLS, and compare every module output against the
golden `.npy` files.

## Repository Layout

```text
LLM_FPGA/
  README.md
  Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md
  qwen25_0p5b_instruct_full_generation_trace/
  qwen25_0p5b_python_reference/
```

### `qwen25_0p5b_instruct_full_generation_trace`

This directory contains the exported golden trace.

```text
qwen25_0p5b_instruct_full_generation_trace/
  00_prefill_full_prompt/
  01_decode_token_01/
  02_decode_token_02/
  ...
  07_decode_token_07/
  generation_token_vocab_logits/
  metadata.json
  name_mapping.json
  flightllm_test_demo/
```

Meaning:

- `00_prefill_full_prompt`: full prompt prefill forward.
- `01_decode_token_01` to `07_decode_token_07`: one-token decode steps.
- `generation_token_vocab_logits`: vocab logits used to select generated tokens.
- `metadata.json`: prompt, token ids, generated ids, output text, and trace metadata.
- `name_mapping.json`: mapping between raw recorder names and FPGA-friendly names.
- `flightllm_test_demo`: reference FPGA LLM demo from FlightLLM.

Current trace summary:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
prompt = 你好，请用一句话介绍你自己。
input_token_count = 36
max_new_tokens = 8
decode = greedy
response = 我是Qwen，由阿里云开发
files = 7386
step_dirs = 8
layer_dirs = 192
```

Key model shape:

```text
layers = 24
hidden_size = 896
num_attention_heads = 14
num_key_value_heads = 2
head_dim = 64
vocab_size = 151936
attention = GQA
norm = RMSNorm
position = RoPE
mlp = SwiGLU
```

### `qwen25_0p5b_python_reference`

This directory contains the Python code used to understand and reproduce the
trace.

```text
qwen25_0p5b_python_reference/
  README.md
  run_full_generate_trace.py
  manual_prefill_decode.py
  qwen2_module_reference.py
  snapshots/
    modeling_qwen2_fpga_instrumented.py
    fpga_recorder.py
```

Important files:

- `run_full_generate_trace.py`: HuggingFace `model.generate()` based trace export.
- `manual_prefill_decode.py`: explicit prefill/decode loop, closest to FPGA runtime.
- `qwen2_module_reference.py`: small module-level PyTorch reference functions.
- `snapshots/modeling_qwen2_fpga_instrumented.py`: instrumented Qwen2 model source.
- `snapshots/fpga_recorder.py`: tensor recorder that writes `.npy` files.

## Reproducing the Python Inference

The H200 server environment used for validation:

```text
conda env = /data3/output/conda_envs/qwen25_fpga_transformers
model path = /data3/models/Qwen2.5-0.5B-Instruct
instrumented transformers = /data3/output/transformers
```

Run the explicit prefill/decode loop:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
export PYTHONIOENCODING=utf-8

cd /data3/output/LLM_FPGA/qwen25_0p5b_python_reference

/data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
  manual_prefill_decode.py \
  --model-path /data3/models/Qwen2.5-0.5B-Instruct \
  --record-dir /data3/output/qwen25_fpga_golden/cases/manual_code_test_001 \
  --max-new-tokens 8 \
  --save-scores
```

Expected output:

```text
response = 我是Qwen，由阿里云开发
generated_ids = [104198, 48, 16948, 3837, 67071, 102661, 99718, 100013]
```

The `model.generate()` path was also tested:

```bash
/data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
  run_full_generate_trace.py \
  --model-path /data3/models/Qwen2.5-0.5B-Instruct \
  --record-dir /data3/output/qwen25_fpga_golden/cases/generate_code_test_001 \
  --max-new-tokens 8
```

Both paths produced the same generated token ids.

## Golden Tensor Examples

Prefill input:

```text
00_prefill_full_prompt/model_input_token_ids.npy
shape = (1, 36)
```

Layer 0 prefill attention:

```text
00_prefill_full_prompt/layer_00_transformer_block/attention_input_after_rmsnorm.npy
shape = (1, 36, 896)

00_prefill_full_prompt/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 36, 36)

00_prefill_full_prompt/layer_00_transformer_block/attention_k_cache_after_update.npy
shape = (1, 2, 36, 64)
```

Decode step 1:

```text
01_decode_token_01/model_input_token_ids.npy
shape = (1, 1)

01_decode_token_01/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 1, 37)
```

Decode step 7:

```text
07_decode_token_07/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 1, 43)
```

Vocab logits:

```text
generation_token_vocab_logits/prefill_next_token_vocab_logits.npy
shape = (1, 151936)
```

## FPGA Implementation Order

Recommended first-pass module order:

```text
1. RMSNorm
2. Linear Q/K/V projection
3. RoPE
4. KV cache update
5. GQA KV repeat
6. QK matmul and scale
7. causal mask
8. softmax
9. attention weighted value sum
10. attention output projection
11. residual add
12. MLP RMSNorm
13. MLP gate/up projection
14. SiLU
15. gate * up
16. MLP down projection
17. final RMSNorm
18. lm_head logits
```

Suggested first verification target:

```text
00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_input.npy
00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_output.npy
```

Start with layer 0 prefill, then layer 0 decode, then expand to all 24 layers.

## Notes

- This repository stores a small-model demo trace, not the full Qwen2.5-7B trace.
- The current model is `Qwen2.5-0.5B-Instruct`.
- The golden tensors are intended for correctness verification before performance tuning.
- One included FlightLLM bitstream file is about 80 MB, which is below GitHub's
  100 MB hard limit but above the 50 MB recommendation.
- If more large traces are added later, Git LFS or external artifact storage is recommended.
