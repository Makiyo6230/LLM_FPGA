# LLM_FPGA

中文 | [English](#english-version)

## 中文版本

这是一个面向 FPGA 大模型推理验证的 Qwen2.5-0.5B-Instruct 参考工程。

本仓库收集了 FPGA 按模块实现和验证所需的三类内容：

1. Qwen2.5-0.5B-Instruct 完整 PyTorch golden trace。
2. 从 prompt 到输出 token 的 Python 推理参考代码，以及模块级数学参考实现。
3. FlightLLM FPGA LLM demo 工程，作为 FPGA 工程组织和部署流程参考。

当前目标不是搭建生产推理服务，而是建立一个 correctness-first 的开发闭环：先导出 PyTorch 中间量，再用 FPGA/C++/HLS 实现相同模块，并把每个模块输出和 golden `.npy` 文件逐项比较。

### 仓库结构

```text
LLM_FPGA/
  README.md
  Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md
  Qwen2.5-0.5B_FPGA模块拆分与验证指南.md
  qwen25_0p5b_instruct_full_generation_trace/
  qwen25_0p5b_python_reference/
```

关键文档：

- `Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md`：从工程角度规划 FPGA 推理 demo 的阶段路线。
- `Qwen2.5-0.5B_FPGA模块拆分与验证指南.md`：逐模块说明 Python 代码、FPGA 实现任务和 `.npy` golden 验证方法。

### Golden Trace 数据目录

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

含义：

- `00_prefill_full_prompt`：完整 prompt 的 prefill forward。
- `01_decode_token_01` 到 `07_decode_token_07`：逐 token decode。
- `generation_token_vocab_logits`：每一步用于选择生成 token 的 vocab logits。
- `metadata.json`：prompt、input ids、generated ids、输出文本和 trace 元信息。
- `name_mapping.json`：原始 recorder 文件名与 FPGA 友好文件名的映射。
- `flightllm_test_demo`：FlightLLM FPGA LLM demo 参考工程。

当前 trace 摘要：

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

模型结构：

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

### Python 推理参考代码

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

文件说明：

- `run_full_generate_trace.py`：基于 HuggingFace `model.generate()` 的完整 trace 导出脚本。
- `manual_prefill_decode.py`：显式 prefill/decode 循环，不调用 `generate()`，更接近 FPGA runtime。
- `qwen2_module_reference.py`：模块级 PyTorch 参考实现，包括 RMSNorm、Linear、RoPE、GQA repeat、Attention、SwiGLU MLP、lm_head。
- `snapshots/modeling_qwen2_fpga_instrumented.py`：插桩后的 Qwen2 模型源码快照。
- `snapshots/fpga_recorder.py`：负责保存 `.npy` 中间量的 recorder。

### 在 H200 上复现推理

已验证的 H200 环境：

```text
conda env = /data3/output/conda_envs/qwen25_fpga_transformers
model path = /data3/models/Qwen2.5-0.5B-Instruct
instrumented transformers = /data3/output/transformers
```

运行显式 prefill/decode：

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

预期输出：

```text
response = 我是Qwen，由阿里云开发
generated_ids = [104198, 48, 16948, 3837, 67071, 102661, 99718, 100013]
```

`model.generate()` 路径也已经验证：

```bash
/data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
  run_full_generate_trace.py \
  --model-path /data3/models/Qwen2.5-0.5B-Instruct \
  --record-dir /data3/output/qwen25_fpga_golden/cases/generate_code_test_001 \
  --max-new-tokens 8
```

两条路径生成的 token ids 完全一致。

### Golden Tensor 示例

prefill 输入：

```text
00_prefill_full_prompt/model_input_token_ids.npy
shape = (1, 36)
```

第 0 层 prefill attention：

```text
00_prefill_full_prompt/layer_00_transformer_block/attention_input_after_rmsnorm.npy
shape = (1, 36, 896)

00_prefill_full_prompt/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 36, 36)

00_prefill_full_prompt/layer_00_transformer_block/attention_k_cache_after_update.npy
shape = (1, 2, 36, 64)
```

decode 第 1 步：

```text
01_decode_token_01/model_input_token_ids.npy
shape = (1, 1)

01_decode_token_01/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 1, 37)
```

decode 第 7 步：

```text
07_decode_token_07/layer_00_transformer_block/attention_qk_scores_after_mask.npy
shape = (1, 14, 1, 43)
```

vocab logits：

```text
generation_token_vocab_logits/prefill_next_token_vocab_logits.npy
shape = (1, 151936)
```

### FPGA 模块实现顺序

建议第一轮按照下面的顺序实现和验证：

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

建议第一个验证目标：

```text
00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_input.npy
00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_output.npy
```

先做 layer 0 的 prefill，再做 layer 0 的 decode，最后扩展到全部 24 层。

### 注意事项

- 本仓库保存的是小模型 demo trace，不是完整 Qwen2.5-7B trace。
- 当前模型是 `Qwen2.5-0.5B-Instruct`。
- golden tensor 用于 correctness verification，性能优化应放在结果对齐之后。
- 仓库中包含一个约 80 MB 的 FlightLLM bitstream 文件，低于 GitHub 100 MB 硬限制，但高于 50 MB 推荐值。
- 如果后续加入更多大规模 trace，建议使用 Git LFS 或外部 artifact storage。

---

## English Version

This is a Qwen2.5-0.5B-Instruct reference project for FPGA LLM inference verification.

The repository contains three kinds of artifacts needed for module-by-module FPGA implementation and verification:

1. A full PyTorch golden trace for Qwen2.5-0.5B-Instruct generation.
2. Python reference code for prompt-to-token inference and module-level math.
3. A FlightLLM FPGA LLM demo project as an engineering reference.

The current goal is not to build a production inference server. The goal is a correctness-first workflow: export PyTorch intermediate tensors, implement the same modules in FPGA/C++/HLS, and compare every module output against the golden `.npy` files.

### Repository Layout

```text
LLM_FPGA/
  README.md
  Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md
  Qwen2.5-0.5B_FPGA模块拆分与验证指南.md
  qwen25_0p5b_instruct_full_generation_trace/
  qwen25_0p5b_python_reference/
```

Key documents:

- `Qwen2.5-0.5B-Instruct_FPGA推理实现路线.md`: engineering roadmap for the FPGA inference demo.
- `Qwen2.5-0.5B_FPGA模块拆分与验证指南.md`: module-by-module mapping from Python code to FPGA tasks and `.npy` golden verification.

### Golden Trace Directory

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

### Python Inference Reference

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

### Reproducing the Python Inference on H200

Validated H200 environment:

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

### Golden Tensor Examples

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

### FPGA Implementation Order

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

### Notes

- This repository stores a small-model demo trace, not the full Qwen2.5-7B trace.
- The current model is `Qwen2.5-0.5B-Instruct`.
- The golden tensors are intended for correctness verification before performance tuning.
- One included FlightLLM bitstream file is about 80 MB, which is below GitHub's 100 MB hard limit but above the 50 MB recommendation.
- If more large traces are added later, Git LFS or external artifact storage is recommended.
