# Qwen2.5-0.5B-Instruct FPGA 推理实现路线

## 目标

当前阶段目标是先用 `Qwen2.5-0.5B-Instruct` 建立一套可验证、可复现、可逐模块对齐的 FPGA 推理开发闭环。

这套闭环包括：

1. 使用修改后的 HuggingFace Transformers 源码导出 PyTorch golden trace。
2. 覆盖完整生成流程：prompt 输入、prefill、逐 token decode、KV cache 增长、lm_head logits、最终文本输出。
3. 将每个 Transformer 模块的输入输出保存为 `.npy`，作为 FPGA/C++ 模块验证基准。
4. 先在 0.5B 模型上验证模块、数据格式、误差标准和目录组织，再迁移到更大的 Qwen2.5 模型。

## 当前数据与代码位置

本地最终数据目录：

```text
C:\Users\makiyo\Desktop\每日思考\LLM_FPGA\qwen25_0p5b_instruct_full_generation_trace
```

该目录包含：

```text
00_prefill_full_prompt
01_decode_token_01
02_decode_token_02
03_decode_token_03
04_decode_token_04
05_decode_token_05
06_decode_token_06
07_decode_token_07
generation_token_vocab_logits
metadata.json
name_mapping.json
flightllm_test_demo
```

其中：

- `00_prefill_full_prompt`：完整 prompt 的 prefill 阶段。
- `01_decode_token_01` 到 `07_decode_token_07`：逐 token decode 阶段。
- `generation_token_vocab_logits`：每一步生成 token 对应的 vocab logits。
- `flightllm_test_demo`：FlightLLM FPGA LLM demo 参考工程。

服务器原始环境：

```text
/data3/output/conda_envs/qwen25_fpga_transformers
/data3/models/Qwen2.5-0.5B-Instruct
/data3/output/qwen25_fpga_golden/cases/generate_002
/data3/output/qwen25_fpga_golden/cases/generate_002_named
```

本地 Transformers 源码：

```text
C:\Users\makiyo\Desktop\每日思考\transformers
```

已修改文件：

```text
src/transformers/utils/fpga_recorder.py
src/transformers/models/qwen2/modeling_qwen2.py
```

## 模型配置

当前 demo 模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

关键结构参数：

```text
architecture = Qwen2ForCausalLM
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

当前 trace 的生成配置：

```text
prompt = 你好，请用一句话介绍你自己。
input_token_count = 36
max_new_tokens = 8
decode = greedy
do_sample = False
use_cache = True
attn_implementation = eager
response = 我是Qwen，由阿里云开发
```

## 已导出的 Golden Trace

本地 trace 已核对：

```text
files = 7386
step_dirs = 8
layer_dirs = 192
```

其中 `192 = 8 steps * 24 layers`。

关键 shape 抽查：

```text
00_prefill_full_prompt/model_input_token_ids.npy
  shape = (1, 36)

00_prefill_full_prompt/layer_00_transformer_block/attention_input_after_rmsnorm.npy
  shape = (1, 36, 896)

00_prefill_full_prompt/layer_00_transformer_block/attention_qk_scores_after_mask.npy
  shape = (1, 14, 36, 36)

00_prefill_full_prompt/layer_00_transformer_block/attention_k_cache_after_update.npy
  shape = (1, 2, 36, 64)

01_decode_token_01/model_input_token_ids.npy
  shape = (1, 1)

01_decode_token_01/layer_00_transformer_block/attention_qk_scores_after_mask.npy
  shape = (1, 14, 1, 37)

07_decode_token_07/layer_00_transformer_block/attention_qk_scores_after_mask.npy
  shape = (1, 14, 1, 43)

generation_token_vocab_logits/prefill_next_token_vocab_logits.npy
  shape = (1, 151936)
```

## 模块命名与 FPGA 对齐入口

每个 step 下有 24 个层目录：

```text
layer_00_transformer_block
layer_01_transformer_block
...
layer_23_transformer_block
```

每层关键中间量：

```text
transformer_block_input.npy
attention_rmsnorm_input.npy
attention_rmsnorm_output.npy
attention_input_after_rmsnorm.npy
attention_q_projection.npy
attention_k_projection.npy
attention_v_projection.npy
attention_q_before_rope.npy
attention_k_before_rope.npy
attention_q_after_rope.npy
attention_k_after_rope.npy
attention_k_cache_after_update.npy
attention_v_cache_after_update.npy
attention_k_repeated_for_gqa.npy
attention_v_repeated_for_gqa.npy
attention_qk_scores_before_mask.npy
attention_qk_scores_after_mask.npy
attention_softmax_weights.npy
attention_weighted_value_sum.npy
attention_output_projection.npy
attention_residual_add_output.npy
mlp_rmsnorm_input.npy
mlp_rmsnorm_output.npy
mlp_gate_projection.npy
mlp_up_projection.npy
mlp_silu_gate_activation.npy
mlp_silu_gate_times_up.npy
mlp_down_projection.npy
mlp_module_output.npy
transformer_block_output.npy
```

全局 final 阶段：

```text
final_rmsnorm_input.npy
final_rmsnorm_output.npy
lm_head_input.npy
lm_head_vocab_logits.npy
```

## 大任务划分

### 1. Golden Trace 固化

目标：确认当前 0.5B trace 可以作为后续 FPGA 对齐基准。

需要完成：

1. 固定 prompt、decode 配置和 tokenizer 版本。
2. 固定目录结构和文件命名。
3. 使用 `metadata.json` 记录 input ids、generated ids、response、模型路径、dtype 和生成配置。
4. 使用 `name_mapping.json` 记录原始 recorder 名称和当前 FPGA 友好名称的对应关系。
5. 选定第一批对齐 case：优先 `00_prefill_full_prompt/layer_00_transformer_block`。

### 2. Tensor 读取与对比工具

目标：写一个独立工具，读取 FPGA 输出和 golden `.npy` 做误差比较。

建议工具：

```text
compare_tensor.py
```

基础指标：

```text
max_abs_error
mean_abs_error
max_relative_error
cosine_similarity
top1_token_match
top5_overlap
```

建议阈值：

```text
FP16/BF16:
  max_abs_error <= 1e-2
  cosine_similarity >= 0.999

INT8/W8A8:
  cosine_similarity >= 0.99
  top5_overlap >= 80%
```

### 3. 单模块 FPGA/C++ Reference

优先实现顺序：

1. RMSNorm
2. Linear projection
3. RoPE
4. GQA KV repeat
5. QK matmul
6. causal mask
7. softmax
8. attention weighted value sum
9. output projection
10. residual add
11. SwiGLU MLP
12. final RMSNorm
13. lm_head logits

每个模块都应读取 golden 输入 `.npy`，输出结果后与对应 golden `.npy` 比较。

### 4. 单层 Transformer Block

目标：完成 `layer_00_transformer_block` 的完整前向链路。

推荐顺序：

1. `attention_rmsnorm_input` 到 `attention_rmsnorm_output`
2. Q/K/V projection
3. RoPE
4. KV cache update
5. GQA repeat
6. attention score
7. softmax
8. value aggregation
9. output projection
10. attention residual add
11. MLP RMSNorm
12. SwiGLU MLP
13. block output

先对齐 prefill，再对齐 decode。

### 5. 多层与完整 Decode

目标：从单层扩展到 24 层，并完成连续 decode。

阶段：

1. 跑通 `00_prefill_full_prompt/layer_00`。
2. 跑通 prefill 的 24 层。
3. 跑通 `01_decode_token_01/layer_00`。
4. 跑通 decode 的 24 层。
5. 跑通 `01_decode_token_01` 到 `07_decode_token_07`。
6. 对齐每一步 `generation_token_vocab_logits`。

### 6. FlightLLM Demo 参考

参考工程位置：

```text
LLM_FPGA/qwen25_0p5b_instruct_full_generation_trace/flightllm_test_demo
```

用途：

1. 参考 FPGA LLM demo 的工程组织方式。
2. 参考 host runtime、kernel 调用和数据搬运方式。
3. 不直接假设它支持 Qwen2.5，需要逐项核对模型结构、权重格式、量化方式和 runtime 接口。

### 7. 权重导出与量化

当前 trace 主要是激活中间量，还需要为 FPGA 准备权重侧数据。

后续需要导出：

```text
embedding weight
每层 RMSNorm weight
q_proj/k_proj/v_proj/o_proj weight 和 bias
mlp gate/up/down weight
final RMSNorm weight
lm_head weight
```

量化路线建议：

1. 先 FP16/BF16 reference 对齐。
2. 再做 W8A8/INT8。
3. 暂不把 W4A8 作为第一阶段目标。

### 8. 迁移到更大的 Qwen2.5 模型

0.5B 路线验证完成后，迁移到更大参数规模模型时需要替换：

1. 模型权重路径。
2. `metadata.json` 中的结构参数。
3. layer 数、hidden size、head 数、KV head 数、intermediate size。
4. golden trace 导出数据。
5. FPGA HBM layout 和 kernel tile 参数。

迁移原则：目录结构、文件命名、对比工具和验证流程保持一致。

## 第一阶段验收标准

当前 0.5B 阶段完成标准：

1. 本地 golden trace 路径固定在 `LLM_FPGA/qwen25_0p5b_instruct_full_generation_trace`。
2. `metadata.json` 和 `name_mapping.json` 可用于还原输入、输出和命名关系。
3. prefill 与 decode 的每层关键 tensor 均可读取。
4. 第 0 层 RMSNorm、Linear、RoPE、Attention、MLP 均能被独立 C++/FPGA reference 对齐。
5. 至少完成一个完整 `layer_00_transformer_block` 的端到端对齐。
6. 最终 logits 的 top1 token 与 PyTorch golden 一致。

## 注意事项

1. `generate_001` 因中文编码问题无效，不作为中文 trace 使用。
2. 当前有效 trace 来源是 `generate_002`。
3. 当前模型是 `Qwen2.5-0.5B-Instruct`，不是 7B。
4. `00_prefill_full_prompt` 是完整 prompt 输入，不是单 token decode。
5. `01_decode_token_01` 之后每步只输入一个 token，但 attention 中会使用历史 KV cache。
6. 后续新增 case 时，应复用当前目录命名体系。

