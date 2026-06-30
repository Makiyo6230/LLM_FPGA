# Qwen2.5-0.5B Python Inference Reference

这个目录保存完整 Python 推理参考代码，作用是把当前 golden trace 背后的推理流程固定下来，方便后续 FPGA 按模块实现。

## 文件说明

```text
run_full_generate_trace.py
manual_prefill_decode.py
qwen2_module_reference.py
snapshots/modeling_qwen2_fpga_instrumented.py
snapshots/fpga_recorder.py
```

含义：

- `run_full_generate_trace.py`：完整推理入口，从 prompt 到 `model.generate()`，保存 `metadata.json` 和每步 vocab logits。
- `manual_prefill_decode.py`：显式 prefill/decode 推理循环，不调用 `generate()`，直接展示 FPGA runtime 需要复现的逐 token 流程。
- `qwen2_module_reference.py`：模块级 PyTorch reference，拆出 RMSNorm、Linear、RoPE、GQA repeat、Attention、SwiGLU MLP、lm_head。
- `snapshots/modeling_qwen2_fpga_instrumented.py`：我们插桩后的 Qwen2 Transformers 源码快照。
- `snapshots/fpga_recorder.py`：保存 `.npy` 中间量的 recorder 源码快照。

## 对应数据目录

当前本地 golden trace：

```text
../qwen25_0p5b_instruct_full_generation_trace
```

其中：

```text
00_prefill_full_prompt
01_decode_token_01
...
07_decode_token_07
generation_token_vocab_logits
metadata.json
name_mapping.json
```

## 在 H200 上复现导出

```bash
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
export QWEN2_FPGA_RECORD=1

/data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
  run_full_generate_trace.py \
  --model-path /data3/models/Qwen2.5-0.5B-Instruct \
  --record-dir /data3/output/qwen25_fpga_golden/cases/generate_reproduce \
  --max-new-tokens 8
```

注意：环境中的 Transformers 必须是已经插桩的 `/data3/output/transformers` editable install。

如果要看更接近 FPGA runtime 的显式循环：

```bash
/data3/output/conda_envs/qwen25_fpga_transformers/bin/python \
  manual_prefill_decode.py \
  --model-path /data3/models/Qwen2.5-0.5B-Instruct \
  --record-dir /data3/output/qwen25_fpga_golden/cases/manual_reproduce \
  --max-new-tokens 8 \
  --save-scores
```

## FPGA 模块实现建议顺序

参考 `qwen2_module_reference.py` 中的 `layer_flow_order()`：

```text
attention_rmsnorm
qkv_projection
rope
kv_cache_update
gqa_kv_repeat
qk_matmul_scale
causal_mask
softmax
weighted_value_sum
attention_output_projection
attention_residual_add
mlp_rmsnorm
mlp_gate_up_projection
silu
gate_times_up
mlp_down_projection
mlp_residual_add
```

建议第一个对齐模块：

```text
../qwen25_0p5b_instruct_full_generation_trace/00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_input.npy
../qwen25_0p5b_instruct_full_generation_trace/00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_output.npy
```

先做 RMSNorm，再做 Q/K/V Linear 和 RoPE。这样每一步都能直接用 golden `.npy` 验证。
